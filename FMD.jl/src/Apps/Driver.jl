# src/Apps/Driver.jl — machinery shared by every per-step driver.
#
# `npt!` (NPT.jl) and `relax!` (Relax.jl) differ in what they do with the
# forces, not in how they obtain them: both resolve a backend, distribute
# the frame, build the same Coulomb/MLIP/NLH neighbour-list family, and
# rebuild it whenever the skin budget runs out.  That common skeleton lives
# here so a fix to the (intricate) DD rebuild path lands in one place
# instead of two (Issue #207).
#
# What stays with the individual drivers: anything that reads the physics
# of the run — initial velocities, thermostat/barostat, the shell-history
# predictor, the gauge/dipole reporting, and the log format.

using FMD
using FMD.Cores
using FMD.Calculators
using CUDA
using Adapt
using MPI

"""
    conv(arrtyp, val)

Internal conversion used to push numeric `params` entries to the device
array type `arrtyp` at the working precision [`FPPREC`](@ref). Scalars /
arrays of floats are cast and arrays are moved to `arrtyp`. Non-numeric
values are returned unchanged so that `params` can mix arrays, tuples, and
exotic objects (closures, `SplineParam`, `ACEParam`, etc.).

The `arrtyp(FPPREC.(val))` constructor preserves the element type — unlike
`CUDA.cu`, which silently downcasts `Float64` arrays to `Float32`
(Issue #39).
"""
function conv(arrtyp, val)
    return val
end
function conv(arrtyp, val::AbstractArray)
    return arrtyp(FPPREC.(val))
end
function conv(arrtyp, val::AbstractFloat)
    return FPPREC.(val)
end

"""
    _arrtyp(params) -> UnionAll

Resolve the device array type for a driver run from `params[:backend]`:
`:cuda` (the default) selects `CuArray`, `:cpu` selects `Array`. Any other
value errors instead of silently falling back.
"""
function _arrtyp(params)
    backend = get(params, :backend, :cuda)
    if backend === :cuda
        CUDA.functional() ||
            error("params[:backend] = :cuda (the default) but CUDA.functional() == false — " *
                  "pass params[:backend] = :cpu for host execution")
        return CuArray
    end
    backend === :cpu && return Array
    error("params[:backend] must be :cuda or :cpu (got $(repr(backend)))")
end

# ---------------------------------------------------------------------------
#  MLIP type dispatch
#
#  The drivers support both ACE (`ACEParam`) and Spline (`SplineParam`)
#  MLIPs via these helpers.  Spline needs two neighbor lists (one per body
#  order); ACE needs one.  Both expose a `calculate!(mlip, atoms,
#  nlists...)` method.  `nothing` (no MLIP — the pure shell model) is a
#  third, trivial case: every helper degenerates to a no-op.
# ---------------------------------------------------------------------------

"""
    build_mlip_nlists(atoms, parent_nlist, mlip, skin) -> Tuple{NeighborList,...}

Build the neighbour list(s) that `mlip` needs to evaluate forces. Returns a
1-tuple for `ACEParam` (single cutoff `mlip.rmax`), a 2-tuple
`(nl2, nl3)` for `SplineParam` (separate 2-/3-body cutoffs), and the
empty tuple for `nothing` (no MLIP). All sublists are derived from
`parent_nlist` (the Coulomb cutoff list) so the cell-list construction is
shared.
"""
build_mlip_nlists(atoms, parent_nlist, mlip::ACEParam, skin) =
    (neighborlist(atoms, parent_nlist, mlip.rmax, skin),)

function build_mlip_nlists(atoms, parent_nlist, mlip::SplineParam, skin)
    nl2 = neighborlist(atoms, parent_nlist, mlip.rmax2, skin)
    nl3 = neighborlist(atoms, parent_nlist, mlip.rmax3, skin)
    return (nl2, nl3)
end

# No MLIP: nothing to build. The empty tuple also makes
# `mlip_nlists_reusable` vacuously true, so the MLIP-only rebuild branch
# in [`rebuild_lists!`](@ref) never fires.
build_mlip_nlists(atoms, parent_nlist, ::Nothing, skin) = ()

"""
    mlip_nlists_reusable(nlists, atoms) -> Bool

True iff every MLIP neighbour list in `nlists` is still valid given the
current atomic positions (skin still covers max displacement).
"""
mlip_nlists_reusable(nlists::Tuple, atoms) =
    all(isreusable(nl, atoms) for nl in nlists)

"""
    build_mlip_workspace(atoms, params) -> ACEWorkspace | nothing

Persistent evaluation workspace for the MLIP, built once by
[`build_force_stack`](@ref) (after `atoms` is on the device) and reused
every step. ACE gets an [`ACEWorkspace`](@ref) — sized from the host-side
`params[:acep]` and `length(atoms)` — which removes the ~10 device-array
allocations per `calculate!` call. Spline needs none.

Dispatches on `params` (not the device `mlip`) because the workspace is
built from the host-side `ACEParam`: the GEMM refactor preprocesses the
coefficients on the host, which needs host arrays — the `adapt`-converted
device copy cannot provide them.
"""
build_mlip_workspace(atoms, params) =
    haskey(params, :acep) ?
        ACEWorkspace(Cores.get_backend(atoms), params[:acep], length(atoms)) :
        nothing

"""
    apply_mlip!(atoms, mlip, nlists, ws)

Splat the MLIP-specific neighbour-list tuple into [`calculate!`](@ref)
so that ACE (1 list + [`ACEWorkspace`](@ref)), Spline (2 lists, no
workspace), and no-MLIP (`nothing`; a no-op) share a single call site.
"""
apply_mlip!(atoms, mlip, nlists::Tuple, ::Nothing) = calculate!(mlip, atoms, nlists...)
apply_mlip!(atoms, mlip::ACEParam, nlists::Tuple, ws::ACEWorkspace) =
    calculate!(mlip, atoms, nlists[1], ws)
apply_mlip!(atoms, ::Nothing, ::Tuple{}, ::Nothing) = nothing

"""
    build_nlh_nlist(atoms, parent_nlist, nlh, skin) -> NeighborList | nothing

Neighbour list for the NLH baseline ([`NLHParam`](@ref)), derived from
`parent_nlist` (the Coulomb-cutoff list) at `nlh.rc`. `nothing` when no
baseline is configured. Kept separate from the MLIP lists because the
baseline is fixed at fit time while the MLIP is what is being fitted.
"""
build_nlh_nlist(atoms, parent_nlist, nlh::NLHParam, skin) =
    neighborlist(atoms, parent_nlist, nlh.rc, skin)
build_nlh_nlist(atoms, parent_nlist, ::Nothing, skin) = nothing

nlh_nlist_reusable(nl::NeighborList, atoms) = isreusable(nl, atoms)
nlh_nlist_reusable(::Nothing, atoms) = true

"""
    apply_nlh!(atoms, nlh, nlist)

Add the NLH baseline to the force stack; a no-op when no baseline is
configured. Called at the same point as [`apply_mlip!`](@ref) — after the
shells are relaxed, since the term is core–core and does not enter the
shell CG (so the XL-BOMD branch is covered by the same call site).
"""
apply_nlh!(atoms, nlh::NLHParam, nlist::NeighborList) = calculate!(nlh, atoms, nlist)
apply_nlh!(atoms, ::Nothing, ::Nothing) = nothing

"""
    _resolve_nlh(params, arrtyp)

Pick up the NLH baseline from `params[:nlh]` (typically splatted in by
[`loadpotential`](@ref), which rebuilds it from the coefficients the fit
actually subtracted) and move it to the device. `nothing` when the loaded
potential carries no baseline.
"""
_resolve_nlh(params, arrtyp) =
    haskey(params, :nlh) ? adapt(arrtyp, params[:nlh]) : nothing

"""
    _resolve_mlip(params, arrtyp)

Pick the MLIP to use from `params`. Accepts at most one of `:acep`
([`ACEParam`](@ref)) or `:splp` ([`SplineParam`](@ref)) and moves it to the
device via `adapt(arrtyp, ...)`, which preserves the coefficient element
type — the previous `cu(...)` silently downcast `Float64` coefficients to
`Float32` (Issue #39). Throws when both are provided (there is no
principled way to choose); returns `nothing` when neither is — the pure
shell model (Coulomb + Buckingham + Spring) with no MLIP term
(Issue #133).
"""
function _resolve_mlip(params, arrtyp)
    if haskey(params, :acep) && haskey(params, :splp)
        error("at most one of params[:acep] (ACEParam) or params[:splp] " *
              "(SplineParam) may be given — both were provided")
    elseif haskey(params, :acep)
        return adapt(arrtyp, params[:acep])
    elseif haskey(params, :splp)
        return adapt(arrtyp, params[:splp])
    else
        return nothing
    end
end

# ---------------------------------------------------------------------------
#  Run entry
# ---------------------------------------------------------------------------

"""
    _assign_local_gpu!(caller::String)

Rank → GPU binding for GPU domain decomposition: node-local rank (via the
standard shared-memory split; assumes the domain lives on MPI.COMM_WORLD,
as MPIDomain's default does) modulo the visible device count. Must run
before the first CUDA allocation of the run — every allocation after this
lands on the bound device. Identity on single-GPU nodes. Limitation:
device buffers the CALLER allocated before the driver (a GPU-built
umbrella, for instance) are not rebound — on multi-GPU nodes either
construct them after MPI.Init with per-rank CUDA_VISIBLE_DEVICES, or run
one rank per GPU with the launcher pinning visibility (see
docs/src/hpc.md).
"""
function _assign_local_gpu!(caller::String)
    CUDA.functional() ||
        error("$caller: params[:mpi] = true with backend = :cuda but " *
              "CUDA.functional() == false")
    lcomm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED,
                                MPI.Comm_rank(MPI.COMM_WORLD))
    lrank = MPI.Comm_rank(lcomm)
    CUDA.device!(lrank % length(CUDA.devices()))
    MPI.free(lcomm)   # lockstep here on every rank — immediate free is safe
    return nothing
end

"""
    _resolve_alpha(atoms, params, caller, mpi) -> params (with `:alpha`)

Resolve the Ewald splitting parameter for the PME-based drivers
(Issue #216). α is not physics: the exact Coulomb sum is α-independent,
and the cost of a step is fixed by the two knobs the caller controls
(`:rc` for real space, `:pme_meshsize` for the grid). So, unless the
caller passes an explicit `:alpha` (external-code comparison, debugging),
α is chosen to minimize the estimated total RMS force error at that fixed
cost ([`pme_optimal_alpha`](@ref)), for the same grid `PMEInfo` will build
([`pme_fftsize`](@ref)). The estimate itself is logged either way — the
run reports the accuracy it is actually getting, with its caveats.

Chosen ONCE, here, and held fixed for the whole run: re-optimizing along
an NPT trajectory would make the Hamiltonian time-dependent (energy
conservation and the virial identity both break). `relax!`'s
post-convergence escalation tightens `pme_meshsize` at fixed α, which
only shrinks the reciprocal error — monotone, so the choose-once α stays
on the safe side of the escalated solver too. The flip side of sitting at
the optimum: the two branches are BALANCED there, so mesh refinement can
remove only the reciprocal share (about 1/√2 of the total error in
quadrature) — the real-space share is a floor set by (rc, α) that no
`verify_mesh_factor` reaches. An `ftol` below that floor needs a larger
`rc`, not a finer mesh.

`:ewald_tolerance` is NOT consumed here — it is the accuracy knob of the
fitting-path `EwaldInfo`, where it moves α and kcut together and more of
it buys more accuracy. On the PME path it used to move only α against a
fixed grid (non-monotone: tightening it could inflate the reciprocal
error a thousandfold — the Issue #211/#216 measurements), which is why a
leftover one is warned about and ignored rather than honored.

Deterministic in the replicated frame ⇒ rank-uniform under MPI.
"""
function _resolve_alpha(atoms::Atoms, params, caller::String, mpi::Bool)
    root = !mpi || MPI.Comm_rank(MPI.COMM_WORLD) == 0
    if haskey(params, :ewald_tolerance) && !haskey(params, :alpha)
        root && @warn "$caller: :ewald_tolerance is no longer used by the " *
                      "PME-based apps — α is auto-determined from (:rc, " *
                      ":pme_meshsize) by error-estimate minimization " *
                      "(Issue #216). Pass :alpha explicitly to override; " *
                      "the fitting apps still take :ewald_tolerance."
    end
    cell = Array{Float64}(Array(atoms.info[:lattice]))
    a, b, c = norm(cell[1, :]), norm(cell[2, :]), norm(cell[3, :])
    fftsize = Calculators.pme_fftsize(a, b, c, Float64(params[:pme_meshsize]))
    # every charged site, cores and shells alike — the reciprocal solver
    # spreads both families onto the same grid
    elemid = Array(atoms.arrays[:elemid])
    q = vcat(Float64.(Array(params[:qc]))[elemid],
             Float64.(Array(params[:qs]))[elemid])
    α = haskey(params, :alpha) ? Float64(params[:alpha]) :
        Calculators.pme_optimal_alpha(cell, fftsize, Float64(params[:rc]), q)
    est = Calculators.pme_error_estimate(cell, fftsize, α, Float64(params[:rc]), q)
    root && @info "$caller: Ewald α = $(round(α, sigdigits = 4)) " *
                  (haskey(params, :alpha) ? "(caller-supplied)" : "(auto)") *
                  ", PME grid $(fftsize); estimated RMS force error " *
                  "$(round(est.total, sigdigits = 3)) Ha/Bohr " *
                  "(real $(round(est.real, sigdigits = 3)), reciprocal " *
                  "$(round(est.recip, sigdigits = 3))) — homogeneous-system " *
                  "estimate: site RMS, safe-side on thermal crystals, blind " *
                  "to symmetric-lattice cancellation (Issue #211)"
    return (; params..., alpha = FPPREC(α))
end

"""
    _setup_frame(atoms::Atoms, params, caller::String) -> (atoms, params, arrtyp, dom)

Bring a caller-supplied host frame and its `params` into the state every
driver loop assumes:

1. resolve the DD opt-in (`params[:mpi] === true`, never inferred from
   `MPI.Initialized()` — the fitting apps initialize MPI in the same
   process, which must not silently flip a later run into DD mode) and
   the backend array type;
2. bind this rank's GPU *before* the first CUDA allocation below;
3. resolve the Ewald `:alpha` ([`_resolve_alpha`](@ref) — auto-determined
   unless the caller passed one) and push every numeric `params` entry to
   the device;
4. validate a declared `:polarization_gauge` while `atoms` is still the
   whole system — after the split each rank holds a brick and the
   per-atom identity the gauge encodes is no longer comparable (an
   umbrella validates its own gauge in its constructor, also
   pre-distribute; this covers the gauge-only case);
5. split the frame across ranks, move it to the backend, and probe the MPI
   transport (unconditional, so "dom is an MPIDomain ⇒ the device
   transport was probed" holds even for a caller-supplied `:domain`);
6. register the per-atom virial accumulator.

`caller` only labels the error messages (`"npt!"` / `"relax!"`).
"""
function _setup_frame(atoms::Atoms, params, caller::String)
    mpi = get(params, :mpi, false) === true
    mpi && !MPI.Initialized() &&
        error("$caller: params[:mpi] = true but MPI.Init() has not been called — " *
              "run under mpiexec and call MPI.Init() before `using FMD`")
    arrtyp = _arrtyp(params)
    mpi && arrtyp !== Array && _assign_local_gpu!(caller)
    params = _resolve_alpha(atoms, params, caller, mpi)
    params = (; [key => conv(arrtyp, params[key]) for key in keys(params)]...)

    if haskey(params, :polarization_gauge)
        Calculators.validate_gauge(params[:polarization_gauge], atoms)
    end

    if mpi
        # Split on the host frame (the distribute machinery is host-side);
        # the device conversion below carries info[:domain] through
        # (_conv_info) so the DD identity survives the move.
        atoms = change_arrtyp(atoms, Array)
        atoms = Cores.distribute(atoms, Cores.MPIDomain(length(atoms)))
    end
    atoms = change_arrtyp(atoms, arrtyp)
    dom = Cores.domain(atoms)
    Cores.probe_device_mpi!(dom, Cores.get_backend(atoms))
    atoms.arrays[:peratom_virial] = arrtyp(zeros(FPPREC, length(atoms), 6))

    return atoms, params, arrtyp, dom
end

# ---------------------------------------------------------------------------
#  Force stack
# ---------------------------------------------------------------------------

"""
    ForceStack

The per-step evaluation state a driver carries between steps: the device
MLIP / NLH baseline, the MLIP workspace, and the neighbour-list family
(the Coulomb list plus the MLIP and NLH sublists derived from it) and
the PME accelerator. The cell list the Coulomb list is built from is a
local of the builders: `NeighborList` keeps its `ghosts`, not the cell
list, and nothing reads it again.

Built once by [`build_force_stack`](@ref), refreshed in place by
[`rebuild_lists!`](@ref), consumed by [`evaluate_forces!`](@ref).

The fields are deliberately untyped (as in [`PMEInfo`](@ref)): every one
of them is replaced wholesale on a rebuild, and pinning the concrete
neighbour-list / cell-list parametrisations here would turn a harmless
representational change into a `convert` error deep in the rebuild. The
cost is one dynamic dispatch per *step* per call site, against a force
evaluation that is orders of magnitude more expensive.
"""
mutable struct ForceStack
    mlip
    nlh
    mlip_ws
    mlip_ws_n::Int
    nlist
    mlip_nlists::Tuple
    nlh_nlist
    pmeinfo
end

"""
    build_force_stack(atoms::Atoms, params, arrtyp, dom) -> (ForceStack, atoms)

Resolve the MLIP / NLH baseline, build the neighbour-list family and the
PME accelerator for the starting frame, and report the effective term
composition. Returns the stack together with `atoms`, which the cell-list
construction re-sorts (and, on the GPU DD path, replaces with a fresh
frame — no frame-derived identity may be carried across this call).

The composition `@info` exists because a run with no MLIP is legal: a
forgotten `:acep` / `:splp` would otherwise run as different physics with
no written trace (`log.txt` records numbers only). The NLH baseline is
listed for the same reason.
"""
function build_force_stack(atoms::Atoms, params, arrtyp, dom)
    mlip = _resolve_mlip(params, arrtyp)
    nlh = _resolve_nlh(params, arrtyp)
    Cores.isroot(dom) &&
        @info "potential = core-shell + " *
              (mlip === nothing ? "no MLIP (pure shell model)" :
               string(nameof(typeof(mlip)))) *
              (nlh === nothing ? "" : " + NLH baseline")
    # Pure shell model with a ≡ 0 for ALL pairs means bare Coulomb +
    # springs — no short-range repulsion at all. Legitimate when
    # explicitly intended, but far more often a forgotten Buckingham
    # block, so say so once (root only under DD). Partially zero blocks
    # (some pairs) are NOT detected here — per-pair zeros are legitimate
    # in many force fields. With an MLIP the zero default is the normal
    # state: the MLIP carries the short-range physics.
    if mlip === nothing && Cores.isroot(dom) && all(iszero, params[:a])
        @warn "no MLIP (params[:acep]/[:splp]) and Buckingham a ≡ 0 — " *
              "running bare Coulomb + springs with no short-range repulsion; " *
              "the system will likely collapse"
    end
    mlip_ws = build_mlip_workspace(atoms, params)
    mlip_ws_n = length(atoms)   # before the re-sort below, which preserves N

    Cores.isroot(dom) && @info "Building neighbor list..."
    # Under GPU DD this first build round-trips through the host right
    # after the entry conversion — deliberate: keeping a single conversion
    # point at the entry and a single rebuild helper is worth one extra
    # init-only copy (and keeps the probe ahead of the first device MPI).
    cl, atoms = _build_celllist(atoms, arrtyp, dom, params[:rc] * (1 + params[:skin]);
                                do_migrate = false)
    nlist = neighborlist(atoms, cl, params[:rc], params[:skin])
    mlip_nlists = build_mlip_nlists(atoms, nlist, mlip, params[:skin])
    nlh_nlist = build_nlh_nlist(atoms, nlist, nlh, params[:skin])
    Cores.isroot(dom) && @info "Done"
    # Domain-aware construction: under MPI with P > 1 (and pme_dd not
    # disabled) the reciprocal solve runs z-slab distributed (PMESlab.jl)
    # instead of replicated per rank.
    pmeinfo = PMEInfo(atoms.info[:lattice], params, dom)

    fs = ForceStack(mlip, nlh, mlip_ws, mlip_ws_n,
                    nlist, mlip_nlists, nlh_nlist, pmeinfo)
    return fs, atoms
end

# One (re)build of the spatial structures. Under GPU DD the build machinery
# (migrate / DD CellList) is host-side: one round trip per rebuild, with
# everything that depends on the frame re-created by the caller from the
# returned pair. The round-trip conversions drop :remap_locked — correct
# semantics, not an accident: each conversion is a fresh frame sharing no
# mutable state with any info family, hence always remap-safe (Issue #70;
# the design decision is recorded in Issue #102). More generally: NO
# frame-derived array identity may be carried across this call (the fresh
# info Dict / fresh arrays sever them all) — nlist, MLIP lists, and any
# identity-keyed cache must be rebuilt inside the caller's rebuild block.
# (The MLIP workspace may be reused: it depends only on backend, the MLIP
# parameters, and N — never on frame array identities.)
function _build_celllist(atoms, arrtyp, dom, rc_pad; do_migrate::Bool)
    if dom isa Cores.MPIDomain && arrtyp !== Array
        atoms_h = change_arrtyp(atoms, Array)
        do_migrate && (atoms_h = migrate(atoms_h))
        cl_h, atoms_h = CellList(atoms_h, rc_pad)
        return change_arrtyp(cl_h, arrtyp), change_arrtyp(atoms_h, arrtyp)
    end
    do_migrate && (atoms = migrate(atoms))
    cl, atoms = CellList(atoms, rc_pad)
    return cl, atoms
end

"""
    rebuild_lists!(fs::ForceStack, atoms::Atoms, params, arrtyp, dom) -> atoms

Refresh `fs` for the current positions, if it needs refreshing. Three
outcomes, in decreasing cost:

1. the Coulomb list has run out of skin → migrate (ownership first: atoms
   that left the brick move to their new owner before the ghost/list
   rebuild; collective, a no-op in serial), rebuild the cell list, resize
   the MLIP workspace if N-local changed, and rebuild all three lists;
2. only an MLIP / NLH sublist went stale → rebuild that pair (they derive
   from the same parent list and skin, so they go stale together and are
   kept in lockstep);
3. nothing to do.

Returns `atoms` — case 1 replaces the frame (see [`_build_celllist`](@ref)).
"""
function rebuild_lists!(fs::ForceStack, atoms::Atoms, params, arrtyp, dom)
    if !isreusable(fs.nlist, atoms)
        Cores.isroot(dom) && @info "Building neighbor list..."
        cl, atoms = _build_celllist(atoms, arrtyp, dom,
                                    params[:rc] * (1 + params[:skin]);
                                    do_migrate = true)
        if fs.mlip_ws !== nothing && length(atoms) != fs.mlip_ws_n
            fs.mlip_ws = build_mlip_workspace(atoms, params)
            fs.mlip_ws_n = length(atoms)
        end
        fs.nlist = neighborlist(atoms, cl, params[:rc], params[:skin])
        fs.mlip_nlists = build_mlip_nlists(atoms, fs.nlist, fs.mlip, params[:skin])
        fs.nlh_nlist = build_nlh_nlist(atoms, fs.nlist, fs.nlh, params[:skin])
        Cores.isroot(dom) && @info "Done"
    elseif !mlip_nlists_reusable(fs.mlip_nlists, atoms) ||
           !nlh_nlist_reusable(fs.nlh_nlist, atoms)
        fs.mlip_nlists = build_mlip_nlists(atoms, fs.nlist, fs.mlip, params[:skin])
        fs.nlh_nlist = build_nlh_nlist(atoms, fs.nlist, fs.nlh, params[:skin])
    end
    return atoms
end

"""
    evaluate_forces!(atoms::Atoms, fs::ForceStack, params; xl = nothing) -> Int

One full force evaluation into `atoms`: shell relaxation + core-shell
stack, then the MLIP and the NLH baseline on top. Returns the shell-CG
force-evaluation count (`ncg`, the [`calculate!`](@ref) return value).

`xl` selects the shell path: `nothing` is the per-step CG relaxation
(`:scf`), an [`XLState`](@ref) runs one shadow XL-BOMD step instead
(Issue #149). Both leave the same outputs, so the MLIP / NLH call sites
below are shared. Deliberately a KEYWORD: a forgotten positional argument
would silently run the SCF path — different physics, no error.
"""
function evaluate_forces!(atoms::Atoms, fs::ForceStack, params; xl = nothing)::Int
    clear!(atoms)
    ncg = xl === nothing ?
        calculate!(CoreShell(), atoms, fs.nlist, fs.pmeinfo, params) :
        xl_step!(atoms, fs.nlist, fs.pmeinfo, params, xl)
    apply_mlip!(atoms, fs.mlip, fs.mlip_nlists, fs.mlip_ws)
    apply_nlh!(atoms, fs.nlh, fs.nlh_nlist)
    return ncg
end
