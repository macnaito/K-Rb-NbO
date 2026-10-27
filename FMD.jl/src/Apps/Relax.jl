# src/Apps/Relax.jl — structure relaxation with FIRE 2.0.
#
# Why FIRE and not CG / L-BFGS: the core-shell force evaluation relaxes the
# shell displacements to `cgtol` internally before returning core forces
# (CoreShell.jl), so from the outer optimizer's point of view energy and
# force are mutually inconsistent at the O(cgtol) level — a non-variational
# error. Line-search methods, which compare energies across trial steps,
# reject steps and hysterese on that noise. FIRE uses forces only.
#
# Reference: Bitzek et al., PRL 97, 170201 (2006); Guénolé et al.,
# Comput. Mater. Sci. 175, 109584 (2020) for the 2.0 refinements
# (semi-implicit Euler, initial delay, half-step backtracking).

using LinearAlgebra
using Printf
using FMD
using FMD.Cores
using FMD.Calculators
using MPI

# FIRE 2.0 control constants (Guénolé et al. Table 1). Only the step-size
# knobs (`dt0`, `dt_max`, `dmax`) are exposed through params — these are
# the algorithm's own tuning, changed only with a reason.
const FIRE_N_DELAY = 20        # consecutive downhill steps before dt may grow
const FIRE_N_MAX_UPHILL = 20   # consecutive uphill steps before giving up
const FIRE_F_INC = 1.1
const FIRE_F_DEC = 0.5
const FIRE_ALPHA0 = 0.25
const FIRE_F_ALPHA = 0.99
# Floor on dt, as a fraction of dt0. Guénolé et al. carry a Δt_min for the
# same reason: without it a run that keeps going uphill halves dt forever
# and spins out its whole step budget taking steps too small to measure.
const FIRE_DT_MIN_FRAC = 1e-3

"""
    _relax_reductions(dom, v, f) -> (P, vnorm, fnorm)

The three global scalars one FIRE step needs — the power `P = F·v` and the
two Euclidean norms — fused into a single Allreduce (the same pattern as
`_cg_sum2` in CoreShell.jl).

Atom-local partial sums only: an empty rank contributes zeros but still
rides the collective, so the branch FIRE takes from `P` is decided from
rank-uniform values and every rank mixes velocities in lockstep.
"""
function _relax_reductions(dom, v, f, cell = nothing)
    buf = zeros(FPPREC, 3)
    if length(v) > 0
        buf[1] = FPPREC(dot(f, v))
        buf[2] = FPPREC(sum(abs2, v))
        buf[3] = FPPREC(sum(abs2, f))
    end
    Cores.allreduce_sum!(dom, buf)
    # The cell degree of freedom is REPLICATED, not decomposed: `stress`
    # returns a whole-system value on every rank. It must therefore be added
    # AFTER the reduction — folding it into the per-rank partial sums would
    # multiply the cell's contribution by the number of ranks and silently
    # rebalance the atom/cell mixing with P.
    #
    # The inner product is the Frobenius one, `sum(A .* B)`, which counts
    # the off-diagonal components twice. That is exactly right and is why
    # the cell dof is carried as a symmetric 3x3 rather than a Voigt-6
    # vector: `voigt2matrix` stores TENSOR components, so the work is
    # σ:ε = σ₁ε₁+σ₂ε₂+σ₃ε₃ + 2(σ₄ε₄+σ₅ε₅+σ₆ε₆) — a plain `dot` of two
    # Voigt-6 vectors would weight every shear component by 1/2 and quietly
    # under-drive the shear modes.
    if cell !== nothing
        vc, fc = cell          # a pair, so a half-supplied call cannot compile
        buf[1] += FPPREC(sum(fc .* vc))
        buf[2] += FPPREC(sum(abs2, vc))
        buf[3] += FPPREC(sum(abs2, fc))
    end
    return buf[1], sqrt(buf[2]), sqrt(buf[3])
end

"""
    _max_force(atoms, dom) -> FPPREC

Whole-system `max|F|` over the Cartesian force components. Collective
under DD; an empty rank votes with zero.
"""
function _max_force(atoms::Atoms, dom)
    local_max = length(atoms) == 0 ? zero(FPPREC) :
                FPPREC(maximum(abs, atoms.arrays[:forces]))
    return Cores.allreduce_max(dom, local_max)
end

"""
    _relax_done(maxf, maxs, params, stol) -> Bool

The convergence test: `max|F| < ftol` **and** `max|P(σ − σ_target)| < stol`.

Both, never either — and never an energy difference. A flat mode (a soft
mode, exactly what these systems have) leaves the energy stationary to
first order while the forces along it are still finite, so an
energy-difference criterion reports convergence on precisely the direction
that has not converged.

With `cell_dof = :none` the caller passes `stol = Inf` (see [`relax!`](@ref)),
which makes the stress half vacuous — as opposed to `stol = 0`, which would
make it unsatisfiable and hang the run at a structure that is already
relaxed.
"""
_relax_done(maxf, maxs, params, stol) = maxf < params[:ftol] && maxs < stol

"""
    _cell_force(atoms, tstress, cell_dof, cell_factor) -> Matrix

The generalised force conjugate to the strain degree of freedom, projected
onto the subspace `cell_dof` allows and scaled by `cell_factor`.

`virial = −∂E/∂ε` is the repository's convention (pinned by
`test/calculators/test_virial.jl`), so with `stress = −virial/V` the strain
gradient is `∂E/∂ε = +V·σ` (tension positive) and the force conjugate to
the enthalpy `H = E − V·σ_target : ε` is

    F_ε = −∂H/∂ε = V·(σ_target − σ)

which is the same expression the MTTK barostat uses (`_mttk_chain!`), taken
here **without** the kinetic term: the FIRE pseudo-velocity is not a
physical velocity and has no place in a stress.

`F_ε` is symmetric by construction — both terms come through
`voigt2matrix` — so the three rigid-rotation degrees of freedom of a
general 3×3 never enter, and each strain increment is a pure deformation.

Per increment, that is exact. Their *product* `Π(I + dε_k)` is not
symmetric in general: the commutators leave an O(dε²) rigid rotation that
accumulates over a run. For a rotationally invariant Hamiltonian that is
harmless (it changes no energy). It is not harmless for a term that couples
to lab-frame tensors — an anisotropic [`SurfaceTerm`](@ref) — which is one
more reason that combination warns; see
[`_check_cell_dof_support`](@ref).

`cell_factor` is a work-preserving change of coordinates, not a mass: with
`q = cell_factor·ε` and `f_q = F_ε/cell_factor`, `f_q·δq = F_ε·δε`. Its
role is to put the strain dof on a comparable scale to the atomic
displacements so one `dt` serves both; the default `N` is ASE's
`UnitCellFilter` convention.
"""
function _cell_force(atoms::Atoms, tstress, cell_dof::Symbol, cell_factor)
    sigma = voigt2matrix(stress(atoms, false))
    F = (tstress .- sigma) .* volume(atoms)
    return _cell_projector(cell_dof, F) ./ cell_factor
end

"""
    _cell_residual(atoms, tstress, cell_dof) -> FPPREC

`max|P(σ − σ_target)|`, the cell half of the convergence test.

The residual is measured **after** projection because the components
`cell_dof` freezes are held by the constraint: their residual stress is the
constraint reaction force, and demanding that it vanish would mean
demanding the constraint do nothing. Under `:iso`, for instance, a
tetragonal cell keeps a deviatoric residual by construction and only
`|tr(Δσ)/3|` is the run's business.

Replicated: `stress(atoms, false)` performs no collective (the virial is
already whole-system), so every rank computes the same number.
"""
function _cell_residual(atoms::Atoms, tstress, cell_dof::Symbol)
    cell_dof === :none && return zero(FPPREC)
    R = _cell_projector(cell_dof, tstress .- voigt2matrix(stress(atoms, false)))
    return FPPREC(maximum(abs, R))
end

"""
    _cap_displacement!(dx, dmax, dom) -> FPPREC

Scale the Cartesian step `dx` in place so that no atom moves further than
`dmax`, and return the factor applied (`1` when the step already fits).

This is not cosmetic. Under domain decomposition `migrate` routes atoms
through a ±1-brick staged exchange and **errors** on an offset of |Δ| ≥ 2
(Migrate.jl), so a single unbounded step down a steep repulsive wall is a
hard failure, not a slow convergence. The scaling is uniform (one factor
for the whole system) so the step direction is preserved, and the factor
is Allreduce-derived so every rank scales identically.
"""
function _cap_displacement!(dx, dmax, dom)
    local_max = length(dx) == 0 ? zero(FPPREC) : FPPREC(maximum(abs, dx))
    gmax = Cores.allreduce_max(dom, local_max)
    gmax <= dmax && return one(FPPREC)
    scale = dmax / gmax
    dx .*= scale
    return scale
end

"""
    _check_relax_tolerances(params)

Entry guard on `ftol` against the inner shell-relaxation tolerance
`cgtol`.

The residual shell force left by the inner CG propagates into the core
force at O(cgtol) (the Hellmann–Feynman cancellation `∂E/∂s = 0` is only
approximate), so a `ftol` at or below `cgtol` asks the outer loop to
converge below its own force noise — unreachable in principle, and it
would spin to `maxsteps` with no diagnosis. The factor is deliberately
loose (a rule of thumb, not a bound: the true constant is system
dependent) — the authoritative check is the post-convergence
re-evaluation at a tightened `cgtol`, see [`_verify_convergence`](@ref).
"""
function _check_relax_tolerances(params)
    ftol = params[:ftol]
    cgtol = params[:cgtol]
    margin = get(params, :ftol_cgtol_margin, 10.0)
    ftol > 0 || error("relax!: params[:ftol] must be > 0 (got $ftol)")
    ftol >= margin * cgtol ||
        error("relax!: ftol = $ftol is not reachable with cgtol = $cgtol — " *
              "the inner shell relaxation leaves an O(cgtol) error in the core " *
              "forces, so the outer criterion must stay well above it " *
              "(require ftol >= $(margin)*cgtol = $(margin * cgtol); tighten " *
              ":cgtol, loosen :ftol, or override :ftol_cgtol_margin knowingly)")
    return nothing
end

"""
    _verify_convergence!(atoms, fs, params, dom) -> (ok, maxf_tight, tight_params)

Re-evaluate the forces with **both** numerical tolerances tightened and
report whether the convergence criterion still holds:

- the shell relaxation, by `params[:verify_cgtol_factor]` (default `1/10`);
- the PME mesh spacing, by `params[:verify_mesh_factor]` (default `1/2`,
  i.e. 8× the grid points).

This is the acceptance test for a relaxation, not a formality. The outer
loop converges against forces that carry the error of both solvers, and
either one can be what put `max|F|` under `ftol`:

- The inner shell CG leaves an O(cgtol) *non-variational* error in the
  core forces (the Hellmann–Feynman cancellation `∂E/∂s = 0` holds only at
  exact relaxation).
- The PME mesh error is not merely small-and-smooth: on a **high-symmetry
  structure** it breaks the symmetry that makes the exact forces vanish.
  Measured on the ideal NaCl-64 fixture, whose analytic-Ewald forces are
  zero to 1e-17, PME reports `max|F| = 1.8e-5` at one mesh spacing and
  1e-17 at the next — the error tracks the grid's commensurability with
  the crystal, not the nominal spacing. A relaxation can therefore chase a
  spurious distortion that exists only on that grid.

If the criterion does not survive the tightening, the caller does not
report a false convergence — and does not simply give up either: it adopts
the tightened settings and keeps relaxing, which is what the finding
actually calls for. See [`relax!`](@ref).

A caveat the table above makes unavoidable: refining the mesh lowers the
*generic* interpolation error monotonically (measured on a rattled frame:
1.9e-5, 4.7e-6, 2.9e-6, 1.7e-6 as the spacing goes 0.50 → 0.30 → 0.25 →
0.20 Å), but on a symmetric structure it can lose an accidental exact
cancellation — the 1e-17 entries above are grids on which the symmetry
happens to close, not grids that are more accurate in general. So the
refined evaluation is a better *bound*, not a smaller number in every case.
That is the right thing to test against: if the refined solver cannot
certify `max|F| < ftol`, then `ftol` is below this discretisation's noise
floor, which is exactly what the caller needs to be told. The `@info`
records both FFT grids so an anomalous value stays diagnosable.

Mutates: `atoms` holds the tightened evaluation on return, and `fs.pmeinfo`
is **replaced** by the finer accelerator (strictly better physics — there
is no reason to go back to the coarse one).
"""
_tighten_solvers(params) =
    (; params...,
     cgtol = FPPREC(params[:cgtol] * get(params, :verify_cgtol_factor, 0.1)),
     pme_meshsize = FPPREC(params[:pme_meshsize] *
                           get(params, :verify_mesh_factor, 0.5)))

function _verify_convergence!(atoms::Atoms, fs::ForceStack, params, dom)
    tight = _tighten_solvers(params)
    # Collective, and rank-uniform by construction (every rank derives the
    # same mesh from the same replicated cell and params). Adopted into `fs`
    # for good: the finer accelerator is strictly better physics, so if the
    # run continues it continues on this one.
    fs.pmeinfo = PMEInfo(atoms.info[:lattice], tight, dom)
    evaluate_forces!(atoms, fs, tight)
    maxf = _max_force(atoms, dom)
    return maxf < params[:ftol], maxf, tight
end

"""
    _reject_unsupported_relax(params)

Refuse the `params` combinations whose relaxed structure would not mean
what the caller expects.

- `shell_dynamics = :xlbomd` propagates an auxiliary shell field along a
  *time-reversible* trajectory. It is a device for dynamics; there is no
  trajectory here, and the shadow surface it integrates is not the surface
  being minimised.
- `polarization_umbrella` is a bias, not a physical potential. Its minimum
  is a biased minimum — a legitimate object, but not the equilibrium
  structure the app promises. Supporting it needs a separate interface
  that says so.

`surface_term` **is** supported: the depolarization field of an external
medium is physics, contributes to the virial, and belongs in the energy
being minimised.
"""
function _reject_unsupported_relax(params)
    get(params, :shell_dynamics, :scf) === :xlbomd &&
        error("relax!: shell_dynamics = :xlbomd is a dynamics device (a " *
              "time-reversible auxiliary shell field) and has no meaning for " *
              "relaxation — drop it and use the default :scf shell path")
    haskey(params, :polarization_umbrella) &&
        error("relax!: the polarization umbrella is a bias, not a physical " *
              "potential — relaxing under it yields a biased minimum, not the " *
              "equilibrium structure this app reports. Drop it (:surface_term " *
              "is supported: the depolarization field is physical)")
    return nothing
end

"""
    _check_cell_dof_support(params, dom)

Guards specific to running the cell degree of freedom.

**`:efield` is refused outright.** [`ExternalField`](@ref) contributes to
neither the energy nor the virial — it adds a force and nothing else. For
the atoms that is complete: the force IS the gradient of the field term, so
`F = 0` is a genuine stationary point. For the cell it is not: the field's
strain derivative is `∂(−E·M)/∂ε = −M⊗E`, whose symmetric part
`−(M⊗E + E⊗M)/2` is generally non-zero and is simply **absent** from `σ`.
A cell relaxed to `σ = σ_target` would therefore sit at the wrong cell, with
nothing in the output to say so.

**An anisotropic [`SurfaceTerm`](@ref) is warned about.** Its `∂U/∂ε`
carries an antisymmetric part that a Voigt-6 virial cannot represent —
`test/calculators/test_surface_term.jl` only matches the strain finite
difference with `symmetrize = true`. The *symmetric* part is in the virial
and is correct, so relaxation over the symmetric-strain subspace (all this
app parameterises) is exact. What is left un-relaxed is the rigid rotation
of the cell against the lab-frame `eps_out` / `shape` tensors — a real
degree of freedom, but one this app never varies. That makes it a
documented limitation, not a wrong answer, hence a warning. An isotropic
`T` makes the term rotationally invariant and the question moot, so the
check is on `T`.
"""
function _check_cell_dof_support(params, dom)
    get(params, :cell_dof, :none) === :none && return nothing

    ef = get(params, :efield, nothing)
    if ef !== nothing && any(!iszero, Array(ef))
        error("relax!: params[:efield] is non-zero and cell_dof = " *
              "$(repr(params[:cell_dof])) — ExternalField contributes no " *
              "energy and no virial, so the field's strain derivative " *
              "-(M⊗E + E⊗M)/2 is missing from the stress and the cell would " *
              "relax to the wrong shape. Use cell_dof = :none with a field, " *
              "or drop the field")
    end

    if haskey(params, :surface_term)
        T = params[:surface_term].T
        aniso = norm(T - (tr(T) / 3) * I) > sqrt(eps(FPPREC)) * max(norm(T), one(FPPREC))
        aniso && Cores.isroot(dom) &&
            @warn "relax!: surface_term has an anisotropic response tensor T, " *
                  "so its strain derivative has an antisymmetric part the " *
                  "Voigt-6 virial does not carry. The symmetric part is exact, " *
                  "so the relaxation over symmetric strain is correct — but " *
                  "(1) the ORIENTATION of the cell relative to eps_out / shape " *
                  "is not relaxed, (2) the product of symmetric strain " *
                  "increments accumulates an O(de^2) rotation that this term " *
                  "IS sensitive to, and (3) T is frozen at construction and " *
                  "does not follow a large cell deformation. Keep the strain " *
                  "small, or relax with cell_dof = :none"
    end
    return nothing
end

"""
    _maybe_rebuild_pme!(fs, atoms, params, dom, len0) -> Bool

Rebuild the PME accelerator when the cell has deformed enough that its FFT
grid no longer resolves `pme_meshsize`. Returns whether it rebuilt.

The trigger is on the **edge lengths**, not the volume:
`fftsize = pme_fftsize(a, b, c, pme_meshsize)` (PME.jl) is fixed at
construction and depends on each side separately, so an isochoric shear —
or any deformation that lengthens one axis and shortens another — degrades
the mesh spacing along an axis while leaving the volume untouched. A
volume-based trigger would sleep straight through it.

Collective under DD, and rank-uniform by construction: the lattice is
replicated, so every rank reaches the same verdict on the same step and
enters `PMEInfo`'s collectives together.
"""
function _maybe_rebuild_pme!(fs::ForceStack, atoms::Atoms, params, dom, len0)
    tol = get(params, :pme_rebuild_ratio, 1.05)
    lat = Array(atoms.info[:lattice])
    len = (norm(lat[1, :]), norm(lat[2, :]), norm(lat[3, :]))
    any(i -> max(len[i] / len0[i], len0[i] / len[i]) > tol, 1:3) || return false
    fs.pmeinfo = PMEInfo(atoms.info[:lattice], params, dom)
    return true
end

"""
    relax!(atoms::Atoms, params) -> (atoms, converged::Bool)

Relax the atomic positions with FIRE 2.0 at fixed cell, using a core-shell
potential plus an optional fitted MLIP. Returns the final frame and
whether the run converged.

If the criterion is met but does not survive the acceptance test
([`_verify_convergence!`](@ref)), the run does not report a false
convergence and does not stop either: it adopts the tightened solvers and
keeps relaxing on the forces that are then visible. This happens at most
once, so the run stays bounded, and it is reported in the log.

The FIRE pseudo-velocity lives in its own `atoms.arrays[:fire_velo]`, so
the frame's `:velo` keeps meaning what it says (nothing here is a
trajectory, and the relaxed frame must not carry a velocity field that
looks like one). Being a registered per-atom array it migrates and
cell-sorts with everything else.

The pseudo-dynamics is **mass-weighted** (`v += F/m·dt`, as in LAMMPS
`min_style fire`) rather than the unit-mass form of the ASE implementation.
The reason is the meaning of `dt`: with unit mass the step is `dt²·F` and
the stable `dt` is smaller than an MD timestep by `√m` — around 0.2 a.u.
for these systems, where `:timestep` is ~41. A user setting `:dt0` by
analogy with `:timestep` would then take steps ~10⁴ too large, and since
`:dmax` clamps them, every step would saturate the cap and the run would
crawl at a fixed step size instead of converging. Mass-weighting makes
`:dt0` / `:dt_max` mean exactly what `:timestep` means, so the analogy is
sound instead of catastrophic. The `P` sign and the velocity mixing are
unaffected — a positive diagonal metric changes neither.

The shell displacements are warm-started from the previous step's
converged values. The 3-point predictor `npt!` uses
([`_extrapolate_shelldisp!`](@ref)) is *not* applicable here: it assumes
equally spaced steps, while FIRE varies `dt` and backtracks half a step
whenever the power goes negative.

# `params` keys

Everything the force stack needs (see [`npt!`](@ref) — `:rc`, `:skin`,
`:cgtol`, `:cgmaxiter`, `:pme_meshsize`, the shell / Buckingham tables,
optionally `:acep` / `:splp` / `:nlh` / `:surface_term`), plus:

- `:ftol` — convergence threshold on `max|F|` (atomic units). Must stay
  well above `:cgtol`; see [`_check_relax_tolerances`](@ref).
- `:maxsteps` — iteration cap. Reaching it is reported as **not
  converged**, not raised as an error: a partially relaxed structure is a
  legitimate (and often useful) result, as long as nobody mistakes it for
  a converged one.
- `:dt0`, `:dt_max` — initial / maximum FIRE step (atomic units).
- `:dmax` — per-step displacement cap; see [`_cap_displacement!`](@ref).
- `:cell_dof` (optional, default `:none`) — which cell degrees of freedom
  relax: `:none` (fixed), `:iso` (hydrostatic only), `:diag` (axes free, no
  shear) or `:full`. See [`_cell_projector`](@ref).
- `:target_stress`, `:stol` — Voigt-6 target (tension positive, as in
  [`npt!`](@ref)) and the tolerance on `max|P(σ − σ_target)|`. **Required**
  when `cell_dof != :none`, unused otherwise.
- `:cell_factor` (optional, default `N`) — the strain dof's scale; see
  [`_cell_force`](@ref).
- `:emax` (optional, default `1e-3`) — per-step strain cap.
- `:pme_rebuild_ratio` (optional, default `1.05`) — edge-length ratio at
  which the PME accelerator is rebuilt; see [`_maybe_rebuild_pme!`](@ref).
- `:output_interval`, `:output_dir`.
- `:verify_cgtol_factor`, `:verify_mesh_factor` (optional, defaults `0.1`
  and `0.5`) — the shell-CG and PME-mesh tightening factors of the
  post-convergence re-check; see [`_verify_convergence`](@ref).
- `:ftol_cgtol_margin` (optional, default `10`) — the entry-guard factor.
- `:backend`, `:mpi` — as in [`npt!`](@ref).

# Output (under `params[:output_dir]`)

- `log.txt` — per-step `step`, `max|F|`, energy, stress Voigt-6, volume,
  `dt`, `α`, `ncg`, then the 9 lattice components. Units: Hartree/Bohr,
  Hartree, GPa, Å³, Å.
  Note: if the run escalates to tightened solvers (see below), the energy
  and stress columns take a small step at that point — they are evaluated
  on a finer PME mesh from there on. That is a change of quadrature, not
  the relaxation going uphill; the `@info` marks the step it happens at.
  The last line is written from inside the loop, i.e. **before** the
  acceptance test re-evaluates; the returned/reported `max|F|` is the
  verified value and can differ from that column by the solver error.
- `step_<n>.xyz` — snapshots every `output_interval` steps.
- `relaxed.xyz` — the final frame, written whether or not it converged
  (the `@warn` and the return value carry that fact).
"""
function relax!(atoms::Atoms, params)
    _reject_unsupported_relax(params)
    _check_relax_tolerances(params)

    atoms, params, arrtyp, dom = _setup_frame(atoms, params, "relax!")
    _check_cell_dof_support(params, dom)
    fs, atoms = build_force_stack(atoms, params, arrtyp, dom)

    # Pseudo-velocity: registered before any rebuild so migrate/cell-sort
    # carry it. Starts at rest — FIRE's first step is pure steepest descent.
    atoms.arrays[:fire_velo] = arrtyp(zeros(FPPREC, length(atoms), 3))

    # Cell degree of freedom. Host-side symmetric 3x3 throughout (it is a
    # handful of numbers, replicated on every rank); Voigt-6 appears only at
    # the params boundary and in the log.
    cell_dof = get(params, :cell_dof, :none)
    _cell_ndof(cell_dof)          # validates the symbol, loudly
    cell_dof === :none || haskey(params, :stol) ||
        error("relax!: params[:stol] is required when cell_dof = " *
              "$(repr(cell_dof)) — the cell half of the convergence test " *
              "needs a stress tolerance (atomic units, as :target_stress)")
    cell_dof === :none || haskey(params, :target_stress) ||
        error("relax!: params[:target_stress] is required when cell_dof = " *
              "$(repr(cell_dof)) — Voigt-6, tension positive, as in npt!")
    cell_factor = FPPREC(get(params, :cell_factor, Cores.n_global(dom, atoms)))
    tstress = haskey(params, :target_stress) ?
        voigt2matrix(Array(params[:target_stress])) : zeros(FPPREC, 3, 3)
    # Inf, not 0, when the cell is fixed: the stress half of the criterion
    # has to be vacuous, and `0 < 0` is false.
    stol = cell_dof === :none ? FPPREC(Inf) : FPPREC(params[:stol])
    emax = FPPREC(get(params, :emax, 1e-3))
    # Pseudo-velocity of the cell, in the SCALED coordinate q = cell_factor*ε
    # rather than in strain rate. That is what makes the reductions below
    # consistent: `fcell` is the force conjugate to q (F_ε/cell_factor), so
    # its partner in P = F.v, |v| and |F| has to be q̇ = cell_factor*ε̇. Storing
    # ε̇ instead would scale the cell's contribution to P by 1/cell_factor and
    # to |v|² by 1/cell_factor² — with the default cell_factor = N the cell
    # would fade out of the uphill test as the system grows, and FIRE would
    # stop backtracking on a cell going uphill under downhill atoms.
    vcell = zeros(FPPREC, 3, 3)
    lat0 = Array(atoms.info[:lattice])
    len0 = (norm(lat0[1, :]), norm(lat0[2, :]), norm(lat0[3, :]))

    dt = FPPREC(params[:dt0])
    dt_max = FPPREC(params[:dt_max])
    dt_min = FPPREC(params[:dt0] * FIRE_DT_MIN_FRAC)
    dmax = FPPREC(params[:dmax])
    alpha = FPPREC(FIRE_ALPHA0)
    # The (dt, cap factor) the previous step actually moved with — what the
    # uphill branch has to undo half of.
    last_dt = dt
    last_scale = one(FPPREC)
    last_cell_scale = one(FPPREC)
    n_downhill = 0             # CONSECUTIVE steps with P > 0 (gates dt growth)
    n_uphill = 0               # CONSECUTIVE steps with P <= 0 (gives up at the cap)
    converged = false
    step = 0
    maxf = zero(FPPREC)

    ncg = evaluate_forces!(atoms, fs, params)
    maxf = _max_force(atoms, dom)
    maxs = _cell_residual(atoms, tstress, cell_dof)
    _log_relax(atoms, params, dom, 0, maxf, dt, alpha, ncg; init = true)

    # A high-symmetry starting structure has zero force by symmetry and is
    # a stationary point — which may be a saddle, not a minimum. Saying so
    # once is much cheaper than a user reading a saddle as an equilibrium.
    if _relax_done(maxf, maxs, params, stol)
        Cores.isroot(dom) &&
            @warn "relax!: the initial configuration already satisfies " *
                  "the convergence criteria (max|F| = $maxf, " *
                  "max|P(σ-σ_t)| = $maxs). If it is a high-symmetry " *
                  "structure this may be a saddle rather than a minimum — " *
                  "re-relax from a slightly randomised copy to confirm"
    end

    # Outer loop: the inner FIRE loop runs to the criterion, then the
    # acceptance test decides whether that criterion meant anything. If it
    # did not, the tightened solvers are adopted and the inner loop resumes
    # on them (see below) — hence the wrapper rather than a straight-line
    # check afterwards.
    eff = params            # solver settings currently in force
    verified = false        # the acceptance test has run
    escalated = false       # ...and it failed, so the run continued on tight solvers
    fft_before = fs.pmeinfo.fftsize
    while true
        while step < params[:maxsteps] && !_relax_done(maxf, maxs, params, stol)
            step += 1
            v = atoms.arrays[:fire_velo]
            f = atoms.arrays[:forces]

            # --- FIRE 2.0 mixing -------------------------------------------
            # Power of the current force against the current velocity, before
            # any update: this is the sign that decides uphill vs downhill.
            fcell = cell_dof === :none ? nothing :
                    _cell_force(atoms, tstress, cell_dof, cell_factor)
            P, vnorm, fnorm = _relax_reductions(dom, v, f,
                cell_dof === :none ? nothing : (vcell, fcell))
            if P > 0
                n_downhill += 1
                n_uphill = 0
                # Guénolé et al. Algorithm 1: dt may grow (and α decay) only
                # after N_delay CONSECUTIVE downhill steps. The counter resets
                # on every uphill step, so the delay re-arms — that is what
                # distinguishes 2.0's single N_delay from 1.0's N_min, and
                # what stops dt from ratcheting back up right after a wall.
                if n_downhill > FIRE_N_DELAY
                    dt = min(dt * FPPREC(FIRE_F_INC), dt_max)
                    alpha *= FPPREC(FIRE_F_ALPHA)
                end
                # v ← (1-α)·v + α·|v|·F̂ . fnorm == 0 cannot happen here (P > 0
                # requires a non-zero force) but the guard keeps the expression
                # total.
                if fnorm > 0
                    mix = alpha * vnorm / fnorm
                    @. v = (1 - alpha) * v + mix * f
                    # Same mixing on the cell dof, with the SAME global
                    # |v|/|F| — the atoms and the strain are one vector in
                    # one inner product, which is the whole point of relaxing
                    # them together rather than alternating.
                    cell_dof === :none ||
                        (@. vcell = (1 - alpha) * vcell + mix * fcell)
                end
            else
                n_downhill = 0
                n_uphill += 1
                # Uphill: undo half of the step that got us here, drop the
                # velocity, and shrink dt. The half-step backtracking is the
                # 2.0 addition — FIRE 1.0 only zeroed the velocity, which
                # leaves the configuration sitting past the wall.
                #
                # It must undo half of the step that was ACTUALLY APPLIED, not
                # half of `dt·v`: `_cap_displacement!` may have scaled the step
                # down, and backtracking by the uncapped amount would move the
                # atoms further than the step it claims to undo — unbounded, in
                # the direction the forces just rejected. Hence `last_scale`.
                #
                # DEVIATION from Algorithm 1, deliberate: the reference shrinks
                # Δt first and backtracks with the shrunk value, so its
                # "half step back" is not half of the step taken. Undoing the
                # applied step exactly is what the operation means, so the
                # backtrack here uses `last_dt` (and `last_scale`) and the
                # shrink happens after.
                #
                # NOT `@.`: the fractional conversion is matrix algebra
                # (`lattice \ I` and the matrix product), which `@.` would
                # broadcast elementwise. Same idiom as the npt! position update.
                atoms.arrays[:pos] .-=
                    ((v * (last_scale * last_dt / 2)) * (atoms.info[:lattice] \ I))
                v .= zero(FPPREC)
                if cell_dof !== :none
                    # Undo the same fraction of the strain increment. To
                    # first order in the (small, emax-capped) strain the
                    # inverse of L(I + ε) is L(I − ε); using the exact
                    # inverse here would be a different, larger correction
                    # than the half-step this is meant to be.
                    deps = _cell_projector(cell_dof,
                                           vcell .* (last_cell_scale * last_dt /
                                                     (2 * cell_factor)))
                    atoms.info[:lattice] .=
                        atoms.info[:lattice] * conv(arrtyp, I - deps)
                    vcell .= zero(FPPREC)
                end
                # ...but not during the initial delay (an ABSOLUTE step count
                # here, unlike the growth gate above — Algorithm 1's
                # `initialdelay`): dt has not had a chance to prove itself yet,
                # and shrinking it there stalls the run.
                if step > FIRE_N_DELAY
                    dt = max(dt * FPPREC(FIRE_F_DEC), dt_min)
                end
                alpha = FPPREC(FIRE_ALPHA0)
            end

            # Persistently uphill with a collapsed dt is not slow convergence,
            # it is a stuck run: report it instead of burning the remaining
            # budget on steps too small to measure.
            if n_uphill >= FIRE_N_MAX_UPHILL
                Cores.isroot(dom) &&
                    @warn "relax!: $(n_uphill) consecutive uphill steps at " *
                          "dt = $dt (floor $dt_min) — the step size has " *
                          "collapsed without meeting ftol. Stopping; check " *
                          ":dt0 / :dmax, or whether the structure is at a " *
                          "cusp of the potential"
                break
            end

            # --- semi-implicit Euler ---------------------------------------
            # Velocity first, then the position from the NEW velocity (this is
            # what makes it semi-implicit; explicit Euler is unstable here).
            @. v += f / atoms.arrays[:masses] * dt
            dx = v * dt
            last_scale = _cap_displacement!(dx, dmax, dom)
            last_dt = dt
            atoms.arrays[:pos] .+= (dx * (atoms.info[:lattice] \ I))

            if cell_dof !== :none
                # Unit mass in q: q̈ = f_q = F_ε/cell_factor. Converting back
                # to a strain increment divides by cell_factor once more, so
                # the strain accelerates as F_ε/cell_factor² — the ASE
                # UnitCellFilter scaling, reached here without ever leaving a
                # consistent inner product.
                @. vcell += fcell * dt
                deps = _cell_projector(cell_dof, vcell .* (dt / cell_factor))
                # Cap the strain increment. Not cosmetic under DD: a cell that
                # deforms too far in one step drives a brick thinner than
                # rc(1+skin), which the ghost construction refuses.
                emax_now = maximum(abs, deps)
                last_cell_scale = emax_now > emax ? FPPREC(emax / emax_now) : one(FPPREC)
                deps .*= last_cell_scale
                # `:pos` is fractional and the lattice multiplies from the
                # right, so L ← L(I + dε) deforms the cell and carries every
                # atom affinely with it — no separate position update, and no
                # deformation-gradient correction, because the strain is an
                # increment about the CURRENT configuration rather than a
                # cumulative one about a fixed reference.
                atoms.info[:lattice] .=
                    atoms.info[:lattice] * conv(arrtyp, I + deps)
                # `eff`, NOT `params`: after an escalation the run is on
                # tightened solvers and `fs.pmeinfo` is the fine mesh.
                # Rebuilding from `params` here would silently restore the
                # ORIGINAL coarse pme_meshsize — undoing the escalation while
                # the run still reports "verified on tightened solvers".
                pme_rebuilt = _maybe_rebuild_pme!(fs, atoms, eff, dom, len0)
                if pme_rebuilt
                    lat_now = Array(atoms.info[:lattice])
                    len0 = (norm(lat_now[1, :]), norm(lat_now[2, :]),
                            norm(lat_now[3, :]))
                    Cores.isroot(dom) &&
                        @info "relax!: cell deformed past the PME mesh budget " *
                              "at step $step — rebuilt the accelerator " *
                              "(FFT grid $(fs.pmeinfo.fftsize))"
                end
            end

            atoms = rebuild_lists!(fs, atoms, params, arrtyp, dom)
            # rebuild_lists! may hand back a re-sorted (or, under GPU DD, a
            # freshly built) frame — every array reference taken above is stale
            # from here on.
            ncg = evaluate_forces!(atoms, fs, eff)
            maxf = _max_force(atoms, dom)
            maxs = _cell_residual(atoms, tstress, cell_dof)

            _log_relax(atoms, params, dom, step, maxf, dt, alpha, ncg; init = false)
            if step % params[:output_interval] == 0
                _write_relax_frame(atoms, params, dom, "step_$(step).xyz")
            end
        end

        # Out of iterations, or the criterion was met. In the latter case the
    # acceptance test decides whether it meant anything.
    _relax_done(maxf, maxs, params, stol) || break
    verified && break                       # already running on tight solvers
    verified = true
    fft_before = fs.pmeinfo.fftsize
    ok, maxf, eff = _verify_convergence!(atoms, fs, params, dom)
    # _verify_convergence! rebuilt the accelerator at the CURRENT cell, and
    # `len0` means "the edge lengths the live mesh was sized for".
    lat_v = Array(atoms.info[:lattice])
    len0 = (norm(lat_v[1, :]), norm(lat_v[2, :]), norm(lat_v[3, :]))
    maxs = _cell_residual(atoms, tstress, cell_dof)
    ok = ok && maxs < stol
    # `escalated` records that the check FAILED and the run continued — not
    # merely that it ran. A healthy run must not be told to tighten anything.
    ok && break
    escalated = true
    # The criterion was within the solver error. Do NOT report it as
    # converged — but do not stop either: keep relaxing on the forces that
    # are now visible. Escalation happens at most once, so the run stays
    # bounded. The FFT grids are logged because the refinement is a change
    # of quadrature, not a strict improvement of every number (see
    # _verify_convergence!) — knowing which grid a result came from is what
    # makes an anomalous max|F| diagnosable after the fact.
    Cores.isroot(dom) &&
        @info "relax!: max|F| met ftol at the requested solver settings, but " *
              "re-evaluating with cgtol x $(get(params, :verify_cgtol_factor, 0.1)) " *
              "and PME mesh spacing x $(get(params, :verify_mesh_factor, 0.5)) " *
              "(FFT grid $fft_before -> $(fs.pmeinfo.fftsize)) gives " *
              "max|F| = $maxf — the criterion was within the solver error. " *
              "Continuing on the tightened solvers"
    step < params[:maxsteps] || break
    end

    converged = _relax_done(maxf, maxs, params, stol)

    _write_relax_frame(atoms, params, dom, "relaxed.xyz")

    if Cores.isroot(dom)
        if converged
            @info "relax!: converged in $step step(s), max|F| = $maxf " *
                  "(ftol = $(params[:ftol]), verified on solvers tightened by " *
                  "$(get(params, :verify_cgtol_factor, 0.1)) / " *
                  "$(get(params, :verify_mesh_factor, 0.5)))" *
                  (escalated ? " — the run had to tighten them mid-flight, so " *
                               "consider setting :cgtol / :pme_meshsize there " *
                               "from the start. This second convergence was " *
                               "NOT re-verified at a third level" : "")
        else
            @warn "relax!: NOT converged after $step step(s) — max|F| = $maxf " *
                  "still above ftol = $(params[:ftol])" *
                  (escalated ? " (measured on the tightened solvers)" : "") *
                  ". The written structure is partially relaxed; raise " *
                  ":maxsteps or check :dt_max / :dmax" *
                  (escalated ? ", and note that the requested :cgtol / " *
                               ":pme_meshsize were too loose to resolve :ftol" : "")
        end
    end

    return atoms, converged
end

"""
    _log_relax(atoms, params, dom, step, maxf, dt, alpha, ncg; init)

Append one line to `log.txt` (root only). `init = true` opens the file for
writing and emits the step-0 line (the unrelaxed starting point), so the
log always begins at the input structure.

Columns: `step | max|F| | energy | σ Voigt-6 | volume | dt | α | ncg |
lattice(9)`. The lattice is logged because with `cell_dof != :none` it is a
degree of freedom, and a relaxation whose cell went somewhere unexpected is
otherwise undiagnosable after the fact.
The stress is the static one (`kinetic = false`) — the FIRE
pseudo-velocity is not a physical velocity and must never enter it.
"""
function _log_relax(atoms::Atoms, params, dom, step, maxf, dt, alpha, ncg; init::Bool)
    sigma = stress(atoms, false)
    energy = Array(atoms.info[:potential_energy])[1]
    vol = volume(atoms)
    Cores.isroot(dom) || return nothing
    lat = Matrix(atoms.info[:lattice]) .* UNITS[:Å]
    open(joinpath(params[:output_dir], "log.txt"), init ? "w" : "a") do io
        @printf(io, "%6d %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %6d %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e \n",
                step, maxf, energy,
                (sigma .* UNITS[:GPa])...,
                vol * UNITS[:Å]^3, dt, alpha, ncg, lat...)
    end
    return nothing
end

"""
    _write_relax_frame(atoms, params, dom, name)

Gather the whole-system frame (collective — every rank participates) and
write it as extxyz from the root.
"""
function _write_relax_frame(atoms::Atoms, params, dom, name::String)
    frame = Cores.gather_frame(atoms, dom)
    Cores.isroot(dom) || return nothing
    open(joinpath(params[:output_dir], name), "w") do io
        writexyz(io, frame, params)
    end
    return nothing
end

"""
    default_params(::Val{:Relax}, params) -> NamedTuple

Default params for `FMD.app[:Relax]`. The shell-model placeholders are
intentionally zero — production runs splat values loaded from
`SplineP.jld2` / `ACEP.jld2` on top of these defaults, exactly as
`:NPTMD` does.

# Defaults

| Key | Default | Meaning |
|---|---|---|
| `maxsteps` | `2000` | Iteration cap (reaching it = not converged) |
| `ftol` | `1e-4` | Convergence threshold on max\\|F\\| (Ha/Bohr) |
| `dt0`, `dt_max` | `1/UNITS[:fs]`, `10/UNITS[:fs]` | FIRE step size |
| `dmax` | `0.1/UNITS[:Å]` | Per-step displacement cap |
| `output_interval` | `50` | Snapshot interval |
| `cell_dof` | `:none` | Cell degrees of freedom (`:none`/`:iso`/`:diag`/`:full`) |
| `emax` | `1e-3` | Per-step strain cap |
| `cgtol`, `cgmaxiter` | `2e-6`, `30` | Per-step shell relaxation. Tighter than the `:NPTMD` default by 10× — `ftol` has to clear it by the entry-guard margin |
| `efield` | `[0,0,0]` | External field |
| `skin` | `0.1` | Neighbor list skin |
| `rc` | `9/Å` | Coulomb cutoff |
| `e` | `1e5` | Buckingham cap |
| `qc`, `qs`, `k2`, `k4` | zeros | Shell-model placeholders |
| `eta_c`, `eta_s` | `1e-5/Å` | Gaussian widths |
| `a`, `rho`, `b`, `c` | zeros / 1e-10 | Buckingham per pair |

The Ewald α is auto-determined from (`rc`, `pme_meshsize`) by
error-estimate minimization (Issue #216); pass `:alpha` to override.

# Required (caller-provided)

`:input`, `:output_dir`, `:elements`, `:pme_meshsize`, and the shell
parameters from the fit. `:acep` / `:splp` are optional (omit both for the
pure shell model).
"""
function default_params(::Val{:Relax}, params)
    _assert_elements(params)
    nelem = length(params[:elements])

    a = (
        maxsteps = 2000,
        ftol = 1e-4,
        dt0 = 1.0/(UNITS[:fs]),
        dt_max = 10.0/(UNITS[:fs]),
        dmax = 0.1/(UNITS[:Å]),
        output_interval = 50,
        cell_dof = :none,
        emax = 1e-3,
        cgtol = 2e-6,
        cgmaxiter = 30,
        efield = [0.0, 0.0, 0.0],
        skin = 0.1,

        rc = 9.0/(UNITS[:Å]),
        e = 1e5,

        qc = zeros(nelem),
        qs = zeros(nelem),
        k2 = zeros(nelem),
        k4 = zeros(nelem),
        eta_c = fill(1e-5/(UNITS[:Å]), nelem),
        eta_s = fill(1e-5/(UNITS[:Å]), nelem),
        a = zeros(nelem,nelem),
        rho = fill(1e-10, (nelem,nelem)),
        b = zeros(nelem,nelem),
        c = zeros(nelem,nelem),
    )
end

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:Relax](params)`
#
# Structure relaxation (fixed cell) with FIRE 2.0 on the selected backend.
# Reads the first frame of `params[:input]` (extxyz), tags species using
# `params[:elements]`, creates `params[:output_dir]` if needed, and runs
# [`relax!`](@ref).
#
# Returns 0 when the relaxation converged and 1 when it did not — a
# non-converged run is a legitimate outcome to act on, not a crash, and a
# shell script driving a batch of relaxations needs to be able to tell.
# --------------------------------------------------------------------------
app[:Relax] = function (params)
    validate_params(:Relax, params)

    atoms = readxyz(open(params[:input], "r"))[1]

    mkpath(params[:output_dir])

    setelemid!(atoms, params[:elements])

    _, converged = relax!(atoms, params)

    return converged ? 0 : 1
end
