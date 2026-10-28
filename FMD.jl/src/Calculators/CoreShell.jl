using LinearAlgebra
import AcceleratedKernels as AK
import KernelAbstractions as KA
using ..FMD.Cores
using StaticArrays

export CoreShell, calculate!, linear_response!, add_baseline!, subtract_baseline!

"""
    CoreShell

Driver for the core-shell potential. A `CoreShell()` instance is dispatched
through [`calculate!`](@ref) to:

1. relax the shell displacements `atoms.arrays[:shelldisp]` until shell
   forces fall below `prms[:cgtol]` (CG with restart);
2. evaluate cores forces, energy, and virial through the Coulomb +
   Buckingham + Spring + ExternalField stack.

The shell relaxation is the inner loop of every MD step, so the choice of
`(:cgtol, :cgmaxiter)` is the dominant performance / accuracy trade-off.
"""
struct CoreShell end

"""
    calculate!(::CoreShell, atoms::Atoms, prms::NamedTuple) -> Int

Convenience method that builds a fresh [`EwaldInfo`](@ref) and a fresh
neighbour list every call. Suitable for the fitting paths where each frame
has a different cell and no neighbour list is reused. Returns the number
of inner force evaluations used by shell relaxation (same counting as the
workhorse method below).
"""
function calculate!(::CoreShell, atoms::Atoms, prms::NamedTuple)
    prms = (; prms..., alpha = ewaldcoeff(prms))
    ewald = EwaldInfo(atoms.info[:lattice], prms)
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    return calculate!(CoreShell(), atoms, nl, ewald, prms)
end

"""
    add_baseline!(atoms::Atoms, params) -> atoms

Evaluate every **fixed** (non-fitted) term of the potential into `atoms`:
the core-shell stack ([`CoreShell`](@ref)) plus, when `params[:nlh]` is
present, the NLH screened-nuclear repulsion ([`NLHParam`](@ref)).

This is the Δ-learning baseline. `fitting!` subtracts exactly this from the
reference E/F/V (via [`subtract_baseline!`](@ref)) and the prediction / MD
paths add exactly this back, so the round trip is exact by construction —
a term added on one side only would silently bias every fitted coefficient.
Call [`clear!`](@ref) first; the MLIP contribution is added on top
afterwards.

`params[:nlh]` is a shell-independent core–core term, so it does not change
where the shells relax to and is evaluated once, after the shell CG.
"""
function add_baseline!(atoms::Atoms, params)
    calculate!(CoreShell(), atoms, params)
    haskey(params, :nlh) && calculate!(params[:nlh], atoms)
    return atoms
end

"""
    subtract_baseline!(dataset, params) -> dataset

Subtract [`add_baseline!`](@ref) from the reference energy / forces / virial
of every frame in `dataset`, in place, leaving exactly the residual the MLIP
is asked to model. Shared by the Spline and ACE `build_normal_system`
paths so the two cannot drift apart.
"""
function subtract_baseline!(dataset, params)
    for i in eachindex(dataset)
        atoms_copy = deepcopy(dataset[i])
        clear!(atoms_copy)
        add_baseline!(atoms_copy, params)
        dataset[i].arrays[:forces]         .-= atoms_copy.arrays[:forces]
        dataset[i].info[:virial]           .-= atoms_copy.info[:virial]
        dataset[i].info[:potential_energy] .-= atoms_copy.info[:potential_energy]
    end
    return dataset
end

# Fallback for Coulomb solvers with no shell-force-only variant: do the
# full evaluation, which is a superset (it writes the same shell forces
# plus energy/virial/core outputs the OnlyShellForce contract leaves
# unspecified). PMEInfo specializes this in PME.jl (Issue #157); EwaldInfo
# does not — it is the fitting-path solver, not the production one, and
# its reciprocal cost is dominated by the O(N²) structure factor rather
# than by the virial reduction this saves.
calculate!(coulomb::CoulombInfo, ::OnlyShellForce, atoms::Atoms, params) =
    calculate!(coulomb, atoms, params)

"""
    shell_forces!(atoms, nl, coulomb, prms)

Internal: clear and (re-)evaluate the shell forces stack — Coulomb
([`EwaldInfo`](@ref) or [`PMEInfo`](@ref)), [`OnlyShellForce`](@ref) of
Buckingham, [`Spring`](@ref), [`ExternalField`](@ref), and the optional
[`SurfaceTerm`](@ref) / [`PolarizationUmbrella`](@ref). Both bias terms
run inside the CG loop on purpose: the shells must relax against the bias
field (for the surface term that relaxation IS the dielectric screening of
the depolarization field). Used by the CG inner loop.

Every stack member is asked for shell forces only. `atoms.info` and
`atoms.arrays[:forces]` are therefore left in an unspecified state on
return — the CG loop reads `arrays[:shellforces]` alone, and
[`core_forces!`](@ref) recomputes the full stack once the shells are
relaxed. This was already true of the Buckingham and (for a shell-less
species) Spring terms; Issue #157 made it true of the Coulomb term too.
"""
function shell_forces!(atoms, nl, coulomb, prms)
    clear!(atoms, (:potential_energy, :virial, :shellforces, :forces, :peratom_virial))
    calculate!(coulomb, OnlyShellForce(), atoms, prms)
    calculate!(CoulombBuckingham(), OnlyShellForce(), get_backend(atoms), atoms, nl, prms)
    calculate!(Spring(), atoms, prms)
    calculate!(ExternalField(), atoms, prms)
    if haskey(prms, :surface_term)
        surface_term!(atoms, prms[:surface_term], prms)
    end
    if haskey(prms, :polarization_umbrella)
        umbrella_p!(atoms, prms[:polarization_umbrella], prms)
    end
end

"""
    core_forces!(atoms, nl, coulomb, prms)

Internal: full-stack force evaluation (cores + shells, energy, virial)
once the shells are at their relaxed positions. Run once at the end of
each shell CG loop.
"""
function core_forces!(atoms, nl, coulomb, prms)
    clear!(atoms, (:potential_energy, :virial, :shellforces, :forces, :peratom_virial))
    calculate!(coulomb, atoms, prms)
    calculate!(CoulombBuckingham(), atoms, nl, prms)
    calculate!(Spring(), atoms, prms)
    calculate!(ExternalField(), atoms, prms)
    if haskey(prms, :surface_term)
        surface_term!(atoms, prms[:surface_term], prms)
    end
    if haskey(prms, :polarization_umbrella)
        umbrella_p!(atoms, prms[:polarization_umbrella], prms)
    end
end

# CG reduction helpers (Phase 1). The local expressions are exactly the
# historical serial ones; under DD each pair is fused into one Allreduce
# so every convergence branch is decided from rank-uniform values — the
# iteration count and every uphill correction are then structurally
# lockstep across ranks (the AbstractDomain collective invariant).
# Serial (`SerialDomain`) results are bit-identical to the historical
# loop. The CG loop itself is empty-rank safe (zero partial sums that
# still ride every collective); the force stack it drives is hardened for
# empty ranks separately (Ewald/PME are, the remaining short-range kernels
# with the DD driver integration).
_cg_sum2(::Cores.SerialDomain, a, b) = (a, b)
function _cg_sum2(dom::Cores.MPIDomain, a, b)
    buf = FPPREC[a, b]
    Cores.allreduce_sum!(dom, buf)
    return (buf[1], buf[2])
end
_cg_maxabs(dom, x) =
    Cores.allreduce_max(dom, isempty(x) ? zero(FPPREC) : FPPREC(maximum(abs, x)))
_cg_lsum(x) = isempty(x) ? zero(FPPREC) : FPPREC(sum(x))
_cg_lsum_ak(x) = isempty(x) ? zero(FPPREC) : FPPREC(AK.sum(x))
_cg_ldot(x, y) = isempty(x) ? zero(FPPREC) : FPPREC(dot(x, y))

"""
    calculate!(::CoreShell, atoms::Atoms, nl::NeighborList,
               coulomb::CoulombInfo, prms::NamedTuple) -> Int

Workhorse method: relaxes shell displacements with a Polak–Ribière CG (with
a gradient-projection step correction when the search direction goes
uphill — `gₖ·pₖ` against `gₖ₋₁·pₖ`, no energy is involved) until
`max(|shell forces|) < prms[:cgtol]` or `prms[:cgmaxiter]` is reached, then
evaluates the full-stack core forces / energy / virial. Returns the number
of inner force evaluations (identical on every rank under domain
decomposition — all convergence branching uses Allreduce-derived values).
The count covers the CG loop only: the one shell-force evaluation before
the loop and the final core evaluation are not included, so the total
force-evaluation cost of a call is `ncg + 1` shell + 1 core. The loop
guard is `fcount < cgmaxiter` and the uphill-correction branch adds a
second evaluation inside an iteration, so the returned value can reach
`cgmaxiter + 1`.

Also stores the rms of the shell-force components at loop exit in
`atoms.info[:shell_residual_rms]` (1-element vector, a.u. Ha/Bohr — the
units of `cgtol`),
`sqrt(Σ F_s² / 3N)` over the same components the stopping test takes the
maximum of. It is taken from the CG loop's own shell forces, before the
full-stack core evaluation re-evaluates `:shellforces` at the same shell
positions. The stopping test pins the maximum at `≈ cgtol`; the rms is
what it leaves free.

Reuses `nl` (built by the caller via [`neighborlist`](@ref)) and `coulomb`
(an [`EwaldInfo`](@ref) or [`PMEInfo`](@ref) — `cellupdate!` is invoked
internally so passing a stale `coulomb` after a cell change is safe).
"""
function calculate!(::CoreShell, atoms::Atoms, nl::NeighborList, coulomb::CoulombInfo, prms::NamedTuple)
    dom = Cores.domain(atoms)
    cellupdate!(coulomb, atoms.info[:lattice], prms)
    arr = atoms.arrays
    gₖ = similar(arr[:shellforces])
    pₖ = similar(arr[:shellforces])
    gₖ₋₁ = similar(arr[:shellforces])
    xₖ₋₁ = similar(arr[:shellforces])
    kvals = KernelAbstractions.zeros(get_backend(atoms), FPPREC, length(atoms))
    k2tpl = prms[:k2]
    length(atoms) > 0 && AK.map!(e -> k2tpl[e], kvals, atoms.arrays[:elemid])

    shell_forces!(atoms, nl, coulomb, prms)

    gₖ .= -arr[:shellforces]
    pₖ .= -gₖ

    max_force = _cg_maxabs(dom, arr[:shellforces])
    fcount = 0
    while max_force > prms[:cgtol] && fcount < prms[:cgmaxiter]
        fcount += 1

        gₖ₋₁ .= gₖ
        xₖ₋₁ .= arr[:shelldisp]

        num, den = _cg_sum2(dom, _cg_lsum(-gₖ .* pₖ), _cg_lsum_ak(kvals .* pₖ .^ 2))
        α = num / den
        arr[:shelldisp] .= xₖ₋₁ + α * pₖ

        shell_forces!(atoms, nl, coulomb, prms)
        gₖ .= -arr[:shellforces]

        # gp/gmp are computed once and reused in the correction below —
        # the historical loop evaluated the same sums twice with
        # identical values.
        gp, gmp = _cg_sum2(dom, _cg_lsum(gₖ .* pₖ), _cg_lsum(gₖ₋₁ .* pₖ))
        if abs(gp) > 0.5 * abs(gmp)
            α -= gp * α / (gp - gmp)
            @. arr[:shelldisp] = xₖ₋₁ + α * pₖ
            shell_forces!(atoms, nl, coulomb, prms)
            gₖ .= -arr[:shellforces]
            fcount += 1
        end

        βnum, βden = _cg_sum2(dom, _cg_ldot(gₖ, (gₖ - gₖ₋₁)), _cg_ldot(gₖ₋₁, gₖ₋₁))
        β = βnum / βden
        β = max(zero(β), β)
        pₖ .= β * pₖ - gₖ

        max_force = _cg_maxabs(dom, arr[:shellforces])
    end

    sumsq, _ = _cg_sum2(dom, _cg_ldot(arr[:shellforces], arr[:shellforces]), zero(FPPREC))
    # 1-element vector, like :potential_energy: change_arrtyp converts every
    # info value as an array.
    atoms.info[:shell_residual_rms] = [sqrt(sumsq / (3 * Cores.n_global(dom, atoms)))]

    core_forces!(atoms, nl, coulomb, prms)

    return fcount
end

"""
    linear_response!(atoms::Atoms, prms::NamedTuple)
    linear_response!(atoms::Atoms, nl::NeighborList,
                     coulomb::CoulombInfo, prms::NamedTuple; tol, maxiter)

Compute the BEC and dielectric tensor (per atom and per cell) of `atoms`
under the current shell-model parameters in `prms`.

The single-argument convenience method builds a fresh `EwaldInfo` and
neighbour list and clears `prms[:efield]` to ensure a zero-field response.
The full method takes pre-built objects and exposes `tol` / `maxiter` for
the linear solve.

Outputs are written into `atoms.arrays[:becex]`, `:becey`, `:becez` and
`atoms.info[:dielectric_tensor]`. Used by `FMD.app[:FitShellModel]`.
"""
function linear_response!(atoms::Atoms, prms::NamedTuple)
    prms = (; prms..., alpha = ewaldcoeff(prms), efield = zeros(FPPREC, 3))
    ewald = EwaldInfo(atoms.info[:lattice], prms)
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    linear_response!(atoms, nl, ewald, prms)
end

# The linear solve itself (cg_hessian_free!) and the full
# linear_response! method live in HVP.jl: they run on the analytic
# Hessian-vector product (Issue #147), whose workspace type specializes
# on EwaldInfo/PMEInfo — not visible from this file (include order). The
# FD directional derivatives that used to live here (jvp! /
# dF_du_dirderiv!) survive as the test oracles fd_jvp! /
# fd_dFdu_dirderiv! in test/helpers.jl.
