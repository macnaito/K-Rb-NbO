"""
    FMD.Calculators

Energy / force evaluators. Every evaluator follows the convention

```julia
calculate!(evaluator, atoms::Atoms, [neighborlist(s),] [cell info,] params)
```

and **adds** its contribution into `atoms.info[:potential_energy]`,
`atoms.info[:virial]`, `atoms.arrays[:forces]`, and (if relevant)
`atoms.arrays[:shellforces]`. Call [`clear!`](@ref) on `atoms` before the
first evaluator each step.

# Evaluators

| Type | Purpose |
|---|---|
| [`CoreShell`](@ref)         | Top-level driver that relaxes shells (CG / linear response) and wraps the Coulomb + Buckingham + Spring + ExternalField stack |
| [`CoulombBuckingham`](@ref) | Long-range Coulomb (Ewald / PME) + short-range Buckingham repulsion / dispersion |
| [`NLHParam`](@ref)          | NLH screened-nuclear repulsion — Δ-learning baseline for the short range the training data never reaches |
| [`OnlyShellForce`](@ref)    | Marker that asks `CoulombBuckingham` to compute *only* shell forces |
| [`Spring`](@ref)            | Anisotropic core–shell harmonic + quartic spring |
| [`ExternalField`](@ref)     | Constant external electric field |
| [`EwaldInfo`](@ref)         | k-space Ewald accelerator |
| [`PMEInfo`](@ref)           | Particle-mesh Ewald accelerator (recommended for large cells) |
| [`SplineParam`](@ref)       | 2/3-body spline MLIP |
| [`ACEParam`](@ref)          | ACE-style MLIP |
| [`PolarizationUmbrella`](@ref) | Polarization-bias term for umbrella sampling (mode amplitude + homogeneity) |
| [`PolarizationMapUmbrella`](@ref) | Restrains the local polarization to an arbitrary per-group target map |
| [`PolarizationGauge`](@ref) | Declared zero-point of the polarization (shared by every window) |
| [`SurfaceTerm`](@ref)       | Depolarization field of an anisotropic external medium (generalised Ewald surface term) |

The dipole helpers ([`calcdipole!`](@ref),
[`calc_dipole_phys!`](@ref), [`total_mfrac_ref`](@ref)) compute macroscopic
polarization from the same atomic configuration.
"""
module Calculators

using ..FMD: FPPREC
using ..FMD.Cores
import KernelAbstractions

export calculate!, CoreShell

"""
    calculate!(evaluator, atoms::Atoms, ...)

Add the contribution of `evaluator` to `atoms.info[:potential_energy]`,
`atoms.info[:virial]`, and the per-atom force / shell-force arrays.
Implementations live alongside each evaluator type — see [`CoreShell`](@ref),
[`CoulombBuckingham`](@ref), [`Spring`](@ref), [`ExternalField`](@ref),
[`EwaldInfo`](@ref), [`PMEInfo`](@ref), [`SplineParam`](@ref),
[`ACEParam`](@ref), [`PolarizationUmbrella`](@ref).

Always [`clear!`](@ref) `atoms` before the first `calculate!` of an MD or
fitting step; subsequent `calculate!` calls accumulate.
"""
function calculate! end

# Copy a host array to the compute backend (plain copy on the CPU backend).
# Keeps the calculators free of direct CUDA references — FFT plans and
# kernels dispatch on the array type instead. Shared by PME (grid tables,
# grid virial) and the DD reduction below.
function _devcopy(backend, x::AbstractArray)
    y = KernelAbstractions.allocate(backend, eltype(x), size(x))
    copyto!(y, x)
    return y
end

"""
    add_reduced_ev!(atoms, pe_local, virial_local)

Add an energy + Voigt-6 virial contribution to `atoms.info` under the
Phase-1 reduction regime: `pe_local`/`virial_local` are **owned partial
sums** (per-atom sums over this rank's owned rows — the ghost model
guarantees they tile the system), globally summed in one fused Allreduce
under domain decomposition so `info` always holds the whole-system value
on every rank.

Replicated quantities must **bypass** this helper and be added directly,
or they would be multiplied by the rank count. "Replicated" means
*computed from globally-reduced inputs* so the value is already identical
on every rank — e.g. the k-space energy **after** the structure factor
S(k) has been Allreduced, the PME grid virial **after** the charge-grid
Allreduce, the charged-cell correction from the Allreduced total charge.
A per-atom quantity summed over owned rows is never replicated, even when
it "belongs" to a reciprocal-space or bias term (the umbrella's atom-loop
virial, for instance, is an owned partial sum; only its group-lattice
terms are replicated).

The `SerialDomain` branch performs bit-identically the same operations as
the pre-DD code (no host detour), so serial and GPU paths are unaffected.
"""
function add_reduced_ev!(atoms::Atoms, pe_local, virial_local)
    dom = domain(atoms)
    if dom isa MPIDomain
        length(virial_local) == 6 ||
            error("add_reduced_ev!: virial_local must be Voigt-6 (got length " *
                  "$(length(virial_local)))")
        # 7 doubles: a host buffer is the right shape for a scalar-sized
        # Allreduce even on the GPU backend (the Array(virial_local) below
        # is the stream drain for a device-reduced virial).
        buf = Vector{FPPREC}(undef, 7)
        buf[1] = FPPREC(pe_local)
        buf[2:7] .= Array(virial_local)
        allreduce_sum!(dom, buf)
        atoms.info[:potential_energy] .+= buf[1]
        backend = KernelAbstractions.get_backend(atoms.info[:virial])
        if backend isa KernelAbstractions.CPU
            # Exactly the pre-device code path: zero extra allocation.
            atoms.info[:virial] .+= @view buf[2:7]
        else
            # info[:virial] lives on the device; copy the host Voigt-6 over
            # before accumulating. Materialized slice, not @view: fast
            # host→device copyto! only exists for dense Array sources (a
            # SubArray source falls back to scalar setindex! and errors).
            atoms.info[:virial] .+= _devcopy(backend, buf[2:7])
        end
    else
        atoms.info[:potential_energy] .+= pe_local
        atoms.info[:virial] .+= virial_local
    end
    return nothing
end

include("CoulombBuckingham.jl")
include("Spring.jl")
include("ExternalField.jl")
include("Ewald.jl")
include("CoreShell.jl")
include("NLH.jl")
include("PME.jl")
include("PMEError.jl")
include("PMESlab.jl")
include("ACE.jl")
include("ACEFast.jl")
include("ACEFastCUDA.jl")
include("ACEPrecond.jl")
include("Spline.jl")
include("Dipole.jl")
include("PolField.jl")
include("PolarizationGauge.jl")
include("SurfaceTerm.jl")
include("Umbrella.jl")
include("MapUmbrella.jl")
include("PolMap.jl")
# Last on purpose: the analytic-HVP layer specializes on EwaldInfo /
# PMEInfo and orchestrates every term (Issue #147) — see HVP.jl header.
include("HVP.jl")

end
