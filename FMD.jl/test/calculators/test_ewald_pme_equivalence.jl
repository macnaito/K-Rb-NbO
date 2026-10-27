# test/calculators/test_ewald_pme_equivalence.jl
#
# Critical equivalence test: Ewald (CPU + reciprocal direct k-sum) and
# PME (FFT mesh) are two solvers for the same physical Hamiltonian.
# When the PME mesh is fine enough, they must match.
#
# PME is backend-generic (#51): the whole suite runs on `Array` in the
# default profile, and additionally on `CuArray` when FMD_TEST_GPU=1 and
# CUDA is functional.
#
# Issue #9: the original version compared forces only on the perfect
# rocksalt `make_nacl_64()`, where every atom's true force is zero by
# symmetry, so `max|F_ew - F_pme| < 1e-3` passed at the noise floor even
# when the Ewald reciprocal force was identically zero (the #7 bug).
#
# Two design rules learned the hard way and applied here:
#   1. ASYMMETRIC fixture (displaced cores / shells) so the true forces are
#      O(0.1-1) and the comparison is not vacuous.
#   2. Validate the RECIPROCAL part IN ISOLATION (no CoulombBuckingham) and
#      normalize errors by the RECIPROCAL magnitude.  Otherwise the larger
#      real-space contribution dilutes — and hides — a reciprocal error: at
#      alpha=0.30 the reciprocal force is ~66% of the total here, so a 2e-2
#      tolerance on the *combined* force would still admit a ~3% reciprocal
#      error (and far more at larger alpha, where the reciprocal part is
#      small).  The reciprocal-isolated checks below are tight (1e-3) because
#      that is the precision the two solvers actually agree to at this mesh.

using Test
using CUDA
using Random
using LinearAlgebra

# PME arrays types to test: CPU always; GPU additionally under the GPU profile.
_epe_arrtyps() = (get(ENV, "FMD_TEST_GPU", "0") == "1" && CUDA.functional()) ?
    (Array, CuArray) : (Array,)

# reciprocal + self ONLY (no CoulombBuckingham) — isolates the k-space solver.
# The Ewald side is the CPU reference regardless of the PME array type.
function ewald_recip_EFS!(atoms, prms)
    a = deepcopy_atoms(atoms)
    ew = Calculators.EwaldInfo(a.info[:lattice], prms)
    reset!(a)
    Calculators.calculate!(ew, a, prms)
    return (a.info[:potential_energy][1], copy(a.arrays[:forces]),
            copy(a.arrays[:shellforces]), copy(a.info[:virial]))
end

function pme_recip_EFS!(atoms_cpu, prms, arrtyp)
    a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), arrtyp)
    p = arrtyp_prms(arrtyp, prms)
    pme = Calculators.PMEInfo(a.info[:lattice], p)
    reset!(a)
    Calculators.calculate!(pme, a, p)
    return (Array(a.info[:potential_energy])[1], Array(a.arrays[:forces]),
            Array(a.arrays[:shellforces]), Array(a.info[:virial]))
end

# full CoulombBuckingham (real) + reciprocal/self — the production stack.
function ewald_EFS!(atoms, prms)
    a = deepcopy_atoms(atoms)
    nl = neighborlist(a, prms[:rc], 0.0)
    ew = Calculators.EwaldInfo(a.info[:lattice], prms)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
    Calculators.calculate!(ew, a, prms)
    return (a.info[:potential_energy][1], copy(a.arrays[:forces]),
            copy(a.arrays[:shellforces]))
end

function pme_EFS!(atoms_cpu, prms, arrtyp)
    a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), arrtyp)
    p = arrtyp_prms(arrtyp, prms)
    nl = neighborlist(a, p[:rc], 0.0)
    pme = Calculators.PMEInfo(a.info[:lattice], p)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, p)
    Calculators.calculate!(pme, a, p)
    return (Array(a.info[:potential_energy])[1], Array(a.arrays[:forces]),
            Array(a.arrays[:shellforces]))
end

# max-norm error of A vs B, normalized by |B| (caller passes the reciprocal
# magnitude as B so real-space cannot mask the discrepancy).
reln(A, B) = maximum(abs.(A .- B)) / max(maximum(abs, B), eps())

@testset "Ewald (CPU) ↔ PME equivalence [$arrtyp]" for arrtyp in _epe_arrtyps()

    # ----------------------------------------------------------------
    # Symmetric rocksalt: energy agreement + mesh convergence (forces are
    # ~0 by symmetry here, so this block checks ENERGY only).
    # ----------------------------------------------------------------
    @testset "Symmetric NaCl 64: energy agrees at fine mesh" begin
        atoms = make_nacl_64()
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.20 / UNITS[:Å]))
        E_ew, _, _ = ewald_EFS!(atoms, prms)
        E_pme, _, _ = pme_EFS!(atoms, prms, arrtyp)
        @test isapprox(E_ew, E_pme; rtol = 1e-3)
    end

    @testset "PME → Ewald energy as mesh refines" begin
        atoms = make_nacl_64()
        prms_coarse = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                                     rc_factor = 0.45)...,
                       pme_meshsize = FMD.FPPREC(0.50 / UNITS[:Å]))
        prms_fine = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                                   rc_factor = 0.45)...,
                     pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))
        E_ew, _, _ = ewald_EFS!(atoms, prms_coarse)  # reference
        E_coarse, _, _ = pme_EFS!(atoms, prms_coarse, arrtyp)
        E_fine, _, _ = pme_EFS!(atoms, prms_fine, arrtyp)
        @test abs(E_fine - E_ew) < abs(E_coarse - E_ew) + 1e-12
    end

    # ----------------------------------------------------------------
    # RECIPROCAL-ISOLATED: the rigorous reciprocal-space correctness check.
    # No real-space term, errors normalized by the reciprocal magnitude.
    # ----------------------------------------------------------------
    @testset "Reciprocal-only PME ↔ Ewald (recip-normalized, fine mesh)" begin
        Random.seed!(42)
        atoms = make_nacl_64()
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.03 .* randn(size(atoms.arrays[:shelldisp]))
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                              rc_factor = 0.45, with_shell = true)...,
                pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        E_ew, F_ew, SF_ew, W_ew = ewald_recip_EFS!(atoms, prms)
        E_pme, F_pme, SF_pme, W_pme = pme_recip_EFS!(atoms, prms, arrtyp)

        # The reciprocal force / shellforce / virial must be genuinely large
        # — these guards are what the #7 reciprocal-force bug (≡ 0) trips.
        @test maximum(abs, F_ew)  > 1e-2
        @test maximum(abs, SF_ew) > 1e-3
        @test maximum(abs, W_ew)  > 1e-1

        # Normalized by the RECIPROCAL magnitude (not the combined force),
        # PME and the analytic Ewald reciprocal agree to the mesh level.
        @test abs(E_pme - E_ew) / abs(E_ew) < 1e-3
        @test reln(F_pme,  F_ew)  < 1e-3
        @test reln(SF_pme, SF_ew) < 1e-3
        @test reln(W_pme,  W_ew)  < 1e-3
    end

    @testset "Reciprocal force → Ewald as mesh refines (recip-normalized)" begin
        Random.seed!(44)
        atoms = make_nacl_64()
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8, rc_factor = 0.45)
        prms_coarse = (; base..., pme_meshsize = FMD.FPPREC(0.40 / UNITS[:Å]))
        prms_fine   = (; base..., pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        _, F_ew, _, _ = ewald_recip_EFS!(atoms, prms_coarse)   # mesh-independent reference
        _, F_coarse, _, _ = pme_recip_EFS!(atoms, prms_coarse, arrtyp)
        _, F_fine, _, _ = pme_recip_EFS!(atoms, prms_fine, arrtyp)

        @test reln(F_fine, F_ew) < reln(F_coarse, F_ew)
        @test reln(F_fine, F_ew) < 1e-3
    end

    # ----------------------------------------------------------------
    # Combined production stack (CB real + reciprocal).  The real-space
    # CoulombBuckingham is identical on both sides and cancels in the
    # difference, so we still normalize by the reciprocal magnitude — this
    # is the integration-level analogue of the isolated check above and is
    # the assertion that fails under the #7 cos/sin swap.
    # ----------------------------------------------------------------
    @testset "Combined force agreement (recip-normalized)" begin
        Random.seed!(43)
        atoms = make_nacl_64()
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        _, F_recip, _, _ = ewald_recip_EFS!(atoms, prms)   # reciprocal scale
        E_ew, F_ew, _ = ewald_EFS!(atoms, prms)
        E_pme, F_pme, _ = pme_EFS!(atoms, prms, arrtyp)

        scale = maximum(abs, F_recip)
        @test scale > 1e-2
        @test isapprox(E_ew, E_pme; rtol = 1e-3)
        # Difference is purely reciprocal (real-space cancels); divide by the
        # reciprocal scale so it is not masked by the real-space part.
        @test maximum(abs.(F_ew .- F_pme)) / scale < 1e-3
        # Newton III: analytic Ewald is exact to round-off; PME momentum
        # conservation is only approximate (B-spline + FFT).
        @test maximum(abs, vec(sum(F_ew, dims = 1)))  < 1e-6
        @test maximum(abs, vec(sum(F_pme, dims = 1))) < 1e-4
    end

    # ----------------------------------------------------------------
    # Charged cell (Issue #15): both solvers must carry the same
    # charged-cell correction -πQ²/(2α²V), and the corrected PME stack
    # must be α-independent.
    # ----------------------------------------------------------------
    @testset "Charged cell: PME ↔ Ewald + α independence (Issue #15)" begin
        Random.seed!(45)
        atoms = make_nacl_64()
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.03 .* randn(size(atoms.arrays[:shelldisp]))

        # Both charge sums non-zero (Σqs ≠ 0 pins the qs part of Q on the
        # PME side too; all stock fixtures have Σqs = 0).
        chprms(α, qcv) = (; nacl_params(atoms; alpha = α, ewald_tolerance = 1e-8,
                                        rc_factor = 0.45, with_shell = true)...,
                          qc = qcv, qs = FMD.FPPREC[-0.2, +0.3],
                          pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        # Fixed α, reciprocal-isolated: PME must match the corrected Ewald
        # (energy, forces, shell forces, virial) on a non-neutral cell
        # (Σqc = +3.2, Σqs = +3.2 ⇒ Q = +6.4 e).
        prms = chprms(0.30, FMD.FPPREC[+1.0, -0.9])
        E_ew, F_ew, SF_ew, W_ew = ewald_recip_EFS!(atoms, prms)
        E_pme, F_pme, SF_pme, W_pme = pme_recip_EFS!(atoms, prms, arrtyp)
        @test abs(E_pme - E_ew) / abs(E_ew) < 1e-3
        @test reln(F_pme,  F_ew)  < 1e-3
        @test reln(SF_pme, SF_ew) < 1e-3
        @test reln(W_pme,  W_ew)  < 1e-3

        # α independence of the full PME stack (CB real + PME).  Q = +19.2 e
        # (Σqc = 16, Σqs = 3.2) makes the uncorrected drift πQ²/(2V)·Δ(1/α²)
        # ≈ 0.29 Ha — far above mesh/truncation scatter — and we require
        # ≥10× suppression, so a missing, qs-less, or wrong PME-side
        # correction fails loudly here.
        mkprms(α) = chprms(α, FMD.FPPREC[+1.0, -0.5])
        E1, _, _ = pme_EFS!(atoms, mkprms(0.30), arrtyp)
        E2, _, _ = pme_EFS!(atoms, mkprms(0.40), arrtyp)
        p = mkprms(0.30)
        Q = sum(p[:qc][atoms.arrays[:elemid]]) + sum(p[:qs][atoms.arrays[:elemid]])
        V = abs(det(Array(atoms.info[:lattice])))
        drift_uncorrected = π * Q^2 / (2V) * abs(1/0.30^2 - 1/0.40^2)
        @test abs(E1 - E2) < 0.1 * drift_uncorrected
    end
end
