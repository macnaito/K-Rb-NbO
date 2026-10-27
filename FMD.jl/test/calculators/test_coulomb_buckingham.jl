# test/calculators/test_coulomb_buckingham.jl
using Test
using LinearAlgebra
using Random
using SpecialFunctions: erfc
import KernelAbstractions

@testset "CoulombBuckingham" begin
    @testset "erfc approximation accuracy" begin
        # erfc_approx is an internal Chebyshev-rational approximation used in
        # the kernel.  Sanity-check it against the SpecialFunctions.erfc
        # reference on a representative range.
        f = Calculators.erfc_approx
        for x in (1e-6, 1e-3, 0.1, 0.5, 1.0, 2.0, 4.0, 6.0)
            @test isapprox(f(FMD.FPPREC(x)), erfc(x); atol = 1e-6)
        end
    end

    atoms = make_nacl_8()
    prms = nacl_params(atoms)

    # Closure that runs the real-space Coulomb+Buckingham kernel against a
    # naive neighbor list.  Buckingham coefficients are zero, so this
    # isolates the screened Coulomb (real-space part of Ewald).
    function calc!(a)
        nl = neighborlist(a, prms[:rc], 0.0)
        reset!(a)
        Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
    end

    @testset "Physical invariants" begin
        assert_translation_invariant(calc!, atoms; rtol = 1e-10, atol = 1e-12,
                                     label = "translation invariance")
        assert_pbc_image_invariant(calc!, atoms; rtol = 1e-10, atol = 1e-12,
                                   label = "PBC integer-shift invariance")
        assert_force_sum_zero(calc!, atoms; atol = 1e-9,
                              label = "Σ F_i ≈ 0")
        assert_force_matches_finite_diff(calc!, atoms;
                                         h = 1e-4,
                                         atom_indices = 1:8,
                                         axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-6,
                                         label = "force = -∇E (FD)")
    end

    @testset "multi-image regime (rc > L): force = -∇E (FD)" begin
        # The Δ-learning reference (CoreShell 3-arg path on small fitting
        # frames) runs with rc larger than the cell — keep a force-kernel
        # regression with multi-layer ghosts and self-image pairs.
        Random.seed!(23)
        a5 = make_perovskite_5()
        a5.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(a5.arrays[:pos]))
        # with_shell = false: the FD helper differentiates E w.r.t. the CORE
        # position and compares against :forces (core) only, but with shells
        # riding on the cores the true derivative is Fc + Fs — with the
        # fixture's qs = −qc/4 that mismatch is exactly 4/3. Core-only
        # Coulomb keeps the FD identity exact (same convention as the
        # minimum-image FD test above).
        pmi = perovskite_params(a5; alpha = 0.35, rc_factor = 1.6, with_shell = false)
        calcmi! = function (a)
            nl = neighborlist(a, pmi[:rc], 0.0)
            reset!(a)
            Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, pmi)
        end
        let aa = deepcopy_atoms(a5)
            _, F = evaluate(calcmi!, aa)
            @test maximum(abs, F) > 1e-2
        end
        assert_force_sum_zero(calcmi!, a5; atol = 1e-9,
                              label = "Σ F ≈ 0 (multi-image)")
        assert_force_matches_finite_diff(calcmi!, a5; h = 1e-4,
                                         atom_indices = 1:5, axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-6,
                                         label = "multi-image force = -∇E (FD)")
    end
end
