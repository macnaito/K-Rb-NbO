# test/calculators/test_virial.jl
#
# Virial = -∂E/∂ε regression coverage.
#
# Before this file the only virial assertion was `assert_virial_voigt`
# (length-6 + finiteness) — the actual stress *values* were never compared
# against a strain finite-difference, so a sign or normalization error in
# any calculator's virial would have passed silently.  This is the stress
# analogue of the force blind spot from #7 / #9: we deliberately use
# ASYMMETRIC fixtures (displaced cores / shells) so the off-diagonal (shear)
# components are genuinely non-zero and the FD comparison is not vacuous.
#
# Convention (see physical_consistency.jl §4c): stored virial Voigt
# [xx,yy,zz,yz,zx,xy] = ΣF·r = -∂E/∂ε under an affine deformation of the
# lattice and the Cartesian shell displacement with fractional coords fixed.

using Test
using LinearAlgebra
using Random

@testset "Virial = -∂E/∂ε (strain FD)" begin

    @testset "CoulombBuckingham (real-space)" begin
        Random.seed!(11)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10, rc_factor = 0.45)
        calc! = function (a)
            nl = neighborlist(a, prms[:rc], 0.0)
            Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "CoulombBuckingham virial")
    end

    @testset "CoulombBuckingham + EwaldInfo (full Ewald)" begin
        Random.seed!(12)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10, rc_factor = 0.45)
        calc! = function (a)
            nl = neighborlist(a, prms[:rc], 0.0)
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
            Calculators.calculate!(ew, a, prms)
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "Ewald (real + reciprocal) virial")
    end

    @testset "Ewald reciprocal (core + shell displaced)" begin
        Random.seed!(13)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(size(atoms.arrays[:shelldisp]))
        prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10,
                           rc_factor = 0.45, with_shell = true)
        calc! = function (a)
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            Calculators.calculate!(ew, a, prms)
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "Ewald reciprocal + self virial")
    end

    @testset "Ewald reciprocal, charged cell (Issue #15)" begin
        Random.seed!(17)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(size(atoms.arrays[:shelldisp]))
        base = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10,
                           rc_factor = 0.45, with_shell = true)
        # Break neutrality in BOTH charge sets: Σqc = +0.4 e, Σqs = +0.4 e
        # ⇒ Q = +0.8 e.  The charged-cell correction -πQ²/(2α²V) adds
        # E_corr to each diagonal virial component; the FD must see it
        # (∂E_corr/∂ε_αα = -E_corr ≠ 0).
        prms = (; base..., qc = FMD.FPPREC[+1.0, -0.9],
                qs = FMD.FPPREC[-0.2, +0.3])
        calc! = function (a)
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            Calculators.calculate!(ew, a, prms)
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "Ewald reciprocal charged-cell virial")
    end

    @testset "CoulombBuckingham + EwaldInfo, charged cell (Issue #15)" begin
        Random.seed!(18)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        base = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10, rc_factor = 0.45)
        prms = (; base..., qc = FMD.FPPREC[+1.0, -0.9])
        calc! = function (a)
            nl = neighborlist(a, prms[:rc], 0.0)
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
            Calculators.calculate!(ew, a, prms)
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "Ewald full charged-cell virial")
    end

    @testset "Spring (harmonic + quartic)" begin
        Random.seed!(14)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:shelldisp] .= 0.05 .* randn(size(atoms.arrays[:shelldisp]))
        prms = (k2 = FMD.FPPREC[10.0, 30.0, 20.0], k4 = FMD.FPPREC[1.0, 2.0, 3.0])
        calc! = a -> Calculators.calculate!(Calculators.Spring(), a, prms)
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-5, atol = 1e-10,
                                          label = "Spring virial")
    end

    @testset "Spline (2-body + 3-body)" begin
        Random.seed!(15)
        atoms = make_perovskite_5(; a_ang = 4.0)
        # Break inversion symmetry so the off-diagonal shear virial is non-zero.
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        rmax2 = 4.0 / UNITS[:Å]; rmax3 = 3.0 / UNITS[:Å]
        splp = Calculators.SplineParam(FMD.FPPREC(rmax2), FMD.FPPREC(rmax3), 3, 6, 5)
        splp.c1 .= randn(FMD.FPPREC, size(splp.c1)) .* 1e-3
        splp.c2 .= randn(FMD.FPPREC, size(splp.c2)) .* 1e-3
        splp.c3 .= randn(FMD.FPPREC, size(splp.c3)) .* 1e-3
        calc! = a -> Calculators.calculate!(splp, a)
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "Spline virial")
    end

    @testset "NLH baseline" begin
        Random.seed!(17)
        # 2×2×2 (40 atoms) at the documented 5 Å default cutoff: the Ti–O
        # (2.0 Å), O–O / Ba–O (2.83 Å) and Ba–Ba (4.0 Å) shells are all
        # inside, so every Voigt component is fed by many pairs. A cutoff
        # below the nearest-neighbour distance would make the term
        # identically zero, and assert_virial_matches_finite_diff skips
        # components where analytic and FD are both under atol — i.e. the
        # testset would go green with zero assertions. The magnitude guard
        # below pins that shut.
        atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.0, A = "Ba", B = "Ti"),
                                (2, 2, 2))
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        nlh = Calculators.NLHParam(Dict("Ba" => 1, "Ti" => 2, "O" => 3),
                                   5.0 / UNITS[:Å])
        calc! = a -> Calculators.calculate!(nlh, a)
        let a = deepcopy_atoms(atoms)
            reset!(a); calc!(a)
            @test a.info[:potential_energy][1] > 1e-4
            @test minimum(abs, a.info[:virial]) > 1e-6   # shear included
        end
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-6, rtol = 1e-4, atol = 1e-8,
                                          label = "NLH virial")
    end

    @testset "ACE" begin
        Random.seed!(16)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        rmin = 1.0 / UNITS[:Å]; rmax = 4.0 / UNITS[:Å]
        acep = Calculators.ACEParam(FMD.FPPREC(rmin), FMD.FPPREC(rmax), 3, 4, 4, 3, 3)
        for f in fieldnames(typeof(acep))
            x = getfield(acep, f); x isa AbstractArray || continue
            x .= randn(FMD.FPPREC, size(x)) .* 1e-3
        end
        calc! = a -> Calculators.calculate!(acep, a)
        assert_virial_matches_finite_diff(calc!, atoms; h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          label = "ACE virial")
    end
end
