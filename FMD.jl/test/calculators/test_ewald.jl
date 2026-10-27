# test/calculators/test_ewald.jl
using Test
using LinearAlgebra

# Madelung constant for the rocksalt structure (well-known textbook value).
const MADELUNG_NACL = 1.7475645946

# Compose Ewald = CoulombBuckingham (short-range, with self-energy) +
# EwaldInfo reciprocal kernel.
function ewald_total_energy!(atoms, prms)
    nl = neighborlist(atoms, prms[:rc], 0.0)
    ewald = Calculators.EwaldInfo(atoms.info[:lattice], prms)
    reset!(atoms)
    Calculators.calculate!(Calculators.CoulombBuckingham(), atoms, nl, prms)
    Calculators.calculate!(ewald, atoms, prms)
    return atoms.info[:potential_energy][1]
end

@testset "Ewald" begin

    @testset "Madelung constant: NaCl 64-atom (2×2×2 of conventional cell)" begin
        atoms = make_nacl_64()
        prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-10,
                           rc_factor = 0.45)
        E = ewald_total_energy!(deepcopy_atoms(atoms), prms)

        # Expected: per-pair = -α_M / r0  (atomic units, q = ±1).
        # 32 NaCl pairs in the 64-atom supercell (2×2×2 of the 8-atom
        # conventional cubic cell).
        npairs = 32
        r0 = NACL_R0_BOHR
        E_expected = -npairs * MADELUNG_NACL / r0
        @test isapprox(E, E_expected; rtol = 5e-4)
    end

    @testset "α independence of total energy" begin
        atoms = make_nacl_64()
        # The α range must satisfy α·rc ≳ 3 so that the real-space erfc
        # cutoff is below the working tolerance.  rc ≈ 9.6 Bohr here,
        # so α ≥ 0.32 keeps α·rc above 3.
        Es = Float64[]
        for α in (0.32, 0.40, 0.50)
            prms = nacl_params(atoms; alpha = α, ewald_tolerance = 1e-10,
                               rc_factor = 0.45)
            push!(Es, ewald_total_energy!(deepcopy_atoms(atoms), prms))
        end
        # All splittings of Ewald should agree to within Madelung-level precision.
        @test isapprox(Es[1], Es[2]; rtol = 1e-3)
        @test isapprox(Es[2], Es[3]; rtol = 1e-3)
    end

    @testset "Charged-cell correction: α independence (Issue #15)" begin
        # Non-neutral cell with BOTH charge sums non-zero: qc = +1.0/-0.9
        # (Σqc = +3.2 e) and qs = -0.2/+0.3 (Σqs = +3.2 e) ⇒ Q = +6.4 e.
        # Σqs ≠ 0 pins the qs part of Q = Σ(qc+qs): dropping qs changes Q²
        # by 4× and trips the guards below (all stock fixtures have Σqs = 0,
        # which would leave that term untested).
        #
        # Every pairwise comparison is scaled by the drift the MISSING
        # correction would produce, πQ²/(2V)·|Δ(1/α²)|, requiring ≥10×
        # suppression — measured on main (uncorrected), the actual drift
        # matches this analytic scale to ~4%, while the corrected residual
        # is ordinary real-space truncation scatter (~4e-4 Ha at
        # ewald_tolerance = 1e-10, the same magnitude the neutral fixture
        # above shows).
        atoms = make_nacl_64()
        alphas = (0.32, 0.40, 0.50)
        chprms(α) = (; nacl_params(atoms; alpha = α, ewald_tolerance = 1e-10,
                                   rc_factor = 0.45, with_shell = true)...,
                     qc = FMD.FPPREC[+1.0, -0.9],
                     qs = FMD.FPPREC[-0.2, +0.3])
        Es = [ewald_total_energy!(deepcopy_atoms(atoms), chprms(α)) for α in alphas]

        p = chprms(alphas[1])
        Q = sum(p[:qc][atoms.arrays[:elemid]]) + sum(p[:qs][atoms.arrays[:elemid]])
        V = abs(det(Array(atoms.info[:lattice])))
        for i in 1:3, j in (i+1):3
            drift_uncorrected = π * Q^2 / (2V) * abs(1/alphas[i]^2 - 1/alphas[j]^2)
            @test abs(Es[i] - Es[j]) < 0.1 * drift_uncorrected
        end
    end

    @testset "Both loose and tight Ewald are within Madelung tolerance" begin
        # The real-space cutoff (rc = 0.45·L) introduces an O(rc⁻¹) error
        # in any single Ewald summation that does *not* monotonically
        # vanish with `ewald_tolerance` (which only controls the
        # reciprocal-space k_max).  We therefore check absolute proximity
        # to the textbook Madelung value, not monotone convergence.
        atoms = make_nacl_64()
        npairs = 32
        E_madelung = -npairs * MADELUNG_NACL / NACL_R0_BOHR
        for tol in (1e-4, 1e-8, 1e-12)
            prms = nacl_params(atoms; alpha = 0.40,
                               ewald_tolerance = tol, rc_factor = 0.45)
            E = ewald_total_energy!(deepcopy_atoms(atoms), prms)
            @test isapprox(E, E_madelung; rtol = 5e-3)
        end
    end

    @testset "Reciprocal-space force = -∇E (Issue #7)" begin
        # Regression test for the cos/sin swap in the reciprocal force pass
        # (_ewald_forces!).
        #
        # The perfect rocksalt lattice has zero force by symmetry, so the
        # FD check above cannot exercise the reciprocal force at all.  The
        # bug is worse than a wrong sign: the swapped factor is *even*
        # under k → −k while the k prefactor is odd, so the reciprocal
        # force cancels to exactly zero over the inversion-symmetric
        # k-set.  We therefore break the symmetry (displace cores, add
        # shell displacements) so the reciprocal force is genuinely
        # non-zero, and FD-check the EwaldInfo path in isolation.

        # --- core force: displaced cores, no shells (qs = 0) -------------
        atoms = make_nacl_8()
        # Asymmetric (non-degenerate) displacement to break the lattice
        # symmetry; the pass/fail criterion below is tolerance-based, so it
        # does not depend on the exact RNG state, only on F being non-zero.
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-10,
                           rc_factor = 0.45)

        ewald_only!(a) = (ew = Calculators.EwaldInfo(a.info[:lattice], prms);
                          reset!(a); Calculators.calculate!(ew, a, prms))

        # Sanity: the analytic reciprocal force must not be identically
        # zero on this asymmetric cell (it was, before the fix).
        let a = deepcopy_atoms(atoms)
            _, F = evaluate(ewald_only!, a)
            @test maximum(abs, F) > 1e-3
        end

        assert_force_sum_zero(ewald_only!, atoms; atol = 1e-9,
                              label = "Σ F_i ≈ 0 (reciprocal)")
        assert_force_matches_finite_diff(ewald_only!, atoms;
                                         h = 1e-5,
                                         atom_indices = 1:8, axes = (1, 2, 3),
                                         rtol = 1e-6, atol = 1e-8,
                                         label = "core force = -∇E (FD)")

        # --- shell force: displaced shells, non-zero shell charges -------
        satoms = make_nacl_8()
        satoms.arrays[:shelldisp] .= 0.05 .* randn(size(satoms.arrays[:shelldisp]))
        sprms = nacl_params(satoms; alpha = 0.40, ewald_tolerance = 1e-10,
                            rc_factor = 0.45, with_shell = true)

        ewald_shell!(a) = (ew = Calculators.EwaldInfo(a.info[:lattice], sprms);
                           reset!(a); Calculators.calculate!(ew, a, sprms))

        let a = deepcopy_atoms(satoms)
            reset!(a); ewald_shell!(a)
            # qs = ±0.2 ⇒ a modest but well-resolved reciprocal shellforce
            # (≫ the FD atol below, so the comparison is not skipped).
            @test maximum(abs, a.arrays[:shellforces]) > 1e-5
        end

        assert_shellforce_matches_finite_diff(ewald_shell!, satoms;
                                              h = 1e-5,
                                              atom_indices = 1:8, axes = (1, 2, 3),
                                              rtol = 1e-6, atol = 1e-8,
                                              label = "shell force = -∂E/∂shelldisp (FD)")
    end

    @testset "Physical invariants (CoulombBuckingham + EwaldInfo)" begin
        atoms = make_nacl_64()
        prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                           rc_factor = 0.45)
        ewald = Calculators.EwaldInfo(atoms.info[:lattice], prms)

        function calc!(a)
            nl = neighborlist(a, prms[:rc], 0.0)
            # The reciprocal-space prefactors only depend on the cell, not on
            # atom positions, so we can build EwaldInfo once and reuse it
            # across translations of the system.  But for a fresh cell we
            # need cellupdate! — keep it safe by rebuilding.
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            reset!(a)
            Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
            Calculators.calculate!(ew, a, prms)
        end

        assert_translation_invariant(calc!, atoms; rtol = 1e-8, atol = 1e-10,
                                     label = "translation invariance")
        assert_pbc_image_invariant(calc!, atoms; rtol = 1e-8, atol = 1e-10,
                                   label = "PBC integer-shift invariance")
        assert_force_sum_zero(calc!, atoms; atol = 1e-8,
                              label = "Σ F_i ≈ 0")
        # FD on charged system with both real and reciprocal — keep h moderate
        # and only check a handful of atoms for speed.
        assert_force_matches_finite_diff(calc!, atoms;
                                         h = 1e-4,
                                         atom_indices = (1, 5, 17, 25),
                                         axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-6,
                                         label = "force = -∇E (FD)")
    end
end
