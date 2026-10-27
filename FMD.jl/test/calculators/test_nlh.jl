# test/calculators/test_nlh.jl
#
# NLH screened-nuclear repulsion baseline (Issue #124).
#
# The value tier is pinned against the reference implementation shipped with
# the published data set (`nlhpot.c`), so a unit-conversion or parsing slip
# cannot hide behind a self-consistent FD test. The remaining tiers cover
# what makes the term safe to use as a Δ-learning baseline: C¹ truncation,
# force/virial consistency, shell independence, and an exact subtract→add
# round trip.

using Test
using Random
using LinearAlgebra
using Logging

@testset "NLH screened-nuclear repulsion" begin
    ANG = UNITS[:Å]
    EV  = UNITS[:eV]
    ELEMENTS = Dict("Ba" => 1, "Ti" => 2, "O" => 3)
    RC = 5.0 / ANG
    nlh = Calculators.NLHParam(ELEMENTS, RC)

    # Raw (un-shifted) pair value at species ids (sa, sb).
    vraw(r, sa, sb) = Calculators.nlh_phi(r, nlh.zz[sa, sb],
        nlh.a[1, sa, sb], nlh.b[1, sa, sb], nlh.a[2, sa, sb], nlh.b[2, sa, sb],
        nlh.a[3, sa, sb], nlh.b[3, sa, sb])
    draw(r, sa, sb) = Calculators.nlh_dphi(r, nlh.zz[sa, sb],
        nlh.a[1, sa, sb], nlh.b[1, sa, sb], nlh.a[2, sa, sb], nlh.b[2, sa, sb],
        nlh.a[3, sa, sb], nlh.b[3, sa, sb])
    # Shifted-force pair value, i.e. exactly what the kernel evaluates.
    vsf(r, sa, sb) = r >= RC ? 0.0 :
        vraw(r, sa, sb) - vraw(RC, sa, sb) - (r - RC) * draw(RC, sa, sb)

    @testset "matches the reference implementation (nlhpot.c)" begin
        # `nlhpot 8 8` / `nlhpot 8 56` at r = 0.999 Å. These pin the Å→Bohr
        # conversion of the screening rates and the eV→Hartree scale at once:
        # getting either wrong moves the value by orders of magnitude.
        r = 0.999 / ANG
        @test isapprox(vraw(r, 3, 3) * EV, 5.448130;  rtol = 1e-6)
        @test isapprox(vraw(r, 1, 3) * EV, 59.954182; rtol = 1e-6)
        # Symmetric under Z₁ ↔ Z₂ (the table stores Z1 ≤ Z2 only).
        @test vraw(r, 1, 3) == vraw(r, 3, 1)
        @test nlh.a[:, 1, 3] == nlh.a[:, 3, 1]
        @test nlh.b[:, 1, 3] == nlh.b[:, 3, 1]
        # dV/dr against a finite difference of V.
        for (sa, sb) in ((3, 3), (1, 3), (1, 1))
            h = 1e-6
            fd = (vraw(r + h, sa, sb) - vraw(r - h, sa, sb)) / (2h)
            @test isapprox(draw(r, sa, sb), fd; rtol = 1e-6)
        end
    end

    @testset "table integrity" begin
        # Σ_k a_k = 1 ⇔ V → Z₁Z₂/r as r → 0. Checked at load; assert the
        # loaded coefficients really carry it for every stored pair.
        for sb in 1:3, sa in 1:3
            @test isapprox(sum(nlh.a[:, sa, sb]), 1.0; atol = 1e-4)
        end
        # Screening rates are positive in Bohr⁻¹ (the table is Å⁻¹).
        @test all(>=(0), nlh.b)
        @test nlh.Z == [56, 22, 8]
        @test nlh.zz == FMD.FPPREC[56*56 56*22 56*8; 22*56 22*22 22*8; 8*56 8*22 8*8]
    end

    @testset "unusable pairs are errors, never silent zeros" begin
        # Non-decaying screening (a_k ≠ 0 with b_k = 0): 24 of the 4278
        # published pairs, O–Na among them. φ(∞) ≠ 0 leaves an unscreened
        # Coulomb tail (1.27 eV at 5 Å), which is a fit-extrapolation
        # artefact, not physics.
        @test_throws ErrorException Calculators.NLHParam(Dict("O" => 1, "Na" => 2), RC)
        # Beyond the table's Z ≤ 92 coverage.
        @test_throws ErrorException Calculators.NLHParam(Dict("O" => 1, "Cf" => 2), RC)
        # Symbol with no nuclear charge.
        @test_throws ErrorException Calculators.NLHParam(Dict("O" => 1, "Xx" => 2), RC)
        # Non-positive cutoff / missing table file.
        @test_throws ErrorException Calculators.NLHParam(ELEMENTS, -1.0)
        @test_throws ErrorException Calculators.NLHParam(ELEMENTS, RC; table = "/no/such/file")
        # Species ids must be a permutation of 1:ns — a hole would leave a
        # Z = 0 row and look up a nonexistent pair.
        @test_throws ErrorException Calculators.NLHParam(Dict("O" => 1, "Ti" => 1), RC)
        @test_throws ErrorException Calculators.NLHParam(Dict("O" => 1, "Ti" => 3), RC)
    end

    @testset "slowly decaying screening is reported, not silently used" begin
        # `b_k == 0` (an exactly non-decaying term) is a hard error, but the
        # table also holds terms with small non-zero `b_k` — the same
        # artefact through a finite window. Those are legal published fits
        # and the residual MLIP absorbs them, so they are accepted; what
        # must never happen is using one without the user seeing the number.
        # Na–Cl's third term has b = 0.699 Å⁻¹ ⇒ ϕ(5 Å) = 0.162 eV.
        @test_logs (:warn, r"not negligible at its own cutoff") match_mode = :any (
            Calculators.NLHParam(Dict("Na" => 1, "Cl" => 2), 5.0 / ANG))
        # BaTiO₃ at the same cutoff is 2 meV — must stay silent, or the
        # warning would be noise everyone learns to ignore.
        @test maximum(abs, nlh.phi_rc) * EV < 1e-2
        @test_logs min_level = Logging.Warn Calculators.NLHParam(ELEMENTS, RC)
    end

    @testset "shifted-force truncation is C¹ at rc" begin
        for (sa, sb) in ((3, 3), (1, 3), (1, 2))
            # Value and slope both vanish at rc, so no jump in E or F when a
            # pair crosses the cutoff during MD.
            @test isapprox(vsf(RC - 1e-8, sa, sb), 0.0; atol = 1e-12)
            @test isapprox((vsf(RC - 1e-6, sa, sb) - vsf(RC - 2e-6, sa, sb)) / 1e-6,
                           0.0; atol = 1e-8)
            @test vsf(RC, sa, sb) == 0.0
            @test vsf(RC + 1.0, sa, sb) == 0.0
            # Still strongly repulsive where it matters.
            @test vsf(1.0 / ANG, sa, sb) > 0
            @test vsf(1.0 / ANG, sa, sb) > vsf(2.0 / ANG, sa, sb) > 0
        end
    end

    # --- configuration-level checks ---------------------------------- #
    Random.seed!(124)
    atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.0, A = "Ba", B = "Ti"), (2, 2, 2))
    atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    calc_nlh! = a -> Calculators.calculate!(nlh, a)

    @testset "energy equals a direct pair sum" begin
        # Independent O(N²) oracle over periodic images — the kernel's
        # neighbour list, ghost bookkeeping and 1/2 double-counting factor
        # are all outside this expression.
        L = atoms.info[:lattice]
        X = atoms.arrays[:pos] * L
        eid = atoms.arrays[:elemid]
        Eref = 0.0
        for i in axes(X, 1), j in axes(X, 1), n1 in -1:1, n2 in -1:1, n3 in -1:1
            (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
            d = (X[i, 1] - X[j, 1] - (n1*L[1,1] + n2*L[2,1] + n3*L[3,1]),
                 X[i, 2] - X[j, 2] - (n1*L[1,2] + n2*L[2,2] + n3*L[3,2]),
                 X[i, 3] - X[j, 3] - (n1*L[1,3] + n2*L[2,3] + n3*L[3,3]))
            r = sqrt(d[1]^2 + d[2]^2 + d[3]^2)
            Eref += 0.5 * vsf(r, eid[i], eid[j])
        end
        E, _ = evaluate(calc_nlh!, deepcopy_atoms(atoms))
        @test Eref > 1e-4                       # the fixture actually overlaps
        @test isapprox(E, Eref; rtol = 1e-10)
    end

    @testset "force = -∇E (FD)" begin
        a = deepcopy_atoms(atoms)
        let aa = deepcopy_atoms(a)
            _, F = evaluate(calc_nlh!, aa)
            @test maximum(abs, F) > 1e-5
        end
        assert_force_matches_finite_diff(calc_nlh!, a; h = 1e-6,
                                         atom_indices = 1:length(a), axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-8,
                                         label = "NLH force FD")
    end

    @testset "core–core: invariant under shell displacements" begin
        # The term reads core positions only. If it ever gained a shell
        # dependence, the shell CG loop (which never sees it) and the
        # analytic HVP would both become inconsistent, and the Δ-learning
        # round trip would stop being exact.
        a1 = deepcopy_atoms(atoms)
        a2 = deepcopy_atoms(atoms)
        a2.arrays[:shelldisp] .+= 0.05 .* randn(FMD.FPPREC, size(a2.arrays[:shelldisp]))
        reset!(a1); calc_nlh!(a1)
        reset!(a2); calc_nlh!(a2)
        @test a1.info[:potential_energy][1] == a2.info[:potential_energy][1]
        @test a1.arrays[:forces] == a2.arrays[:forces]
        @test a1.info[:virial] == a2.info[:virial]
        # and it writes no shell force at all
        @test all(iszero, a2.arrays[:shellforces])
    end

    @testset "save → load round trip" begin
        saved = Calculators.nlh_saved(nlh)
        back = Calculators.NLHParam(saved)
        @test back.rc == nlh.rc
        @test back.Z == nlh.Z
        @test back.a == nlh.a
        @test back.b == nlh.b
        @test back.zz == nlh.zz
        E1, F1 = evaluate(calc_nlh!, deepcopy_atoms(atoms))
        E2, F2 = evaluate(a -> Calculators.calculate!(back, a), deepcopy_atoms(atoms))
        @test E1 == E2 && F1 == F2

        # A hand-edited block must not install a silently wrong baseline.
        bad_sym = (; saved..., a = copy(saved.a))
        bad_sym.a[1, 1, 3] += 0.1                # (1,3) ≠ (3,1): breaks symmetry
        @test_throws ErrorException Calculators.NLHParam(bad_sym)
        bad_sum = (; saved..., a = copy(saved.a))
        bad_sum.a[1, 3, 3] += 0.1                # diagonal pair: breaks Σa = 1
        @test_throws ErrorException Calculators.NLHParam(bad_sum)
        @test_throws ErrorException Calculators.NLHParam((; saved..., rc = -1.0))
        @test_throws ErrorException Calculators.NLHParam((; saved..., Z = [56, 22]))
        @test_throws ErrorException Calculators.NLHParam((; rc = saved.rc, Z = saved.Z))
    end

    @testset "Δ-learning bookkeeping: subtract → add is exact" begin
        # A frame carrying reference E/F/V; subtract_baseline! removes the
        # baseline, add_baseline! puts it back. Any asymmetry between the two
        # halves biases every fitted coefficient, so pin the identity.
        a = deepcopy_atoms(atoms)
        prms = perovskite_params(a; with_shell = true)
        # The baseline cutoff must stay inside the Coulomb cutoff (the MD path
        # derives the NLH list from the rc list), so use a cutoff that fits
        # this fixture's smaller rc.
        nlh_in = Calculators.NLHParam(ELEMENTS, 0.9 * prms[:rc])
        prms = (; prms..., nlh = nlh_in, cgtol = 1e-10, cgmaxiter = 200)
        ref = deepcopy_atoms(a)
        reset!(ref)
        Calculators.add_baseline!(ref, prms)
        E0 = ref.info[:potential_energy][1]
        F0 = copy(ref.arrays[:forces])
        V0 = copy(ref.info[:virial])

        ds = [deepcopy_atoms(a)]
        ds[1].info[:potential_energy] = FMD.FPPREC[E0]
        ds[1].arrays[:forces] = copy(F0)
        ds[1].info[:virial] = copy(V0)
        Calculators.subtract_baseline!(ds, prms)
        # Reference == baseline ⇒ residual is identically zero.
        @test isapprox(ds[1].info[:potential_energy][1], 0.0; atol = 1e-10 * max(1, abs(E0)))
        @test maximum(abs, ds[1].arrays[:forces]) < 1e-10 * max(1, maximum(abs, F0))
        @test maximum(abs, ds[1].info[:virial]) < 1e-10 * max(1, maximum(abs, V0))

        # ... and the NLH part is genuinely present in that baseline.
        prms_no = Base.structdiff(prms, NamedTuple{(:nlh,)})
        ref_no = deepcopy_atoms(a)
        reset!(ref_no)
        Calculators.add_baseline!(ref_no, prms_no)
        @test !isapprox(ref_no.info[:potential_energy][1], E0; rtol = 1e-8)
    end
end
