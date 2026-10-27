# test/calculators/test_spline.jl
using Test
using Random
using LinearAlgebra

@testset "Spline (B-spline 2-body + 3-body)" begin
    Random.seed!(0)
    atoms = make_perovskite_5(; a_ang = 4.0)
    # Spline takes cutoffs in atomic (Bohr) length units.
    rmax2 = 4.0 / UNITS[:Å]
    rmax3 = 3.0 / UNITS[:Å]
    splp = Calculators.SplineParam(FMD.FPPREC(rmax2), FMD.FPPREC(rmax3),
                                   3, 6, 5)
    splp.c1 .= randn(FMD.FPPREC, size(splp.c1)) .* 1e-3
    splp.c2 .= randn(FMD.FPPREC, size(splp.c2)) .* 1e-3
    splp.c3 .= randn(FMD.FPPREC, size(splp.c3)) .* 1e-3

    @testset "force = -∇E (FD, 2-body + 3-body)" begin
        # Replaces the former in-`src` `checkforce` helper with the shared
        # test-side FD assertion.  Use a displaced (asymmetric) configuration
        # so the forces are genuinely non-zero (the old single-atom check on
        # the symmetric perovskite was effectively vacuous).
        Random.seed!(40)
        a = deepcopy_atoms(atoms)
        a.arrays[:pos] .+= 0.03 .* randn(size(a.arrays[:pos]))
        calc! = x -> Calculators.calculate!(splp, x)
        let aa = deepcopy_atoms(a)
            _, F = evaluate(calc!, aa)
            @test maximum(abs, F) > 1e-5
        end
        assert_force_matches_finite_diff(calc!, a; h = 1e-5,
                                         atom_indices = 1:length(a), axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-7,
                                         label = "Spline force (2+3 body) FD")
    end

    # applyA must equal calculate! on the same configuration: the adjoint /
    # FD tests are closed within the applyA path, so a self-consistent
    # mis-wiring (e.g. swapped nl2/nl3 bundles) would pass them. Uses the
    # symmetrized c3 (the 3-body kernel reads c3 without canonicalization;
    # neighbour-enumeration-order invariance needs c3 ∈ im(P) — Issue #1).
    splp_sym = deepcopy(splp)
    Calculators._spl_symmetrize_c3!(splp_sym.c3)

    @testset "applyA ≡ calculate! wiring ($label)" for (label, mk) in (
        ("40-atom shuffled", () -> begin
            Random.seed!(71)
            a = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (2, 2, 2);
                                shuffle_order = true)
            a.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(a.arrays[:pos]))
            a
        end),
        ("5-atom multi-image (rmax2 > a)", () -> begin
            Random.seed!(72)
            a = make_perovskite_5(; a_ang = 3.5)   # a < rmax2: self/multi images
            a.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(a.arrays[:pos]))
            a
        end),
    )
        a0 = mk()
        aC = deepcopy_atoms(a0)
        reset!(aC)
        Calculators.calculate!(splp_sym, aC,
                               neighborlist(aC, splp_sym.rmax2, 0.0),
                               neighborlist(aC, splp_sym.rmax3, 0.0))

        E_A, F_A, V_A = Calculators.applyA(
            splp_sym, Calculators.makearg_spline(deepcopy_atoms(a0),
                                                 splp_sym.rmax2, splp_sym.rmax3))

        tol(X) = 1e-11 + 1e-9 * maximum(abs, X)
        @test maximum(abs, aC.arrays[:forces]) > 1e-5
        @test isapprox(E_A, aC.info[:potential_energy][1]; rtol = 1e-10)
        @test maximum(abs, F_A .- aC.arrays[:forces]) <= tol(aC.arrays[:forces])
        @test maximum(abs, V_A .- aC.info[:virial]) <= tol(aC.info[:virial])
    end

    @testset "applyA + applyAT! self-adjoint" begin
        # Because Spline is linear in (c1, c2, c3),
        #     <applyA(splp) · u_data, u_param>  ==
        #     <u_data, applyAT!(splp_grad) · u_param>
        # Concretely, applyA produces (E, F, V); applyAT! accumulates the
        # adjoint of given (u_E, u_F, u_V) into a parameter-shaped gradient.
        arg = Calculators.makearg_spline(atoms, splp.rmax2, splp.rmax3)
        E1, F1, V1 = Calculators.applyA(splp, arg)

        # Pick arbitrary dual values.
        u_E = randn()
        u_F = randn(size(F1))
        u_V = randn(size(V1))

        splp_grad = deepcopy(splp)
        splp_grad.c1 .= 0; splp_grad.c2 .= 0; splp_grad.c3 .= 0
        Calculators.applyAT!(splp_grad, arg, u_E, u_F, u_V)

        # Adjoint identity: <(E, F, V), (u_E, u_F, u_V)>
        #                 = <splp, splp_grad>  (parameter-side inner product)
        lhs = u_E * E1 + sum(u_F .* F1) + sum(u_V .* V1)
        rhs = sum(splp.c1 .* splp_grad.c1) + sum(splp.c2 .* splp_grad.c2) +
              sum(splp.c3 .* splp_grad.c3)
        @test isapprox(lhs, rhs; rtol = 1e-8)
    end

    @testset "force = -∇E (FD, 2-body only / 3-body only)" begin
        Random.seed!(41)
        a = deepcopy_atoms(atoms)
        a.arrays[:pos] .+= 0.03 .* randn(size(a.arrays[:pos]))
        splp_3body = deepcopy(splp); splp_3body.c1 .= 0; splp_3body.c2 .= 0  # only c3
        splp_2body = deepcopy(splp); splp_2body.c3 .= 0                      # only c1, c2
        for (sp, lbl) in ((splp_3body, "3-body only"), (splp_2body, "2-body only"))
            calc! = x -> Calculators.calculate!(sp, x)
            let aa = deepcopy_atoms(a)
                _, F = evaluate(calc!, aa)
                @test maximum(abs, F) > 1e-5
            end
            assert_force_matches_finite_diff(calc!, a; h = 1e-5,
                                             atom_indices = 1:length(a), axes = (1, 2, 3),
                                             rtol = 1e-4, atol = 1e-7,
                                             label = "Spline force ($lbl) FD")
        end
    end

    @testset "inversion symmetry: 3-body kernel + symmetric c3 (Issue #1)" begin
        # B-site (Ti) of `make_perovskite_5` sits at fractional (½,½,½)
        # and is an inversion center for the 5-atom cell.  Ti–O distances
        # 0.5·a = 2.0 Å < rmax3 = 3.0 Å, so Ti–O–O triples (sj == sk = O)
        # populate the diagonal slice that previously broke under inversion.
        splp_3 = deepcopy(splp); splp_3.c1 .= 0; splp_3.c2 .= 0
        Calculators._spl_symmetrize_c3!(splp_3.c3)
        calc! = a -> Calculators.calculate!(splp_3, a)
        assert_inversion_invariant(calc!, atoms;
                                   center_frac = (0.5, 0.5, 0.5),
                                   rtol = 1e-12, atol = 1e-14,
                                   label = "Spline 3-body inversion (sym c3)")
    end
end

@testset "Spline c3 symmetrize (Issue #1)" begin
    Random.seed!(3)
    nb = 6; nθ = 6; ns = 3; nsi = ns
    c3 = randn(Float64, nb, nb, nθ, ns, ns, nsi)
    c3_orig = copy(c3)

    @testset "involution σ symmetry achieved" begin
        c3 .= c3_orig
        Calculators._spl_symmetrize_c3!(c3)
        for si in 1:nsi, sb in 1:ns, sa in 1:ns
            slc_ab = c3[:, :, :, sa, sb, si]
            slc_ba = c3[:, :, :, sb, sa, si]
            @test isapprox(slc_ab, permutedims(slc_ba, (2, 1, 3));
                           atol = 1e-12)
        end
    end

    @testset "idempotency P² = P" begin
        c3 .= c3_orig
        Calculators._spl_symmetrize_c3!(c3)
        c1 = copy(c3)
        Calculators._spl_symmetrize_c3!(c3)
        @test c3 == c1
    end

    @testset "self-adjoint <c, P d> = <P c, d>" begin
        c = randn(Float64, size(c3))
        d = randn(Float64, size(c3))
        Pc = copy(c); Calculators._spl_symmetrize_c3!(Pc)
        Pd = copy(d); Calculators._spl_symmetrize_c3!(Pd)
        @test isapprox(sum(c .* Pd), sum(Pc .* d); rtol = 1e-12)
    end

    @testset "_spl_c3_asymmetry orbit coverage (Issue #63)" begin
        # A projected c3 measures exactly zero (bit-exact, not just small).
        c3 .= c3_orig
        Calculators._spl_symmetrize_c3!(c3)
        @test Calculators._spl_c3_asymmetry(c3) == 0.0
        # One planted asymmetric entry per σ-orbit class — each must be
        # detected with its exact magnitude (a scan that misses a class
        # would return 0 for that case).
        planted = (
            (CartesianIndex(1, 2, 1, 1, 1, 1), 0.25),   # sa == sb, a < b
            (CartesianIndex(1, 1, 1, 1, 2, 1), 0.50),   # sa < sb, a == b
            (CartesianIndex(2, 1, 1, 1, 2, 1), 0.75),   # sa < sb, a ≠ b
        )
        for (idx, val) in planted
            fill!(c3, 0.0)
            c3[idx] = val
            @test Calculators._spl_c3_asymmetry(c3) == val
        end
    end
end

@testset "Spline taper smoothness at cutoff" begin
    # Verify that the radial taper (active only in [α·rc, rc]) drives the
    # spline value and force to zero smoothly as r → rc, so MD doesn't see
    # a discontinuous step when neighbours cross the cutoff sphere.
    Random.seed!(1)
    L = FMD.FPPREC(60.0)              # large cubic box: no PBC interference
    rmax2 = FMD.FPPREC(8.0)
    rmax3 = FMD.FPPREC(4.0)
    nb2, nb3 = 12, 6
    splp = Calculators.SplineParam(rmax2, rmax3, 1, nb2, nb3)
    splp.c2 .= randn(FMD.FPPREC, size(splp.c2))
    splp.c3 .= randn(FMD.FPPREC, size(splp.c3))

    lat = Matrix{FMD.FPPREC}(I, 3, 3) .* L

    # ---- 2-body: slide atom 2 radially while atom 1 sits at origin ---
    function eval_2body(r_x)
        splp_2 = deepcopy(splp); splp_2.c3 .= 0
        pos = Matrix{FMD.FPPREC}([0 0 0; r_x/L 0 0])
        atoms2 = make_atoms(lat, pos, [1, 1])
        arg = Calculators.makearg_spline(atoms2, rmax2, rmax3)
        E, F, _ = Calculators.applyA(splp_2, arg)
        return E, F[2, 1]
    end

    # Beyond rc: pair is outside neighbour list → exactly zero
    E_out, F_out = eval_2body(rmax2 + FMD.FPPREC(0.5))
    @test E_out == 0
    @test F_out == 0

    # Approach rc from inside: taper drives spline value/force toward zero
    E_near, F_near = eval_2body(rmax2 - FMD.FPPREC(1e-6))
    @test abs(E_near) < 1e-10
    @test abs(F_near) < 1e-4

    # FD ↔ analytic at multiple radii spanning r_inner = rc/2 and rc
    h = FMD.FPPREC(1e-4)
    for r0 in (FMD.FPPREC(2.0),                         # interior, tap = 1
               rmax2 * FMD.FPPREC(0.5) - FMD.FPPREC(0.1),   # just inside r_inner
               rmax2 * FMD.FPPREC(0.5) + FMD.FPPREC(0.1),   # just outside r_inner
               rmax2 * FMD.FPPREC(0.75),                    # mid taper
               rmax2 - FMD.FPPREC(0.1))                     # just inside rc
        Ep, _ = eval_2body(r0 + h)
        Em, _ = eval_2body(r0 - h)
        _, F_an = eval_2body(r0)
        F_num = -(Ep - Em) / (2h)
        @test isapprox(F_an, F_num; rtol = 1e-3, atol = 1e-5)
    end

    # ---- 3-body: keep atom 3 close to atom 1, slide atom 2 radially ----
    rik_fix = FMD.FPPREC(2.0)  # fixed, well inside rmax3
    function eval_3body(rij_x)
        splp_3 = deepcopy(splp); splp_3.c2 .= 0
        pos = Matrix{FMD.FPPREC}([0 0 0;
                                  rij_x/L 0 0;
                                  0 rik_fix/L 0])
        atoms3 = make_atoms(lat, pos, [1, 1, 1])
        arg = Calculators.makearg_spline(atoms3, rmax2, rmax3)
        E, F, _ = Calculators.applyA(splp_3, arg)
        return E, F[2, 1]
    end

    # Beyond rmax3 → no triplet, exact zero
    E_out, F_out = eval_3body(rmax3 + FMD.FPPREC(0.3))
    @test E_out == 0
    @test F_out == 0

    # Approach rmax3 from inside: taper kicks in only above 0.9·rmax3
    E_near, F_near = eval_3body(rmax3 - FMD.FPPREC(1e-6))
    @test abs(E_near) < 1e-10
    @test abs(F_near) < 1e-4

    # FD ↔ analytic for the 3-body taper, spanning r_inner = 0.9·rmax3 and rmax3
    for r0 in (FMD.FPPREC(2.0),                              # tap_ij = 1
               rmax3 * FMD.FPPREC(0.9) - FMD.FPPREC(0.05),   # just inside r_inner
               rmax3 * FMD.FPPREC(0.9) + FMD.FPPREC(0.05),   # just outside r_inner
               rmax3 - FMD.FPPREC(0.05))                     # just inside rmax3
        Ep, _ = eval_3body(r0 + h)
        Em, _ = eval_3body(r0 - h)
        _, F_an = eval_3body(r0)
        F_num = -(Ep - Em) / (2h)
        @test isapprox(F_an, F_num; rtol = 1e-3, atol = 1e-5)
    end
end

@testset "Spline boundary pin (ϕ(rc) penalty)" begin
    # The pin penalty is (λ/2) · ϕ(rc)² with
    #   ϕ(rc) = (c[nb-3] + 4·c[nb-2] + c[nb-1]) / 6
    # Gradient should be (λv/3, 4λv/3, λv/3) at indices (nb-3, nb-2, nb-1),
    # zero elsewhere.
    Random.seed!(2)

    @testset "axis-1 pin gradient: analytic match" begin
        nb = 8
        ns = 2
        c = randn(Float64, nb, ns, ns)
        g = zeros(Float64, nb, ns, ns)
        λ = 1.7
        Calculators._spl_add_pin_axis1!(g, c, λ)
        for sb in 1:ns, sa in 1:ns
            v = (c[nb-3, sa, sb] + 4 * c[nb-2, sa, sb] + c[nb-1, sa, sb]) / 6
            @test isapprox(g[nb-3, sa, sb], λ * v / 3;     atol = 1e-12)
            @test isapprox(g[nb-2, sa, sb], 4 * λ * v / 3; atol = 1e-12)
            @test isapprox(g[nb-1, sa, sb], λ * v / 3;     atol = 1e-12)
            # All other indices untouched
            for k in vcat(1:nb-4, nb:nb)
                @test g[k, sa, sb] == 0
            end
        end
    end

    @testset "axis-1 pin: symmetric posdef + null space" begin
        nb = 10
        c1 = randn(Float64, nb, 1, 1)
        c2 = randn(Float64, nb, 1, 1)
        Mc1 = zeros(Float64, nb, 1, 1); Calculators._spl_add_pin_axis1!(Mc1, c1, 1.0)
        Mc2 = zeros(Float64, nb, 1, 1); Calculators._spl_add_pin_axis1!(Mc2, c2, 1.0)
        # Symmetry: <c1, M c2> = <M c1, c2>
        @test isapprox(sum(c1 .* Mc2), sum(Mc1 .* c2); atol = 1e-10)
        # Posdef: c^T M c ≥ 0
        @test sum(c1 .* Mc1) ≥ -1e-12
        # Null space: any c with ϕ(rc) = 0 yields zero gradient.
        c0 = zeros(Float64, nb, 1, 1)
        c0[nb-3, 1, 1] = 1.0
        c0[nb-2, 1, 1] = 0.0
        c0[nb-1, 1, 1] = -1.0  # ϕ(rc) = (1 + 0 - 1)/6 = 0
        Mc0 = zeros(Float64, nb, 1, 1); Calculators._spl_add_pin_axis1!(Mc0, c0, 1.0)
        @test maximum(abs, Mc0) < 1e-12
    end

    @testset "3-body pin: axis 1 + axis 2 symmetric application" begin
        nb = 6
        ns = 2
        c3 = randn(Float64, nb, nb, nb, ns, ns, ns)
        g_ax1only = zeros(Float64, nb, nb, nb, ns, ns, ns)
        Calculators._spl_add_pin_axis1!(g_ax1only, c3, 1.0)
        # Verify axis-1 pin only modifies indices nb-3..nb-1 along axis 1
        for I in CartesianIndices((nb, nb, ns, ns, ns))
            @test g_ax1only[1, I] == 0
            @test g_ax1only[nb, I] == 0
        end

        # Full 3-body pin: axis 1 + axis 2 symmetric. Permuting c3 (1↔2)
        # before/after should give the same result.
        g_full = zeros(Float64, nb, nb, nb, ns, ns, ns)
        Calculators._spl_add_pin_3body!(g_full, c3, 1.0)
        c3_perm = permutedims(c3, (2, 1, 3, 4, 5, 6))
        g_perm = zeros(Float64, nb, nb, nb, ns, ns, ns)
        Calculators._spl_add_pin_3body!(g_perm, c3_perm, 1.0)
        @test isapprox(g_full, permutedims(g_perm, (2, 1, 3, 4, 5, 6));
                       atol = 1e-12)
    end
end
