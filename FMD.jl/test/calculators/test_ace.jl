# test/calculators/test_ace.jl
using Test
using Random
using LinearAlgebra

@testset "ACE" begin
    Random.seed!(0)
    atoms = make_perovskite_5(; a_ang = 4.0)
    rmin = 1.0 / UNITS[:Å]
    rmax = 4.0 / UNITS[:Å]
    # ACEParam(rmin, rmax, ns, nc0, nc1, nc2, nc3); enforce nc0 ≥ nc1 ≥ nc2 ≥ nc3 ≥ 3.
    acep = Calculators.ACEParam(FMD.FPPREC(rmin), FMD.FPPREC(rmax),
                                3, 4, 4, 3, 3)

    # Initialize all parameter tensors with a small random seed; ACE is
    # not linear in its parameters but the gradient checks still apply.
    for f in fieldnames(typeof(acep))
        x = getfield(acep, f)
        x isa AbstractArray || continue
        x .= randn(FMD.FPPREC, size(x)) .* 1e-3
    end

    @testset "geometry stubs: displacement ≡ xcart difference (exact)" begin
        # The whole ghost migration of the ACE kernels rests on this
        # identity: with IdLattice/ZeroShift, displacement must return the
        # plain Cartesian differences BIT-exactly (×true/×false and +Int32(0)
        # are exact), so the ghost path is the same numerical path.
        Random.seed!(70)
        for _ in 1:20
            xi, yi, zi = randn(FMD.FPPREC), randn(FMD.FPPREC), randn(FMD.FPPREC)
            xj, yj, zj = randn(FMD.FPPREC), randn(FMD.FPPREC), randn(FMD.FPPREC)
            dx, dy, dz, r = Calculators.displacement(
                xi, yi, zi, xj, yj, zj,
                Calculators.ZeroShift()[1, 1], Calculators.ZeroShift()[1, 1],
                Calculators.ZeroShift()[1, 1], Calculators.IdLattice())
            @test dx === xi - xj && dy === yi - yj && dz === zi - zj
            @test r === sqrt((xi - xj)^2 + (yi - yj)^2 + (zi - zj)^2)
        end
    end

    @testset "ACE force = -∇E (FD)" begin
        # Replaces the former in-`src` `checkforce` helper.  Use a displaced
        # (asymmetric) configuration so the forces are genuinely non-zero —
        # on the symmetric perovskite the old single-atom check was vacuous.
        Random.seed!(32)
        a = deepcopy_atoms(atoms)
        a.arrays[:pos] .+= 0.05 .* randn(size(a.arrays[:pos]))
        calc! = x -> Calculators.calculate!(acep, x)
        let aa = deepcopy_atoms(a)
            _, F = evaluate(calc!, aa)
            @test maximum(abs, F) > 1e-3
        end
        assert_force_matches_finite_diff(calc!, a; h = 1e-5,
                                         atom_indices = 1:length(a), axes = (1, 2, 3),
                                         rtol = 1e-4, atol = 1e-6,
                                         label = "ACE force = -∇E (FD)")
    end

    @testset "Analytic adjoint = ∂L/∂θ (FD directional derivative)" begin
        # `scatteradd_analytic!` accumulates the VJP  g = ∂L/∂θ  of the scalar
        #     L(θ) = u_E·E + ⟨u_F, F⟩ + ⟨u_V, V⟩,   θ = vec(acep)
        # used in the CG fit.  This was previously cross-checked against
        # Enzyme reverse-mode AD; we now validate it with a central finite
        # difference along random parameter-space directions — a directional
        # derivative covers all parameters statistically and needs no AD
        # reference (so the Enzyme dependency can be dropped).  Reuses the
        # production `applyA` / `vec` / `setval!` machinery.
        Random.seed!(7)
        arg = Calculators.makearg(atoms, acep.rmax)
        _, F0, V0 = Calculators.applyA(acep, arg)
        u_E = randn(); u_F = randn(size(F0)); u_V = randn(size(V0))

        g = deepcopy(acep)
        for f in fieldnames(typeof(g))
            x = getfield(g, f); x isa AbstractArray && (x .= 0)
        end
        # Production adjoint wrapper (ghost format: extends u_F to the
        # extended rows before the scatter).
        Calculators.applyAT!(g, arg, u_E, u_F, u_V)
        gvec = vec(g)
        θ = vec(acep)
        @test maximum(abs, gvec) > 1e-6   # non-trivial gradient

        L(θ′) = begin
            a = deepcopy(acep)
            Calculators.setval!(a, θ′)
            E, F, V = Calculators.applyA(a, arg)
            u_E * E + sum(u_F .* F) + sum(u_V .* V)
        end

        h = 1e-6
        for _ in 1:4
            δ = randn(length(θ)); δ ./= norm(δ)
            dir_fd = (L(θ .+ h .* δ) - L(θ .- h .* δ)) / (2h)
            dir_an = dot(gvec, δ)
            @test isapprox(dir_an, dir_fd; rtol = 1e-5, atol = 1e-8)
        end
    end

    @testset "ACEFast (KA) ↔ reference parity (E, F, virial)" begin
        # `calculate!(acep, atoms, nl)` (ACEFast.jl, KernelAbstractions) is
        # the production force/energy/virial path; the hand-derived reference
        # lives in ACE.jl as
        #   calculate!(acep, forces, virial, pos, elemid, lattice, list, …).
        # This parity (KA path vs reference) used to be cross-checked by a
        # `@debug_block` inside `src` (gated by the `debug` Preference); that
        # in-production check was removed, so this hard test on an asymmetric
        # configuration is now its single source of verification.
        Random.seed!(31)
        a = deepcopy_atoms(atoms)
        a.arrays[:pos] .+= 0.05 .* randn(size(a.arrays[:pos]))
        nl = neighborlist(a, acep.rmax, 0.0)

        a_ka = deepcopy_atoms(a); reset!(a_ka)
        Calculators.calculate!(acep, a_ka, nl)
        E_ka = a_ka.info[:potential_energy][1]
        F_ka = copy(a_ka.arrays[:forces])
        V_ka = copy(a_ka.info[:virial])

        gh = nl.ghosts
        Fext = zeros(FMD.FPPREC, length(gh), 3)
        V_ref = zeros(FMD.FPPREC, 6)
        E_ref = Calculators.calculate!(acep, Fext, V_ref, gh, nl.list)
        F_ref = zeros(FMD.FPPREC, length(a), 3)
        Cores.fold_ghosts!(F_ref, Fext, gh)

        @test maximum(abs, F_ka) > 1e-3   # non-trivial forces being compared
        @test isapprox(E_ka, E_ref; rtol = 1e-6)
        @test isapprox(F_ka, F_ref; rtol = 1e-6, atol = 1e-10)
        @test isapprox(V_ka, V_ref; rtol = 1e-6, atol = 1e-10)
    end

    @testset "ACEWorkspace reuse purity" begin
        # A persistent workspace must give the same results as a fresh one,
        # including after evaluating a different configuration in between
        # (no state leaking across calls).
        Random.seed!(33)
        a1 = deepcopy_atoms(atoms)
        a1.arrays[:pos] .+= 0.05 .* randn(FMD.FPPREC, size(a1.arrays[:pos]))
        a2 = deepcopy_atoms(atoms)
        a2.arrays[:pos] .+= 0.05 .* randn(FMD.FPPREC, size(a2.arrays[:pos]))
        nl1 = neighborlist(a1, acep.rmax, 0.0)
        nl2 = neighborlist(a2, acep.rmax, 0.0)

        ws = Calculators.ACEWorkspace(Cores.get_backend(a1), acep, length(a1))
        function eval_with(a, nl)
            aa = deepcopy_atoms(a)
            reset!(aa)
            Calculators.calculate!(acep, aa, nl, ws)
            return (aa.info[:potential_energy][1],
                    copy(aa.arrays[:forces]), copy(aa.info[:virial]))
        end

        E1a, F1a, V1a = eval_with(a1, nl1)
        E2,  F2,  _   = eval_with(a2, nl2)   # different config through the same ws
        E1b, F1b, V1b = eval_with(a1, nl1)   # back to config 1

        # rtol 1e-10, not bitwise: forces/virials accumulate via @atomic, so
        # a multi-threaded KA CPU backend may reorder the sums between runs.
        @test !isapprox(E2, E1a; rtol = 1e-6)  # the two configs actually differ
        @test isapprox(E1b, E1a; rtol = 1e-10)
        @test isapprox(F1b, F1a; rtol = 1e-10, atol = 1e-14)
        @test isapprox(V1b, V1a; rtol = 1e-10, atol = 1e-14)

        # Persistent-workspace result matches the fresh-workspace (3-arg) path.
        aref = deepcopy_atoms(a1)
        reset!(aref)
        Calculators.calculate!(acep, aref, nl1)
        @test isapprox(E1a, aref.info[:potential_energy][1]; rtol = 1e-10)
        @test isapprox(F1a, aref.arrays[:forces]; rtol = 1e-10, atol = 1e-14)

        # Mismatched dims must fail loudly, not corrupt: wrong Nat, and
        # wrong channel counts (workspace built for a smaller ACEParam).
        ws_small = Calculators.ACEWorkspace(Cores.get_backend(a1), acep, length(a1) - 1)
        @test_throws ErrorException Calculators.calculate!(acep, deepcopy_atoms(a1), nl1, ws_small)
        acep_small = Calculators.ACEParam(acep.rmin, acep.rmax, 3, 4, 3, 3, 3)
        ws_wrongch = Calculators.ACEWorkspace(Cores.get_backend(a1), acep_small, length(a1))
        @test_throws ErrorException Calculators.calculate!(acep, deepcopy_atoms(a1), nl1, ws_wrongch)

        # The workspace evaluates a coefficient snapshot: wholesale mutation
        # of the source acep afterwards must be caught by the cp/c0
        # fingerprint, not silently ignored. (Only the fully-consumed fields
        # are fingerprinted — higher-order tensors have garbage triangles by
        # contract, see the NaN-injection test.)
        acep_mut = deepcopy(acep)
        ws_mut = Calculators.ACEWorkspace(Cores.get_backend(a1), acep_mut, length(a1))
        acep_mut.c0 .*= 2
        @test_throws ErrorException Calculators.calculate!(acep_mut, deepcopy_atoms(a1), nl1, ws_mut)
    end

    @testset "GEMM path: layout / coefficient-region parity" begin
        # Reference-vs-ACEFast comparisons on configurations that stress the
        # species-sorted GEMM layout of the 3-stage split.
        rmin4 = FMD.FPPREC(1.0 / UNITS[:Å])
        rmax4 = FMD.FPPREC(4.0 / UNITS[:Å])

        function ref_EFV(acep_, a, nl)
            gh = nl.ghosts
            Fext = zeros(FMD.FPPREC, length(gh), 3)
            V = zeros(FMD.FPPREC, 6)
            E = Calculators.calculate!(acep_, Fext, V, gh, nl.list)
            F = zeros(FMD.FPPREC, length(a), 3)
            Cores.fold_ghosts!(F, Fext, gh)
            return E, F, V
        end
        function ka_EFV(acep_, a, nl; ws = nothing)
            aa = deepcopy_atoms(a)
            reset!(aa)
            ws === nothing ? Calculators.calculate!(acep_, aa, nl) :
                             Calculators.calculate!(acep_, aa, nl, ws)
            return aa.info[:potential_energy][1], copy(aa.arrays[:forces]),
                   copy(aa.info[:virial])
        end
        function checkparity(acep_, a)
            nl = neighborlist(a, acep_.rmax, 0.0)
            E_ref, F_ref, V_ref = ref_EFV(acep_, a, nl)
            E, F, V = ka_EFV(acep_, a, nl)
            @test maximum(abs, F_ref) > 1e-4
            @test isfinite(E)
            @test isapprox(E, E_ref; rtol = 1e-6)
            @test isapprox(F, F_ref; rtol = 1e-6, atol = 1e-10)
            @test isapprox(V, V_ref; rtol = 1e-6, atol = 1e-10)
        end

        # (i) 40-atom shuffled supercell (elemid neither species- nor
        # cell-sorted), all four Chebyshev orders distinct (n0>n1>n2>n3).
        Random.seed!(51)
        acep_d = Calculators.ACEParam(rmin4, rmax4, 3, 6, 5, 4, 3)
        for f in fieldnames(typeof(acep_d))
            x = getfield(acep_d, f)
            x isa AbstractArray || continue
            x .= randn(FMD.FPPREC, size(x)) .* 1e-3
        end
        a40 = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (2, 2, 2);
                              shuffle_order = true)
        a40.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(a40.arrays[:pos]))
        checkparity(acep_d, a40)

        # (ii) missing species: ns = 3 but species 2 absent — its row block
        # is empty and every per-species GEMM must skip it.
        a_miss = deepcopy_atoms(a40)
        a_miss.arrays[:elemid] .= replace(a_miss.arrays[:elemid], 2 => 3)
        checkparity(acep_d, a_miss)

        # (iii) NaN in the never-consumed coefficient regions: strict lower
        # triangle of the quadratic tensors, ν < λ side of the 4-body
        # tensors. Both the reference and the GEMM preprocessing must read
        # only the upper regions, so results stay finite and in parity.
        acep_nan = deepcopy(acep_d)
        for c in (acep_nan.c00, acep_nan.c11, acep_nan.c22, acep_nan.c33)
            nch = size(c, 1)
            for s in axes(c, 3), ν in 1:nch, μ in (ν+1):nch
                c[μ, ν, s] = NaN
            end
        end
        for c in (acep_nan.c011, acep_nan.c022, acep_nan.c033)
            nch = size(c, 2)
            for s in axes(c, 4), λ in 1:nch, ν in 1:(λ-1), μ in axes(c, 1)
                c[μ, ν, λ, s] = NaN
            end
        end
        checkparity(acep_nan, a40)

        # (iv) elemid replacement invalidates the species-sort cache:
        # evaluate a permuted copy (new arrays ⇒ the identity check fires)
        # through the SAME workspace — being the same physical system, E is
        # unchanged and forces are the permutation of the original.
        Random.seed!(52)
        perm = randperm(length(a40))
        a_perm = deepcopy_atoms(a40)
        a_perm.arrays[:pos] = a40.arrays[:pos][perm, :]
        a_perm.arrays[:elemid] = a40.arrays[:elemid][perm]
        ws40 = Calculators.ACEWorkspace(Cores.get_backend(a40), acep_d, length(a40))
        nl40 = neighborlist(a40, acep_d.rmax, 0.0)
        E1, F1, _ = ka_EFV(acep_d, a40, nl40; ws = ws40)
        nlp = neighborlist(a_perm, acep_d.rmax, 0.0)
        E2, F2, _ = ka_EFV(acep_d, a_perm, nlp; ws = ws40)
        @test isapprox(E2, E1; rtol = 1e-10)
        @test isapprox(F2, F1[perm, :]; rtol = 1e-8, atol = 1e-12)
    end
end
