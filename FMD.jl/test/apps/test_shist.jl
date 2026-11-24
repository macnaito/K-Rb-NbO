# test/apps/test_shist.jl
#
# Unit tests for the npt! shell-displacement extrapolation history helpers
# (_init_shist! / _extrapolate_shelldisp! / _rotate_shist!).  npt! itself has
# no test harness (it needs a fitted potential), so the warmup gating, the
# predictor values, and the follow-the-resort behaviour are pinned down here.
using Test
using LinearAlgebra
using FMD.Apps: _init_shist!, _extrapolate_shelldisp!, _rotate_shist!,
                _init_shell_work!, _shell_work_step!

@testset "shist helpers (npt! extrapolation history)" begin
    N = 4
    lat = Matrix{FMD.FPPREC}(10.0I, 3, 3)
    frac = FMD.FPPREC[0 0 0; 0.5 0 0; 0 0.5 0; 0 0 0.5]
    sval(k) = FMD.FPPREC.(k .* reshape(1:3N, N, 3))

    @testset "warmup gate and predictor values" begin
        atoms = make_atoms(lat, frac, ones(Int, N))
        sd = atoms.arrays[:shelldisp]
        cnt = _init_shist!(atoms)
        @test cnt == 0

        # Steps 1-3: extrapolation must be a strict no-op (history warming up).
        for k in 1:3
            sd .= sval(k)
            before = copy(sd)
            _extrapolate_shelldisp!(atoms, cnt)
            @test sd == before
            cnt = _rotate_shist!(atoms, cnt)
            @test cnt == k
        end

        # History now holds [s(3), s(2), s(1)] — the initialisation copies
        # have been fully shifted out without ever being read.
        @test atoms.arrays[:shist1] == sval(3)
        @test atoms.arrays[:shist2] == sval(2)
        @test atoms.arrays[:shist3] == sval(1)

        # Step 4: 3-point predictor 3·s(t) − 3·s(t−1) + s(t−2).
        _extrapolate_shelldisp!(atoms, cnt)
        @test sd ≈ 3 .* sval(3) .- 3 .* sval(2) .+ sval(1)
    end

    @testset "history follows the CellList-style resort (atoms[idx])" begin
        atoms = make_atoms(lat, frac, ones(Int, N))
        sd = atoms.arrays[:shelldisp]
        cnt = _init_shist!(atoms)
        for k in 1:3
            sd .= sval(k)
            cnt = _rotate_shist!(atoms, cnt)
        end

        perm = [3, 1, 4, 2]
        sorted = atoms[perm]
        @test sorted.arrays[:shist1] == sval(3)[perm, :]
        @test sorted.arrays[:shist2] == sval(2)[perm, :]
        @test sorted.arrays[:shist3] == sval(1)[perm, :]

        # Predicting after the resort equals resorting the prediction.
        _extrapolate_shelldisp!(sorted, cnt)
        @test sorted.arrays[:shelldisp] ≈ (3 .* sval(3) .- 3 .* sval(2) .+ sval(1))[perm, :]
    end

    @testset "selectable predictor order 0:4 (shell_extrap_order)" begin
        binom(n, k) = factorial(n) ÷ (factorial(k) * factorial(n - k))
        for order in 0:4
            atoms = make_atoms(lat, frac, ones(Int, N))
            sd = atoms.arrays[:shelldisp]
            cnt = _init_shist!(atoms; order = order)
            nslots = max(3, order + 1)
            @test all(haskey(atoms.arrays, Symbol(:shist, j)) for j in 1:nslots)
            @test !haskey(atoms.arrays, Symbol(:shist, nslots + 1))
            # Smooth history s(k) = c0 + c1·k + … + c_order·k^order: an
            # order-k predictor reproduces a degree-k polynomial exactly.
            coef = [randn(FMD.FPPREC, N, 3) for _ in 0:order]
            spoly(k) = sum(coef[p + 1] .* FMD.FPPREC(k)^p for p in 0:order)
            for k in 1:order + 1
                sd .= spoly(k)
                before = copy(sd)
                _extrapolate_shelldisp!(atoms, cnt; order = order)
                @test sd == before                 # warmup: no-op
                cnt = _rotate_shist!(atoms, cnt)
            end
            sd .= 0
            _extrapolate_shelldisp!(atoms, cnt; order = order)
            @test sd ≈ spoly(order + 2) rtol = 1e-10
            # Generic coefficient form on an arbitrary (non-polynomial) history.
            hist = [atoms.arrays[Symbol(:shist, j)] for j in 1:order + 1]
            @test sd ≈ sum((-1)^(j + 1) * binom(order + 1, j) .* hist[j] for j in 1:order + 1)
            # Rotation shifts every registered slot.
            snap = [copy(atoms.arrays[Symbol(:shist, j)]) for j in 1:nslots]
            sd .= sval(7)
            cnt = _rotate_shist!(atoms, cnt)
            @test atoms.arrays[:shist1] == sval(7)
            @test all(atoms.arrays[Symbol(:shist, j)] == snap[j - 1] for j in 2:nslots)
        end
        # Default keyword equals the historical 3-point predictor bitwise.
        a1 = make_atoms(lat, frac, ones(Int, N)); a2 = make_atoms(lat, frac, ones(Int, N))
        c1 = _init_shist!(a1); c2 = _init_shist!(a2; order = 2)
        for k in 1:3
            a1.arrays[:shelldisp] .= sval(k) .+ k^2; a2.arrays[:shelldisp] .= a1.arrays[:shelldisp]
            c1 = _rotate_shist!(a1, c1); c2 = _rotate_shist!(a2, c2)
        end
        _extrapolate_shelldisp!(a1, c1); _extrapolate_shelldisp!(a2, c2; order = 2)
        @test a1.arrays[:shelldisp] == a2.arrays[:shelldisp]
        @test a1.arrays[:shelldisp] == 3 .* a1.arrays[:shist1] .- 3 .* a1.arrays[:shist2] .+ a1.arrays[:shist3]
        @test_throws ErrorException _init_shist!(make_atoms(lat, frac, ones(Int, N)); order = 5)
    end

    @testset "shell-residual work ΔW and direction cosθ (_shell_work_step!)" begin
        # Quadratic shell energy E(s) = ½ k |s − s★|², residual F = −k (s − s★).
        # F is linear in s, so the trapezoid ΔW = ½(F₀ + F₁)·(s₁ − s₀) is the
        # exact line integral: ΔW == −(E₁ − E₀), and conserved + W_d sees no
        # leak from the shell update.
        atoms = make_atoms(lat, frac, ones(Int, N))
        dom = FMD.Cores.domain(atoms)
        a = atoms.arrays
        k = FMD.FPPREC(0.7)
        sstar = FMD.FPPREC.(0.01 .* sin.(reshape(1:3N, N, 3)))
        E(s) = k / 2 * sum(abs2, s .- sstar)
        F(s) = -k .* (s .- sstar)
        s0 = FMD.FPPREC.(0.02 .* cos.(reshape(1:3N, N, 3)))
        s1 = s0 .+ FMD.FPPREC.(0.005 .* reshape(1:3N, N, 3) ./ 3N)
        a[:shelldisp] .= s0
        a[:shellforces] = F(s0)
        _init_shist!(atoms)
        _init_shell_work!(atoms)
        @test a[:shres_prev] == F(s0)
        @test a[:shres_prev] !== a[:shellforces]       # a copy, not an alias

        a[:shelldisp] .= s1
        a[:shellforces] .= F(s1)
        dW, c = _shell_work_step!(atoms, dom)
        @test dW ≈ -(E(s1) - E(s0)) rtol = 1e-10
        fbar = (F(s0) .+ F(s1)) ./ 2
        ds = s1 .- s0
        @test c ≈ sum(fbar .* ds) / (norm(fbar) * norm(ds)) rtol = 1e-10
        @test -1 <= c <= 1
        # The previous residual has advanced to the current one.
        @test a[:shres_prev] == F(s1)

        # Residual exactly along / against the shell motion: cosθ = ±1.
        _rotate_shist!(atoms, 0)
        d = FMD.FPPREC.(reshape(1:3N, N, 3)) .* 1e-3
        a[:shres_prev] .= d; a[:shellforces] .= d
        a[:shelldisp] .= a[:shist1] .+ d
        @test _shell_work_step!(atoms, dom)[2] ≈ 1 rtol = 1e-10
        _rotate_shist!(atoms, 0)
        a[:shres_prev] .= -d; a[:shellforces] .= -d
        a[:shelldisp] .= a[:shist1] .+ d
        @test _shell_work_step!(atoms, dom)[2] ≈ -1 rtol = 1e-10

        # No shell motion: ΔW = 0 and cosθ undefined (NaN), never a fake 0.
        _rotate_shist!(atoms, 0)
        dW0, c0 = _shell_work_step!(atoms, dom)
        @test dW0 == 0
        @test isnan(c0)
    end
end
