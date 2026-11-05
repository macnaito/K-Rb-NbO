# test/apps/test_shist.jl
#
# Unit tests for the npt! shell-displacement extrapolation history helpers
# (_init_shist! / _extrapolate_shelldisp! / _rotate_shist!).  npt! itself has
# no test harness (it needs a fitted potential), so the warmup gating, the
# predictor values, and the follow-the-resort behaviour are pinned down here.
using Test
using LinearAlgebra
using FMD.Apps: _init_shist!, _extrapolate_shelldisp!, _rotate_shist!

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

    @testset "ASPC predictor (shell_predictor = :aspc)" begin
        using FMD.Apps: _aspc_coeffs, _shell_predictor_npoints
        # Kolafa (2004) table.
        @test _aspc_coeffs(0) == [2, -1]
        @test _aspc_coeffs(1) == [5//2, -2, 1//2]
        @test _aspc_coeffs(2) == [14//5, -14//5, 6//5, -1//5]
        @test _aspc_coeffs(3) == [3, -24//7, 27//14, -4//7, 1//14]
        @test _aspc_coeffs(4) == [22//7, -55//14, 55//21, -22//21, 5//21, -1//42]
        for k in 0:4
            B = _aspc_coeffs(k)
            @test sum(B) == 1                                   # constant history
            @test sum(B[j] * (1 - j) for j in eachindex(B)) == 1  # linear: s(t−j+1) = 1−j → s(t+1) = 1
            # Not exact for a quadratic (by design): s = τ², τ = 1−j, target 1.
            @test sum(B[j] * (1 - j)^2 for j in eachindex(B)) != 1
            @test _shell_predictor_npoints(:aspc, k) == k + 2
        end

        for k in 0:4
            atoms = make_atoms(lat, frac, ones(Int, N))
            sd = atoms.arrays[:shelldisp]
            cnt = _init_shist!(atoms; predictor = :aspc, order = k)
            nslots = max(3, k + 2)
            @test all(haskey(atoms.arrays, Symbol(:shist, j)) for j in 1:nslots)
            @test !haskey(atoms.arrays, Symbol(:shist, nslots + 1))
            c0 = randn(FMD.FPPREC, N, 3); c1 = randn(FMD.FPPREC, N, 3)
            slin(t) = c0 .+ FMD.FPPREC(t) .* c1
            for t in 1:k + 2
                sd .= slin(t)
                before = copy(sd)
                _extrapolate_shelldisp!(atoms, cnt; predictor = :aspc, order = k)
                @test sd == before                                  # warmup: no-op
                cnt = _rotate_shist!(atoms, cnt)
            end
            _extrapolate_shelldisp!(atoms, cnt; predictor = :aspc, order = k)
            @test sd ≈ slin(k + 3) rtol = 1e-12                     # linear history: exact
            hist = [atoms.arrays[Symbol(:shist, j)] for j in 1:k + 2]
            @test sd ≈ sum(FMD.FPPREC(_aspc_coeffs(k)[j]) .* hist[j] for j in 1:k + 2)
        end
        @test_throws ErrorException _init_shist!(make_atoms(lat, frac, ones(Int, N)); predictor = :aspc, order = 5)
        @test_throws ErrorException _init_shist!(make_atoms(lat, frac, ones(Int, N)); predictor = :foo, order = 2)
    end
end
