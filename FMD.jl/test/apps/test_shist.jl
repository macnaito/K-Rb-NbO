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
end
