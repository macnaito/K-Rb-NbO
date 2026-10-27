# test/calculators/test_ace_merge.jl
#
# `+(::ACEParam, ::ACEParam)` — Δ-learning merge の等価性テスト．
# ACE エネルギーは係数について線形，かつ Chebyshev radial 基底は (rmin, rmax)
# のみで決まる入れ子なので，「マージ後の 1 回の評価」と「個別評価の和」は
# 機械精度で一致しなければならない（異なる基底数の埋め込みを含めて）．
using Test
using Random

@testset "ACEParam + (Δ-learning merge)" begin
    Random.seed!(11)
    rmin = FMD.FPPREC(1.0 / UNITS[:Å])
    rmax = FMD.FPPREC(4.0 / UNITS[:Å])

    randomize!(p) = begin
        for f in fieldnames(typeof(p))
            x = getfield(p, f)
            x isa AbstractArray && (x .= randn(FMD.FPPREC, size(x)) .* 1e-3)
        end
        p
    end

    # E / F / virial をまとめて返す評価ヘルパ
    function eval_efv(acep, atoms)
        a = deepcopy_atoms(atoms)
        reset!(a)
        Calculators.calculate!(acep, a)
        return a.info[:potential_energy][1], copy(a.arrays[:forces]), copy(a.info[:virial])
    end

    acep_a = randomize!(Calculators.ACEParam(rmin, rmax, 3, 6, 5, 4, 3))
    acep_b = randomize!(Calculators.ACEParam(rmin, rmax, 3, 4, 4, 3, 3))
    merged = acep_a + acep_b

    @testset "union orders and hierarchy" begin
        @test Calculators._ace_nk(merged) == (6, 5, 4, 3)
        n0, n1, n2, n3 = Calculators._ace_nk(merged)
        @test n0 >= n1 >= n2 >= n3 >= 3
    end

    @testset "E/F/V additivity across different nk (machine precision)" begin
        for trial in 1:3
            atoms = make_perovskite_5(; a_ang = 4.0)
            atoms.arrays[:pos] .+= 0.04 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
            Ea, Fa, Va = eval_efv(acep_a, atoms)
            Eb, Fb, Vb = eval_efv(acep_b, atoms)
            Em, Fm, Vm = eval_efv(merged, atoms)
            # 非空虚性: E だけでなく力・virial も実際に非零で，delta が効いていること
            @test maximum(abs, Fa) > 0 && maximum(abs, Fb) > 0
            @test maximum(abs, Va) > 0 && maximum(abs, Vb) > 0
            @test !isapprox(Em, Ea; rtol = 1e-6)
            @test isapprox(Em, Ea + Eb; rtol = 1e-12, atol = 1e-15)
            @test isapprox(Fm, Fa .+ Fb; rtol = 1e-12, atol = 1e-14)
            @test isapprox(Vm, Va .+ Vb; rtol = 1e-12, atol = 1e-14)
        end
    end

    @testset "mixed union (embedding non-trivial on both sides)" begin
        # union がどちらの入力とも一致しないケース: (6,4,4,3) + (5,5,3,3) → (6,5,4,3)．
        # 両側の埋め込みが恒等写像にならないので，レベルごとに max の出所が異なる
        # 経路（a 側・b 側両方のブロック再配置）を検証する．
        acep_p = randomize!(Calculators.ACEParam(rmin, rmax, 3, 6, 4, 4, 3))
        acep_q = randomize!(Calculators.ACEParam(rmin, rmax, 3, 5, 5, 3, 3))
        mpq = acep_p + acep_q
        @test Calculators._ace_nk(mpq) == (6, 5, 4, 3)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.04 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        Ep, Fp, Vp = eval_efv(acep_p, atoms)
        Eq, Fq, Vq = eval_efv(acep_q, atoms)
        Em, Fm, Vm = eval_efv(mpq, atoms)
        @test maximum(abs, Fp) > 0 && maximum(abs, Fq) > 0
        @test isapprox(Em, Ep + Eq; rtol = 1e-12, atol = 1e-15)
        @test isapprox(Fm, Fp .+ Fq; rtol = 1e-12, atol = 1e-14)
        @test isapprox(Vm, Vp .+ Vq; rtol = 1e-12, atol = 1e-14)
    end

    @testset "_ACE_FIELD_LEVELS covers every coefficient field" begin
        arrfields = Tuple(n for n in fieldnames(Calculators.ACEParam)
                          if getfield(acep_a, n) isa AbstractArray)
        @test arrfields == keys(Calculators._ACE_FIELD_LEVELS)
    end

    @testset "equal nk merge = plain coefficient sum" begin
        acep_c = randomize!(Calculators.ACEParam(rmin, rmax, 3, 6, 5, 4, 3))
        m2 = acep_a + acep_c
        for f in fieldnames(typeof(m2))
            x = getfield(m2, f)
            x isa AbstractArray || continue
            @test x == getfield(acep_a, f) .+ getfield(acep_c, f)   # 埋め込みが恒等
        end
    end

    @testset "zero delta is exact identity" begin
        zero_delta = Calculators.ACEParam(rmin, rmax, 3, 4, 4, 3, 3)   # all-zero
        m3 = acep_a + zero_delta
        for f in fieldnames(typeof(m3))
            x = getfield(m3, f)
            x isa AbstractArray || continue
            @test x == getfield(acep_a, f)                             # ビット等価
        end
    end

    @testset "incompatible bases error" begin
        other_domain = Calculators.ACEParam(rmin, FMD.FPPREC(1.1) * rmax, 3, 4, 4, 3, 3)
        @test_throws ErrorException acep_a + other_domain
        other_rmin = Calculators.ACEParam(FMD.FPPREC(0.9) * rmin, rmax, 3, 4, 4, 3, 3)
        @test_throws ErrorException acep_a + other_rmin
        other_species = Calculators.ACEParam(rmin, rmax, 2, 4, 4, 3, 3)
        @test_throws ErrorException acep_a + other_species
    end
end
