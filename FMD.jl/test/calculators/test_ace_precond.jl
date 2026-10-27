# test/calculators/test_ace_precond.jl
#
# blockchol-si preconditioner (Issues #177, #183): the feature-syrk assembled
# per-(field × si) LIVE-column blocks must EQUAL the corresponding diagonal
# sub-blocks of the operator Hmul! applies — probed column by column (small
# blocks) or through random block-supported vectors (wide c0ll blocks; exact
# identity because columns outside a block never feed rows inside it), and
# the structurally dead columns (upper-triangle convention) must respond with
# exactly Γ·e_j. This pins the row kernel, the E/F/V weights, the ghost fold,
# the vec layout, and the live/dead split at once.

using Test
using Random
using LinearAlgebra
using MPI

@testset "ACE blockchol-si preconditioner" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(41)

    # ---- fixture: 3 frames of jittered 3-species perovskite + one 2×1×1
    # supercell (Nat = 10), random labels. The mixed frame sizes are
    # load-bearing: the row buffer is sized by maxNat while the virial-row
    # base vb = 1 + 3Nat is per-frame, so a maxNat/Nat mix-up is invisible
    # to a fixture whose frames are all the same size. ----
    frames = Atoms[]
    for _ in 1:3
        at = make_perovskite_5(; a_ang = 4.0)
        at.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(at.arrays[:pos]))
        at.info[:potential_energy] = FMD.FPPREC[0.1 * randn()]
        at.arrays[:forces] .= 1e-2 .* randn(FMD.FPPREC, size(at.arrays[:forces]))
        at.info[:virial] = 1e-3 .* randn(FMD.FPPREC, 6)
        push!(frames, at)
    end
    let at = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (2, 1, 1))
        at.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(at.arrays[:pos]))
        at.info[:potential_energy] = FMD.FPPREC[0.1 * randn()]
        at.arrays[:forces] .= 1e-2 .* randn(FMD.FPPREC, size(at.arrays[:forces]))
        at.info[:virial] = 1e-3 .* randn(FMD.FPPREC, 6)
        push!(frames, at)
    end

    rmin = FMD.FPPREC(1.0 / UNITS[:Å])
    rmax = FMD.FPPREC(4.0 / UNITS[:Å])
    mk() = Calculators.ACEParam(rmin, rmax, 3, 4, 4, 3, 3)

    Λ = FMD.FPPREC(1e-3)   # small on purpose: the data Gram must carry the test
    base = perovskite_params(frames[1]; with_shell = false)
    params = (; base...,
        weight_force  = 10.0,
        weight_stress = 1.0,
        method_weight = (; test = 1.0),
        rigid = (; cp = 1e-8, c0 = Λ, c00 = Λ, c11 = Λ, c22 = Λ, c33 = Λ,
                   c011 = Λ, c022 = Λ, c033 = Λ),
        nsteps = 400,
    )

    acep = mk()
    sys = Calculators.build_normal_system(acep, frames, params)
    n = length(sys.x0)
    layoutall = Calculators._ace_block_layout(acep)

    @testset "layout partitions 1:n, live maps match the triangle convention" begin
        @test first(layoutall[1].range) == 1
        @test last(layoutall[end].range) == n
        for k in 2:length(layoutall)
            @test first(layoutall[k].range) == last(layoutall[k-1].range) + 1
        end
        for blk in layoutall
            w = length(blk.range)
            @test issorted(blk.live) && allunique(blk.live)
            @test all(j -> 1 <= j <= w, blk.live)
            # independent derivation of the live set as a Bool mask over one
            # si-slab (same dead definition as the NaN-injection parity test in
            # test_ace.jl) — the source builds it via _pairindex ranks, so a
            # convention drift on either side breaks this comparison
            x = getfield(acep, blk.field)
            mask = if ndims(x) == 1
                trues(1)
            elseif ndims(x) == 2
                trues(size(x, 1))
            elseif ndims(x) == 3
                [μ <= ν for μ in 1:size(x, 1), ν in 1:size(x, 2)]
            else
                [ν >= λ for _ in 1:size(x, 1), ν in 1:size(x, 2), λ in 1:size(x, 3)]
            end
            @test findall(vec(mask)) == blk.live
        end
    end

    asm = Calculators._assemble_block_grams(acep, params, sys.precond_ctx)
    grams = Dict(idx => copy(G) for (idx, G) in enumerate(asm.grams) if G !== nothing)
    @test length(grams) == length(layoutall)   # single rank owns everything

    Γof(field) = FMD.FPPREC(params[:rigid][field])

    @testset "assembled block ≡ probed Hmul! block" begin
        q = zeros(n); e = zeros(n)
        for (idx, blk) in enumerate(layoutall)
            G = Symmetric(grams[idx], :U)
            w = length(blk.range)
            Γ = Γof(blk.field)
            livepos = Dict(j => k for (k, j) in enumerate(blk.live))
            if w <= 200
                # probe every stored column: live columns must match the
                # assembled live×live Gram (+Γ on the diagonal); dead columns
                # must return exactly Γ·e_j — their data contribution is a
                # structural zero (never read forward, never written adjoint),
                # so the identity is exact, not a tolerance
                for (jloc, jcol) in enumerate(blk.range)
                    e[jcol] = 1.0
                    sys.Hmul!(q, e)
                    e[jcol] = 0.0
                    if haskey(livepos, jloc)
                        ref = zeros(w)
                        ref[blk.live] .= G[:, livepos[jloc]]
                        ref[jloc] += Γ
                        @test norm(q[blk.range] .- ref) <= 1e-10 * (1 + norm(ref))
                    else
                        @test count(!iszero, q) == 1
                        @test q[jcol] == Γ
                    end
                end
            else
                # wide block: 10 random block-supported probes (live part
                # statistical; the dead part enters through Γ·v)
                for _ in 1:10
                    v = randn(w)
                    fill!(e, 0.0)
                    e[blk.range] .= v
                    sys.Hmul!(q, e)
                    ref = Γ .* v
                    ref[blk.live] .+= G * v[blk.live]
                    @test norm(q[blk.range] .- ref) <= 1e-10 * (1 + norm(ref))
                end
                # pin the 4-body dead convention exactly on a wide c0ll block:
                # dead columns respond with Γ·e_j and nothing else
                fill!(e, 0.0)   # clear the random support of the last probe
                deadcols = setdiff(1:w, blk.live)
                for jloc in deadcols[unique(round.(Int, range(1, length(deadcols); length = 5)))]
                    jcol = blk.range[jloc]
                    e[jcol] = 1.0
                    sys.Hmul!(q, e)
                    e[jcol] = 0.0
                    @test count(!iszero, q) == 1
                    @test q[jcol] == Γ
                end
            end
        end
    end

    @testset "group budget invariance" begin
        # budget_bytes = 1 forces one block per group (27 groups, the
        # production-default multi-group path): exercises the offs = -1
        # column-map skips and the per-center partial-block continue logic.
        # Per-block arithmetic is identical, so the Grams must match to
        # summation-order roundoff.
        asm2 = Calculators._assemble_block_grams(acep, params, sys.precond_ctx;
                                                 budget_bytes = 1)
        @test count(!isnothing, asm2.grams) == length(layoutall)
        for idx in eachindex(layoutall)
            @test isapprox(Symmetric(asm2.grams[idx], :U), Symmetric(grams[idx], :U);
                           rtol = 1e-13)
        end
    end

    @testset "panel invariance" begin
        # panel_bytes sized so the widest block lands in 3 panels: the gemm
        # (above-diagonal strip) + syrk (diagonal tile) accumulation and the
        # multi-sweep frame loop must reproduce the single-panel Grams to
        # summation-order roundoff. Also crossed with budget_bytes = 1
        # (1-block groups × panels — group and panel splitting are
        # orthogonal).
        S = sizeof(FMD.FPPREC)
        wmax = maximum(length(blk.live) for blk in layoutall)
        pb = S * wmax * cld(wmax, 3)
        for kw in ((; panel_bytes = pb), (; panel_bytes = pb, budget_bytes = 1))
            asmp = Calculators._assemble_block_grams(acep, params, sys.precond_ctx;
                                                     kw...)
            @test count(!isnothing, asmp.grams) == length(layoutall)
            for idx in eachindex(layoutall)
                @test isapprox(Symmetric(asmp.grams[idx], :U),
                               Symmetric(grams[idx], :U); rtol = 1e-13)
                # no zero-fill in the assembly: every window element is
                # written by exactly one panel Reduce, so the strict lower
                # triangle must be exactly zero
                @test all(iszero, tril(asmp.grams[idx], -1))
            end
        end
    end

    @testset "_precond_plan invariants" begin
        # pure planning — a production-scale oversized width costs nothing
        S = 8
        widths = [25_200, 9_750, 936, 936, 540, 78, 12, 1]
        budget = 2^31
        plan = Calculators._precond_plan(widths; budget_bytes = budget,
                                         panel_bytes = budget, S = S)
        # an oversized block (S·w² > budget) is always alone in its group
        for group in plan.groups
            if any(idx -> S * widths[idx]^2 > budget, group)
                @test length(group) == 1
            end
        end
        # panels tile each block; single full-width panel ⟺ S·w² ≤ panel_bytes
        for idx in eachindex(widths)
            pans = plan.panels[idx]
            @test first(first(pans)) == 1 && last(last(pans)) == widths[idx]
            @test all(first(pans[k]) == last(pans[k-1]) + 1 for k in 2:length(pans))
            @test (pans == [1:widths[idx]]) == (S * widths[idx]^2 <= budget)
        end
        # every (block, panel) item appears exactly once, in its own group
        seen = Set{Tuple{Int, UnitRange{Int}}}()
        for (g, group) in enumerate(plan.groups), sweep in plan.sweeps[g], item in sweep
            @test item.idx in group
            @test !((item.idx, item.cols) in seen)
            push!(seen, (item.idx, item.cols))
        end
        @test length(seen) == sum(length, plan.panels)
        # the rank-private sweep peak is bounded by the panel budget (escape:
        # a single item may exceed it) — the deterministic version of "the
        # per-rank transient fits the budget" for nk8765-scale widths
        for gs in plan.sweeps, sweep in gs
            total = sum(S * widths[item.idx] * length(item.cols) for item in sweep)
            @test total <= max(budget,
                               maximum(S * widths[item.idx] * length(item.cols)
                                       for item in sweep))
        end
        # degenerate budgets: every block escapes to its own group, panels
        # become single columns
        plan2 = Calculators._precond_plan(widths[3:end]; budget_bytes = 1,
                                          panel_bytes = 1, S = S)
        @test all(g -> length(g) == 1, plan2.groups)
        @test all(idx -> all(c -> length(c) == 1, plan2.panels[idx]),
                  eachindex(widths[3:end]))
    end

    @testset "ldiv! contract" begin
        P = Calculators.build_blockchol_si_preconditioner(acep, params, sys.precond_ctx)
        # resident factors are live-width — the memory claim of the live/dead split
        @test all(size(b.chol, 1) == length(b.live) for b in P.owned)
        r = randn(n)
        rkeep = copy(r)
        z = ldiv!(zeros(n), P, r)
        @test r == rkeep                       # ldiv! must not touch r
        @test dot(r, z) > 0                    # SPD
        # Residual form: a solution comparison against `\` would be amplified
        # by κ(block) (up to ~1e12 at this deliberately weak Λ); the Cholesky
        # solve is backward-stable, so H·z ≈ r is the κ-independent contract.
        # Href embeds the live×live Gram into the stored width and puts Γ on
        # the whole diagonal, so the dead rows Γ·z_dead = r_dead are checked too.
        for (idx, blk) in enumerate(layoutall)
            w = length(blk.range)
            Href = Matrix{Float64}(Γof(blk.field) * I, w, w)
            Href[blk.live, blk.live] .+= Matrix(Symmetric(grams[idx], :U))
            @test norm(Href * z[blk.range] .- r[blk.range]) <=
                  1e-10 * (1 + norm(Href) * norm(z[blk.range]))
        end
        # dead coordinates are solved analytically — bitwise identical to r/Γ
        for b in P.owned
            isempty(b.dead) && continue
            @test z[b.range][b.dead] == r[b.range][b.dead] ./ b.Γ
        end
    end

    @testset "banner reports live-column resident factors" begin
        banner = mktemp() do path, io
            redirect_stdout(io) do
                Calculators.build_blockchol_si_preconditioner(acep, params, sys.precond_ctx)
            end
            flush(io)
            read(path, String)
        end
        expected = round(sum(8 * length(blk.live)^2 for blk in layoutall) / 2^30; digits = 2)
        @test occursin("resident factors <= $(expected) GiB per rank", banner)
    end

    @testset "PCG solution ≡ CG solution" begin
        # generously regularised so both solves converge within nsteps
        # (3.0e5 for the 4-frame fixture: at 3.0e4 the PCG true residual
        # stalls near 3e-6 on this conditioning and misses the 1e-9 guard)
        Λc = FMD.FPPREC(3.0e5)
        pconv = (; params...,
            rigid = (; cp = Λc, c0 = Λc, c00 = Λc, c11 = Λc, c22 = Λc, c33 = Λc,
                       c011 = Λc, c022 = Λc, c033 = Λc))

        acep_cg = mk()
        Calculators.fitting!(acep_cg, frames, (; pconv..., cg_preconditioner = :none))
        acep_pcg = mk()
        Calculators.fitting!(acep_pcg, frames, (; pconv..., cg_preconditioner = :blockchol_si))

        # converged-fit guard: both must actually solve the same system
        probe = Calculators.build_normal_system(mk(), frames, pconv)
        Hx = zeros(n)
        for (x, tag) in ((vec(acep_cg), "cg"), (vec(acep_pcg), "pcg"))
            probe.Hmul!(Hx, copy(x))
            @test norm(Hx .- probe.y) <= 1e-9 * norm(probe.y)
        end
        @test isapprox(vec(acep_pcg), vec(acep_cg); rtol = 1e-6, atol = 1e-10)
    end

    @testset "default off is byte-identical" begin
        acep_a = mk()
        Calculators.fitting!(acep_a, frames, (; params..., nsteps = 30))
        acep_b = mk()
        Calculators.fitting!(acep_b, frames, (; params..., nsteps = 30,
                                                cg_preconditioner = :none))
        @test vec(acep_a) == vec(acep_b)
    end

    @testset "rigid guard" begin
        pbad = (; params..., rigid = (; cp = 1e-8, c0 = Λ, c00 = Λ, c11 = Λ,
                                        c22 = Λ, c33 = Λ, c011 = Λ, c022 = Λ,
                                        c033 = 0.0))
        err = try
            Calculators.build_blockchol_si_preconditioner(acep, pbad, sys.precond_ctx)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("c033", err.msg)
    end

    @testset "unknown mode errors" begin
        @test_throws ErrorException Calculators.fitting!(mk(), frames,
            (; params..., cg_preconditioner = :jacobi))
    end
end
