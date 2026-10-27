# test/core/test_ghosts.jl
#
# Unit tests for the Ghosts primitives (build_ghosts / sync_ghosts! /
# extend_ghosts! / fold_ghosts!) — the extended-row-space contract every
# ghost-based force kernel will rely on.
using Test
using LinearAlgebra

@testset "Ghosts" begin
    atoms = make_nacl_8()
    a = atoms.info[:lattice][1, 1]
    rc = FMD.FPPREC(0.49 * a)

    @testset "build: selection matches brute force, xcart carries the shift" begin
        g = Cores.build_ghosts(atoms, rc)
        Cores.sync_ghosts!(g, atoms)
        N = length(atoms)
        G = length(g.src)
        @test g.n_owned == N
        @test G > 0

        # Brute-force reference: enumerate every image in the ±1 layer and
        # keep those inside the fractional halo window [-δ, 1+δ).
        lat = Array(atoms.info[:lattice])
        δ = Cores._halo_frac(lat, rc)
        pos = Array(atoms.arrays[:pos])
        expected = Set{NTuple{4,Int}}()
        for i in 1:N, sx in -1:1, sy in -1:1, sz in -1:1
            (sx == 0 && sy == 0 && sz == 0) && continue
            w = pos[i, :] .- floor.(pos[i, :]) .+ [sx, sy, sz]
            if all(-δ[d] <= w[d] < 1 + δ[d] for d in 1:3)
                ws = Int.(-floor.(pos[i, :]))
                push!(expected, (i, ws[1] + sx, ws[2] + sy, ws[3] + sz))
            end
        end
        got = Set((Int(Array(g.src)[k]), Int.(Array(g.shift)[N+k, :])...) for k in 1:G)
        @test got == expected

        # Owned rows of shift are the frozen wrap offsets (all zero here —
        # nacl_8 positions are inside [0,1)); xcart of a ghost is its
        # owner's xcart plus the image lattice vector.
        @test all(==(0), Array(g.shift)[1:N, :])
        xc = Array(g.xcart)
        shift = Array(g.shift)
        src = Array(g.src)
        for k in 1:G
            o = Int(src[k])
            img = xc[o, :] .+ vec(Float64.(shift[N+k, :] .- shift[o, :])' * lat)
            @test isapprox(xc[N+k, :], img; atol = 1e-10)
        end

        # elemid replicated correctly.
        @test Array(g.elemid_ext)[N+1:end] == Array(atoms.arrays[:elemid])[Array(g.src)]
    end

    @testset "frozen wrap offsets keep xcart continuous across a boundary" begin
        a2 = deepcopy_atoms(atoms)
        a2.arrays[:pos][1, :] .= FMD.FPPREC[0.999, 0.5, 0.5]
        g = Cores.build_ghosts(a2, rc)
        Cores.sync_ghosts!(g, a2)
        x_before = Array(g.xcart)[1, :]

        # Drift across the box face WITHOUT a rebuild: xcart must move by
        # exactly the small Cartesian step, not jump by a lattice vector.
        a2.arrays[:pos][1, 1] += FMD.FPPREC(0.002)   # 0.999 → 1.001
        Cores.sync_ghosts!(g, a2)
        x_after = Array(g.xcart)[1, :]
        step = 0.002 * Array(a2.info[:lattice])[1, :]
        @test isapprox(x_after .- x_before, step; atol = 1e-10)
    end

    @testset "extend_ghosts! replicates owner values" begin
        g = Cores.build_ghosts(atoms, rc)
        N = length(atoms)
        G = length(g.src)
        vals = FMD.FPPREC.(reshape(1:3N, N, 3))
        ext = zeros(FMD.FPPREC, N + G, 3)
        Cores.extend_ghosts!(ext, vals, g)
        @test ext[1:N, :] == vals
        @test ext[N+1:end, :] == vals[Array(g.src), :]
    end

    @testset "fold_ghosts! adds ghost rows onto owners (with duplicates)" begin
        g = Cores.build_ghosts(atoms, rc)
        N = length(atoms)
        G = length(g.src)
        ext = FMD.FPPREC.(reshape(1:(3 * (N + G)), N + G, 3))
        dst = ones(FMD.FPPREC, N, 3)
        Cores.fold_ghosts!(dst, ext, g)

        expected = ones(FMD.FPPREC, N, 3) .+ ext[1:N, :]
        for k in 1:G
            expected[Int(Array(g.src)[k]), :] .+= ext[N+k, :]
        end
        @test isapprox(dst, expected; rtol = 1e-12)
    end

    @testset "multi-layer images for a cell smaller than the cutoff" begin
        small = make_perovskite_5()
        rc_big = FMD.FPPREC(1.6 * small.info[:lattice][1, 1])
        g = Cores.build_ghosts(small, rc_big)
        # δ = 1.6 → image layers up to ±2 must appear.
        @test maximum(abs, Array(g.shift)) == 2

        # Full ±2-layer brute-force reference with exact (src, shift) set
        # match. The naive↔CellList cross-check shares build_ghosts, so this
        # is the only independent verification of the image enumeration.
        lat = Array(small.info[:lattice])
        δ = Cores._halo_frac(lat, rc_big)
        pos = Array(small.arrays[:pos])
        N = length(small)
        expected = Set{NTuple{4,Int}}()
        for i in 1:N, sx in -2:2, sy in -2:2, sz in -2:2
            (sx == 0 && sy == 0 && sz == 0) && continue
            w = pos[i, :] .- floor.(pos[i, :]) .+ [sx, sy, sz]
            if all(-δ[d] <= w[d] < 1 + δ[d] for d in 1:3)
                ws = Int.(-floor.(pos[i, :]))
                push!(expected, (i, ws[1] + sx, ws[2] + sy, ws[3] + sz))
            end
        end
        got = Set((Int(Array(g.src)[k]), Int.(Array(g.shift)[N+k, :])...)
                  for k in 1:length(g.src))
        @test got == expected
    end
end
