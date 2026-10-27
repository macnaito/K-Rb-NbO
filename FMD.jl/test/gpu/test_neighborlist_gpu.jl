# test/gpu/test_neighborlist_gpu.jl
#
# GPU parity for the ghost-based neighbour machinery: the CuArray-built
# lists must reduce to the same canonical pair map as the CPU-built ones
# (the ghost_pairmap helper comes from
# test/core/test_neighborlist.jl, included earlier in the same session).
using Test
using CUDA
using LinearAlgebra

@testset "Ghost-based neighbour lists (GPU)" begin
    @testset "GPU ↔ CPU pair parity: $name" for (name, mk, rcof) in (
        ("NaCl 8", make_nacl_8, a -> 0.55 * a.info[:lattice][1, 1]),
        ("perovskite 40, shuffled",
         () -> replicate_atoms(make_perovskite_5(), (2, 2, 2); shuffle_order = true),
         a -> 0.55 * a.info[:lattice][1, 1] / 2),
    )
        atoms = mk()
        rc = FMD.FPPREC(rcof(atoms))
        ref = ghost_pairmap(atoms, neighborlist(atoms, rc, 0.0))
        @test !isempty(ref)

        # Regression for Issue #54: the fixtures carry info[:species]
        # (frame-level table); per-atom arrays are all isbits, so the
        # CellList re-sort device-gathers without a host fallback.
        a_gpu = Cores.change_arrtyp(atoms, CuArray)

        nl_naive = neighborlist(a_gpu, rc, 0.0)
        @test ghost_pairmap(a_gpu, nl_naive) == ref

        cl, a_sorted, _ = CellList(a_gpu, rc)
        nl_cl = neighborlist(a_sorted, cl, rc, 0.0)
        @test ghost_pairmap(a_sorted, nl_cl) == ref
    end

    @testset "Ghost primitives on CuArray (fold with duplicates)" begin
        atoms = make_nacl_8()
        rc = FMD.FPPREC(0.49 * atoms.info[:lattice][1, 1])
        g_cpu = Cores.build_ghosts(atoms, rc)
        N = length(atoms)
        G = length(g_cpu.src)

        a_gpu = Cores.change_arrtyp(atoms, CuArray)
        g = Cores.build_ghosts(a_gpu, rc)
        @test length(g.src) == G

        ext = CuArray(FMD.FPPREC.(reshape(1:(3 * (N + G)), N + G, 3)))
        dst = CUDA.zeros(FMD.FPPREC, N, 3)
        Cores.fold_ghosts!(dst, ext, g)

        ext_h = FMD.FPPREC.(reshape(1:(3 * (N + G)), N + G, 3))
        expected = copy(ext_h[1:N, :])
        src_h = Array(g.src)
        for k in 1:G
            expected[Int(src_h[k]), :] .+= ext_h[N+k, :]
        end
        @test isapprox(Array(dst), expected; rtol = 1e-12)
    end
end
