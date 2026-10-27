# test/gpu/test_atoms_gpu.jl
#
# Reached only when CUDA.functional() == true (gated by runtests.jl).

using Test
using CUDA
using KernelAbstractions: get_backend, Backend

@testset "Atoms GPU roundtrip" begin
    atoms_cpu = make_nacl_8()

    @testset "change_arrtyp(CuArray) → change_arrtyp(Array)" begin
        atoms_gpu = Cores.change_arrtyp(atoms_cpu, CuArray)

        # GPU backend reported.
        @test get_backend(atoms_gpu) isa CUDA.CUDAKernels.CUDABackend

        # Non-isbits info entries stay host-side (Issue #54): the id →
        # symbol table must not become a CuArray{String}.
        @test atoms_gpu.info[:species] isa Vector{String}
        @test atoms_gpu.info[:species] == atoms_cpu.info[:species]

        # Round-trip.
        atoms_back = Cores.change_arrtyp(atoms_gpu, Array)
        @test atoms_back.arrays[:pos] == atoms_cpu.arrays[:pos]
        @test atoms_back.arrays[:elemid] == atoms_cpu.arrays[:elemid]
        @test atoms_back.info[:lattice] == atoms_cpu.info[:lattice]
        @test atoms_back.info[:species] == atoms_cpu.info[:species]
    end

    @testset "setelemid! remap on device-resident Atoms" begin
        # The remap path round-trips :elemid through the host
        # (Array → copyto!) — exercise it on a CuArray-backed frame.
        a_gpu = Cores.change_arrtyp(atoms_cpu, CuArray)
        setelemid!(a_gpu, Dict("Cl" => 1, "Na" => 2))
        @test Array(a_gpu.arrays[:elemid]) == 3 .- atoms_cpu.arrays[:elemid]
        @test a_gpu.info[:species] == ["Cl", "Na"]
        # The CPU original is untouched (fresh elemid array, rebound table).
        @test atoms_cpu.info[:species] == ["Na", "Cl"]
    end

    @testset "change_arrtyp drops the remap lock (Issue #70)" begin
        # A sliced (locked) family: the device copy severs the sharing
        # (fresh info + arrays), so it unlocks and can be remapped without
        # touching the CPU family, which itself stays locked.
        a = deepcopy_atoms(atoms_cpu)
        _ = a[1:4]
        @test haskey(a.info, :remap_locked)
        g = Cores.change_arrtyp(a, CuArray)
        @test !haskey(g.info, :remap_locked)
        setelemid!(g, Dict("Cl" => 1, "Na" => 2))
        @test Array(g.arrays[:elemid]) == 3 .- a.arrays[:elemid]
        @test a.info[:species] == ["Na", "Cl"]
        @test_throws ErrorException setelemid!(a, Dict("Cl" => 1, "Na" => 2))
    end
end
