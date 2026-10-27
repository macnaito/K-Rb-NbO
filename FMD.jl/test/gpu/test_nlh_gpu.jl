# test/gpu/test_nlh_gpu.jl
#
# The NLH baseline runs through KernelAbstractions on either backend; the
# coefficient tensors have to be adapted to the device alongside `Atoms`.
# Verify energy / forces / virial agree between CPU and GPU.

using Test
using CUDA
using Adapt
using Random
using LinearAlgebra

@testset "NLH CPU vs GPU parity" begin
    Random.seed!(124)
    elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3)
    atoms_cpu = replicate_atoms(make_perovskite_5(; a_ang = 4.0, A = "Ba", B = "Ti"),
                                (2, 2, 2))
    atoms_cpu.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:pos]))
    nlh_cpu = Calculators.NLHParam(elements, 5.0 / UNITS[:Å])

    a_cpu = deepcopy_atoms(atoms_cpu)
    reset!(a_cpu)
    Calculators.calculate!(nlh_cpu, a_cpu)
    E_cpu = a_cpu.info[:potential_energy][1]
    F_cpu = copy(a_cpu.arrays[:forces])
    V_cpu = copy(a_cpu.info[:virial])
    @test abs(E_cpu) > 1e-6            # the fixture actually overlaps

    nlh_gpu = adapt(CuArray, nlh_cpu)
    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
    reset!(a_gpu)
    Calculators.calculate!(nlh_gpu, a_gpu)
    E_gpu = Array(a_gpu.info[:potential_energy])[1]
    F_gpu = Array(a_gpu.arrays[:forces])
    V_gpu = Array(a_gpu.info[:virial])

    @test isapprox(E_cpu, E_gpu; rtol = 1e-8, atol = 1e-10)
    @test isapprox(F_cpu, F_gpu; rtol = 1e-6, atol = 1e-9)
    @test isapprox(V_cpu, V_gpu; rtol = 1e-6, atol = 1e-9)
end
