# test/gpu/test_ace_spline_gpu.jl
#
# Spline is wired through KernelAbstractions and runs on either CPU
# (LoopVectorization paths) or GPU (CUDABackend kernels via @kernel).
# Verify forces/energy match between the two backends within tolerance.
# ACE GPU parity lives in test_ace_gpu.jl (the KA path in ACEFast.jl vs
# the CPU reference in ACE.jl).

using Test
using CUDA
using Adapt
using Random
using LinearAlgebra

@testset "Spline CPU vs GPU parity" begin
    Random.seed!(0)
    atoms_cpu = make_perovskite_5(; a_ang = 4.0)
    rmax2 = 4.0 / UNITS[:Å]
    rmax3 = 3.0 / UNITS[:Å]
    splp_cpu = Calculators.SplineParam(FMD.FPPREC(rmax2), FMD.FPPREC(rmax3),
                                       3, 6, 5)
    splp_cpu.c1 .= randn(FMD.FPPREC, size(splp_cpu.c1)) .* 1e-3
    splp_cpu.c2 .= randn(FMD.FPPREC, size(splp_cpu.c2)) .* 1e-3
    splp_cpu.c3 .= randn(FMD.FPPREC, size(splp_cpu.c3)) .* 1e-3

    # CPU run.
    a_cpu = deepcopy_atoms(atoms_cpu)
    reset!(a_cpu)
    Calculators.calculate!(splp_cpu, a_cpu)
    E_cpu = a_cpu.info[:potential_energy][1]
    F_cpu = copy(a_cpu.arrays[:forces])

    # GPU run.  Adapt SplineParam itself to GPU since its tensors must
    # live on the device.
    splp_gpu = Calculators.SplineParam(FMD.FPPREC(rmax2), FMD.FPPREC(rmax3),
                                       3, 6, 5)
    splp_gpu.c1 .= splp_cpu.c1
    splp_gpu.c2 .= splp_cpu.c2
    splp_gpu.c3 .= splp_cpu.c3
    splp_gpu = adapt(CuArray, splp_gpu)

    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
    reset!(a_gpu)
    Calculators.calculate!(splp_gpu, a_gpu)
    E_gpu = Array(a_gpu.info[:potential_energy])[1]
    F_gpu = Array(a_gpu.arrays[:forces])

    @test isapprox(E_cpu, E_gpu; rtol = 1e-5, atol = 1e-8)
    @test isapprox(F_cpu, F_gpu; rtol = 1e-5, atol = 1e-8)
end
