# test/gpu/test_surface_term_gpu.jl
#
# SurfaceTerm on the GPU backend — CPU↔GPU serial parity for surface_term!
# (energy / virial / forces / shellforces / results).  The struct itself is
# host-only (T, M_frac_ref are 3×3 / 3-vector scalar-side data); what runs
# on the device is the dipole reduction inside calcdipole! and the
# per-charge force scatter, so those are what parity pins here.
#
# Fixture: rattled NaCl-64 with shells (distinct qs:qc ratios per element,
# neutral cell) and a fully anisotropic ε_out + rotated ellipsoid so no
# force/virial component vanishes by symmetry.
#
# Tolerances follow the other GPU parity tests (rtol 1e-6, atol 1e-10):
# the device dipole reduction reassociates the sum (~1e-14 relative).

using Test
using CUDA
using Random
using LinearAlgebra

@testset "SurfaceTerm (GPU) parity vs CPU" begin
    Random.seed!(179)
    proto = make_nacl_64()
    rattled = deepcopy_atoms(proto)
    rattled.arrays[:pos] .+= FMD.FPPREC(0.01) .*
        randn(FMD.FPPREC, size(rattled.arrays[:pos]))
    rattled.arrays[:shelldisp] .= FMD.FPPREC(0.05) .*
        randn(FMD.FPPREC, size(rattled.arrays[:shelldisp]))

    qc = FMD.FPPREC[+1.0, -1.0]
    qs = FMD.FPPREC[-0.2, +0.2]      # Σ(qc+qs) = 0, distinct qs:qc ratios
    prms_cpu = (qc = qc, qs = qs)
    prms_gpu = arrtyp_prms(CuArray, prms_cpu)

    # Deterministic rotation (Rodrigues) — same helper as the CPU test file,
    # local here so this file stands alone under the GPU profile.
    axis = [1.0, 2.0, 3.0] ./ norm([1.0, 2.0, 3.0])
    K = [0.0 -axis[3] axis[2]; axis[3] 0.0 -axis[1]; -axis[2] axis[1] 0.0]
    R = Matrix{Float64}(I, 3, 3) + sin(0.7) .* K + (1 - cos(0.7)) .* (K * K)
    ε = R * Diagonal([1.5, 3.0, 6.0]) * R'
    A = R * Diagonal([1.0, 1.6, 0.7]) * R'

    # Two instances so CPU and GPU evaluations do not share the results Dict.
    st_cpu = Calculators.SurfaceTerm(rattled, qc, qs;
                                     eps_out = ε, shape = A, gauge = proto)
    st_gpu = Calculators.SurfaceTerm(rattled, qc, qs;
                                     eps_out = ε, shape = A, gauge = proto)
    # Construction is host-side and deterministic: bit-equal tensors.
    @test st_gpu.T == st_cpu.T
    @test st_gpu.M_frac_ref == st_cpu.M_frac_ref

    a_cpu = deepcopy_atoms(rattled)
    reset!(a_cpu)
    Calculators.surface_term!(a_cpu, st_cpu, prms_cpu)

    a_gpu = Cores.change_arrtyp(deepcopy_atoms(rattled), CuArray)
    clear!(a_gpu, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.surface_term!(a_gpu, st_gpu, prms_gpu)

    # Non-vacuous reference: the rattle produced a real δM, forces on both
    # cores and shells, and live shear virial components.
    E_cpu = a_cpu.info[:potential_energy][1]
    @test E_cpu > 1e-8
    @test maximum(abs, a_cpu.arrays[:forces]) > 1e-8
    @test maximum(abs, a_cpu.arrays[:shellforces]) > 1e-8
    @test maximum(abs, Array(a_cpu.info[:virial])[4:6]) > 1e-8

    # Evaluation parity.
    @test st_gpu.results["bias"] ≈ st_cpu.results["bias"] rtol = 1e-6 atol = 1e-12
    @test isapprox(st_gpu.results["dM"], st_cpu.results["dM"];
                   rtol = 1e-6, atol = 1e-12)
    @test Array(a_gpu.info[:potential_energy])[1] ≈ E_cpu rtol = 1e-6 atol = 1e-12
    @test isapprox(Array(a_gpu.info[:virial]), Array(a_cpu.info[:virial]);
                   rtol = 1e-6, atol = 1e-10)
    @test isapprox(Array(a_gpu.arrays[:forces]), a_cpu.arrays[:forces];
                   rtol = 1e-6, atol = 1e-10)
    @test isapprox(Array(a_gpu.arrays[:shellforces]), a_cpu.arrays[:shellforces];
                   rtol = 1e-6, atol = 1e-10)
end
