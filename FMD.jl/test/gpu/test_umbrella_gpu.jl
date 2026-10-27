# test/gpu/test_umbrella_gpu.jl
#
# PolarizationUmbrella on the GPU backend — CPU↔GPU serial parity for
# umbrella_p! (cv / bias / U_homo / sum_r2 / energy / virial / forces /
# shellforces).  First GPU coverage for the umbrella: Phase 2 stacks
# umbrella × GPU × DD on top of this, so the serial-GPU baseline must be
# pinned first.
#
# Fixture notes:
#  - TWO atoms per unit cell sharing one groupid, so the Atomix.@atomic
#    accumulation in group_p really contends on the GPU (a 1-atom-per-group
#    fixture would never race and the atomic path would go untested);
#  - two elements with DIFFERENT qs:qc ratios (0.5 vs 0.25) and a neutral
#    cell (Σ(qc+qs) = 0): with a single element shellforces ≡ (qs/qc)·forces
#    identically and their comparison would carry no independent
#    information;
#  - rattled cores + nonzero shelldisp so cv / forces / virial are
#    genuinely nonzero (thresholds below sit ~2 orders under the measured
#    magnitudes, far above the parity atol);
#  - non-cubic group grid (3,4,5): CUFFT and FFTW must agree across
#    distinct per-dimension sizes (a cubic grid could hide a dimension
#    -layout mixup in the rFFT round trip);
#  - config (a) uniform mode + homogeneity penalty + Gaussian smoothing,
#    (b) cosine mode m=(0,0,1) with k_homo > 0 so the FFT pass feeds the
#    compared energy/forces (with k_homo = 0 the cosine-mode cv comes from
#    the analytic ghat_at_m identity and the FFT output would be compared
#    nowhere), (c) flat-bottom (r_th > 0) with σ = 0, driving the
#    ifelse(rmag > r_th) device branch and the chain term that vanishes
#    analytically in the quadratic case.
#
# Tolerances: with 2 atoms/group the GPU atomic accumulation order differs
# from the CPU loop (reassociation ~1e-14 relative here); CUFFT vs FFTW
# adds the same order.  rtol 1e-6 / atol 1e-10 follows the other GPU
# parity tests and sits far above both.

using Test
using CUDA
using Random

# ASE-style supercell of a TWO-atom unit cell (flat group index
# iz-fastest, same layout convention as test/calculators/test_umbrella.jl):
# element 1 at the cell corner, element 2 at the body center, both mapped
# to the same groupid.
function _umb_gpu_supercell(reps::NTuple{3,Int}; a_bohr = 4.0)
    Nx, Ny, Nz = reps
    n = 2 * Nx * Ny * Nz
    pos = zeros(FMD.FPPREC, n, 3)
    groupid = zeros(Int, n)
    elemid = zeros(Int, n)
    k = 0
    for ix in 0:Nx-1, iy in 0:Ny-1, iz in 0:Nz-1
        gid = ix * Ny * Nz + iy * Nz + iz + 1
        for (e, off) in ((1, 0.0), (2, 0.5))
            k += 1
            pos[k, 1] = (ix + off) / Nx
            pos[k, 2] = (iy + off) / Ny
            pos[k, 3] = (iz + off) / Nz
            groupid[k] = gid
            elemid[k] = e
        end
    end
    info = Dict{Symbol,Any}(
        :lattice => FMD.FPPREC.([Nx*a_bohr 0 0; 0 Ny*a_bohr 0; 0 0 Nz*a_bohr]),
        :potential_energy => FMD.FPPREC[0],
        :virial => zeros(FMD.FPPREC, 6),
        :method => Symbol[:test],
    )
    arrays = Dict{Symbol,Array}(
        :pos => pos,
        :elemid => elemid,
        :groupid => groupid,
        :forces => zeros(FMD.FPPREC, n, 3),
        :shellforces => zeros(FMD.FPPREC, n, 3),
        :shelldisp => zeros(FMD.FPPREC, n, 3),
        :velo => zeros(FMD.FPPREC, n, 3),
        :id => collect(1:n),
        :masses => ones(FMD.FPPREC, n),
    )
    return Atoms{Array}(info, arrays)
end

@testset "PolarizationUmbrella (GPU) parity vs CPU" begin
    reps = (3, 4, 5)
    # Neutral cell (1.0 + 0.5 - 1.2 - 0.3 = 0), distinct qs:qc ratios.
    qc = FMD.FPPREC[1.0, -1.2]
    qs = FMD.FPPREC[0.5, -0.3]
    prms_cpu = (qc = qc, qs = qs)
    prms_gpu = arrtyp_prms(CuArray, prms_cpu)

    # Gauge reference frame (unrattled) and the evaluated frame (rattled
    # cores + rattled shells).  :pos is rattled without wrapping (the
    # polarization gauge convention — CLAUDE.md).
    atoms_ref = _umb_gpu_supercell(reps)
    rng = Random.Xoshiro(7)
    rattled = deepcopy_atoms(atoms_ref)
    rattled.arrays[:pos] .+= FMD.FPPREC(0.02) .* randn(rng, FMD.FPPREC, size(rattled.arrays[:pos]))
    rattled.arrays[:shelldisp] .= FMD.FPPREC(0.05) .* randn(rng, FMD.FPPREC, size(rattled.arrays[:shelldisp]))

    # Sanity for the fixture premise: the two elements give shellforces
    # that are NOT a scalar multiple of forces (checked below per config).
    ratio(a) = a.arrays[:shellforces] ./ a.arrays[:forces]

    # (label, k_par, k_homo, amplitude, sigma, m, r_th)
    configs = [
        ("uniform + homo + sigma", 1e3, 5e1, 0.02, 1.0, (0, 0, 0), 0.0),
        ("cosine m=(0,0,1) + homo", 1e3, 5e1, 0.02, 0.5, (0, 0, 1), 0.0),
        ("flat-bottom r_th, σ=0",   1e3, 5e1, 0.02, 0.0, (0, 0, 0), 2e-3),
    ]

    for (label, k_par, k_homo, amp, sigma, m, r_th) in configs
        @testset "$label" begin
            build_pu(; kw...) = Calculators.PolarizationUmbrella(
                atoms_ref, k_par, k_homo, amp, FMD.FPPREC[1, 0, 0], reps, qc, qs;
                gauge = atoms_ref, sigma = sigma, m = m, r_th = r_th, kw...)

            pu_cpu = build_pu()                    # default backend = KA.CPU()
            a_cpu = deepcopy_atoms(rattled)
            Calculators.umbrella_p!(a_cpu, pu_cpu, prms_cpu)

            pu_gpu = build_pu(backend = CUDA.CUDABackend())
            a_gpu = Cores.change_arrtyp(deepcopy_atoms(rattled), CuArray)
            Calculators.umbrella_p!(a_gpu, pu_gpu, prms_gpu)

            # Construction parity: host-side scalars baked in at build time
            # must agree exactly; the device tables are copies of the same
            # host computation, so they must be bit-equal.
            @test pu_gpu.ghat_at_m == pu_cpu.ghat_at_m
            @test pu_gpu.denom_inv == pu_cpu.denom_inv
            @test Array(pu_gpu.Ghat) == pu_cpu.Ghat
            @test Array(pu_gpu.M_frac_ref) == pu_cpu.M_frac_ref

            # Non-vacuous reference (thresholds calibrated ~2 orders below
            # the measured magnitudes of this fixture — see PR #97): the
            # bias engaged, forces reach both cores and shells, the virial
            # picked up the group-lattice and reference-cell terms, and the
            # homogeneity pass saw a real residual field.
            E_cpu = a_cpu.info[:potential_energy][1]
            @test abs(E_cpu) > 1e-3               # measured 0.24 – 0.63
            @test maximum(abs, a_cpu.arrays[:forces]) > 1e-4       # 7e-3 – 2.6e-2
            @test maximum(abs, a_cpu.arrays[:shellforces]) > 1e-4  # 2.9e-3 – 1.1e-2
            @test maximum(abs, a_cpu.info[:virial]) > 1e-3         # 2.9e-2 – 0.48
            @test pu_cpu.results["sum_r2"] > 1e-5                  # 8.8e-4 – 2.1e-2
            # Distinct qs:qc ratios really decouple shell from core forces
            # (the per-atom ratio spread is structurally |0.25 − 0.5| = 0.25).
            r = ratio(a_cpu)
            @test maximum(abs, r .- r[1, 1]) > 0.1

            # Evaluation parity.
            @test pu_gpu.results["cv"] ≈ pu_cpu.results["cv"] rtol = 1e-6 atol = 1e-12
            @test pu_gpu.results["bias"] ≈ pu_cpu.results["bias"] rtol = 1e-6 atol = 1e-12
            @test pu_gpu.results["U_homo"] ≈ pu_cpu.results["U_homo"] rtol = 1e-6 atol = 1e-12
            @test pu_gpu.results["sum_r2"] ≈ pu_cpu.results["sum_r2"] rtol = 1e-6 atol = 1e-12
            @test Array(a_gpu.info[:potential_energy])[1] ≈ E_cpu rtol = 1e-6 atol = 1e-12
            @test isapprox(Array(a_gpu.info[:virial]), a_cpu.info[:virial];
                           rtol = 1e-6, atol = 1e-10)
            @test isapprox(Array(a_gpu.arrays[:forces]), a_cpu.arrays[:forces];
                           rtol = 1e-6, atol = 1e-10)
            @test isapprox(Array(a_gpu.arrays[:shellforces]), a_cpu.arrays[:shellforces];
                           rtol = 1e-6, atol = 1e-10)
        end
    end
end

# PolarizationMapUmbrella on the GPU — same CPU↔GPU parity contract as above.
# Shares the smoothing machinery (PolField.jl) with the mode umbrella, so what
# is exercised here beyond that is the target-map/weight grids being built on
# the host and copied to the device, and the adjoint without a chain term.
@testset "PolarizationMapUmbrella CPU/GPU parity" begin
    reps = (3, 4, 5)
    ng = prod(reps)
    atoms_ref = _umb_gpu_supercell(reps)
    rng = Random.Xoshiro(11)
    rattled = deepcopy_atoms(atoms_ref)
    rattled.arrays[:pos] .+= FMD.FPPREC(0.015) .* randn(rng, FMD.FPPREC, size(rattled.arrays[:pos]))
    rattled.arrays[:shelldisp] .= FMD.FPPREC(0.05) .* randn(rng, FMD.FPPREC, size(rattled.arrays[:shelldisp]))
    qc = FMD.FPPREC[1.0, -0.5]
    qs = FMD.FPPREC[0.5, -1.0]          # per element qc+qs = ±1.5 → group-neutral
    prms_cpu = (qc = qc, qs = qs)
    prms_gpu = arrtyp_prms(CuArray, prms_cpu)

    gauge = Calculators.PolarizationGauge(atoms_ref, reps, qc, qs)
    # Structured, non-trivial target and a mask with a released band.
    jit(n) = 2 * (n * 0.6180339887498949 - floor(n * 0.6180339887498949)) - 1
    tgt = FMD.FPPREC[0.003 * jit(2i + 5d) for i in 1:ng, d in 1:3]
    w = FMD.FPPREC[(i % 7 == 0 ? 0.0 : 0.3 + 0.7 * (i % 3)) for i in 1:ng]

    for (label, sigma, r_th) in (("sigma>0", 0.8, 0.0),
                                 ("flat-bottom, sigma=0", 0.0, 1e-3))
        @testset "$label" begin
            build(; kw...) = Calculators.PolarizationMapUmbrella(
                atoms_ref, 5e3, tgt, reps, qc, qs;
                gauge = gauge, weight = w, sigma = sigma, r_th = r_th, kw...)

            pu_cpu = build()
            a_cpu = deepcopy_atoms(rattled)
            Calculators.umbrella_p!(a_cpu, pu_cpu, prms_cpu)

            pu_gpu = build(backend = CUDA.CUDABackend())
            a_gpu = Cores.change_arrtyp(deepcopy_atoms(rattled), CuArray)
            Calculators.umbrella_p!(a_gpu, pu_gpu, prms_gpu)

            # The host-built grids must reach the device bit-for-bit.
            @test Array(pu_gpu.Ghat) == pu_cpu.Ghat
            @test Array(pu_gpu.M_frac_ref) == pu_cpu.M_frac_ref
            @test Array(pu_gpu.P_target_x) == pu_cpu.P_target_x
            @test Array(pu_gpu.P_target_y) == pu_cpu.P_target_y
            @test Array(pu_gpu.P_target_z) == pu_cpu.P_target_z
            @test Array(pu_gpu.wgrid) == pu_cpu.wgrid
            @test pu_gpu.wsum == pu_cpu.wsum

            # Non-vacuous.
            E_cpu = a_cpu.info[:potential_energy][1]
            @test abs(E_cpu) > 1e-6
            @test maximum(abs, a_cpu.arrays[:forces]) > 1e-5
            @test maximum(abs, a_cpu.arrays[:shellforces]) > 1e-5
            @test maximum(abs, a_cpu.info[:virial]) > 1e-6
            @test pu_cpu.results["cv"] > 1e-5

            @test pu_gpu.results["cv"] ≈ pu_cpu.results["cv"] rtol = 1e-6 atol = 1e-12
            @test pu_gpu.results["bias"] ≈ pu_cpu.results["bias"] rtol = 1e-6 atol = 1e-12
            @test pu_gpu.results["sum_r2"] ≈ pu_cpu.results["sum_r2"] rtol = 1e-6 atol = 1e-12
            @test Array(a_gpu.info[:potential_energy])[1] ≈ E_cpu rtol = 1e-6 atol = 1e-12
            @test isapprox(Array(a_gpu.info[:virial]), a_cpu.info[:virial];
                           rtol = 1e-6, atol = 1e-10)
            @test isapprox(Array(a_gpu.arrays[:forces]), a_cpu.arrays[:forces];
                           rtol = 1e-6, atol = 1e-10)
            @test isapprox(Array(a_gpu.arrays[:shellforces]), a_cpu.arrays[:shellforces];
                           rtol = 1e-6, atol = 1e-10)
        end
    end
end
