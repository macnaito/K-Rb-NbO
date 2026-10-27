# test/gpu/test_hvp_gpu.jl
#
# Analytic HVP on the GPU backend (Issue #147): CPU ↔ GPU parity of
# hvp! over the PME path (dipole spread / convolve / gathers are KA
# kernels; the short-range and spring/intra kernels are backend-generic
# by construction). Tolerance band rtol 1e-6 / atol 1e-10 like the other
# GPU parity tests: atomic-accumulation reassociation (~1e-14) + CUFFT vs
# FFTW, two orders of headroom. The fixture is rattled (asymmetric) so
# the products are genuinely non-zero. The Ewald cache is CPU-only by
# design and must refuse a GPU frame loudly.

using Test
using CUDA
using Random
using LinearAlgebra

@testset "analytic HVP (GPU)" begin
    Random.seed!(51)
    atoms_cpu = make_nacl_8()
    atoms_cpu.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:pos]))
    atoms_cpu.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:shelldisp]))
    prms = nacl_params(atoms_cpu; alpha = 0.30, with_shell = true)
    prms = (; prms..., pme_meshsize = FMD.FPPREC(0.5))
    N = length(atoms_cpu)
    v_cpu = randn(FMD.FPPREC, N, 3)
    v_cpu ./= sqrt(sum(abs2, v_cpu))

    # CPU reference (PME path).
    nl_cpu = neighborlist(atoms_cpu, prms[:rc], prms[:skin])
    pme_cpu = Calculators.PMEInfo(atoms_cpu.info[:lattice], prms)
    ws_cpu = Calculators.hvp_workspace(atoms_cpu, nl_cpu, pme_cpu, prms)
    Jref = zeros(FMD.FPPREC, N, 3)
    Kref = zeros(FMD.FPPREC, N, 3)
    Wref = zeros(FMD.FPPREC, N, 6)
    Wrecref = Calculators.hvp!(Jref, Kref, Wref, v_cpu, atoms_cpu, ws_cpu, prms)

    @testset "PME-path hvp! CPU ↔ GPU parity" begin
        a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        p = arrtyp_prms(CuArray, prms)
        # Naive list (no CellList re-sort): rows keep the CPU order, so
        # the products compare without an :id permutation.
        nl = neighborlist(a, p[:rc], p[:skin])
        pme = Calculators.PMEInfo(a.info[:lattice], p)
        ws = Calculators.hvp_workspace(a, nl, pme, p)
        Jv = CUDA.zeros(FMD.FPPREC, N, 3)
        Kv = CUDA.zeros(FMD.FPPREC, N, 3)
        Wv = CUDA.zeros(FMD.FPPREC, N, 6)
        v = CuArray(v_cpu)
        Wrec = Calculators.hvp!(Jv, Kv, Wv, v, a, ws, p)

        scale = max(one(FMD.FPPREC), maximum(abs, Jref))
        @test maximum(abs, Jref) > 1e-3            # non-vacuous fixture
        @test maximum(abs, Array(Jv) .- Jref) < 1e-6 * scale + 1e-10
        @test maximum(abs, Array(Kv) .- Kref) < 1e-6 * scale + 1e-10

        # Virial derivative (Issue #151): per-atom part and the grid
        # remainder, both backend-generic by construction.
        wtot_ref = vec(sum(Wref, dims = 1)) .+ Wrecref
        wtot = vec(sum(Array(Wv), dims = 1)) .+ Wrec
        wscale = max(one(FMD.FPPREC), maximum(abs, wtot_ref))
        @test maximum(abs, wtot_ref) > 1e-4        # non-vacuous
        @test maximum(abs, Array(Wv) .- Wref) < 1e-6 * wscale + 1e-10
        @test maximum(abs, wtot .- wtot_ref) < 1e-6 * wscale + 1e-10

        # Kv-skipping fast path stays consistent on the device.
        Jv2 = CUDA.zeros(FMD.FPPREC, N, 3)
        Wv2 = CUDA.zeros(FMD.FPPREC, N, 6)
        Calculators.hvp!(Jv2, nothing, Wv2, v, a, ws, p)
        @test Array(Jv2) == Array(Jv)
        @test Array(Wv2) == Array(Wv)
    end

    # Surface term (Issue #181): the block is a device reduction plus a
    # uniform per-charge scatter, and its virial derivative rides the
    # host-side return value. Parity therefore has to cover BOTH outputs
    # — a device reduction that silently sums the wrong rows would still
    # produce a plausible Jv.
    @testset "surface-term hvp! CPU ↔ GPU parity" begin
        proto = make_nacl_8()
        # |q_s| = 1 (qc + qs = ±0.8, still neutral): the block scales as
        # q_s², and the fixture's ±0.2 would put it near the tolerance.
        sp = (; prms..., qc = FMD.FPPREC[1.8, -1.8], qs = FMD.FPPREC[-1.0, 1.0])
        R = [0.8 -0.6 0.0; 0.6 0.8 0.0; 0.0 0.0 1.0]
        mkst() = Calculators.SurfaceTerm(atoms_cpu, sp[:qc], sp[:qs];
            eps_out = R * Diagonal([1.5, 3.0, 8.0]) * R',
            shape = [1.0, 1.4, 0.7], gauge = proto)
        # The gauge is the unrattled prototype so δM ≠ 0 — otherwise the
        # virial derivative is identically zero and its parity vacuous.
        sp_cpu = (; sp..., surface_term = mkst())

        nl_c = neighborlist(atoms_cpu, sp_cpu[:rc], sp_cpu[:skin])
        pme_c = Calculators.PMEInfo(atoms_cpu.info[:lattice], sp_cpu)
        ws_c = Calculators.hvp_workspace(atoms_cpu, nl_c, pme_c, sp_cpu)
        Jc = zeros(FMD.FPPREC, N, 3); Kc = zeros(FMD.FPPREC, N, 3)
        Wc = zeros(FMD.FPPREC, N, 6)
        Wrec_c = Calculators.hvp!(Jc, Kc, Wc, v_cpu, atoms_cpu, ws_c, sp_cpu)

        a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        # A separate SurfaceTerm instance: its `results` Dict must not be
        # shared between the two evaluations (and hvp! pins the instance
        # its workspace was built from).
        p = (; arrtyp_prms(CuArray, sp)..., surface_term = mkst())
        nl = neighborlist(a, p[:rc], p[:skin])
        pme = Calculators.PMEInfo(a.info[:lattice], p)
        ws = Calculators.hvp_workspace(a, nl, pme, p)
        Jv = CUDA.zeros(FMD.FPPREC, N, 3); Kv = CUDA.zeros(FMD.FPPREC, N, 3)
        Wv = CUDA.zeros(FMD.FPPREC, N, 6)
        Wrec = Calculators.hvp!(Jv, Kv, Wv, CuArray(v_cpu), a, ws, p)

        # Non-vacuous: the same fixture WITHOUT the term gives a
        # measurably different product, so the parity below is about the
        # surface block and not about everything else agreeing.
        ws_c0 = Calculators.hvp_workspace(atoms_cpu, nl_c, pme_c, sp)
        J0 = zeros(FMD.FPPREC, N, 3); K0 = zeros(FMD.FPPREC, N, 3)
        W0 = zeros(FMD.FPPREC, N, 6)
        Wrec0 = Calculators.hvp!(J0, K0, W0, v_cpu, atoms_cpu, ws_c0, sp)
        jscale = max(one(FMD.FPPREC), maximum(abs, Jc))
        @test maximum(abs, Jc .- J0) > 1e-6 * jscale
        @test maximum(abs, Wrec_c .- Wrec0) > 1e-8

        @test maximum(abs, Array(Jv) .- Jc) < 1e-6 * jscale + 1e-10
        @test maximum(abs, Array(Kv) .- Kc) < 1e-6 * jscale + 1e-10
        wscale = max(one(FMD.FPPREC), maximum(abs, Wrec_c))
        @test maximum(abs, Array(Wv) .- Wc) < 1e-6 * wscale + 1e-10
        @test maximum(abs, Wrec .- Wrec_c) < 1e-6 * wscale + 1e-10
    end

    @testset "Ewald cache refuses the GPU backend" begin
        a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        p = arrtyp_prms(CuArray, prms)
        nl = neighborlist(a, p[:rc], p[:skin])
        ew_cpu = Calculators.EwaldInfo(atoms_cpu.info[:lattice], prms)
        @test_throws ErrorException Calculators.hvp_workspace(a, nl, ew_cpu, p)
    end
end
