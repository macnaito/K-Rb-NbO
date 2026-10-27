# test/mpi/test_bundle_gpu_mpi.jl
#
# GPU-tier parent driver for the bundled MPI tests (Issue #113). Same two
# stages as test_bundle_mpi.jl, with backend = cuda: the children pass
# device buffers straight to MPI (CUDA-aware build required) and the
# serial npt references here are GPU *serial* runs, isolating DD
# deviations from backend deviations.
using Test
using JLD2
using CUDA
using FMD.Apps: npt!

include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "_npt_dd_common.jl"))
include(joinpath(@__DIR__, "_verify_checks.jl"))

# Light checks first, npt last (incremental artifacts — see
# test_bundle_mpi.jl). The GPU-specialized JIT (~30 s of kernel codegen
# that pkgimages cannot bake) is now paid once per bundle instead of once
# per check.
const CUDA_BUNDLES = [
    (1, ["allreduce_gpu", "ghosts_gpu/normal", "reduce/normal", "umbrella",
         "coreshell", "npt_dd/normal"]),
    (2, ["allreduce_gpu", "collectives", "ghosts_gpu/normal", "ghosts_gpu/bigrc",
         "reduce/normal", "pmeslab/normal", "umbrella", "umbrella/map",
         "surface_term", "coreshell", "npt_dd/normal", "npt_dd/umbrella"]),
    (6, ["ghosts_gpu/tiny", "reduce/tiny", "pmeslab/tiny", "hvp_dd/tiny",
         "npt_dd/tiny"]),
]

# ---- stage 1: run the bundles ---------------------------------------------
cuda_bundles = Dict{Int,Any}()
for (P, names) in CUDA_BUNDLES
    @testset "bundle execution (cuda, P = $P)" begin
        outdir = mktempdir()
        # Below the HPC job elapse (45 min) so a hang still yields the
        # [TIMEOUT] diagnostics — see test_bundle_mpi.jl.
        ok, log = run_mpi_script(joinpath(@__DIR__, "_run_bundle.jl"), P;
                                 args = vcat([outdir, "cuda"], names),
                                 timeout_sec = 1500)
        @test ok
        ok || @info "bundle subprocess output (cuda, P=$P):\n$log"
        cuda_bundles[P] = (; ok, log, outdir)
    end
end

# ---- stage 2: per-feature verification ------------------------------------

@testset "device-buffer Allreduce through Domain helpers (P ∈ {1, 2})" begin
    for P in (1, 2)
        verify_allreduce_gpu(load_check(cuda_bundles[P], "allreduce_gpu"),
                             cuda_bundles[P].log, P)
    end
end

@testset "z-slab distributed PME on device (P = 2 normal, P = 6 tiny)" begin
    # Device grids through reduce_scatter / alltoallv / allgatherv with
    # CUFFT slab plans; tiny additionally hits the zero-plane / zero-ky /
    # empty-atom degeneracy on the CUDA-aware transport.
    verify_pmeslab(load_check(cuda_bundles[2], "pmeslab/normal"), 2, "normal")
    verify_pmeslab(load_check(cuda_bundles[6], "pmeslab/tiny"), 6, "tiny")
end

@testset "device-buffer segmented collectives (P = 2)" begin
    # reduce_scatter_sum!/allgatherv!/alltoallv! on CuArrays: the raw
    # MPI_Reduce_scatter binding and the VBuffer collectives through the
    # CUDA-aware transport (slab-PME prerequisite, Issue #191). One rank
    # count — the count-layout coverage lives in the CPU tier.
    verify_collectives(load_check(cuda_bundles[2], "collectives"), 2)
end

@testset "device DD ghost exchange ≡ host DD exchange" begin
    for (mode, P) in (("normal", 1), ("normal", 2), ("bigrc", 2), ("tiny", 6))
        @testset "$mode (P = $P)" begin
            verify_ghosts_gpu(load_check(cuda_bundles[P], "ghosts_gpu/$mode"),
                              P, mode)
        end
    end
end

@testset "GPU DD parity: ghost-free calculators (P ∈ {1, 2})" begin
    for P in (1, 2)
        verify_reduce(load_check(cuda_bundles[P], "reduce/normal"), P, "normal", :cuda)
    end
    @testset "empty ranks (GPU, NaCl-8, P = 6)" begin
        verify_reduce(load_check(cuda_bundles[6], "reduce/tiny"), 6, "tiny", :cuda)
    end
end

@testset "polarization umbrella GPU DD parity (P ∈ {1, 2})" begin
    for P in (1, 2)
        verify_umbrella(load_check(cuda_bundles[P], "umbrella"), P)
    end
    # Target-map bias on the same tier: this is where the device-resident
    # target/weight grids meet CUDA-aware MPI, which neither the serial GPU
    # parity test nor the CPU DD test covers on its own.  Only P = 2 — the
    # DD logic is shared, so a second rank count would buy repetition rather
    # than coverage (and the CUDA bundles are the memory-tight ones).
    verify_umbrella(load_check(cuda_bundles[2], "umbrella/map"), 2)
end

@testset "surface term GPU DD parity (P = 2)" begin
    # The dipole Allreduce is a host-buffer collective even on the GPU
    # backend (calcdipole! drains the device reduction to the host first),
    # so one rank count suffices — the device-side content is pinned by the
    # serial GPU parity test and the DD logic by the CPU tier.
    verify_surface_term(load_check(cuda_bundles[2], "surface_term"), 2)
end

@testset "shell-CG GPU DD parity (P ∈ {1, 2})" begin
    for P in (1, 2)
        verify_coreshell(load_check(cuda_bundles[P], "coreshell"), P, :cuda)
    end
end

@testset "npt! GPU DD tier-2 (driver integration)" begin
    # GPU serial reference in THIS process (no MPI.Init → SerialDomain).
    ref = npt_dd_serial_ref(:cuda)

    @testset "trajectory parity vs GPU serial (P = $P)" for P in (1, 2)
        verify_npt_normal(load_check(cuda_bundles[P], "npt_dd/normal"),
                          joinpath(cuda_bundles[P].outdir, bundle_slug("npt_dd/normal")),
                          ref, P, :cuda)
        # The CUDA-aware transport probe announced itself and passed
        # somewhere in this bundle (process-level property).
        @test occursin("CUDA-aware MPI probe passed", cuda_bundles[P].log)
    end

    # Umbrella through the whole GPU DD driver (P = 2): Pg device
    # Allreduce inside every CG force evaluation; Bias.log must match the
    # GPU serial run (root-only, per-step CV/bias).
    @testset "umbrella through the GPU DD driver (P = 2)" begin
        ref_dir_u = npt_dd_serial_ref_umbrella(:cuda)
        verify_npt_umbrella(load_check(cuda_bundles[2], "npt_dd/umbrella"),
                            joinpath(cuda_bundles[2].outdir, bundle_slug("npt_dd/umbrella")),
                            ref_dir_u, :cuda)
    end

    # NaCl-8 at P = 6: empty ranks ride the whole device driver
    # (distribute, probe, device collectives, host round trip, gather).
    @testset "empty ranks drive the full GPU loop (NaCl-8, P = 6)" begin
        verify_npt_tiny(load_check(cuda_bundles[6], "npt_dd/tiny"),
                        joinpath(cuda_bundles[6].outdir, bundle_slug("npt_dd/tiny")),
                        :cuda)
    end
end
