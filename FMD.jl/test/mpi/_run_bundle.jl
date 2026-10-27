# test/mpi/_run_bundle.jl
#
# Bundled child runner (Issue #113): executes a list of registered checks
# in ONE mpiexec process, amortizing the per-process fixed cost (startup +
# package load + JIT; on cuda also the CUDA context) that used to be paid
# once per check.
#
#   mpiexec -n P julia --project _run_bundle.jl <outdir> <backend> <check>...
#
# Contract with the parent drivers (test_bundle_mpi.jl / test_bundle_gpu_mpi.jl):
#   * exit 0 + sentinel = the RUNNER completed every check slot. It does
#     NOT mean the checks passed — each check writes its own artifact
#     <outdir>/<slug>.jld2 (rank 0), and a check that throws leaves
#     <outdir>/<slug>.error.r<rank>.txt instead. The parent asserts on the
#     artifacts, so a red feature stays localized to its own testset.
#   * "[bundle] begin/ok/FAIL <name>" progress lines (rank 0) localize
#     hangs: on a timeout, the last "begin" without a matching "ok/FAIL"
#     names the culprit.
#   * A check that throws on EVERY rank is caught, recorded, and the
#     bundle moves on. If only some ranks throw mid-collective, the ranks
#     desynchronize and the bundle hangs until the parent timeout —
#     accepted trade-off; the progress lines + partial artifacts keep the
#     failure localized.

using MPI
using JLD2
MPI.Init()
using FMD
using FMD.Cores
using FMD.Calculators
using FMD.Apps: npt!
using Random
using LinearAlgebra
using Logging
using CUDA
import Adapt
import KernelAbstractions
include(joinpath(@__DIR__, "driver.jl"))    # bundle_slug + sentinel constant
include(joinpath(@__DIR__, "_checks.jl"))   # helpers, fixtures, check registry

length(ARGS) >= 3 || error("usage: _run_bundle.jl outdir backend check...")
const OUTDIR = ARGS[1]
const BACKEND = Symbol(ARGS[2])
const NAMES = ARGS[3:end]
BACKEND in (:cpu, :cuda) || error("backend must be cpu or cuda (got $BACKEND)")

# Deliberate-abort checks kill their process by design (asymmetrically —
# e.g. farjump moves an atom on rank 0 only), which inside a bundle would
# desynchronize the ranks and hang until the parent timeout. Refuse them
# here instead of relying on the parent to know better.
const ABORT_ONLY = Set(["migrate/keyset", "migrate/farjump"])

for name in NAMES
    # ARGS are identical on every rank, so these errors are symmetric.
    haskey(CHECKS, name) || error("unknown check: $name")
    name in ABORT_ONLY && error("$name is abort-only; run it via _run_single.jl")
end

function run_one(name, comm)
    rank = MPI.Comm_rank(comm)
    fn, mode = CHECKS[name]
    s = bundle_slug(name)
    chkout = joinpath(OUTDIR, s * ".jld2")
    chkdir = joinpath(OUTDIR, s)
    rank == 0 && mkpath(chkdir)
    MPI.Barrier(comm)          # chkdir visible + previous check fully done
    if rank == 0
        println("[bundle] begin $name")
        flush(stdout)
    end
    Random.seed!(0)            # hygiene; checks also re-seed themselves
    failed = 0
    try
        fn(chkout, chkdir, mode, BACKEND, comm)
    catch err
        failed = 1
        open(joinpath(OUTDIR, "$s.error.r$rank.txt"), "w") do io
            showerror(io, err, catch_backtrace())
        end
    end
    failed = MPI.Allreduce(failed, max, comm)
    if rank == 0
        println(failed == 0 ? "[bundle] ok $name" : "[bundle] FAIL $name")
        flush(stdout)
    end
    # Synchronized GC on every backend: each check builds its own MPIDomain,
    # whose Cartesian comm carries an MPI_Comm_free finalizer — a collective
    # operation (see Domain.jl "construct once per run"). Running the GC at
    # this barrier-adjacent point pins the finalizers to the same program
    # point on every rank instead of leaving them to rank-local allocation
    # pressure.
    GC.gc()
    BACKEND === :cuda && CUDA.reclaim()   # device-memory hygiene between checks
    return nothing
end

comm = MPI.COMM_WORLD

if BACKEND === :cuda
    # Loud failure, not silent skip (Issue #18 convention).
    CUDA.functional() || error("child: CUDA.functional() == false under mpiexec")
    # Node-local rank → GPU binding before any allocation, same rule as
    # npt!'s _assign_local_gpu! (a global-rank binding can diverge from it
    # on multi-node runs with non-block rank placement). Identity on
    # 1-GPU nodes.
    let lcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
        CUDA.device!(MPI.Comm_rank(lcomm) % length(CUDA.devices()))
        MPI.free(lcomm)   # lockstep here on every rank — immediate free is safe
    end
end

for name in NAMES
    run_one(name, comm)
end

MPI.Comm_rank(comm) == 0 && println(FMD_MPI_SENTINEL)
MPI.Finalize()
