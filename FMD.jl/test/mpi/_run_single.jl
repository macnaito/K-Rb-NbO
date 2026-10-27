# test/mpi/_run_single.jl
#
# Single-check child runner: executes ONE registered check in its own
# mpiexec process, with NO try/catch — an error propagates to a nonzero
# exit without the sentinel. Two uses:
#   * the deliberate-abort checks (migrate/keyset, migrate/farjump), whose
#     parent testsets assert `!ok` plus the error text in the merged
#     process output — they must kill the process and so cannot ride a
#     bundle;
#   * ad-hoc reproduction / timing of one check outside a bundle, e.g.
#       mpiexec -n 2 julia --project test/mpi/_run_single.jl \
#           npt_dd/normal out.jld2 workdir cpu
#     (from Julia, launch via run(`$(MPI.mpiexec()) ...`) — see driver.jl.)
#
#   ARGS: check_name out_path workdir [backend]

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
include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "_checks.jl"))

length(ARGS) >= 3 || error("usage: _run_single.jl check_name out_path workdir [backend]")
const NAME = ARGS[1]
const OUT = ARGS[2]
const DIR = ARGS[3]
const BACKEND = length(ARGS) >= 4 ? Symbol(ARGS[4]) : :cpu
BACKEND in (:cpu, :cuda) || error("backend must be cpu or cuda (got $BACKEND)")
haskey(CHECKS, NAME) || error("unknown check: $NAME")

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

if BACKEND === :cuda
    # Loud failure, not silent skip (Issue #18 convention). Node-local
    # binding — same rule as npt!'s _assign_local_gpu! (see _run_bundle.jl).
    CUDA.functional() || error("child: CUDA.functional() == false under mpiexec")
    let lcomm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, MPI.Comm_rank(comm))
        CUDA.device!(MPI.Comm_rank(lcomm) % length(CUDA.devices()))
        MPI.free(lcomm)
    end
end

rank == 0 && mkpath(DIR)
MPI.Barrier(comm)
Random.seed!(0)
fn, mode = CHECKS[NAME]
fn(OUT, DIR, mode, BACKEND, comm)

rank == 0 && println(FMD_MPI_SENTINEL)
MPI.Finalize()
