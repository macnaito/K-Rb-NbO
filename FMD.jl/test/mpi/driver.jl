# test/mpi/driver.jl
#
# Subprocess driver: launch `mpiexec -n N julia --project=. <script> <args...>`
# from the parent (single-process) test runner.  Returns true on success
# (exit code 0 + sentinel printed by child).
#
# Each child script is expected to print "===FMD_MPI_DONE===" on its last
# rank-0 line. For the bundled runner (_run_bundle.jl, Issue #113) the
# sentinel means "the runner completed every check slot", NOT "the checks
# passed" — per-check verdicts live in the artifacts the checks write
# (see _run_bundle.jl).

using MPI

const FMD_MPI_SENTINEL = "===FMD_MPI_DONE==="

# Artifact-naming contract shared by the bundle runner (child side) and
# the tier drivers (parent side): check name "ghosts/bigrc" → file stem
# "ghosts_bigrc" (artifact <outdir>/<stem>.jld2, workdir <outdir>/<stem>).
bundle_slug(name::AbstractString) = replace(name, "/" => "_")

"""
    run_mpi_script(script_path, n; args=String[], timeout_sec=180)

Spawn `mpiexec -n n julia --project=<project> <script_path> <args...>` and
collect stdout/stderr.  Returns (ok::Bool, output::String).

Each rank is launched with a `--heap-size-hint` of roughly its fair share
of host RAM. Julia sizes its heap against *total* memory, so `n` ranks on
one machine each believe they own it: measured on a 15 GB / 6-rank CUDA
bundle the ranks peaked at 14.2 GB resident and the OOM killer took the
parent down, turning a 100 s bundle into a 20 min timeout. The hint costs
nothing in wall clock (102 s either way at P = 6) and bounds the peak to
12.2 GB. It is a hint, not a limit — a rank that genuinely needs more
still grows.
"""
function run_mpi_script(script_path::AbstractString, n::Integer;
                        args::Vector{String} = String[],
                        timeout_sec::Real = 180)
    project = abspath(joinpath(@__DIR__, "..", ".."))
    mpiexec_path = MPI.mpiexec()
    heap_hint = max(512 * 1024^2, round(Int, 0.3 * Sys.total_memory() / n))
    cmd = `$(mpiexec_path) -n $n $(Base.julia_cmd().exec[1]) -t1
           --heap-size-hint=$(heap_hint) --project=$project $script_path $args`

    io_out = IOBuffer()
    p = run(pipeline(cmd, stdout = io_out, stderr = io_out), wait = false)
    t0 = time()
    while process_running(p) && (time() - t0) < timeout_sec
        sleep(0.1)
    end
    if process_running(p)
        kill(p)
        return false, String(take!(io_out)) * "\n[TIMEOUT]"
    end
    output = String(take!(io_out))
    ok = success(p) && occursin(FMD_MPI_SENTINEL, output)
    return ok, output
end
