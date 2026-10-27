# test/mpi/_check_npt_dd.jl
#
# The full DD NPT driver — distribute, per-step collectives, rebuild +
# migrate (a host round trip on the cuda backend), root-only I/O,
# gather_frame snapshots. Saves the gathered final state (per-id) for the
# parent to compare against its serial reference; trajectory outputs
# (log.txt / Bias.log / step_N.xyz) land in `dir`.
# mode ∈ normal | tiny | umbrella | csvr | scr | xlbomd* | nlh*.

# Forwarding logger that counts @info messages containing `needle` and
# passes everything through to `parent` (so npt!'s progress lines stay in
# the process log for hang diagnostics). Replaces the pre-#113 parent-side
# stdout grep `count("Building neighbor list", log) >= 2`, which would
# overcount in a bundled process. The message is root-only (NPT.jl), so
# counting on rank 0 is exact.
mutable struct CountingLogger <: Logging.AbstractLogger
    parent::Logging.AbstractLogger
    needle::String
    count::Int
end
Logging.min_enabled_level(l::CountingLogger) = Logging.min_enabled_level(l.parent)
Logging.shouldlog(l::CountingLogger, args...) = Logging.shouldlog(l.parent, args...)
Logging.catch_exceptions(l::CountingLogger) = Logging.catch_exceptions(l.parent)
function Logging.handle_message(l::CountingLogger, level, message, args...; kwargs...)
    occursin(l.needle, string(message)) && (l.count += 1)
    return Logging.handle_message(l.parent, level, message, args...; kwargs...)
end

function check_npt_dd(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    full = _npt_dd_fixture(mode in ("tiny", "nlh_tiny"); umbrella = mode == "umbrella")
    params = _npt_dd_params(full, dir; tiny = mode in ("tiny", "nlh_tiny"), mpi = true,
                            umbrella = mode == "umbrella",
                            csvr = mode == "csvr", scr = mode == "scr",
                            xl = mode in ("xlbomd", "xlbomd_cheb"),
                            cheb = mode == "xlbomd_cheb",
                            xlbaro = mode == "xlbomd_baro",
                            nlh = mode in ("nlh", "nlh_tiny"),
                            surface = mode == "xlbomd_surface",
                            backend = backend)

    lg = CountingLogger(Logging.current_logger(), "Building neighbor list", 0)
    res = Logging.with_logger(() -> npt!(deepcopy_atoms(full), params), lg)

    dom = Cores.domain(res)
    nmin = MPI.Allreduce(length(res), min, comm)
    frame = Cores.gather_frame(res, dom)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, nmin = nmin, backend = String(backend),
                     nl_rebuilds = lg.count,
                     ids = collect(Int, frame.arrays[:id]),
                     pos = Array(frame.arrays[:pos]),
                     velo = Array(frame.arrays[:velo]),
                     shelldisp = Array(frame.arrays[:shelldisp]),
                     lattice = Matrix(Array(res.info[:lattice])))
    end
    return nothing
end
