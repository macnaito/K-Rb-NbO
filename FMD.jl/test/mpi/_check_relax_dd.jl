# test/mpi/_check_relax_dd.jl
#
# The FIRE relaxation driver under domain decomposition (Issue #209).
#
# relax! introduces four collectives that only execute at P > 1 and that
# every FIRE branch depends on:
#
#   _relax_reductions   the fused (F·v, |v|², |F|²) Allreduce — its sign
#                       decides downhill vs uphill, so a rank-local value
#                       would put ranks on different branches
#   _max_force          the convergence test itself
#   _cap_displacement!  the step cap — a rank-local factor would deform the
#                       structure instead of scaling the step
#   _verify_convergence!  a collective PMEInfo rebuild, entered only from a
#                       rank-uniform condition
#
# The check runs the same frame serially and distributed and compares the
# gathered final state per :id. Reduction reassociation across ranks is
# ~1e-16 per scalar and FIRE is a contraction toward the minimum, so the
# two runs agree far tighter than the tolerances asserted by the parent.

function check_relax_dd(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    tiny = mode == "tiny"
    cell_dof = mode == "cell" ? :full : (mode == "cell_iso" ? :iso : :none)
    full = _relax_dd_fixture(tiny)
    params = _relax_dd_params(full, dir; tiny = tiny, mpi = true,
                              cell_dof = cell_dof, backend = backend)

    res, converged = FMD.Apps.relax!(deepcopy_atoms(full), params)

    dom = Cores.domain(res)
    nmin = MPI.Allreduce(length(res), min, comm)
    frame = Cores.gather_frame(res, dom)
    # Whole-system on every rank; recorded so the parent can check the
    # ranks agreed on the verdict rather than merely finishing.
    maxf = FMD.Apps._max_force(res, dom)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, nmin = nmin, backend = String(backend),
                     converged = converged, maxf = maxf,
                     ids = collect(Int, frame.arrays[:id]),
                     pos = Array(frame.arrays[:pos]),
                     shelldisp = Array(frame.arrays[:shelldisp]),
                     fire_velo = Array(frame.arrays[:fire_velo]),
                     lattice = Matrix(Array(res.info[:lattice])))
    end
    return nothing
end
