# test/mpi/_check_allreduce.jl
#
# MPI Allreduce identity: each rank constructs the same vector
# v_local = ones(8) * (rank+1); after Allreduce(SUM), every rank should
# hold sum_{r=0..P-1}(r+1) * ones(8).

function check_allreduce(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    v = fill(Float64(rank + 1), 8)
    MPI.Allreduce!(v, MPI.SUM, comm)

    expected = sum(1:P) * 1.0
    ok = all(isapprox.(v, expected; atol = 1e-12))

    # Each rank is consistent — only rank 0 writes the artifact.
    rank == 0 && JLD2.jldsave(out; v = v, expected = expected, ok = ok, mpisize = P)
    return nothing
end
