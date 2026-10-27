# test/mpi/_check_allreduce_gpu.jl
#
# Device-buffer MPI transport through the Domain helpers —
# probe_device_mpi!, then allreduce_sum! of a CuVector, of vec(CuMatrix)
# (the PME grid-reduction call shape: a contiguous reshape sharing
# storage), and of a host array (which must keep taking the plain CPU path
# under the same helper).
#
# CUDA-aware MPI required: on a non-aware build the probe crashes the
# process right after its announcement — the parent asserts the probe
# lines in the bundle log.

function check_allreduce_gpu(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    dom = Cores.MPIDomain(1)   # nglobal is irrelevant here; constructor is collective
    Cores.probe_device_mpi!(dom, CUDA.CUDABackend())

    expected = FMD.FPPREC(sum(1:P))

    # Device-written CuVector: fill! launches an asynchronous kernel, so this
    # is the production shape — pending device work immediately reduced. The
    # stream drain inside allreduce_sum! is what makes it correct; a host-
    # uploaded buffer (like M below) would pass even without the drain.
    v = KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, 8)
    fill!(v, FMD.FPPREC(rank + 1))
    Cores.allreduce_sum!(dom, v)
    ok_v = all(Array(v) .== expected)

    # vec(CuMatrix): reduces through the shared storage, M sees the result.
    M = CuArray(fill(FMD.FPPREC(rank + 1), 4, 3))
    Cores.allreduce_sum!(dom, vec(M))
    ok_m = all(Array(M) .== expected)

    # Host array through the same helper (the Phase 1 path, must stay intact).
    h = fill(FMD.FPPREC(rank + 1), 5)
    Cores.allreduce_sum!(dom, h)
    ok_h = all(h .== expected)

    ok = MPI.Allreduce(ok_v && ok_m && ok_h, &, comm)

    if rank == 0
        JLD2.jldsave(out; mpisize = P, ok = ok, has_cuda = MPI.has_cuda())
    end
    return nothing
end
