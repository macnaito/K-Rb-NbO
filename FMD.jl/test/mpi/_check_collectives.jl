# test/mpi/_check_collectives.jl
#
# Segmented collectives (reduce_scatter_sum! / allgatherv! / alltoallv!,
# Issue #191) — known-value identities with uneven counts, a zero-count
# rank, and padded buffers (buffers longer than the true count, the
# max(count,1) allocation pattern of the slab PME). Runs on both backends:
# :cuda passes device buffers straight through the Domain helpers, which
# is the CUDA-aware proof for the raw MPI_Reduce_scatter binding and the
# VBuffer collectives.
#
# Count layout (0-based rank r): counts[r+1] = r + 2, except the last
# rank gets 0 — so P=2 gives [2, 0] (uneven + trailing empty rank) and
# P=6 gives [2, 3, 4, 5, 6, 0]. Every rank participates in every
# collective regardless of its count (the empty-rank invariant).

function check_collectives(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    T = FMD.FPPREC

    dom = Cores.MPIDomain(1)   # nglobal irrelevant; constructor is collective
    _dev(x) = backend == :cuda ? CuArray(x) : x

    counts = Cint[r == P - 1 ? 0 : r + 2 for r in 0:(P-1)]
    offs = cumsum([0; Int.(counts[1:end-1])])
    ntot = sum(Int, counts)
    myn = Int(counts[rank+1])
    myoff = offs[rank+1]

    # --- reduce_scatter_sum!: sendbuf[i] = (rank+1)·i, so the reduced
    # segment on rank r must be S·(global index), S = Σ(rank+1) = P(P+1)/2.
    S = T(P * (P + 1) ÷ 2)
    send_rs = _dev(T[(rank + 1) * i for i in 1:ntot])
    recv_rs = _dev(zeros(T, max(myn, 1)))          # padded when myn == 0
    Cores.reduce_scatter_sum!(dom, send_rs, recv_rs, counts)
    h_rs = Array(recv_rs)
    # Padding past the own count must be untouched — the zero-count rank's
    # whole (max(count,1)-sized) buffer rides on this.
    ok_rs = all(h_rs[1:myn] .== S .* T.(myoff .+ (1:myn))) &&
            all(h_rs[myn+1:end] .== 0)

    # --- allgatherv!: rank r contributes counts[r] copies of (r+1); the
    # send buffer carries 2 extra sentinel elements past the true count,
    # which must NOT be transmitted (pins the Buffer-count logic).
    send_ag = _dev(T[fill(T(rank + 1), myn); fill(T(-999), 2)])
    recv_ag = _dev(zeros(T, ntot))
    Cores.allgatherv!(dom, send_ag, recv_ag, counts)
    h_ag = Array(recv_ag)
    ok_ag = all(all(h_ag[offs[q+1]+1:offs[q+1]+Int(counts[q+1])] .== T(q + 1))
                for q in 0:(P-1))

    # --- alltoallv!: route-encoded values. s_{r→q} = (r+q) % 3 elements
    # (symmetric ⇒ rcounts = scounts elementwise on each rank), segment to
    # q filled with r·100 + q; the segment received from q must read
    # q·100 + r.
    sc = Cint[(rank + q) % 3 for q in 0:(P-1)]
    soffs = cumsum([0; Int.(sc[1:end-1])])
    send_a2a_h = zeros(T, max(sum(Int, sc), 1))
    for q in 0:(P-1), i in 1:Int(sc[q+1])
        send_a2a_h[soffs[q+1]+i] = T(rank * 100 + q)
    end
    send_a2a = _dev(send_a2a_h)
    recv_a2a = _dev(zeros(T, max(sum(Int, sc), 1)))
    Cores.alltoallv!(dom, send_a2a, sc, recv_a2a, sc)
    h_a2a = Array(recv_a2a)
    ok_a2a = all(all(h_a2a[soffs[q+1]+1:soffs[q+1]+Int(sc[q+1])] .== T(q * 100 + rank))
                 for q in 0:(P-1))

    ok = MPI.Allreduce(ok_rs && ok_ag && ok_a2a, &, comm)

    if rank == 0
        JLD2.jldsave(out; mpisize = P, ok = ok, ntot = ntot,
                     nzero = count(==(Cint(0)), counts))
    end
    return nothing
end
