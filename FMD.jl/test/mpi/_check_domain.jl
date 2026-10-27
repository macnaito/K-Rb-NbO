# test/mpi/_check_domain.jl
#
# MPIDomain construction, brick partition of fractional space, and the
# collective reduction helpers. Writes a JLD2 verdict from rank 0.

function check_domain(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    dom = Cores.MPIDomain(1000; comm = comm)

    # --- local structural checks (reduced with & below) --------------------
    ok_dims = prod(dom.dims) == P
    ok_bounds = all(0.0 <= dom.lo[d] < dom.hi[d] <= 1.0 for d in 1:3)
    ok_formula = all(dom.lo[d] == FMD.FPPREC(dom.coords[d] / dom.dims[d]) &&
                     dom.hi[d] == FMD.FPPREC((dom.coords[d] + 1) / dom.dims[d])
                     for d in 1:3)
    # nbr order semantics: nbr[d][1] must be the rank whose d-coordinate is
    # mine - 1 (mod dims[d]), nbr[d][2] mine + 1; the other coordinates equal.
    # A swapped (minus, plus) pair would silently cross-wire the future halo
    # exchange, so pin the order via Cart_coords, not just the range.
    function nbr_ok(dom)
        for d in 1:3, (s, off) in ((1, -1), (2, +1))
            r = dom.nbr[d][s]
            0 <= r < MPI.Comm_size(dom.comm) || return false
            c = Int.(MPI.Cart_coords(dom.comm, r))
            expect = [mod(dom.coords[k] + (k == d ? off : 0), dom.dims[k]) for k in 1:3]
            c == expect || return false
        end
        return true
    end
    ok_nbr = nbr_ok(dom)

    ok_dims = MPI.Allreduce(ok_dims, &, comm)
    ok_bounds = MPI.Allreduce(ok_bounds, &, comm)
    ok_formula = MPI.Allreduce(ok_formula, &, comm)
    ok_nbr = MPI.Allreduce(ok_nbr, &, comm)

    # --- disjoint cover: every test point owned by exactly one rank --------
    # Deterministic point set incl. the FP edges wrapfrac is specified to
    # handle (w == 1.0 fold, brick boundaries).
    rng = Random.Xoshiro(0)
    pts = [rand(rng, FMD.FPPREC, 3) for _ in 1:200]
    push!(pts, FMD.FPPREC[0.0, 0.0, 0.0])
    push!(pts, FMD.FPPREC[1.0 - eps(FMD.FPPREC), 0.5, 0.5])
    push!(pts, FMD.FPPREC[0.5, 0.25, 0.75])
    push!(pts, Cores.wrapfrac.(FMD.FPPREC[-eps(FMD.FPPREC) / 4, 2.5, -1.25]))
    # owns wraps internally — unwrapped points must resolve to exactly one
    # owner too (each is an independent probe of its wrapped location).
    push!(pts, FMD.FPPREC[2.5, -1.25, 0.9])
    push!(pts, FMD.FPPREC[-eps(FMD.FPPREC) / 4, 1.0, -0.5])
    # brick boundaries of every rank along each dimension
    for c in 0:maximum(dom.dims)-1, d in 1:3
        c < dom.dims[d] || continue
        w = FMD.FPPREC(c / dom.dims[d])
        p = FMD.FPPREC[0.5, 0.5, 0.5]
        p[d] = w
        push!(pts, p)
    end
    counts = Int32[Cores.owns(dom, p[1], p[2], p[3]) for p in pts]
    MPI.Allreduce!(counts, +, comm)
    ok_partition = all(==(1), counts)

    # --- reduction helpers -------------------------------------------------
    s = Cores.allreduce_sum(dom, rank)
    sv = FMD.FPPREC[rank + 1, 2rank]
    Cores.allreduce_sum!(dom, sv)
    mx = Cores.allreduce_max(dom, FMD.FPPREC(rank))
    land_all = Cores.allreduce_and(dom, true)
    land_one = Cores.allreduce_and(dom, rank != 0)   # rank 0 votes false
    roots = MPI.Allreduce(Cores.isroot(dom) ? 1 : 0, +, comm)
    ng = Cores.n_global(dom, nothing)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, dims = collect(dom.dims),
                     ok_dims = ok_dims, ok_bounds = ok_bounds,
                     ok_formula = ok_formula, ok_nbr = ok_nbr,
                     ok_partition = ok_partition,
                     s = s, sv = sv, mx = mx,
                     land_all = land_all, land_one = land_one,
                     roots = roots, ng = ng)
    end
    return nothing
end
