# test/mpi/_check_ghosts.jl
#
# DD ghost build oracle, P=1 ≡ serial pairmap equality, CoulombBuckingham
# force parity (exercising the MPI sync), the fold adjoint identity
# (reverse chains), shelldisp transport, extend replication, and the
# naive-builder guard. mode ∈ normal | deadlock | bigrc | tiny.
#   bigrc — rc = L: multi-round exchanges on dims 2 (shared-peer tags) and
#           dims 1 (self-stages), δ > 1.
#   tiny  — NaCl-8 at P = 6: brick x = [2/3, 1) holds no atoms → empty
#           ranks must ride every collective (CB parity skipped).

function check_ghosts(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    # Identical replicated frame on every rank.
    Random.seed!(0)
    full = mode == "tiny" ? make_nacl_8() : make_nacl_64()
    full.arrays[:pos] .+= (mode == "tiny" ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    if mode != "tiny"
        full.arrays[:pos][1, :] .+= 1.0    # gauge-carrying unwrapped rows
        full.arrays[:pos][2, 1] -= 2.0
    end
    orig = deepcopy_atoms(full)   # serial reference (never gets a :domain)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = mode == "bigrc" ? 1.0 : 0.45, with_shell = false)
    rc = prms[:rc]
    sk = FMD.FPPREC(0.1)
    rc_eff = FMD.FPPREC(rc * (1 + sk))

    dom = Cores.MPIDomain(length(full); comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)   # no movement here; keeps the production call order

    cl, loc, _ = CellList(loc, rc_eff)
    nl = neighborlist(loc, cl, rc, sk)
    g = nl.ghosts
    dd = g.dd
    N = g.n_owned
    idsl = Array(loc.arrays[:id])
    nfull = length(orig)

    if mode == "deadlock"
        # Asymmetric displacement: only rank 0's atoms exceed skin/2. The
        # collective isreusable must return the SAME false on every rank —
        # a missing Allreduce would split the votes (rank 0 false, others
        # true) and, in production, deadlock at the next collective.
        if rank == 0 && length(loc) > 0
            loc.arrays[:pos][1, 1] += FMD.FPPREC(1.1) * FMD.FPPREC(nl.skin) /
                                      FMD.FPPREC(Array(orig.info[:lattice])[1, 1])
        end
        reusable = isreusable(nl, loc)
        votes = MPI.Allgather(reusable ? Int32(1) : Int32(0), comm)
        if rank == 0
            JLD2.jldsave(out; mpisize = P, votes = collect(votes))
        end
        return nothing
    end

    # --- 1. ghost oracle: (id, shift) set equality vs brute force ----------
    δ = Cores._halo_frac(Array(orig.info[:lattice]), rc_eff)
    wfull = [Cores.wrapfrac(orig.arrays[:pos][i, d]) for i in 1:nfull, d in 1:3]
    myids = Set(idsl)
    expected = Set{NTuple{4,Int}}()
    for i in 1:nfull, sx in -3:3, sy in -3:3, sz in -3:3
        u1 = wfull[i, 1] + sx
        u2 = wfull[i, 2] + sy
        u3 = wfull[i, 3] + sz
        inreg = (dom.lo[1] - δ[1] <= u1 < dom.hi[1] + δ[1]) &&
                (dom.lo[2] - δ[2] <= u2 < dom.hi[2] + δ[2]) &&
                (dom.lo[3] - δ[3] <= u3 < dom.hi[3] + δ[3])
        inreg || continue
        id = orig.arrays[:id][i]
        (sx == 0 && sy == 0 && sz == 0 && id in myids) && continue
        push!(expected, (id, sx, sy, sz))
    end
    actual = Set{NTuple{4,Int}}()
    for k in 1:g.n_ghost
        push!(actual, (dd.ghost_id[k], Int(dd.gshift[k, 1]), Int(dd.gshift[k, 2]),
                       Int(dd.gshift[k, 3])))
    end
    ok_oracle = MPI.Allreduce(actual == expected && length(actual) == g.n_ghost, &, comm)

    # --- 2. P = 1: DD build ≡ serial build (pairmap equality) --------------
    ok_p1 = true
    if P == 1
        cl0, a0, _ = CellList(deepcopy_atoms(orig), rc_eff)
        nl0 = neighborlist(a0, cl0, rc, sk)
        ok_p1 = ghost_pairmap(a0, nl0) == ghost_pairmap(loc, nl)
    end
    ok_p1 = MPI.Allreduce(ok_p1, &, comm)

    # --- 3. CoulombBuckingham force parity (exercises the MPI sync) --------
    ok_forces = true
    fmax = FMD.FPPREC(1.0)
    if mode != "tiny"
        a0 = deepcopy_atoms(orig)
        cl0, a0, _ = CellList(a0, rc_eff)
        nl0 = neighborlist(a0, cl0, rc, sk)
        Calculators.calculate!(Calculators.CoulombBuckingham(), a0, nl0, prms)
        F_ref = Array(a0.arrays[:forces])
        ids0 = Array(a0.arrays[:id])
        byid = zeros(FMD.FPPREC, nfull, 3)
        byid[ids0, :] .= F_ref

        fill!(loc.arrays[:forces], 0)
        Calculators.calculate!(Calculators.CoulombBuckingham(), loc, nl, prms)
        F_loc = Array(loc.arrays[:forces])
        fmax = max(one(FMD.FPPREC), maximum(abs, F_ref))
        ok_forces = length(loc) == 0 ||
                    maximum(abs, F_loc .- byid[idsl, :]) < 1e-12 * fmax
        fmax = maximum(abs, F_ref)
    end
    ok_forces = MPI.Allreduce(ok_forces, &, comm)

    # --- 4. fold adjoint identity (exercises the reverse chains) -----------
    f(id, s1, s2, s3, c) = (id + 0.1s1 + 0.01s2 + 0.001s3) * c
    fill!(g.fext, 0)
    for r in 1:N, c in 1:3
        g.fext[r, c] = f(idsl[r], 0, 0, 0, c)
    end
    for k in 1:g.n_ghost, c in 1:3
        g.fext[N+k, c] = f(dd.ghost_id[k], dd.gshift[k, 1], dd.gshift[k, 2],
                           dd.gshift[k, 3], c)
    end
    dst = zeros(FMD.FPPREC, N, 3)
    Cores.fold_ghosts!(dst, g.fext, g)

    los = [ntuple(d -> FMD.FPPREC(Int(MPI.Cart_coords(dom.comm, r)[d]) / dom.dims[d]), 3)
           for r in 0:P-1]
    his = [ntuple(d -> FMD.FPPREC((Int(MPI.Cart_coords(dom.comm, r)[d]) + 1) / dom.dims[d]), 3)
           for r in 0:P-1]
    ok_fold = true
    for r in 1:N
        id = idsl[r]
        i = findfirst(==(id), orig.arrays[:id])
        w = (wfull[i, 1], wfull[i, 2], wfull[i, 3])
        for c in 1:3
            exp_val = f(id, 0, 0, 0, c)   # own row contribution
            for rr in 1:P, sx in -3:3, sy in -3:3, sz in -3:3
                u = (w[1] + sx, w[2] + sy, w[3] + sz)
                inreg = all(los[rr][d] - δ[d] <= u[d] < his[rr][d] + δ[d] for d in 1:3)
                inreg || continue
                owned_here = sx == 0 && sy == 0 && sz == 0 &&
                             all(los[rr][d] <= w[d] < his[rr][d] for d in 1:3)
                owned_here && continue
                exp_val += f(id, sx, sy, sz, c)
            end
            abs(dst[r, c] - exp_val) < 1e-8 || (ok_fold = false)
        end
    end
    ok_fold = MPI.Allreduce(ok_fold, &, comm)

    # --- 5. shelldisp transport (bitwise: no seam shift applies to it) -----
    h(id, c) = FMD.FPPREC(0.01id + 0.1c)
    for r in 1:N, c in 1:3
        loc.arrays[:shelldisp][r, c] = h(idsl[r], c)
    end
    Cores.sync_ghosts!(g, loc)
    ok_sdisp = true
    for k in 1:g.n_ghost, c in 1:3
        g.sdisp_ext[N+k, c] == h(dd.ghost_id[k], c) || (ok_sdisp = false)
    end
    for r in 1:N, c in 1:3
        g.sdisp_ext[r, c] == h(idsl[r], c) || (ok_sdisp = false)
    end
    ok_sdisp = MPI.Allreduce(ok_sdisp, &, comm)

    # --- 6. extend replication (adjoint-forward path used by fitting) ------
    owned2 = FMD.FPPREC[h(idsl[r], c) + 1 for r in 1:N, c in 1:2]
    ext2 = zeros(FMD.FPPREC, length(g), 2)
    Cores.extend_ghosts!(ext2, owned2, g)
    ok_extend = true
    for k in 1:g.n_ghost, c in 1:2
        ext2[N+k, c] == h(dd.ghost_id[k], c) + 1 || (ok_extend = false)
    end
    for r in 1:N, c in 1:2
        ext2[r, c] == owned2[r, c] || (ok_extend = false)
    end
    ok_extend = MPI.Allreduce(ok_extend, &, comm)

    # --- 7. the naive builder refuses to run under DD ----------------------
    ok_naive = try
        neighborlist(loc, rc, sk)
        false
    catch e
        occursin("naive builder", sprint(showerror, e))
    end
    ok_naive = MPI.Allreduce(ok_naive, &, comm)

    min_nlocal = MPI.Allreduce(N, min, comm)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, dims = collect(dom.dims),
                     nghost = g.n_ghost, min_nlocal = min_nlocal,
                     ok_oracle = ok_oracle, ok_p1 = ok_p1,
                     ok_forces = ok_forces, ok_fold = ok_fold,
                     ok_sdisp = ok_sdisp, ok_extend = ok_extend,
                     ok_naive = ok_naive,
                     fmax = fmax)
    end
    return nothing
end
