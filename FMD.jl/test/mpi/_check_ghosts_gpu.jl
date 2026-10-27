# test/mpi/_check_ghosts_gpu.jl
#
# The device DD ghost exchange against the host DD exchange on the SAME
# frozen plan. The host DD build/exchange is already pinned against the
# brute-force oracle by check_ghosts, so comparing device against host-DD
# isolates exactly what this path adds: the converter
# (change_arrtyp(::Ghosts)), the lazily materialized DDStageExec, the
# pack/unpack kernels, and the device MPI transfers.
#
# The transport values (upos, sdisp_ext, fold sums, extend copies) are
# pure copies / one-add-per-slot — the device results must be BIT-equal
# to the host ones. xcart adds a device matvec (compared at 1e-12, far
# below physics, in case a backend contracts mul+add).
#
# mode ∈ normal | bigrc | tiny
#   bigrc — rc = L: multi-round exchanges / self-stages on the device path.
#   tiny  — NaCl-8 at P = 6: empty ranks ride every device collective.

function check_ghosts_gpu(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    # Identical replicated frame on every rank (same fixture family as the
    # CPU ghost check, including gauge-carrying unwrapped rows).
    Random.seed!(0)
    full = mode == "tiny" ? make_nacl_8() : make_nacl_64()
    full.arrays[:pos] .+= (mode == "tiny" ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    if mode != "tiny"
        full.arrays[:pos][1, :] .+= 1.0
        full.arrays[:pos][2, 1] -= 2.0
    end
    prms = nacl_params(full; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = mode == "bigrc" ? 1.0 : 0.45, with_shell = false)
    rc = prms[:rc]
    sk = FMD.FPPREC(0.1)
    rc_eff = FMD.FPPREC(rc * (1 + sk))

    # --- host DD build + host reference exchange ---------------------------
    dom = Cores.MPIDomain(length(full); comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)
    cl, loc, _ = CellList(loc, rc_eff)   # includes the ghost cell-sort permute
    nl = neighborlist(loc, cl, rc, sk)
    g = nl.ghosts
    N = g.n_owned
    G = g.n_ghost
    idsl = Array(loc.arrays[:id])

    h(id, c) = FMD.FPPREC(0.01id + 0.1c)
    for r in 1:N, c in 1:3
        loc.arrays[:shelldisp][r, c] = h(idsl[r], c)
    end
    Cores.sync_ghosts!(g, loc)
    upos_h = copy(g.dd.upos)
    sdisp_h = copy(g.sdisp_ext)
    xcart_h = copy(g.xcart)

    # Convert NOW — post-sync, pre-fold — so the device replays the identical
    # operations from the identical starting state.
    Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
    loc_d = Cores.change_arrtyp(loc, CuArray)
    g_d = Cores.change_arrtyp(g, CuArray)

    # Host fold reference (mutates g.fext — after the converter snapshot).
    f(id, s1, s2, s3, c) = FMD.FPPREC((id + 0.1s1 + 0.01s2 + 0.001s3) * c)
    fext0 = zeros(FMD.FPPREC, N + G, 3)
    for r in 1:N, c in 1:3
        fext0[r, c] = f(idsl[r], 0, 0, 0, c)
    end
    for k in 1:G, c in 1:3
        fext0[N+k, c] = f(g.dd.ghost_id[k], g.dd.gshift[k, 1], g.dd.gshift[k, 2],
                          g.dd.gshift[k, 3], c)
    end
    g.fext .= fext0
    dst_h = zeros(FMD.FPPREC, N, 3)
    Cores.fold_ghosts!(dst_h, g.fext, g)

    # Host extend reference (3 columns — a narrower-than-DD_NCOL payload).
    # Typed comprehension: on an empty rank (N = 0) an untyped one infers
    # Matrix{Any}, which CuArray refuses.
    owned3 = FMD.FPPREC[h(idsl[r], c) + 1 for r in 1:N, c in 1:3]
    ext3_h = zeros(FMD.FPPREC, N + G, 3)
    Cores.extend_ghosts!(ext3_h, owned3, g)

    # --- device replay -----------------------------------------------------
    # sync: same shelldisp state (converted with the frame) → the transported
    # values must be bit-equal to the host exchange.
    Cores.sync_ghosts!(g_d, loc_d)
    ok_upos = Array(g_d.dd.upos) == upos_h
    ok_sdisp = Array(g_d.sdisp_ext) == sdisp_h
    ok_xcart = G + N == 0 || maximum(abs, Array(g_d.xcart) .- xcart_h) <= 1e-12

    # fold: same pattern, device buffers. The mutated ext (the chain's
    # intermediate state) must match the host's too, and a SECOND fold on the
    # identically mutated ext must keep matching (buffer/exec reuse on the
    # accumulating path).
    copyto!(g_d.fext, fext0)
    dst_d = KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N, 3)
    Cores.fold_ghosts!(dst_d, g_d.fext, g_d)
    ok_fold = Array(dst_d) == dst_h
    ok_fext = Array(g_d.fext) == g.fext
    dst2_h = zeros(FMD.FPPREC, N, 3)
    Cores.fold_ghosts!(dst2_h, g.fext, g)
    dst2_d = KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N, 3)
    Cores.fold_ghosts!(dst2_d, g_d.fext, g_d)
    ok_fold2 = Array(dst2_d) == dst2_h && Array(g_d.fext) == g.fext

    # extend: 3-column payload through the (·, DD_NCOL) buffer views.
    owned3_d = CuArray(owned3)
    ext3_d = KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N + G, 3)
    Cores.extend_ghosts!(ext3_d, owned3_d, g_d)
    ok_extend = Array(ext3_d) == ext3_h

    # A payload wider than DD_NCOL must be a loud error, not a truncation —
    # on both the fold and the extend entry.
    ok_ncol = try
        Cores.fold_ghosts!(KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N, 7),
                           KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N + G, 7),
                           g_d)
        false
    catch e
        occursin("DD_NCOL", sprint(showerror, e))
    end
    ok_ncol &= try
        Cores.extend_ghosts!(KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N + G, 7),
                             KernelAbstractions.zeros(CUDA.CUDABackend(), FMD.FPPREC, N, 7),
                             g_d)
        false
    catch e
        occursin("DD_NCOL", sprint(showerror, e))
    end

    # Second sync on the SAME device exchange (exec now materialized —
    # exercises buffer reuse across calls, the self-stage explicit copy
    # included) must reproduce identical values.
    Cores.sync_ghosts!(g_d, loc_d)
    ok_resync = Array(g_d.dd.upos) == upos_h && Array(g_d.sdisp_ext) == sdisp_h

    ok_all = MPI.Allreduce(ok_upos && ok_sdisp && ok_xcart && ok_fold && ok_fext &&
                           ok_fold2 && ok_extend && ok_ncol && ok_resync, &, comm)
    min_nlocal = MPI.Allreduce(N, min, comm)
    nghost_tot = MPI.Allreduce(G, +, comm)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, dims = collect(dom.dims),
                     min_nlocal = min_nlocal, nghost_tot = nghost_tot,
                     ok = ok_all,
                     ok_upos = ok_upos, ok_sdisp = ok_sdisp, ok_xcart = ok_xcart,
                     ok_fold = ok_fold, ok_fext = ok_fext, ok_fold2 = ok_fold2,
                     ok_extend = ok_extend, ok_ncol = ok_ncol,
                     ok_resync = ok_resync)
    end
    return nothing
end
