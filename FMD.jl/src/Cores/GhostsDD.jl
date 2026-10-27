# src/Cores/GhostsDD.jl — included from Ghosts.jl
#
# Domain-decomposition ghost machinery: a staged 6-direction halo exchange
# replaces the serial image enumeration. The build runs on host frames;
# the per-step exchange (sync/fold/extend) is backend-generic — KA
# pack/unpack kernels over lazily materialized per-stage resources
# (DDStageExec), with device buffers handed to MPI directly (CUDA-aware
# transport, Domain.jl contract).  For
# each dimension d (in fixed order x → y → z) and each round
# r = 1..H_d = ceil(δ_d · dims_d), an "up" stage ships the band within
# δ_d of the upper brick face to the plus neighbour (filling that rank's
# lower halo) and a "down" stage mirrors it.  Round r ≥ 2 forwards the
# rows received in round r−1 of the same stream, so bricks thinner than
# the cutoff are covered by more rounds; dimension order makes corner
# ghosts flow automatically (a dim-2 stage forwards dim-1 ghosts) and
# uniquely (dim-1 stages never see dim-2 ghosts — no duplicates).
#
# A dimension with dims[d] == 1 degenerates to self-stages (peer == this
# rank): the "exchange" is a local copy with the seam shift applied, which
# reproduces exactly the serial periodic images.  With P == 1 the whole
# build is self-stages and must agree with the serial builder as a set.
#
# Coordinates: every row carries `u`, its fractional coordinate in THIS
# rank's halo frame — owned rows have u = pos + frozen wrap ∈ [lo, hi)
# (up to one ulp below `lo` at the folded FP edge, see `_frozen_wrap`;
# ownership predicates wrap internally and the cell clamp absorbs it),
# ghosts live in [lo−δ, hi+δ) and may lie outside [0,1).  Crossing the
# periodic seam adds the per-stage integer `seamshift` on dimension d
# only.  The plan (stages, row lists, region tags) is frozen at build;
# only the values (u, shelldisp, forces) flow at sync/fold time.

"""
    DDStage

One frozen pairwise exchange of the DD ghost plan: `sendrows` (extended
row indices — owned rows or ghosts from earlier stages, enabling forward
chains) are packed in list order, shipped to `peer_send`, and the rows
received from `peer_recv` land in `recvrows` (in list order — the
"pack order = row-list order" contract). `seamshift` is the integer box
shift applied to `u[dim]` when the message crosses the periodic seam.
With `dims[dim] == 2` both peers are the same rank and concurrent stages
are distinguished by `tag` alone; with `dims[dim] == 1` the peers are
this rank itself and the exchange is a local copy.
"""
struct DDStage
    dim::Int
    peer_send::Int
    peer_recv::Int
    tag::Int
    seamshift::Int
    sendrows::Vector{Int32}
    recvrows::Vector{Int32}
end

# Widest column count any exchange payload uses: sync = 6 (u + shelldisp),
# fold = 3 (fext) / 6 (vext), extend = 3 (sdisp / adjoint u_F). The
# preallocated stage buffers are (n, DD_NCOL); narrower payloads go through
# a contiguous first-columns view. A wider payload is a hard error (raise
# DD_NCOL), never a silent truncation.
const DD_NCOL = 6

"""
    DDStageExec

Per-stage execution resources on the compute backend, materialized lazily
from the (host) plan by [`_ensure_exec!`](@ref): the row lists copied to
the backend for the pack/unpack kernels, and the preallocated send/receive
buffers (`(ns, DD_NCOL)` / `(nr, DD_NCOL)`; the current per-call
allocation is gone). Forward transfers pack into `sbuf` and receive into
`rbuf`; reverse (fold) transfers swap the roles. The plan itself
(`DDStage`) stays on the host as the single source of truth — a cell-sort
permutation rebuilds the stages and simply drops `exec` (see
`_permute_ghosts`), so there is no device-side remap to get wrong.
"""
struct DDStageExec{TI<:AbstractVector{Int32},TM<:AbstractMatrix{FPPREC}}
    sendrows::TI
    recvrows::TI
    sbuf::TM
    rbuf::TM
end

"""
    DDExchange

Frozen DD communication plan plus the per-ghost bookkeeping:

- `stages` — executed in order for forward comm (`sync_ghosts!`,
  `extend_ghosts!`), in reverse with accumulation for reverse comm
  (`fold_ghosts!`).
- `upos` — `(G, 3)` ghost fractional coordinates in this rank's halo
  frame; refreshed by every sync.
- `region` — `(G, 3)` `Int8` halo-band signs per dimension
  (`-1`/`0`/`+1`), frozen at build; the CellList uses them to classify
  ghost rows into halo cells (never all-zero — a ghost is outside the
  brick in at least one dimension).
- `ghost_id`, `gshift` — owner's global `:id` and the accumulated integer
  shift relative to the owner's wrapped coordinate
  (`u_ghost = wrap(pos_owner) + gshift`); diagnostics and the test oracle.

`upos` follows the compute backend (the only per-step mutable state the
exchange owns); `region`/`ghost_id`/`gshift` stay host-side (build-time
binning and diagnostics only). `exec` caches the per-stage backend
resources ([`DDStageExec`](@ref)); empty means "not materialized yet" —
the element type is deliberately abstract, and every consumer crosses a
per-stage function barrier.
"""
struct DDExchange{TU<:AbstractMatrix{FPPREC}}
    dom::MPIDomain
    stages::Vector{DDStage}
    upos::TU
    region::Matrix{Int8}
    ghost_id::Vector{Int}
    gshift::Matrix{Int32}
    exec::Vector{DDStageExec}
end

# Plan-only constructor: exec starts empty and is materialized on first
# use (build and _permute_ghosts both go through this — a permutation
# invalidates the row lists, so dropping exec is exactly right).
DDExchange(dom::MPIDomain, stages, upos, region, ghost_id, gshift) =
    DDExchange(dom, stages, upos, region, ghost_id, gshift, DDStageExec[])

"""
    _ensure_exec!(g::Ghosts, dd::DDExchange) -> Vector{DDStageExec}

Materialize the per-stage backend resources (row-list copies + transfer
buffers) on the backend of `dd.upos`, verifying — once per rebuild — the
invariants the `inbounds` kernels rely on: row lists inside the extended
row space (an out-of-bounds device write would silently corrupt other
arrays), `sendrows`/`recvrows` unique within a stage (the pack gather,
the unpack scatters, and especially the atomics-free fold scatter-add —
a duplicate recvrow would double-count a ghost's force at the owner), and
self-stages shipping equal counts (`rbuf .= sbuf` requires it).

The errors here are deliberately rank-local (no lockstep verdict): a
violation is a bug in the deterministic build logic, not a data-dependent
runtime condition — collectivizing would spend an Allreduce per rebuild
on a programmer-error case. The hang of the peer ranks on such an error
is accepted.
"""
function _ensure_exec!(g::Ghosts, dd::DDExchange)
    isempty(dd.exec) || return dd.exec
    backend = get_backend(dd.upos)
    me = MPI.Comm_rank(dd.dom.comm)
    N = g.n_owned
    ntot = N + g.n_ghost
    size(dd.upos, 1) == g.n_ghost ||
        error("_ensure_exec!: upos has $(size(dd.upos, 1)) rows but the plan " *
              "carries $(g.n_ghost) ghosts")
    for st in dd.stages
        ns = length(st.sendrows)
        nr = length(st.recvrows)
        allunique(st.sendrows) && allunique(st.recvrows) ||
            error("_ensure_exec!: duplicate rows within a stage — pack/unpack " *
                  "and the fold scatter-add rely on their uniqueness (atomics-free)")
        all(r -> 1 <= r <= ntot, st.sendrows) ||
            error("_ensure_exec!: sendrows outside the extended row space 1:$ntot")
        all(r -> N < r <= ntot, st.recvrows) ||
            error("_ensure_exec!: recvrows must be ghost rows ($(N+1):$ntot)")
        st.peer_send == me && st.peer_recv == me && ns != nr &&
            error("_ensure_exec!: self-stage with ns=$ns ≠ nr=$nr")
        sr = KernelAbstractions.allocate(backend, Int32, ns)
        copyto!(sr, st.sendrows)
        rr = KernelAbstractions.allocate(backend, Int32, nr)
        copyto!(rr, st.recvrows)
        sbuf = KernelAbstractions.allocate(backend, FPPREC, ns, DD_NCOL)
        rbuf = KernelAbstractions.allocate(backend, FPPREC, nr, DD_NCOL)
        push!(dd.exec, DDStageExec(sr, rr, sbuf, rbuf))
    end
    return dd.exec
end

# Contiguous first-`ncol`-columns view of a stage buffer (column-major, so
# the wire layout equals a dense (n, ncol) matrix — bitwise identical to
# the former exact-size per-call buffers).
_bufcols(M::AbstractMatrix, ncol::Int) = ncol == size(M, 2) ? M : view(M, :, 1:ncol)

"""
    _dd_transfer!(src, dst, st, dom, tagoff, reverse) -> dst

One pairwise transfer over preallocated stage buffers. `src` was packed by
a kernel on this task's stream; `_device_mpi_prepare!` (Domain.jl) drains
the stream before MPI reads the raw pointer — that same drain also orders
any still-pending unpack kernel of the previous call against MPI's write
into `dst` (single task stream). Self-stages copy explicitly (`dst .= src`
— never alias: the buffers persist across calls and directions). MPI is
called unconditionally for peer stages, zero-length payloads included (the
peer's matching Sendrecv must pair up — the empty-rank deadlock rule; a
zero-row device buffer converts to a NULL device pointer, which MPI
accepts at count = 0).
"""
function _dd_transfer!(src::AbstractMatrix, dst::AbstractMatrix, st::DDStage,
                       dom::MPIDomain, tagoff::Int, reverse::Bool)
    me = MPI.Comm_rank(dom.comm)
    if st.peer_send == me && st.peer_recv == me
        dst .= src
        return dst
    end
    _device_mpi_prepare!(parent(src))
    dest = reverse ? st.peer_recv : st.peer_send
    source = reverse ? st.peer_send : st.peer_recv
    MPI.Sendrecv!(src, dst, dom.comm;
                  dest = dest, sendtag = st.tag + tagoff,
                  source = source, recvtag = st.tag + tagoff)
    return dst
end

"""
    _frozen_wrap(pos) -> Matrix{Int32}

The frozen wrap offsets `-floor.(pos)` with the FP edge folded: where
`pos + ws` rounds to exactly `1.0` (tiny negative `pos`) the offset is
reduced by one, so the frozen `u = pos + ws` classifies the atom at its
true (lower) face — an unfolded `u = 1.0` would land it in the wrong
exchange band. Single source of truth for the DD build and the DD
CellList binning (a second implementation drifting by one ulp is how
atoms get mis-banded). Note the folded `u` can sit one ulp *below* zero /
`lo` — the ownership predicates wrap internally and the cell clamp
absorbs it.
"""
function _frozen_wrap(pos::AbstractMatrix)
    ws = Int32.(.-floor.(pos))
    @inbounds for j in axes(pos, 2), i in axes(pos, 1)
        if pos[i, j] + ws[i, j] >= 1
            ws[i, j] -= Int32(1)
        end
    end
    return ws
end

"""
    _build_ghosts_dd(atoms, rc_eff, dom::MPIDomain) -> Ghosts

Staged halo-exchange ghost build (collective; `migrate` must have run so
every owned wrapped coordinate lies in the brick). Freezes the wrap
offsets, the stage plan, and the per-ghost bookkeeping; the returned
object is not synced.
"""
function _build_ghosts_dd(atoms::Atoms, rc_eff, dom::MPIDomain)
    KernelAbstractions.get_backend(atoms) isa KernelAbstractions.CPU ||
        error("build_ghosts (DD): the staged build runs on host frames — build on " *
              "the host and convert the result with change_arrtyp(ghosts, arrtyp); " *
              "the CellList/NeighborList converters that complete this workflow " *
              "arrive with the remaining Phase 2 PRs")
    haskey(atoms.arrays, :id) ||
        error("build_ghosts (DD): atoms.arrays[:id] is required (global atom identity)")
    FPPREC === Float32 && dom.nglobal >= 2^24 &&
        error("build_ghosts (DD): the build payload carries atom ids in FPPREC — " *
              "ids ≥ 2^24 are not exactly representable in Float32")
    N = length(atoms)
    pos = atoms.arrays[:pos]
    δ = _halo_frac(Array(atoms.info[:lattice]), rc_eff)
    H = ntuple(d -> max(ceil(Int, δ[d] * dom.dims[d]), 1), 3)
    maximum(H) <= 50 ||
        error("build_ghosts (DD): H = $H exchange rounds — brick absurdly thin " *
              "relative to the cutoff (tag scheme headroom exceeded)")
    # The stage sequence (and hence deadlock freedom) requires every rank
    # to derive the identical (δ, H) — i.e. bit-identical lattices. Verify
    # instead of assuming (one cheap collective per rebuild).
    sig = hash((δ, H))
    MPI.Allreduce(sig, min, dom.comm) == MPI.Allreduce(sig, max, dom.comm) ||
        error("build_ghosts (DD): halo geometry (δ, H) differs across ranks — the " *
              "lattice is not bit-identical (it must be updated only from " *
              "Allreduce-derived values; see the AbstractDomain invariant)")

    ws = _frozen_wrap(pos)
    u_own = Matrix{FPPREC}(undef, N, 3)
    u_own .= pos .+ ws
    bad = ""
    for i in 1:N
        if !owns(dom, u_own[i, 1], u_own[i, 2], u_own[i, 3]) && isempty(bad)
            bad = "atom :id=$(atoms.arrays[:id][i])"
        end
    end
    # Collective: a lone rank erroring here would leave the others blocking
    # in the first Sendrecv of the stage loop.
    allreduce_and(dom, isempty(bad)) ||
        error("build_ghosts (DD): owned atom(s) outside this rank's brick" *
              (isempty(bad) ? " (on another rank)" : " — $bad") *
              " — call migrate before rebuilding")
    ids_own = Int.(atoms.arrays[:id])
    elem_own = atoms.arrays[:elemid]

    # Growing ghost tables (build-time only; modest sizes).
    gu = Matrix{FPPREC}(undef, 0, 3)
    gid = Int[]
    gelem = eltype(elem_own)[]
    ggs = Matrix{Int32}(undef, 0, 3)
    stages = DDStage[]

    # Row accessors over the current extended set (owned + ghosts so far).
    ucur(row, c) = row <= N ? u_own[row, c] : gu[row-N, c]

    for d in 1:3
        rows_before_dim = N + size(gu, 1)
        prev_up = Int32[]
        prev_down = Int32[]
        for r in 1:H[d]
            for dir in (:up, :down)
                # Band selection: round 1 scans everything that existed
                # before this dimension started (owned + earlier-dim
                # ghosts → corners); round r ≥ 2 forwards only the rows
                # received in round r−1 of the same stream (same-dim
                # ghosts of the opposite stream belong to the peer and
                # must never be reflected back — that would duplicate).
                if dir === :up
                    peer_send = dom.nbr[d][2]
                    peer_recv = dom.nbr[d][1]
                    seamshift = dom.coords[d] == dom.dims[d] - 1 ? -1 : 0
                    candidates = r == 1 ? Int32.(1:rows_before_dim) : prev_up
                    sendrows = Int32[row for row in candidates
                                     if ucur(row, d) >= dom.hi[d] - δ[d]]
                else
                    peer_send = dom.nbr[d][1]
                    peer_recv = dom.nbr[d][2]
                    seamshift = dom.coords[d] == 0 ? 1 : 0
                    candidates = r == 1 ? Int32.(1:rows_before_dim) : prev_down
                    sendrows = Int32[row for row in candidates
                                     if ucur(row, d) < dom.lo[d] + δ[d]]
                end
                tag = 1000 + (d - 1) * 100 + (r - 1) * 2 + (dir === :up ? 0 : 1)

                # Build-time payload: u(3) + gshift(3) + elemid(1) + id(1),
                # small integers carried exactly in FPPREC.
                ns = length(sendrows)
                sbuf = Matrix{FPPREC}(undef, ns, 8)
                for (j, row) in enumerate(sendrows)
                    for c in 1:3
                        sbuf[j, c] = ucur(row, c)
                        sbuf[j, 3+c] = row <= N ? zero(FPPREC) : FPPREC(ggs[row-N, c])
                    end
                    sbuf[j, d] += seamshift
                    sbuf[j, 3+d] += seamshift
                    sbuf[j, 7] = row <= N ? FPPREC(elem_own[row]) : FPPREC(gelem[row-N])
                    sbuf[j, 8] = row <= N ? FPPREC(ids_own[row]) : FPPREC(gid[row-N])
                end

                # Count exchange, then the payload.
                me = MPI.Comm_rank(dom.comm)
                if peer_send == me && peer_recv == me
                    nr = ns
                    rbuf = sbuf
                else
                    nrref = Ref(0)
                    MPI.Sendrecv!(Ref(ns), nrref, dom.comm;
                                  dest = peer_send, sendtag = tag,
                                  source = peer_recv, recvtag = tag)
                    nr = nrref[]
                    rbuf = Matrix{FPPREC}(undef, nr, 8)
                    MPI.Sendrecv!(sbuf, rbuf, dom.comm;
                                  dest = peer_send, sendtag = tag,
                                  source = peer_recv, recvtag = tag)
                end

                # Append received ghosts; freeze their plan rows.
                g0 = size(gu, 1)
                recvrows = Int32.(N .+ g0 .+ (1:nr))
                gu = vcat(gu, rbuf[:, 1:3])
                ggs = vcat(ggs, Int32.(rbuf[:, 4:6]))
                append!(gelem, eltype(elem_own).(rbuf[:, 7]))
                append!(gid, Int.(rbuf[:, 8]))
                push!(stages, DDStage(d, peer_send, peer_recv, tag, seamshift,
                                      sendrows, collect(recvrows)))
                if dir === :up
                    prev_up = collect(recvrows)
                else
                    prev_down = collect(recvrows)
                end
            end
        end
    end

    G = size(gu, 1)
    # Region tags + halo-band sanity: every ghost must lie inside the
    # extended region and outside the brick in at least one dimension.
    region = Matrix{Int8}(undef, G, 3)
    badg = ""
    for k in 1:G
        allzero = true
        for d in 1:3
            w = gu[k, d]
            if !(dom.lo[d] - δ[d] <= w < dom.hi[d] + δ[d]) && isempty(badg)
                badg = "ghost id=$(gid[k]) outside the halo region along dim $d (u=$w)"
            end
            s = w < dom.lo[d] ? Int8(-1) : (w >= dom.hi[d] ? Int8(1) : Int8(0))
            region[k, d] = s
            s == 0 || (allzero = false)
        end
        if allzero && isempty(badg)
            badg = "ghost id=$(gid[k]) lies inside the brick (would shadow an owned atom)"
        end
    end
    # Collective verdict — a lone erroring rank would leave the others
    # blocking in the sync that follows the build.
    allreduce_and(dom, isempty(badg)) ||
        error("build_ghosts (DD): stage plan inconsistency" *
              (isempty(badg) ? " (on another rank)" : " — $badg"))

    dd = DDExchange(dom, stages, gu, region, gid, ggs)
    shift = vcat(ws, ggs)
    elemid_ext = vcat(elem_own, gelem)
    xcart = zeros(FPPREC, N + G, 3)
    sdisp_ext = zeros(FPPREC, N + G, 3)
    fext = zeros(FPPREC, N + G, 3)
    vext = zeros(FPPREC, N + G, 6)
    return Ghosts(N, G, Int32[], shift, xcart, sdisp_ext, elemid_ext, fext, vext, dd)
end

# Forward communication: ship current u + shelldisp along the frozen plan,
# then refresh xcart from u · lattice.
function _sync_ghosts_dd!(g::Ghosts, atoms::Atoms)
    # The plan's row lists are frozen against the build-time row count; a
    # frame that migrated without a rebuild would make the @inbounds loops
    # below ship other atoms' coordinates silently.
    size(atoms.arrays[:pos], 1) == g.n_owned ||
        error("sync_ghosts! (DD): atoms has $(size(atoms.arrays[:pos], 1)) rows but " *
              "the plan was built for $(g.n_owned) — rebuild the ghosts after migrate")
    # Function barrier: atoms.arrays / atoms.info are Dicts with abstract
    # value types; the per-row loops must run on concretely typed arrays.
    return _sync_stages_dd!(g, g.dd, atoms.arrays[:pos], atoms.arrays[:shelldisp],
                            atoms.info[:lattice])
end

# Sync pack: owned rows ship u = pos + frozen shift, ghost rows forward
# the previously received upos (the forward chain); the seam shift applies
# to the stage dimension only. Value-order matches the former host loop
# exactly ((pos + shift) + seamshift), so the CPU backend is bit-identical.
@kernel inbounds = true function kernel_dd_pack_sync!(sbuf, @Const(rows), @Const(pos),
        @Const(shift), @Const(upos), @Const(sdisp), @Const(sdisp_ext),
        n_owned, dim, seamshift)
    j, c = @index(Global, NTuple)
    row = Int(rows[j])
    if row <= n_owned
        u = pos[row, c] + shift[row, c]
        s = sdisp[row, c]
    else
        k = row - n_owned
        u = upos[k, c]
        s = sdisp_ext[row, c]
    end
    sbuf[j, c] = c == dim ? u + seamshift : u
    sbuf[j, 3+c] = s
end

# Sync unpack: recvrows are always ghost rows (build guarantee), each row
# unique within a stage — plain scatter, no atomics.
@kernel inbounds = true function kernel_dd_unpack_sync!(upos, sdisp_ext, @Const(rows),
                                                        @Const(rbuf), n_owned)
    j, c = @index(Global, NTuple)
    row = Int(rows[j])
    upos[row-n_owned, c] = rbuf[j, c]
    sdisp_ext[row, c] = rbuf[j, 3+c]
end

# xcart refresh for the DD extended set: owned rows from pos + shift,
# ghost rows from upos (serial's kernel_sync_xcart! reads the owner via
# src — under DD the owner lives on another rank, hence this variant).
# Same matvec association as the former host loop: bit-identical on CPU.
@kernel inbounds = true function kernel_sync_xcart_dd!(xcart, @Const(pos), @Const(shift),
                                                       @Const(upos), @Const(lat), n_owned)
    r = @index(Global)
    if r <= n_owned
        f1 = pos[r, 1] + shift[r, 1]
        f2 = pos[r, 2] + shift[r, 2]
        f3 = pos[r, 3] + shift[r, 3]
    else
        k = r - n_owned
        f1 = upos[k, 1]
        f2 = upos[k, 2]
        f3 = upos[k, 3]
    end
    xcart[r, 1] = f1 * lat[1, 1] + f2 * lat[2, 1] + f3 * lat[3, 1]
    xcart[r, 2] = f1 * lat[1, 2] + f2 * lat[2, 2] + f3 * lat[3, 2]
    xcart[r, 3] = f1 * lat[1, 3] + f2 * lat[2, 3] + f3 * lat[3, 3]
end

function _sync_stages_dd!(g::Ghosts, dd::DDExchange, pos::AbstractMatrix,
                          sdisp::AbstractMatrix, lat::AbstractMatrix)
    N = g.n_owned
    backend = get_backend(dd.upos)
    get_backend(pos) == backend ||
        error("sync_ghosts! (DD): atoms and ghosts live on different backends — " *
              "convert both (change_arrtyp) before syncing")
    exec = _ensure_exec!(g, dd)
    for i in eachindex(dd.stages)
        _stage_sync!(dd.stages[i], exec[i], g, dd, pos, sdisp, backend)
    end
    g.sdisp_ext[1:N, :] .= sdisp
    ntot = N + g.n_ghost
    ntot > 0 &&
        kernel_sync_xcart_dd!(backend)(g.xcart, pos, g.shift, dd.upos, lat, N;
                                       ndrange = ntot)
    return g
end

# Per-stage function barrier (exec has an abstract element type): kernels
# only launch for nonzero row counts (a zero-size device launch errors),
# the transfer runs unconditionally (empty-rank deadlock rule).
function _stage_sync!(st::DDStage, ex::DDStageExec, g::Ghosts, dd::DDExchange,
                      pos, sdisp, backend)
    ns = length(st.sendrows)
    nr = length(st.recvrows)
    ns > 0 &&
        kernel_dd_pack_sync!(backend)(ex.sbuf, ex.sendrows, pos, g.shift, dd.upos,
                                      sdisp, g.sdisp_ext, g.n_owned, st.dim,
                                      st.seamshift; ndrange = (ns, 3))
    _dd_transfer!(ex.sbuf, ex.rbuf, st, dd.dom, 0, false)
    nr > 0 &&
        kernel_dd_unpack_sync!(backend)(dd.upos, g.sdisp_ext, ex.recvrows, ex.rbuf,
                                        g.n_owned; ndrange = (nr, 3))
    return nothing
end

# Generic gather-pack / scatter-unpack over a frozen row list. Rows are
# unique within a stage (asserted at _ensure_exec!), so every (j, c)
# thread touches a distinct (row, c) — including the accumulating fold
# unpack, which therefore needs no atomics.
@kernel inbounds = true function kernel_dd_pack!(sbuf, @Const(rows), @Const(ext))
    j, c = @index(Global, NTuple)
    sbuf[j, c] = ext[Int(rows[j]), c]
end

@kernel inbounds = true function kernel_dd_unpack_set!(ext, @Const(rows), @Const(rbuf))
    j, c = @index(Global, NTuple)
    ext[Int(rows[j]), c] = rbuf[j, c]
end

@kernel inbounds = true function kernel_dd_unpack_add!(ext, @Const(rows), @Const(rbuf))
    j, c = @index(Global, NTuple)
    row = Int(rows[j])
    ext[row, c] += rbuf[j, c]
end

# Forward replication of an arbitrary owned array into the extended rows
# (adjoint vectors etc.) — the same plan, pure copy, no seam shift.
function _extend_ghosts_dd!(ext, owned, g::Ghosts)
    N = g.n_owned
    ncol = size(ext, 2)
    ncol <= DD_NCOL ||
        error("extend_ghosts! (DD): payload has $ncol columns > DD_NCOL = $DD_NCOL " *
              "— raise DD_NCOL (GhostsDD.jl)")
    ext[1:N, :] .= owned
    dd = g.dd
    backend = get_backend(dd.upos)
    get_backend(ext) == backend ||
        error("extend_ghosts! (DD): ext and the ghosts live on different backends")
    exec = _ensure_exec!(g, dd)
    for i in eachindex(dd.stages)
        _stage_extend!(dd.stages[i], exec[i], ext, ncol, dd.dom, backend)
    end
    return ext
end

function _stage_extend!(st::DDStage, ex::DDStageExec, ext, ncol, dom, backend)
    ns = length(st.sendrows)
    nr = length(st.recvrows)
    sb = _bufcols(ex.sbuf, ncol)
    rb = _bufcols(ex.rbuf, ncol)
    ns > 0 && kernel_dd_pack!(backend)(sb, ex.sendrows, ext; ndrange = (ns, ncol))
    _dd_transfer!(sb, rb, st, dom, 2000, false)
    nr > 0 && kernel_dd_unpack_set!(backend)(ext, ex.recvrows, rb; ndrange = (nr, ncol))
    return nothing
end

# Reverse communication: run the stages backwards, adding the received
# contributions at the original send rows so ghost-of-ghost chains fold
# towards the owner, then accumulate the owned rows into dst. The buffer
# roles swap: pack from recvrows into the (nr, ·) buffer, receive into the
# (ns, ·) one.
function _fold_ghosts_dd!(dst, ext, g::Ghosts)
    N = g.n_owned
    ncol = size(ext, 2)
    ncol <= DD_NCOL ||
        error("fold_ghosts! (DD): payload has $ncol columns > DD_NCOL = $DD_NCOL " *
              "— raise DD_NCOL (GhostsDD.jl)")
    dd = g.dd
    backend = get_backend(dd.upos)
    get_backend(ext) == backend ||
        error("fold_ghosts! (DD): ext and the ghosts live on different backends")
    exec = _ensure_exec!(g, dd)
    for i in reverse(eachindex(dd.stages))
        _stage_fold!(dd.stages[i], exec[i], ext, ncol, dd.dom, backend)
    end
    dst .+= @view ext[1:N, :]
    return dst
end

function _stage_fold!(st::DDStage, ex::DDStageExec, ext, ncol, dom, backend)
    ns = length(st.sendrows)
    nr = length(st.recvrows)
    sb = _bufcols(ex.rbuf, ncol)   # reverse: send what the recvrows hold
    rb = _bufcols(ex.sbuf, ncol)
    nr > 0 && kernel_dd_pack!(backend)(sb, ex.recvrows, ext; ndrange = (nr, ncol))
    _dd_transfer!(sb, rb, st, dom, 4000, true)
    ns > 0 && kernel_dd_unpack_add!(backend)(ext, ex.sendrows, rb; ndrange = (ns, ncol))
    return nothing
end
