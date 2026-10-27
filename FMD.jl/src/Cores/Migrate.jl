# src/Cores/Migrate.jl
#
# Rebuild-time atom migration between domain bricks (Phase 1), and the
# initial full-frame → rank-local distribution.  Communication pattern:
# staged x→y→z pairwise exchanges with the ±1 Cartesian neighbours
# (diagonal moves route automatically through consecutive stages).  Runs
# only at neighbour-list rebuilds — the skin/2 rebuild criterion bounds
# how far atoms drift between calls, so a ±1-brick exchange suffices; an
# atom that nevertheless lands outside its target brick is a hard error,
# never a silent misplacement.

using MPI
using ..FMD: FPPREC

export migrate, distribute, gather_frame

"""
    distribute(atoms::Atoms, dom::MPIDomain) -> Atoms

Split a fully replicated frame into this rank's local frame: installs
`dom` into the (shared) `info[:domain]` and keeps the rows whose wrapped
fractional position falls in this rank's brick. `:pos` itself stays
unwrapped — the wrap is an ownership predicate only (polarization gauge).
Collective over `dom.comm`: verifies that every rank holds the same
replicated frame and that the bricks partition it exactly.

Call [`setelemid!`](@ref) on the pristine frame *before* distributing —
`distribute` slices, which stamps the shared `info` with the remap lock
(Issue #70).
"""
function distribute(atoms::Atoms, dom::MPIDomain)
    KernelAbstractions.get_backend(atoms) isa KernelAbstractions.CPU ||
        error("distribute: the split machinery runs on host frames — distribute " *
              "first, then change_arrtyp the local frame (info[:domain] survives " *
              "the conversion; this is what npt! does)")
    length(atoms) == dom.nglobal ||
        error("distribute: frame has $(length(atoms)) rows but dom.nglobal = " *
              "$(dom.nglobal) — pass the full frame, not a slice")
    # Every rank must slice the *same* replicated frame — a rank-dependent
    # frame (different seed, different file) passes the count check below
    # yet corrupts the system. hash of a Float64 matrix is content-based
    # and process-stable (Julia does not randomize hash seeds).
    hpos = hash(atoms.arrays[:pos])
    MPI.Allreduce(hpos, min, dom.comm) == MPI.Allreduce(hpos, max, dom.comm) ||
        error("distribute: the replicated frames differ across ranks — every rank " *
              "must construct or read the identical full frame")
    pos = atoms.arrays[:pos]
    mask = [owns(dom, pos[i, 1], pos[i, 2], pos[i, 3]) for i in axes(pos, 1)]
    nlocal = count(mask)
    total = allreduce_sum(dom, nlocal)
    total == dom.nglobal ||
        error("distribute: bricks did not partition the frame (Σ owned = $total, " *
              "expected $(dom.nglobal)) — ownership predicate inconsistency")
    atoms.info[:domain] = dom
    return atoms[mask]
end

"""
    migrate(atoms::Atoms) -> Atoms

Move atoms that left this rank's brick to the neighbouring owner ranks.
No-op on a [`SerialDomain`](@ref) frame. Call at neighbour-list rebuilds,
immediately before the `CellList` construction, and **replace the frame
with the return value** — the input's row set and any `NeighborList` /
`Ghosts` built from it are stale after migration. The whole per-atom
state travels with the atom because everything lives in `atoms.arrays`
(the registered-arrays rule; keys are exchanged in `sort(keys(...))`
order, verified identical across ranks first). `:pos` is transferred
unwrapped (polarization gauge).

Routing uses the **minimal-image brick offset**: the target brick index is
derived from the same `k/P` expressions as the brick bounds, so a hop
across the periodic seam (last brick → first) is a legitimate `+1` move.
An offset of `|Δ| ≥ 2` bricks cannot be routed by the ±1 staged exchange
and errors on every rank (collectively, so no rank is left blocking in a
later reduction), as does a violated global atom count.
"""
function migrate(atoms::Atoms)
    dom = domain(atoms)
    dom isa SerialDomain && return atoms
    return _migrate(atoms, dom::MPIDomain)
end

function _migrate(atoms::Atoms, dom::MPIDomain)
    KernelAbstractions.get_backend(atoms) isa KernelAbstractions.CPU ||
        error("migrate: the migration machinery runs on host frames — do the " *
              "rebuild round trip change_arrtyp(atoms, Array) → migrate → " *
              "CellList → change_arrtyp back (this is what npt!'s rebuild does)")
    # A per-atom array present on some ranks only — or with a mismatched
    # eltype / column count — would silently skew or garble rows for
    # migrated atoms. Verify the full array signatures agree before any
    # row moves (hash is content-based and process-stable).
    ks = sort!(collect(keys(atoms.arrays)))
    for k in ks
        ndims(atoms.arrays[k]) <= 2 ||
            error("migrate: arrays[:$k] has ndims = $(ndims(atoms.arrays[k])); " *
                  "per-atom arrays must be 1- or 2-D")
    end
    sig = hash([(k, string(eltype(atoms.arrays[k])), ndims(atoms.arrays[k]),
                 ndims(atoms.arrays[k]) == 1 ? 1 : size(atoms.arrays[k], 2))
                for k in ks])
    MPI.Allreduce(sig, min, dom.comm) == MPI.Allreduce(sig, max, dom.comm) ||
        error("migrate: atoms.arrays key sets differ across ranks (name, eltype, " *
              "or column count) — a runtime-added per-atom array (e.g. " *
              ":peratom_virial, :shist1..3) exists on some ranks only; register " *
              "it identically on every rank before migrating")
    for d in 1:3
        dom.dims[d] == 1 && continue   # own_dim is always true along a 1-brick dimension
        atoms = _migrate_dim(atoms, dom, d, ks)
    end
    # Backstop (sender-side routing already errors on |Δ| ≥ 2): every
    # surviving row must be owned. Collective, so a detection on one rank
    # never leaves the others blocking in the count reduction below.
    pos = atoms.arrays[:pos]
    bad = ""
    for i in axes(pos, 1)
        if !owns(dom, pos[i, 1], pos[i, 2], pos[i, 3]) && isempty(bad)
            bad = "atom :id=$(_atomid(atoms, i))"
        end
    end
    allreduce_and(dom, isempty(bad)) ||
        error("migrate: atom(s) landed outside their rank's brick" *
              (isempty(bad) ? " (on another rank)" : " — $bad") *
              " — routing/ownership inconsistency")
    total = allreduce_sum(dom, length(atoms))
    total == dom.nglobal ||
        error("migrate: global atom count changed ($total ≠ $(dom.nglobal)) — " *
              "atoms lost or duplicated")
    return atoms
end

_atomid(atoms::Atoms, i::Int) =
    haskey(atoms.arrays, :id) ? atoms.arrays[:id][i] : i

# Target brick index along one dimension, derived from the same k/P
# expressions as the brick bounds lo/hi — floor(w*P) alone can disagree
# with those bounds by one ulp at a boundary, so bracket-correct against
# FPPREC(k / P) itself.
function _brick_index(w::FPPREC, P::Int)
    k = min(floor(Int, w * P), P - 1)
    if w < FPPREC(k / P)
        k -= 1
    elseif w >= FPPREC((k + 1) / P)
        k += 1
    end
    (0 <= k < P && FPPREC(k / P) <= w && (k + 1 == P || w < FPPREC((k + 1) / P))) ||
        error("_brick_index: inconsistent bracket for w = $w, P = $P")
    return k
end

# One staged exchange along dimension d: rows whose wrapped coordinate
# left the brick go to the ±1 neighbour, with the direction decided by the
# *minimal-image brick offset* — a naive `w < lo → minus / w ≥ hi → plus`
# inverts at the periodic seam (an atom leaving the last brick upward
# wraps to w ≈ 0 < lo, yet its owner is the *plus* neighbour).  The two
# directions use distinct tags — with dims[d] == 2 the minus and plus
# peers are the same rank and the four concurrent messages are
# distinguishable only by tag.
function _migrate_dim(atoms::Atoms, dom::MPIDomain, d::Int, ks::Vector{Symbol})
    pos = atoms.arrays[:pos]
    n = size(pos, 1)
    stay = trues(n)
    tominus = falses(n)
    toplus = falses(n)
    Pd = dom.dims[d]
    c = dom.coords[d]
    far = ""   # first |Δ| ≥ 2 offender, reported collectively below
    for i in 1:n
        w = wrapfrac(pos[i, d])
        Δ = _brick_index(w, Pd) - c
        Δ -= Pd * round(Int, Δ / Pd)   # minimal image, Δ ∈ (-Pd/2, Pd/2]
        if Δ == -1
            stay[i] = false
            tominus[i] = true
        elseif Δ == 1
            stay[i] = false
            toplus[i] = true
        elseif Δ != 0 && isempty(far)
            far = "atom :id=$(_atomid(atoms, i)) moved Δ=$Δ bricks along dim $d"
        end
    end
    # |Δ| ≥ 2 cannot be routed by the ±1 exchange. Collective error before
    # any communication, so no rank is left blocking in a Sendrecv.
    allreduce_and(dom, isempty(far)) ||
        error("migrate: an atom moved more than one brick between rebuilds" *
              (isempty(far) ? " (on another rank)" : " — $far"))

    minus_peer, plus_peer = dom.nbr[d]
    tag_minus = 100 + d   # traffic flowing toward minus
    tag_plus = 200 + d    # traffic flowing toward plus
    # Counts first (the exchange is unconditional — this rank may receive
    # even when it sends nothing).
    nrecv_m = Ref(0)      # arriving from the plus peer (their minus-bound rows)
    nrecv_p = Ref(0)      # arriving from the minus peer (their plus-bound rows)
    MPI.Sendrecv!(Ref(count(tominus)), nrecv_m, dom.comm;
                  dest = minus_peer, sendtag = tag_minus,
                  source = plus_peer, recvtag = tag_minus)
    MPI.Sendrecv!(Ref(count(toplus)), nrecv_p, dom.comm;
                  dest = plus_peer, sendtag = tag_plus,
                  source = minus_peer, recvtag = tag_plus)

    recv_m = Dict{Symbol,Array}()
    recv_p = Dict{Symbol,Array}()
    for key in ks   # deterministic order, identical on every rank
        arr = atoms.arrays[key]
        sm = ndims(arr) == 1 ? arr[tominus] : arr[tominus, :]
        sp = ndims(arr) == 1 ? arr[toplus] : arr[toplus, :]
        rm = ndims(arr) == 1 ? Vector{eltype(arr)}(undef, nrecv_m[]) :
             Matrix{eltype(arr)}(undef, nrecv_m[], size(arr, 2))
        rp = ndims(arr) == 1 ? Vector{eltype(arr)}(undef, nrecv_p[]) :
             Matrix{eltype(arr)}(undef, nrecv_p[], size(arr, 2))
        MPI.Sendrecv!(sm, rm, dom.comm;
                      dest = minus_peer, sendtag = tag_minus,
                      source = plus_peer, recvtag = tag_minus)
        MPI.Sendrecv!(sp, rp, dom.comm;
                      dest = plus_peer, sendtag = tag_plus,
                      source = minus_peer, recvtag = tag_plus)
        recv_m[key] = rm
        recv_p[key] = rp
    end
    return _vcat_rows(atoms[stay], recv_m, recv_p)
end

"""
    gather_frame(atoms::Atoms, dom) -> Atoms | nothing

I/O counterpart of [`distribute`](@ref): collect the rank-local frames
onto the root rank as one full frame with rows sorted by `:id` (the
original file order), for `writexyz` snapshots. Returns the input frame
unchanged on a `SerialDomain`; under DD the root receives the assembled
frame and every other rank receives `nothing`. Collective over
`dom.comm` — every rank must call it whenever any rank does. Device
frames are pulled to the host internally (I/O-bound by nature; the
branch is rank-uniform since every rank shares one backend).
"""
gather_frame(atoms::Atoms, ::SerialDomain) = atoms
function gather_frame(atoms::Atoms, dom::MPIDomain)
    # I/O-bound by nature: device frames are pulled to the host here (every
    # rank has the same backend, so the branch is rank-uniform — collective
    # safety is preserved).
    KernelAbstractions.get_backend(atoms) isa KernelAbstractions.CPU ||
        (atoms = change_arrtyp(atoms, Array))
    comm = dom.comm
    isrootrank = MPI.Comm_rank(comm) == 0
    counts = MPI.Allgather(Int32(length(atoms)), comm)
    ks = sort!(collect(keys(atoms.arrays)))
    sig = hash([(k, string(eltype(atoms.arrays[k])), ndims(atoms.arrays[k]),
                 ndims(atoms.arrays[k]) == 1 ? 1 : size(atoms.arrays[k], 2))
                for k in ks])
    MPI.Allreduce(sig, min, comm) == MPI.Allreduce(sig, max, comm) ||
        error("gather_frame: atoms.arrays signatures differ across ranks")
    newarrays = Dict{Symbol,Array}()
    for k in ks
        arr = atoms.arrays[k]
        ncol = ndims(arr) == 1 ? 1 : size(arr, 2)
        send = ndims(arr) == 1 ? arr : vec(arr)   # column-major block
        if isrootrank
            recv = Vector{eltype(arr)}(undef, Int(sum(counts)) * ncol)
            MPI.Gatherv!(send, MPI.VBuffer(recv, counts .* Int32(ncol)), 0, comm)
            # Reassemble the per-rank column-major blocks.
            parts = Vector{Array}(undef, length(counts))
            ofs = 0
            for (r, c) in enumerate(counts)
                block = recv[ofs+1:ofs+Int(c)*ncol]
                parts[r] = ndims(arr) == 1 ? block : reshape(block, Int(c), ncol)
                ofs += Int(c) * ncol
            end
            newarrays[k] = reduce(vcat, parts)
        else
            MPI.Gatherv!(send, nothing, 0, comm)
        end
    end
    isrootrank || return nothing
    # The same conservation invariant distribute/migrate enforce: a lost
    # or duplicated atom must never leave the building as a silently
    # short snapshot.
    Int(sum(counts)) == dom.nglobal ||
        error("gather_frame: gathered $(Int(sum(counts))) rows but nglobal = " *
              "$(dom.nglobal) — atoms lost or duplicated")
    order = sortperm(newarrays[:id])
    for k in ks
        newarrays[k] = ndims(newarrays[k]) == 1 ? newarrays[k][order] :
                       newarrays[k][order, :]
    end
    newarrays[:id] == collect(1:dom.nglobal) ||
        error("gather_frame: gathered :id set is not a permutation of 1:nglobal")
    # The returned frame is a detached snapshot, NOT a decomposed frame:
    # its info drops :domain (a root-only collective call on it would
    # hang) and is a fresh Dict (no aliasing into the live run).
    info = Dict{Symbol,Any}(k => v for (k, v) in atoms.info if k !== :domain)
    info[:remap_locked] = [true]
    return Atoms(info, newarrays)
end

# Row-concatenate received per-atom arrays onto a frame.  Internal to
# migration — the public API never grows rows.  Shares `info` with the
# input, so it stamps :remap_locked per the Atoms invariant (Issue #70);
# `atoms[stay]` upstream already did, this keeps the helper safe on its
# own.
function _vcat_rows(atoms::Atoms, recvs::Dict{Symbol,Array}...)
    haskey(atoms.info, :remap_locked) || (atoms.info[:remap_locked] = [true])
    newarrays = typeof(atoms.arrays)()
    for key in keys(atoms.arrays)
        newarrays[key] = vcat(atoms.arrays[key], (r[key] for r in recvs)...)
    end
    return Atoms(atoms.info, newarrays)
end
