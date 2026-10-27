using LinearAlgebra
using KernelAbstractions
using Atomix: @atomic
using ..FMD: FPPREC

export neighborlist, NeighborList, isreusable, CellList, sync!

"""
    NeighborList

Ghost-based per-atom neighbour list. `list[i, k]` is the `k`-th neighbour
of owned atom `i` as a row index into the **extended** (owned + ghost)
arrays of `ghosts::Ghosts` (`0` terminates the row). There are no shift
columns: a neighbour that is a periodic image is simply a ghost row whose
`xcart` already carries the shift, so force kernels compute plain
Cartesian differences.

- `saved_positions` — Cartesian core positions captured at construction
  time, used by [`isreusable`](@ref) for the half-skin re-build check.
- `skin` — half-displacement budget; the list (and the ghost *selection*)
  stays valid as long as no core or shell has moved more than `skin/2`.
- `ghosts` — the [`Ghosts`](@ref) this list indexes into. Derived lists
  built with the [`neighborlist(atoms, nl_src, ...)`](@ref) filter share
  the parent's `Ghosts` object.

Call [`sync!`](@ref)`(nl, atoms)` at the start of every force evaluation
to refresh `ghosts.xcart` / `ghosts.sdisp_ext`.
"""
struct NeighborList{TL<:AbstractMatrix{Int32},TS<:AbstractMatrix,TG<:Ghosts}
    list::TL
    saved_positions::TS
    skin::FPPREC
    rc_build::FPPREC   # effective cutoff the list was built with
    ghosts::TG
end

"""
    sync!(nl, atoms)

Refresh the extended position/shell scratch of `nl` from the current
`atoms` state via [`sync_ghosts!`](@ref).
"""
sync!(nl::NeighborList, atoms::Atoms) = (sync_ghosts!(nl.ghosts, atoms); nl)

"""
    CellList{T}

Cell-list bookkeeping over the **extended** domain: interior cells cover
the primary cell `[0,1)³` (fractional), surrounded by `halo[d]` layers of
halo cells holding the ghost rows. Interior and halo cells are disjoint by
construction (a ghost always lies outside `[0,1)` in at least one
direction), so a single `start`/`count` pair per cell suffices: interior
cells point at owned rows `1:N`, halo cells at ghost rows `N+1:N+G`.

- `start[cx, cy, cz]`, `count[cx, cy, cz]` — extended-grid row ranges.
- `ncells` — interior grid dimensions.
- `halo` — halo cell layers per dimension (`= ceil(δ·ncells)`; 1 for the
  usual MD case where the cell side ≥ the cutoff, larger only for cells
  smaller than the cutoff).
- `ghosts` — the [`Ghosts`](@ref) built alongside (ghost rows cell-sorted).

Construct with [`CellList(atoms, cutoff)`](@ref); the constructor re-sorts
the owned atoms by cell and returns `(celllist, sorted_atoms, perm)`.
"""
struct CellList{TA<:AbstractArray,TG<:Ghosts}
    start::TA
    count::TA
    ncells::NTuple{3,Int}
    halo::NTuple{3,Int}
    rc_build::FPPREC
    ghosts::TG
end

"""
    change_arrtyp(cl::CellList, arrtyp) -> CellList

Move the cell tables and the embedded [`Ghosts`](@ref) to `arrtyp` — the
device-enablement step after a host-side DD build: build `CellList` on
the host frame, convert atoms and cell list together, then rebuild the
neighbour list on the device (`neighborlist` is backend-generic and opens
with a `sync_ghosts!`, which runs the device DD exchange). A
`NeighborList` converter is
deliberately absent — rebuilding from the converted `CellList` is the
supported path.
"""
change_arrtyp(cl::CellList, arrtyp) =
    CellList(arrtyp(cl.start), arrtyp(cl.count), cl.ncells, cl.halo, cl.rc_build,
             change_arrtyp(cl.ghosts, arrtyp))

# Cell assignment in the extended grid. The clamp range depends on the row's
# image offset per dimension (s = shift[r] − shift[owner]; 0 for owned rows):
# s == 0 → interior band, s > 0 → upper halo, s < 0 → lower halo. This pins
# the interior/halo disjointness even at the FP edge where a wrapped
# coordinate rounds to exactly 1.0 (an owned atom must never fall into a
# halo cell — the scan kernel only visits interior cells — and a ghost must
# never fall into an interior cell — the merged start/count would hide the
# owned rows there).
@inline function _extcell_dim(w, s, nc::Int32, hc::Int32)
    craw = floor(Int32, w * nc) + Int32(1) + hc
    lo = s > 0 ? hc + nc + Int32(1) : (s < 0 ? Int32(1) : hc + Int32(1))
    hi = s > 0 ? nc + Int32(2) * hc : (s < 0 ? hc : hc + nc)
    return clamp(craw, lo, hi)
end

@kernel inbounds = true function kernel_extcellidx!(cellidx, @Const(fracs), @Const(shift), @Const(src),
                                                    n_owned, ncells, halo, first_row)
    I = @index(Global)
    r = I + first_row - 1
    o = r <= n_owned ? r : Int(src[r-n_owned])
    cx = _extcell_dim(fracs[o, 1] + shift[r, 1], shift[r, 1] - shift[o, 1], ncells[1], halo[1])
    cy = _extcell_dim(fracs[o, 2] + shift[r, 2], shift[r, 2] - shift[o, 2], ncells[2], halo[2])
    cz = _extcell_dim(fracs[o, 3] + shift[r, 3], shift[r, 3] - shift[o, 3], ncells[3], halo[3])
    E1 = ncells[1] + Int32(2) * halo[1]
    E2 = ncells[2] + Int32(2) * halo[2]
    cellidx[I] = cx + (cy - Int32(1)) * E1 + (cz - Int32(1)) * E1 * E2
end

@kernel inbounds = true function kernel_hist32!(hist, @Const(cellidx))
    I = @index(Global)
    @atomic hist[Int(cellidx[I])] += one(eltype(hist))
end

"""
    CellList(atoms::Atoms, cutoff) -> (CellList, Atoms, perm)

Build the extended-domain cell list: bin the owned atoms into interior
cells sized ≥ `cutoff` (re-sorting `atoms` along the cell order — every
entry of `atoms.arrays` is permuted together, so per-atom state must live
there; the returned `perm` exposes the old→new mapping for diagnostics),
materialise the ghosts with [`build_ghosts`](@ref), and cell-sort the
ghost rows into the halo cells. Works for cells smaller than `cutoff` too
(the halo simply grows to several layers).
"""
function CellList(atoms::Atoms, cutoff)
    dom = domain(atoms)
    dom isa MPIDomain && return _celllist_dd(atoms, cutoff, dom)
    backend = get_backend(atoms)
    N = length(atoms)

    lattice_cpu = Array(atoms.info[:lattice])
    δ = _halo_frac(lattice_cpu, cutoff)
    ncells = Tuple(max.(floor.(Int, 1 ./ δ), 1))
    halo = Tuple(ceil.(Int, δ .* ncells))
    extdims = ncells .+ 2 .* halo

    # --- owned: bin (using per-atom wrap offsets), sort, permute atoms ----
    ws = Int32.(.-floor.(atoms.arrays[:pos]))
    cellidx = KernelAbstractions.zeros(backend, Int32, N)
    kernel_extcellidx!(backend, 256, N)(cellidx, atoms.arrays[:pos], ws,
                                        KernelAbstractions.zeros(backend, Int32, 0),
                                        N, Int32.(ncells), Int32.(halo), 1)
    idx = similar(cellidx)
    sortperm!(idx, cellidx)
    cellidx = cellidx[idx]
    atoms = atoms[idx]

    hist_o = KernelAbstractions.zeros(backend, Int32, prod(extdims))
    N > 0 && kernel_hist32!(backend, 256, N)(hist_o, cellidx)
    start_o = accumulate(+, hist_o) .- hist_o .+ Int32(1)

    # --- ghosts: build on the sorted atoms, then cell-sort into halo cells --
    g = build_ghosts(atoms, cutoff)
    G = length(g.src)
    if G > 0
        gcell = KernelAbstractions.zeros(backend, Int32, G)
        kernel_extcellidx!(backend, 256, G)(gcell, atoms.arrays[:pos], g.shift, g.src,
                                            N, Int32.(ncells), Int32.(halo), N + 1)
        gperm = similar(gcell)
        sortperm!(gperm, gcell)
        gcell = gcell[gperm]
        g = _permute_ghosts(g, gperm)

        ghist = KernelAbstractions.zeros(backend, Int32, prod(extdims))
        kernel_hist32!(backend, 256, G)(ghist, gcell)
        start_g = Int32(N) .+ accumulate(+, ghist) .- ghist .+ Int32(1)

        # interior and halo cells are disjoint → merge the two histograms.
        # _extcell_dim pins this by construction; verify it anyway (a
        # violation would silently hide owned rows behind ghost rows).
        if Bool(any((hist_o .> Int32(0)) .& (ghist .> Int32(0))))
            error("CellList: an owned and a ghost row were binned into the same cell " *
                  "(interior/halo disjointness violated)")
        end
        count = hist_o .+ ghist
        start = ifelse.(ghist .> Int32(0), start_g, start_o)
    else
        count = hist_o
        start = start_o
    end

    sync_ghosts!(g, atoms)

    return CellList(reshape(start, extdims), reshape(count, extdims), ncells, halo,
                    FPPREC(cutoff), g), atoms, idx
end

# Brick-relative fractional coordinate: maps this rank's brick to [0,1).
@inline _brickv(u, dom::MPIDomain, d::Int) = (u - dom.lo[d]) * dom.dims[d]

# DD CellList (CPU): the interior grid covers this rank's brick in
# brick-relative coordinates v = (u − lo)·dims; the halo fraction becomes
# δ′ = δ_global · dims. Ghost rows are classified into the halo bands via
# the frozen region signs (the serial shift-difference trick assumes
# ghosts lie outside the *global* box, which no longer holds under DD).
# The same _extcell_dim clamp pins interior/halo disjointness at FP edges.
function _celllist_dd(atoms::Atoms, cutoff, dom::MPIDomain)
    get_backend(atoms) isa KernelAbstractions.CPU ||
        error("CellList (DD): CPU-only in Phase 1 (GPU decomposition is Phase 2)")
    N = length(atoms)
    lattice_cpu = Array(atoms.info[:lattice])
    δg = _halo_frac(lattice_cpu, cutoff)
    δ = ntuple(d -> FPPREC(δg[d] * dom.dims[d]), 3)
    ncells = Tuple(max.(floor.(Int, 1 ./ δ), 1))
    halo = Tuple(ceil.(Int, δ .* ncells))
    extdims = ncells .+ 2 .* halo
    nc = Int32.(ncells)
    hc = Int32.(halo)
    E1 = Int32(extdims[1])
    E2 = Int32(extdims[2])

    # --- owned: bin in brick coordinates, sort, permute atoms -------------
    # _frozen_wrap is the single wrap implementation shared with the DD
    # ghost build — binning and exchange bands must agree bit-for-bit.
    pos = atoms.arrays[:pos]
    ws = _frozen_wrap(pos)
    cellidx = Vector{Int32}(undef, N)
    @inbounds for i in 1:N
        cx = _extcell_dim(_brickv(pos[i, 1] + ws[i, 1], dom, 1), Int32(0), nc[1], hc[1])
        cy = _extcell_dim(_brickv(pos[i, 2] + ws[i, 2], dom, 2), Int32(0), nc[2], hc[2])
        cz = _extcell_dim(_brickv(pos[i, 3] + ws[i, 3], dom, 3), Int32(0), nc[3], hc[3])
        cellidx[i] = cx + (cy - Int32(1)) * E1 + (cz - Int32(1)) * E1 * E2
    end
    idx = sortperm(cellidx)
    cellidx = cellidx[idx]
    atoms = atoms[idx]

    hist_o = zeros(Int32, prod(extdims))
    @inbounds for ci in cellidx
        hist_o[ci] += Int32(1)
    end
    start_o = accumulate(+, hist_o) .- hist_o .+ Int32(1)

    # --- ghosts: staged halo exchange, then cell-sort into halo cells -----
    g = build_ghosts(atoms, cutoff)   # dispatches to the DD build
    G = g.n_ghost
    if G > 0
        dd = g.dd
        gcell = Vector{Int32}(undef, G)
        @inbounds for k in 1:G
            cx = _extcell_dim(_brickv(dd.upos[k, 1], dom, 1), Int32(dd.region[k, 1]), nc[1], hc[1])
            cy = _extcell_dim(_brickv(dd.upos[k, 2], dom, 2), Int32(dd.region[k, 2]), nc[2], hc[2])
            cz = _extcell_dim(_brickv(dd.upos[k, 3], dom, 3), Int32(dd.region[k, 3]), nc[3], hc[3])
            gcell[k] = cx + (cy - Int32(1)) * E1 + (cz - Int32(1)) * E1 * E2
        end
        gperm = sortperm(gcell)
        gcell = gcell[gperm]
        g = _permute_ghosts(g, gperm)

        ghist = zeros(Int32, prod(extdims))
        @inbounds for ci in gcell
            ghist[ci] += Int32(1)
        end
        start_g = Int32(N) .+ accumulate(+, ghist) .- ghist .+ Int32(1)

        # Collective verdict: the sync below is collective, so a lone rank
        # erroring here would leave the others blocking in it.
        disjoint = !any((hist_o .> Int32(0)) .& (ghist .> Int32(0)))
        allreduce_and(dom, disjoint) ||
            error("CellList (DD): an owned and a ghost row were binned into the same " *
                  "cell (interior/halo disjointness violated" *
                  (disjoint ? " on another rank)" : ")"))
        count = hist_o .+ ghist
        start = ifelse.(ghist .> Int32(0), start_g, start_o)
    else
        allreduce_and(dom, true) ||
            error("CellList (DD): interior/halo disjointness violated on another rank")
        count = hist_o
        start = start_o
    end

    sync_ghosts!(g, atoms)

    return CellList(reshape(start, extdims), reshape(count, extdims), ncells, halo,
                    FPPREC(cutoff), g), atoms, idx
end

@kernel inbounds = true function kernel_nbcount!(counter, @Const(xcart), rc, @Const(clstart), @Const(clcount), halo)
    bx, by, bz = @index(Group, NTuple)
    tid = @index(Local, Linear)
    W = prod(@groupsize())
    rc² = rc * rc

    ex = bx + halo[1]
    ey = by + halo[2]
    ez = bz + halo[3]
    start = clstart[ex, ey, ez]
    stop = start + clcount[ex, ey, ez] - 1

    for i in start+(tid-1):W:stop
        xi = xcart[i, 1]
        yi = xcart[i, 2]
        zi = xcart[i, 3]
        cnt = Int32(0)
        for dz in -halo[3]:halo[3], dy in -halo[2]:halo[2], dx in -halo[1]:halo[1]
            cs = clstart[ex+dx, ey+dy, ez+dz]
            ce = cs + clcount[ex+dx, ey+dy, ez+dz] - 1
            for j in cs:ce
                rx = xi - xcart[j, 1]
                ry = yi - xcart[j, 2]
                rz = zi - xcart[j, 3]
                r² = rx * rx + ry * ry + rz * rz
                cnt += ifelse((r² < rc²) & (j != i), Int32(1), Int32(0))
            end
        end
        counter[i] = cnt
    end
end

@kernel inbounds = true function kernel_nblistup!(list, @Const(xcart), rc, @Const(clstart), @Const(clcount), halo)
    bx, by, bz = @index(Group, NTuple)
    tid = @index(Local, Linear)
    W = prod(@groupsize())
    rc² = rc * rc

    ex = bx + halo[1]
    ey = by + halo[2]
    ez = bz + halo[3]
    start = clstart[ex, ey, ez]
    stop = start + clcount[ex, ey, ez] - 1

    for i in start+(tid-1):W:stop
        xi = xcart[i, 1]
        yi = xcart[i, 2]
        zi = xcart[i, 3]
        cnt = 0
        for dz in -halo[3]:halo[3], dy in -halo[2]:halo[2], dx in -halo[1]:halo[1]
            cs = clstart[ex+dx, ey+dy, ez+dz]
            ce = cs + clcount[ex+dx, ey+dy, ez+dz] - 1
            for j in cs:ce
                rx = xi - xcart[j, 1]
                ry = yi - xcart[j, 2]
                rz = zi - xcart[j, 3]
                r² = rx * rx + ry * ry + rz * rz
                if (r² < rc²) & (j != i)
                    list[i, cnt+1] = j
                    cnt += 1
                end
            end
        end
    end
end

"""
    neighborlist(atoms::Atoms, celllist::CellList, cutoff, skinratio) -> NeighborList

Build a ghost-based neighbour list with the effective cutoff
`rc = cutoff·(1 + skinratio)` from a [`CellList`](@ref) built with cells
≥ `rc`. `atoms` must be the re-sorted atoms returned by the `CellList`
constructor. Neighbour entries are extended row indices (owned or ghost);
a smaller-cutoff list can be derived with the
[`neighborlist(atoms, nl_src, ...)`](@ref) filter sharing the same ghosts.
"""
function neighborlist(atoms::Atoms, celllist::CellList, cutoff, skinratio)
    backend = get_backend(atoms)
    # The convert-after-host-build workflow's most likely slip is converting
    # atoms but not the cell list (or vice versa) — fail with a plain
    # message instead of an inscrutable kernel error.
    get_backend(celllist.start) == backend ||
        error("neighborlist: atoms and CellList live on different backends — " *
              "convert both (change_arrtyp) together")
    N = length(atoms)
    rc = FPPREC(cutoff * (1 + skinratio))
    if rc > celllist.rc_build * (1 + sqrt(eps(FPPREC)))
        error("neighborlist: effective cutoff $(rc) exceeds the CellList build cutoff " *
              "$(celllist.rc_build) — the ghost shell and the scan stencil would be too " *
              "small (silent pair loss). Build the CellList with cutoff*(1+skinratio).")
    end
    g = celllist.ghosts
    sync_ghosts!(g, atoms)

    counter = KernelAbstractions.zeros(backend, Int32, N)
    halo = Int32.(celllist.halo)
    kernel_nbcount!(backend, (8, 8, 8), celllist.ncells .* (8, 8, 8))(
        counter, g.xcart, rc, celllist.start, celllist.count, halo)

    maxcount = N > 0 ? Int(reduce(max, counter)) : 0
    list = KernelAbstractions.zeros(backend, Int32, N, maxcount)
    kernel_nblistup!(backend, (8, 8, 8), celllist.ncells .* (8, 8, 8))(
        list, g.xcart, rc, celllist.start, celllist.count, halo)

    return NeighborList(list, atoms.arrays[:pos] * atoms.info[:lattice],
                        cutoff * skinratio, rc, g)
end

"""
    isreusable(nl, atoms::Atoms) -> Bool

True if every core and shell in `atoms` is still within `nl.skin / 2` of
the position captured when `nl` was built (the same criterion also keeps
the ghost *selection* valid — ghost positions themselves are refreshed by
[`sync!`](@ref) at every evaluation). When this returns `false` the caller
must rebuild the cell list, the ghosts, and the neighbour list.
"""
function isreusable(nl::NeighborList, atoms::Atoms)
    pos0 = nl.saved_positions
    dr2max = if length(atoms) == 0
        zero(FPPREC)   # an empty rank still votes in the collective below
    else
        cpos = atoms.arrays[:pos] * atoms.info[:lattice]
        spos = cpos .+ atoms.arrays[:shelldisp]
        FPPREC(max(maximum(sum((cpos .- pos0) .^ 2, dims = 2)),
                   maximum(sum((spos .- pos0) .^ 2, dims = 2))))
    end
    # Collective under DD: one rank exceeding the budget makes every rank
    # rebuild together (the Allreduce result is rank-uniform, so the
    # decision is lockstep — see the AbstractDomain collective invariant).
    dr2max = allreduce_max(domain(atoms), dr2max)
    return sqrt(dr2max) < nl.skin / 2
end
