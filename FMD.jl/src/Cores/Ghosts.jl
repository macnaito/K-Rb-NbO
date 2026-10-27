using LinearAlgebra
using KernelAbstractions
using Atomix: @atomic
using ..FMD: FPPREC

export Ghosts, build_ghosts, sync_ghosts!, fold_ghosts!, extend_ghosts!

"""
    Ghosts

Materialised periodic images ("ghosts") of the owned atoms, plus the
extended per-atom scratch arrays force kernels read. In every extended
array, rows `1:n_owned` are the owned atoms and rows `n_owned+1:end` the
ghosts; neighbour lists index directly into this extended row space, so
kernels never apply a periodic wrap themselves.

# Fields

- `n_owned` — number of owned atoms `N`.
- `src` — `(G,)` owner row of each ghost (row index into the owned range,
  *not* the `:id` array).
- `shift` — `(N+G, 3)` `Int32` lattice shifts. Rows `1:N` hold the owned
  **wrap offsets** `-floor.(pos)` frozen at build time; rows `N+1:end` the
  ghost **total** shifts (owner wrap offset + image offset). The uniform
  sync rule is `xcart[r] = (pos[owner(r)] + shift[r]) * lattice` with
  `owner(r) = r` for owned rows. Freezing the wrap offset keeps `xcart`
  continuous while an owned atom drifts across a box boundary between
  rebuilds (`:pos` is never wrapped — wrapping would jump the polarization
  gauge used by the dipole/umbrella machinery).
- `xcart` — `(N+G, 3)` Cartesian core positions; refreshed by
  [`sync_ghosts!`](@ref) at every force evaluation (the lattice changes
  every NPT step).
- `sdisp_ext` — `(N+G, 3)` shell displacements (owned values replicated to
  ghost rows); refreshed by [`sync_ghosts!`](@ref).
- `elemid_ext` — `(N+G,)` element ids; fixed at build time.
- `fext` — `(N+G, 3)` accumulation target for scatter-type force kernels
  (a ghost row is folded back onto its owner with [`fold_ghosts!`](@ref)).
  **The scatter user must `fill!(fext, 0)` before its kernel** — neither
  `sync_ghosts!` nor `fold_ghosts!` clears it.
- `vext` — `(N+G, 6)` same for per-atom virial scatter (same zero-clear
  contract).

Under domain decomposition (`domain(atoms) isa MPIDomain`) the ghost rows
come from the halo exchange instead of local image enumeration: `src` is
empty (owners live on other ranks), `dd` carries the frozen communication
plan ([`DDExchange`](@ref)), and rows `N+1:end` of `shift` hold the
accumulated integer shift relative to the owner's *wrapped* coordinate
(diagnostics — the sync path reads `dd.upos`, not `shift`). Force kernels
are oblivious to the difference: they only read
`xcart`/`sdisp_ext`/`elemid_ext`/`fext`/`vext`/`n_owned`.

Built by [`build_ghosts`](@ref) (and internally by the `CellList`
constructor, which additionally cell-sorts the ghost rows). Rebuilt
whenever the neighbour list is rebuilt; between rebuilds only
[`sync_ghosts!`](@ref) runs.
"""
struct Ghosts{TI<:AbstractVector{Int32},TS<:AbstractMatrix{Int32},
              TF<:AbstractMatrix,TE<:AbstractVector,TD}
    n_owned::Int
    n_ghost::Int
    src::TI
    shift::TS
    xcart::TF
    sdisp_ext::TF
    elemid_ext::TE
    fext::TF
    vext::TF
    dd::TD    # Nothing (serial) | DDExchange (domain decomposition)
end

Base.length(g::Ghosts) = g.n_owned + g.n_ghost

"""
    _halo_frac(lattice_cpu, rc_eff) -> NTuple{3}

Fractional halo thickness per lattice direction: `rc_eff` divided by the
perpendicular width of the cell along that direction. An owned atom sees
all neighbours within `rc_eff` iff the ghost shell covers `[-δ, 1+δ)` in
fractional coordinates.
"""
function _halo_frac(lattice_cpu::AbstractMatrix, rc_eff)
    v = abs(det(lattice_cpu))
    w1 = v / norm(cross(lattice_cpu[2, :], lattice_cpu[3, :]))
    w2 = v / norm(cross(lattice_cpu[3, :], lattice_cpu[1, :]))
    w3 = v / norm(cross(lattice_cpu[1, :], lattice_cpu[2, :]))
    return (FPPREC(rc_eff / w1), FPPREC(rc_eff / w2), FPPREC(rc_eff / w3))
end

@kernel inbounds = true function kernel_ghostcount!(gcount, @Const(wpos), δ, H)
    i = @index(Global)
    cnt = Int32(0)
    for sz in -H[3]:H[3], sy in -H[2]:H[2], sx in -H[1]:H[1]
        (sx == 0 && sy == 0 && sz == 0) && continue
        w1 = wpos[i, 1] + sx
        w2 = wpos[i, 2] + sy
        w3 = wpos[i, 3] + sz
        inside = (-δ[1] <= w1 < 1 + δ[1]) & (-δ[2] <= w2 < 1 + δ[2]) & (-δ[3] <= w3 < 1 + δ[3])
        cnt += ifelse(inside, Int32(1), Int32(0))
    end
    gcount[i] = cnt
end

@kernel inbounds = true function kernel_ghostfill!(src, gshift, @Const(offsets), @Const(gcount),
                                                   @Const(wpos), @Const(ws), δ, H)
    i = @index(Global)
    ofs = offsets[i] - gcount[i]   # exclusive prefix → deterministic slots, no atomics
    for sz in -H[3]:H[3], sy in -H[2]:H[2], sx in -H[1]:H[1]
        (sx == 0 && sy == 0 && sz == 0) && continue
        w1 = wpos[i, 1] + sx
        w2 = wpos[i, 2] + sy
        w3 = wpos[i, 3] + sz
        inside = (-δ[1] <= w1 < 1 + δ[1]) & (-δ[2] <= w2 < 1 + δ[2]) & (-δ[3] <= w3 < 1 + δ[3])
        if inside
            ofs += Int32(1)
            src[ofs] = i
            gshift[ofs, 1] = ws[i, 1] + sx
            gshift[ofs, 2] = ws[i, 2] + sy
            gshift[ofs, 3] = ws[i, 3] + sz
        end
    end
end

"""
    build_ghosts(atoms::Atoms, rc_eff) -> Ghosts

Materialise every periodic image relevant within `rc_eff` of the primary
cell. Handles arbitrarily small cells (`rc_eff` larger than the box) by
generating multiple image layers — the image range per direction is
`ceil(δ)` where `δ` is the fractional halo thickness. Ghost rows are in
owner-major order; the `CellList` constructor re-sorts them by cell.

The returned object is **not** synced — call [`sync_ghosts!`](@ref) before
reading `xcart`/`sdisp_ext`.

Under domain decomposition (`domain(atoms) isa MPIDomain`) this dispatches
to the staged halo exchange ([`_build_ghosts_dd`](@ref)) — collective over
the domain communicator; `migrate` must have run first.
"""
function build_ghosts(atoms::Atoms, rc_eff)
    dom = domain(atoms)
    dom isa MPIDomain && return _build_ghosts_dd(atoms, rc_eff, dom)
    backend = get_backend(atoms)
    N = length(atoms)
    pos = atoms.arrays[:pos]

    δ = _halo_frac(Array(atoms.info[:lattice]), rc_eff)
    H = Int32.(ceil.(Int, δ))

    ws = Int32.(.-floor.(pos))          # frozen wrap offsets (owned rows)
    wpos = pos .+ ws                    # wrapped coords in [0,1)

    gcount = KernelAbstractions.zeros(backend, Int32, N)
    kernel_ghostcount!(backend, 256, N)(gcount, wpos, δ, H)
    offsets = accumulate(+, gcount)
    G = Int(sum(gcount))

    src = KernelAbstractions.zeros(backend, Int32, G)
    gshift = KernelAbstractions.zeros(backend, Int32, G, 3)
    kernel_ghostfill!(backend, 256, N)(src, gshift, offsets, gcount, wpos, ws, δ, H)

    return _assemble_ghosts(atoms, ws, src, gshift)
end

# Assemble a Ghosts from the owned wrap offsets + ghost (src, shift) table.
function _assemble_ghosts(atoms::Atoms, ws, src, gshift)
    backend = get_backend(atoms)
    N = length(atoms)
    G = length(src)
    shift = vcat(ws, gshift)
    elemid_ext = vcat(atoms.arrays[:elemid], atoms.arrays[:elemid][src])
    xcart = KernelAbstractions.zeros(backend, FPPREC, N + G, 3)
    sdisp_ext = KernelAbstractions.zeros(backend, FPPREC, N + G, 3)
    fext = KernelAbstractions.zeros(backend, FPPREC, N + G, 3)
    vext = KernelAbstractions.zeros(backend, FPPREC, N + G, 6)
    return Ghosts(N, G, src, shift, xcart, sdisp_ext, elemid_ext, fext, vext, nothing)
end

# Re-order the ghost rows of `g` by permutation `p` (used by the CellList
# constructor to cell-sort ghosts). Owned rows are untouched. In the DD
# case the communication plan's row lists are remapped with the inverse
# permutation — the remap is entirely rank-local (a peer's sendrows are
# row indices on the peer), and the "pack order = row-list order" contract
# is preserved because the lists themselves keep their order.
function _permute_ghosts(g::Ghosts, p)
    N = g.n_owned
    ws = g.shift[1:N, :]
    gshift = g.shift[N+1:end, :][p, :]
    shift = vcat(ws, gshift)
    elemid_ext = vcat(g.elemid_ext[1:N], g.elemid_ext[N+1:end][p])
    if g.dd === nothing
        src = g.src[p]
        return Ghosts(N, g.n_ghost, src, shift, g.xcart, g.sdisp_ext, elemid_ext,
                      g.fext, g.vext, nothing)
    end
    dd = g.dd
    invp = invperm(p)
    remap(row) = row <= N ? Int32(row) : Int32(N + invp[row-N])
    stages = [DDStage(st.dim, st.peer_send, st.peer_recv, st.tag, st.seamshift,
                      Int32[remap(r) for r in st.sendrows],
                      Int32[remap(r) for r in st.recvrows]) for st in dd.stages]
    dd2 = DDExchange(dd.dom, stages, dd.upos[p, :], dd.region[p, :],
                     dd.ghost_id[p], dd.gshift[p, :])
    return Ghosts(N, g.n_ghost, g.src, shift, g.xcart, g.sdisp_ext, elemid_ext,
                  g.fext, g.vext, dd2)
end

"""
    change_arrtyp(g::Ghosts, arrtyp) -> Ghosts

Move the extended arrays (and, under DD, the exchange's `upos`) to
`arrtyp` — the device-enablement step after a host-side DD build. The
plan (stages / region / ghost_id / gshift) stays on the host; the cached
per-stage execution resources start empty and are re-materialized on the
new backend at the first exchange (`_ensure_exec!`).
"""
function change_arrtyp(g::Ghosts, arrtyp)
    dd2 = g.dd === nothing ? nothing :
          DDExchange(g.dd.dom, g.dd.stages, arrtyp(g.dd.upos), g.dd.region,
                     g.dd.ghost_id, g.dd.gshift)
    return Ghosts(g.n_owned, g.n_ghost, arrtyp(g.src), arrtyp(g.shift),
                  arrtyp(g.xcart), arrtyp(g.sdisp_ext), arrtyp(g.elemid_ext),
                  arrtyp(g.fext), arrtyp(g.vext), dd2)
end

@kernel inbounds = true function kernel_sync_xcart!(xcart, @Const(pos), @Const(shift), @Const(src),
                                                    @Const(lattice), n_owned)
    r = @index(Global)
    o = r <= n_owned ? r : Int(src[r-n_owned])
    f1 = pos[o, 1] + shift[r, 1]
    f2 = pos[o, 2] + shift[r, 2]
    f3 = pos[o, 3] + shift[r, 3]
    xcart[r, 1] = f1 * lattice[1, 1] + f2 * lattice[2, 1] + f3 * lattice[3, 1]
    xcart[r, 2] = f1 * lattice[1, 2] + f2 * lattice[2, 2] + f3 * lattice[3, 2]
    xcart[r, 3] = f1 * lattice[1, 3] + f2 * lattice[2, 3] + f3 * lattice[3, 3]
end

"""
    sync_ghosts!(g::Ghosts, atoms::Atoms) -> Ghosts

Refresh the extended scratch arrays from the current `atoms` state:
`xcart[r] = (pos[owner(r)] + shift[r]) * lattice` for every row, and
`sdisp_ext` = shell displacements replicated to ghost rows. Must run at
the start of every force evaluation (positions, shells, and — under NPT —
the lattice all change between evaluations); the ghost *selection* stays
valid under the same skin-/2 criterion as the neighbour list itself.

Under domain decomposition this *is* the MPI forward communication: the
frozen stage plan ships current `u = pos + shift` (owned) / `upos` (ghost,
for forward chains) plus shell displacements to the neighbour ranks.
"""
function sync_ghosts!(g::Ghosts, atoms::Atoms)
    g.dd === nothing || return _sync_ghosts_dd!(g, atoms)
    n = length(g)
    n == 0 && return g
    backend = get_backend(atoms)
    kernel_sync_xcart!(backend, 256, n)(g.xcart, atoms.arrays[:pos], g.shift, g.src,
                                        atoms.info[:lattice], g.n_owned)
    extend_ghosts!(g.sdisp_ext, atoms.arrays[:shelldisp], g)
    return g
end

@kernel inbounds = true function kernel_extend!(ext, @Const(owned), @Const(src), n_owned)
    r, c = @index(Global, NTuple)
    o = r <= n_owned ? r : Int(src[r-n_owned])
    ext[r, c] = owned[o, c]
end

"""
    extend_ghosts!(ext, owned, g::Ghosts) -> ext

Replicate the owned per-atom array `owned` (`(N, c)`) into the extended
array `ext` (`(N+G, c)`): owned rows are copied, ghost rows receive their
owner's values. Used for `sdisp_ext` and for the adjoint `u_F` vectors of
the fitting machinery.
"""
function extend_ghosts!(ext, owned, g::Ghosts)
    g.dd === nothing || return _extend_ghosts_dd!(ext, owned, g)
    length(g) == 0 && return ext
    backend = get_backend(ext)
    kernel_extend!(backend)(ext, owned, g.src, g.n_owned; ndrange = size(ext))
    return ext
end

@kernel inbounds = true function kernel_fold_add!(dst, @Const(ext), @Const(src), n_owned)
    k, c = @index(Global, NTuple)
    @atomic dst[Int(src[k]), c] += ext[n_owned+k, c]
end

"""
    fold_ghosts!(dst, ext, g::Ghosts) -> dst

Fold the extended accumulation array `ext` (`(N+G, c)`) into the owned
array `dst` (`(N, c)`): `dst .+= ext[1:N, :]`, plus every ghost row added
onto its owner (atomically — several ghosts may share an owner). Used
after scatter-type force/virial kernels. `ext` is **not** cleared here —
the caller owns the `fill!(ext, 0)` before its scatter kernel (stale
contents would silently double the forces).

Under domain decomposition this *is* the MPI reverse communication: the
stages run in reverse order and the received contributions are **added**
into `ext` at the original send rows, so ghost-of-ghost chains fold
towards the owner before the final owned-row accumulation.
"""
function fold_ghosts!(dst, ext, g::Ghosts)
    g.dd === nothing || return _fold_ghosts_dd!(dst, ext, g)
    g.n_owned == 0 && return dst
    dst .+= @view ext[1:g.n_owned, :]
    if length(g.src) > 0
        backend = get_backend(dst)
        kernel_fold_add!(backend)(dst, ext, g.src, g.n_owned;
                                  ndrange = (length(g.src), size(dst, 2)))
    end
    return dst
end

include("GhostsDD.jl")
