# src/Calculators/PMESlab.jl
#
# z-slab distributed PME reciprocal solve (Phase 1, Issue #193). The
# replicated pipeline keeps the whole charge grid on every rank and
# Allreduces it before a replicated 3D FFT; here the grid is split into
# contiguous z-slabs (dim 3 is the slowest in column-major, so a slab is a
# contiguous range of `vec(grid)`) and the solve becomes
#
#   reduce_scatter → 2D rfft (x,y) → pack → alltoallv (transpose)
#     → 1D fft (z) → k-space multiply → 1D bfft (z) → alltoallv
#     → unpack → 2D brfft (x,y) → allgatherv
#
# The k-space arrays are split along ky (dim 2): the alltoallv reception
# blocks — (nx2, nyloc, zcounts[s]) from each sender s, ordered by rank =
# z order — then land as the contiguous (nx2, nyloc, nz) `khat` with no
# unpack, and the backward send needs no pack for the same reason. Every
# transform leg is unnormalized (rfft/fft forward, bfft/brfft inverse),
# one round trip per dimension, so the total scaling matches the
# replicated `brfft ∘ rfft` exactly — the distributed solve differs from
# the replicated one only in floating-point summation order.
#
# Phase 1 keeps `spread!`/`gather!` on full-size local grids: the
# reduce_scatter consumes the locally spread full grid, and the
# allgatherv redistributes the full energy grid for the unchanged
# `gather!`. Sparse/halo spread-gather is a later phase.
#
# This file provides the machinery only; the `calculate!` integration is
# Issue #193's follow-up PR.

"""
    PMESlab

State of the z-slab distributed reciprocal solve, held in `PMEInfo.dd`
(`nothing` on the replicated path). Built once per run — `fftsize` and
the partition never change under NPT (the plans are fixed; `cellupdate!`
refreshes the k-factors in place). All rank-count tables are indexed by
**comm rank** (0-based rank r ↔ 1-based table entry r+1); the Cartesian
`coords` of `MPIDomain` must not be used here.
"""
struct PMESlab
    comm                        # dom.comm identity guard (nothing in serial force_dd)
    P::Int
    myrank::Int                 # 0-based comm rank
    fftsize::NTuple{3,Int}
    # real-space z-slab partition (uneven: remainder to the leading ranks;
    # zero counts legal when P > nz)
    zcounts::Vector{Int}
    zoffs::Vector{Int}
    # k-space ky partition (same scheme over ny)
    ycounts::Vector{Int}
    yoffs::Vector{Int}
    nzloc::Int
    nyloc::Int
    # collective count tables (element counts, host-resident)
    rs_counts::Vector{Cint}     # nx·ny·zcounts — reduce_scatter / allgatherv
    a2a_scounts::Vector{Cint}   # forward send:    nx2·ycounts[s]·nzloc
    a2a_rcounts::Vector{Cint}   # forward receive: nx2·nyloc·zcounts[s] (backward swaps)
    # device tables for the transpose pack/unpack
    ydest                       # Int32 (ny): global ky → destination rank (1-based)
    ylocidx                     # Int32 (ny): global ky → 0-based local ky at the destination
    # yc/sdisp stay Int64: the pack position arithmetic multiplies them up
    # to ~grid-size magnitudes, which can overflow Int32 on large grids.
    yc_dev                      # Int (P): ycounts
    sdisp_dev                   # Int (P): 0-based element displacement of each send block
    # slab-shaped plans (nothing when the corresponding local count is 0)
    rplan2d                     # rfft (nx, ny, nzloc) dims (1,2)
    irplan2d                    # brfft (nx2, ny, nzloc) dims (1,2)
    zplan                       # in-place fft (nx2, nyloc, nz) dim 3
    izplan                      # in-place bfft (nx2, nyloc, nz) dim 3
    # preallocated buffers (no per-call allocation)
    slabreal                    # (nx, ny, nzloc) FPPREC
    slabhat                     # (nx2, ny, nzloc) Complex
    sendbuf                     # nx2·ny·nzloc Complex (fwd send / bwd receive)
    khat                        # (nx2, nyloc, nz) Complex — Q̂, kept intact for the virial
    kwork                       # (nx2, nyloc, nz) Complex — convolve/backward scratch
    wscratch                    # (nx2, nyloc, nz) FPPREC — virial weight
    gridenergy                  # (nx, ny, nz) FPPREC — allgatherv target, read by gather!
    dummy_r                     # 1-element FPPREC buffer for zero-length MPI arguments
    # Two complex dummies, one per direction: a rank with nzloc == nyloc
    # == 0 substitutes BOTH alltoallv arguments, and the MPI standard
    # forbids aliased send/recv buffers in non-in-place collectives even
    # at zero counts.
    dummy_cs                    # 1-element Complex send-side dummy
    dummy_cr                    # 1-element Complex recv-side dummy
    # ky-slab k-factors (the replicated full-grid factors stay in PMEInfo
    # for the HVP path)
    expfactor_dd                # (nx2, nyloc, nz)
    bsp2factor_dd               # (nx2, nyloc, nz)
    presfactor_dd               # (nx2, nyloc, nz, 6)
end

"""
    _slab_partition(n, P) -> (counts, offs)

Split `n` items over `P` ranks: `n ÷ P` each, remainder to the leading
ranks. Zero counts are legal (P > n). `offs[r+1]` is the 0-based offset
of rank r's segment; `sum(counts) == n`.
"""
function _slab_partition(n::Int, P::Int)
    counts = [n ÷ P + (r <= n % P ? 1 : 0) for r in 1:P]
    offs = cumsum([0; counts[1:end-1]])
    return counts, offs
end

# MPI-facing view of a possibly-empty buffer: MPI must never see a
# zero-length device pointer (implementation-defined), so empty buffers
# are substituted by a 1-element dummy — legal because every count the
# collective reads for this rank is 0 then.
_mpibuf(x, dummy) = length(x) == 0 ? dummy : vec(x)

function _build_pmeslab(backend, fftsize::NTuple{3,Int}, rfftsize::NTuple{3,Int},
                        bsp2_host, P::Int, myrank::Int, comm)
    nx, ny, nz = fftsize
    nx2 = rfftsize[1]
    zcounts, zoffs = _slab_partition(nz, P)
    ycounts, yoffs = _slab_partition(ny, P)
    nzloc = zcounts[myrank+1]
    nyloc = ycounts[myrank+1]

    rs_counts   = Cint.(nx * ny .* zcounts)
    a2a_scounts = Cint.(nx2 .* ycounts .* nzloc)
    a2a_rcounts = Cint.(nx2 * nyloc .* zcounts)

    ydest_h   = zeros(Int32, ny)
    ylocidx_h = zeros(Int32, ny)
    for s in 1:P, j in 1:ycounts[s]
        ydest_h[yoffs[s]+j]   = Int32(s)
        ylocidx_h[yoffs[s]+j] = Int32(j - 1)
    end
    blocksizes = nx2 .* ycounts .* nzloc
    sdisp_h = cumsum([0; blocksizes[1:end-1]])

    slabreal   = KA.zeros(backend, FPPREC, (nx, ny, nzloc))
    slabhat    = KA.zeros(backend, Complex{FPPREC}, (nx2, ny, nzloc))
    sendbuf    = KA.zeros(backend, Complex{FPPREC}, nx2 * ny * nzloc)
    khat       = KA.zeros(backend, Complex{FPPREC}, (nx2, nyloc, nz))
    kwork      = KA.zeros(backend, Complex{FPPREC}, (nx2, nyloc, nz))
    wscratch   = KA.zeros(backend, FPPREC, (nx2, nyloc, nz))
    gridenergy = KA.zeros(backend, FPPREC, fftsize)

    # Plans only where this rank owns data; the transforms are skipped in
    # lockstep-safe fashion (the skip condition is the partition, identical
    # on every rank, never a data value).
    rplan2d  = nzloc > 0 ? AFFTs.plan_rfft(slabreal, (1, 2)) : nothing
    irplan2d = nzloc > 0 ? AFFTs.plan_brfft(slabhat, nx, (1, 2)) : nothing
    zplan    = nyloc > 0 ? AFFTs.plan_fft!(khat, 3) : nothing
    izplan   = nyloc > 0 ? AFFTs.plan_bfft!(kwork, 3) : nothing

    yr = yoffs[myrank+1]+1 : yoffs[myrank+1]+nyloc
    return PMESlab(comm, P, myrank, fftsize,
                   zcounts, zoffs, ycounts, yoffs, nzloc, nyloc,
                   rs_counts, a2a_scounts, a2a_rcounts,
                   _devcopy(backend, ydest_h), _devcopy(backend, ylocidx_h),
                   _devcopy(backend, collect(ycounts)), _devcopy(backend, sdisp_h),
                   rplan2d, irplan2d, zplan, izplan,
                   slabreal, slabhat, sendbuf, khat, kwork, wscratch, gridenergy,
                   KA.zeros(backend, FPPREC, 1),
                   KA.zeros(backend, Complex{FPPREC}, 1),
                   KA.zeros(backend, Complex{FPPREC}, 1),
                   KA.zeros(backend, FPPREC, (nx2, nyloc, nz)),
                   _devcopy(backend, bsp2_host[:, yr, :]),
                   KA.zeros(backend, FPPREC, (nx2, nyloc, nz, 6)))
end

"""
    PMEInfo(cell, params, dom) -> PMEInfo

Domain-aware constructor: on an `MPIDomain` with more than one rank (and
`params[:pme_dd] != false`) the returned object carries the z-slab
distributed solver in `.dd`; otherwise it is exactly `PMEInfo(cell,
params)`. The replicated plans and full-grid factors are always built —
the HVP path stays on them — so `dd` only ever *adds* state.

The `force_dd` hook builds the P = 1 slab machinery on a `SerialDomain`
(collectives degrade to single-segment copies): the distributed transform
becomes unit-testable in the default profile.
"""
function PMEInfo(cell, params, dom::Cores.SerialDomain; force_dd::Bool = false)
    pme = PMEInfo(cell, params)
    force_dd || return pme
    return _pme_with_slab(pme, cell, params, dom, nothing)
end
function PMEInfo(cell, params, dom::Cores.MPIDomain; force_dd::Bool = false)
    pme = PMEInfo(cell, params)
    # Collectivized arm decision: a rank-local early return with the other
    # ranks entering _pme_with_slab's collectives would be an undiagnosed
    # hang, so a disagreeing :pme_dd (or force_dd) across ranks is refused
    # in lockstep instead.
    armed = force_dd || (Cores.nranks(dom) > 1 && get(params, :pme_dd, true))
    all_on = Cores.allreduce_and(dom, armed)
    all_off = Cores.allreduce_and(dom, !armed)
    (all_on || all_off) ||
        error("PMEInfo: pme_dd/force_dd differs across ranks — every rank must " *
              "make the same choice")
    all_on || return pme
    return _pme_with_slab(pme, cell, params, dom, dom.comm)
end

function _pme_with_slab(pme::PMEInfo, cell, params, dom, comm)
    backend = get_backend(cell)
    # The count tables assume one fftsize on every rank; each rank derives
    # it independently from its own cell copy, so a non-replicated cell
    # would silently build inconsistent tables (alltoallv hang or
    # corruption). Cheap once-per-run insurance, lockstep verdict.
    consistent = all(d -> Cores.allreduce_max(dom, pme.fftsize[d]) == pme.fftsize[d] &&
                          -Cores.allreduce_max(dom, -pme.fftsize[d]) == pme.fftsize[d], 1:3)
    Cores.allreduce_and(dom, consistent) ||
        error("PMEInfo: fftsize differs across ranks — the cell is not replicated")
    # bsp2factor is cell-independent: slice the ky range straight off the
    # replicated device array (values identical to a host-side rebuild).
    dd = _build_pmeslab(backend, pme.fftsize, pme.rfftsize, Array(pme.bsp2factor),
                        Cores.nranks(dom), Cores.rankid(dom), comm)
    pme2 = PMEInfo(pme.rplan, pme.irplan, pme.fftsize, pme.rfftsize,
                   pme.expfactor, pme.bsp2factor, pme.presfactor, dd)
    # Fill the slab factors through the ordinary cellupdate! (the
    # replicated factors are refreshed a second time — construction-only
    # cost — so the slab hook has exactly one call site and one
    # rcell/volume derivation).
    cellupdate!(pme2, cell, params)
    return pme2
end

# ky-slab counterpart of `cellupdate!`'s kernel launch: same
# kernel_cellupdate!, launched over the local (nx2, nyloc, nz) shape with
# this rank's global ky offset. `rcell_sm`/`volume`/`alpha` come from the
# caller (cellupdate!, the single call site) so the cell geometry has one
# derivation. Backend from the slab buffers, NOT from a cell array —
# cellupdate! rebinds `cell` to its host copy before this hook runs.
function _cellupdate_slab!(dd::PMESlab, rcell_sm::SMatrix{3,3,FPPREC},
                           volume::FPPREC, alpha::FPPREC)
    backend = get_backend(dd.gridenergy)
    dims = size(dd.expfactor_dd)::NTuple{3,Int}
    n = prod(dims)
    n == 0 && return nothing
    kernel_cellupdate!(backend, 512, n)(
        dd.expfactor_dd, dd.presfactor_dd, rcell_sm, volume, alpha,
        dims, dd.yoffs[dd.myrank+1], dd.fftsize
    )
    return nothing
end

# Transpose pack: slabhat (nx2, ny, nzloc) → sendbuf, destination-major.
# The flat kernel index IS slabhat's linear index; only the write position
# is table-driven. Block layout on the wire is (i, local ky, z) column-
# major, which is exactly the contiguous layout of the receiver's khat
# z-range — hence no unpack on the forward path (and no pack backward).
@kernel inbounds = true function kernel_slab_pack!(buf, @Const(slabhat), @Const(ydest),
                                                   @Const(ylocidx), @Const(yc), @Const(sdisp),
                                                   dims)
    index = @index(Global)
    i0 = (index - 1) % dims[1]
    j0 = ((index - 1) ÷ dims[1]) % dims[2]
    z0 = (index - 1) ÷ (dims[1] * dims[2])
    s = ydest[j0+1]
    pos = sdisp[s] + (z0 * yc[s] + Int(ylocidx[j0+1])) * dims[1] + i0 + 1
    buf[pos] = slabhat[index]
end

# Transpose unpack (backward): mirror of kernel_slab_pack!.
@kernel inbounds = true function kernel_slab_unpack!(slabhat, @Const(buf), @Const(ydest),
                                                     @Const(ylocidx), @Const(yc), @Const(sdisp),
                                                     dims)
    index = @index(Global)
    i0 = (index - 1) % dims[1]
    j0 = ((index - 1) ÷ dims[1]) % dims[2]
    z0 = (index - 1) ÷ (dims[1] * dims[2])
    s = ydest[j0+1]
    pos = sdisp[s] + (z0 * yc[s] + Int(ylocidx[j0+1])) * dims[1] + i0 + 1
    slabhat[index] = buf[pos]
end

# The solver state is bound to the communicator (and its size) it was
# built for; a frame carrying a different domain would read the wrong
# count tables — hang or silent corruption. Lockstep-safe: the verdict is
# derived from construction-time state, identical on every rank.
function _dd_domain_check(dd::PMESlab, dom)
    ok = dd.comm === nothing ? dom isa Cores.SerialDomain :
         (dom isa Cores.MPIDomain && dd.comm == dom.comm)
    ok || error("PMESlab: the slab solver was built for a different domain/communicator")
    return nothing
end

"""
    _pme_forward_dd!(pmeinfo, dom, gridcharge) -> khat

Distributed forward transform: locally spread full `gridcharge` (partial
sums) → reduce-scattered z-slab → 2D rfft (x, y) → transpose → 1D fft
(z). Leaves `Q̂`'s ky-slab in `pmeinfo.dd.khat` (also returned).
`gridcharge` is read, not mutated.
"""
function _pme_forward_dd!(pmeinfo::PMEInfo, dom, gridcharge)
    dd = pmeinfo.dd
    _dd_domain_check(dd, dom)
    backend = get_backend(dd.gridenergy)
    Cores.reduce_scatter_sum!(dom, vec(gridcharge), _mpibuf(dd.slabreal, dd.dummy_r),
                              dd.rs_counts)
    if dd.nzloc > 0
        mul!(dd.slabhat, dd.rplan2d, dd.slabreal)
        kernel_slab_pack!(backend, 512, length(dd.slabhat))(
            dd.sendbuf, dd.slabhat, dd.ydest, dd.ylocidx, dd.yc_dev, dd.sdisp_dev,
            size(dd.slabhat))
    end
    Cores.alltoallv!(dom, _mpibuf(dd.sendbuf, dd.dummy_cs), dd.a2a_scounts,
                     _mpibuf(dd.khat, dd.dummy_cr), dd.a2a_rcounts)
    dd.nyloc > 0 && (dd.zplan * dd.khat)
    return dd.khat
end

"""
    _pme_backward_dd!(pmeinfo, dom) -> gridenergy

Distributed inverse transform of `pmeinfo.dd.kwork` (the convolved
k-grid): 1D bfft (z) → transpose back → 2D brfft (x, y) → allgatherv, so
every rank ends with the full real-space energy grid in
`pmeinfo.dd.gridenergy` (also returned) ready for the unchanged
`gather!`. Destroys `kwork` (in-place bfft) but leaves `khat` intact for
the virial.
"""
function _pme_backward_dd!(pmeinfo::PMEInfo, dom)
    dd = pmeinfo.dd
    backend = get_backend(dd.gridenergy)
    dd.nyloc > 0 && (dd.izplan * dd.kwork)
    Cores.alltoallv!(dom, _mpibuf(dd.kwork, dd.dummy_cs), dd.a2a_rcounts,
                     _mpibuf(dd.sendbuf, dd.dummy_cr), dd.a2a_scounts)
    if dd.nzloc > 0
        kernel_slab_unpack!(backend, 512, length(dd.slabhat))(
            dd.slabhat, dd.sendbuf, dd.ydest, dd.ylocidx, dd.yc_dev, dd.sdisp_dev,
            size(dd.slabhat))
        # brfft destroys its input (FFTW c2r) — slabhat is scratch here.
        mul!(dd.slabreal, dd.irplan2d, dd.slabhat)
    end
    Cores.allgatherv!(dom, _mpibuf(dd.slabreal, dd.dummy_r), vec(dd.gridenergy),
                      dd.rs_counts)
    return dd.gridenergy
end

"""
    _pme_convolve_dd!(pmeinfo, dom, gridcharge) -> gridenergy

Distributed counterpart of [`_pme_convolve`](@ref). Unlike the replicated
pair, the full solve ([`_pme_solve_dd!`](@ref)) *calls* this — the
convolution is structurally one expression, so within one run the shell-
only and full paths produce bit-identical energy grids by construction
(the replicated pair needs a warning and a test to pin that).
"""
function _pme_convolve_dd!(pmeinfo::PMEInfo, dom, gridcharge)
    dd = pmeinfo.dd
    _pme_forward_dd!(pmeinfo, dom, gridcharge)
    @. dd.kwork = dd.khat * dd.expfactor_dd * dd.bsp2factor_dd
    return _pme_backward_dd!(pmeinfo, dom)
end

"""
    _pme_presfactor_contract_dd(dd, w) -> Vector{FPPREC} (Voigt-6, rank-local)

ky-slab counterpart of [`_pme_presfactor_contract`](@ref): the same gemv
over the local k-portion. The result is a rank-local partial sum — the
caller must Allreduce it before treating it as the replicated virial.
"""
function _pme_presfactor_contract_dd(dd::PMESlab, w)
    Mloc = length(w)
    Mloc == 0 && return zeros(FPPREC, 6)   # ky-empty rank (P > ny)
    return Array(reshape(dd.presfactor_dd, Mloc, 6)' * reshape(w, Mloc))::Vector{FPPREC}
end

"""
    _pme_solve_dd!(pmeinfo, dom, gridcharge) -> (gridenergy, virial6)

Distributed counterpart of [`_pme_solve`](@ref): full energy grid on
every rank plus the reciprocal virial as a host Voigt-6 vector,
**already Allreduced** (replicated across ranks, like the return of the
replicated `_pme_solve` under the old grid-Allreduce scheme).
"""
function _pme_solve_dd!(pmeinfo::PMEInfo, dom, gridcharge)
    dd = pmeinfo.dd
    gridenergy = _pme_convolve_dd!(pmeinfo, dom, gridcharge)
    @. dd.wscratch = -dd.bsp2factor_dd * abs2(dd.khat)
    vloc = _pme_presfactor_contract_dd(dd, dd.wscratch)
    Cores.allreduce_sum!(dom, vloc)
    return gridenergy, vloc
end
