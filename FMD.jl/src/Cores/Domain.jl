# src/Cores/Domain.jl
#
# Domain layer for MPI domain decomposition (Phase 1).  A domain describes
# which part of the periodic box this process owns and carries the
# communicator for collective reductions.  `SerialDomain` (the default)
# performs zero MPI calls, so serial runs never require `MPI.Init()`;
# fitting's data-parallel MPI never installs a domain either, so it keeps
# running on `SerialDomain` even under mpiexec.

using MPI
import KernelAbstractions
using ..FMD: FPPREC

export SerialDomain, MPIDomain, domain,
       allreduce_sum, allreduce_sum!, allreduce_max, allreduce_and,
       reduce_scatter_sum!, allgatherv!, alltoallv!,
       n_global, isroot, nranks, rankid, probe_device_mpi!

"""
    AbstractDomain

Ownership + communication context for domain decomposition.

# Collective invariant (the deadlock rule)

The MPI standard guarantees that the result of an `Allreduce` is
bit-identical on every rank. The codebase relies on the contract that
**every control-flow branch that ranks must take in lockstep (shell-CG
convergence, `isreusable`, MTTK / lattice updates) is decided only from
values that came out of a collective reduction** — never from rank-local
data. Replicated state (the thermostat/barostat structs, the lattice, CG scalars) is therefore
updated exclusively from Allreduce-derived values and stays bit-identical
across ranks, which is also why the lattice never needs to be exchanged
with the ghost halos. Guarding a collective call with `isroot` (or any
rank-dependent condition) deadlocks — see also the fitting-side rule in
`build_normal_system`.

Two corollaries: (a) never derive a lockstep branch from a rank-local sum
that skipped `allreduce_sum` — the ranks will disagree in FP even when
they "should" agree; (b) an Allreduce result is bit-identical *across
ranks of one run*, not across different process counts — parity tests
between P values must compare with tolerances, never bitwise.
"""
abstract type AbstractDomain end

"""
    SerialDomain()

The no-op fallback domain: this process owns the whole box, every
reduction helper is an identity, and no MPI function is ever called.
"""
struct SerialDomain <: AbstractDomain end

"""
    MPIDomain

Brick ownership on a periodic MPI Cartesian grid. The brick is the
axis-aligned box `[lo, hi)` in **fractional space** `[0,1)³`, so ownership
predicates are invariant under NPT lattice deformation (only halo widths,
which depend on the lattice, change between rebuilds). `hi[d]` of one
brick and `lo[d]` of its neighbour are computed by the same expression
`(c+1)/P_d`, hence bit-equal — the bricks tile `[0,1)³` without gaps or
overlaps.

- `comm`   — Cartesian communicator (all-periodic, non-reordered)
- `dims`   — process grid (`MPI.Dims_create`)
- `coords` — this rank's 0-based grid coordinates
- `nbr[d]` — `(minus, plus)` neighbour ranks along dimension `d`
- `nglobal` — global atom count (fixed during an NPT run)

Degenerate neighbours are the norm, not the exception: `dims[d] == 1`
makes both entries of `nbr[d]` this rank itself (halo exchange degenerates
to a local image copy), and `dims[d] == 2` makes `nbr[d][1] == nbr[d][2]`
(the same peer receives both directions — exchanges must be separated by
message tag or per-direction buffers, never by peer rank alone).

Construct **once per run** and install in `atoms.info[:domain]`: the
Cartesian communicator carries an `MPI_Comm_free` finalizer, and comm
destruction is itself collective — churning MPIDomains makes GC timing a
collective hazard. The domain is geometry-agnostic on purpose; checking
that a brick is thick enough for a cutoff (`rc` + skin vs brick width) is
the responsibility of the halo/ghost construction, which must error on
violation rather than silently drop interactions.
"""
struct MPIDomain <: AbstractDomain
    comm::MPI.Comm
    dims::NTuple{3,Int}
    coords::NTuple{3,Int}
    lo::NTuple{3,FPPREC}
    hi::NTuple{3,FPPREC}
    nbr::NTuple{3,NTuple{2,Int}}
    nglobal::Int
end

"""
    MPIDomain(nglobal; comm = MPI.COMM_WORLD) -> MPIDomain

Build the Cartesian process grid (`MPI.Dims_create` picks the most cubic
factorization) and this rank's fractional-space brick. Collective over
`comm` — every rank must call it.
"""
function MPIDomain(nglobal::Integer; comm::MPI.Comm = MPI.COMM_WORLD)
    MPI.Initialized() ||
        error("MPIDomain: MPI.Init() has not been called (call it before `using FMD` " *
              "in driver scripts, as the test/mpi/_run_*.jl convention does)")
    P = MPI.Comm_size(comm)
    dims = Int.(MPI.Dims_create(P, (0, 0, 0)))
    cart = MPI.Cart_create(comm, dims; periodic = (true, true, true), reorder = false)
    coords_v = Int.(MPI.Cart_coords(cart))
    coords = ntuple(d -> coords_v[d], 3)
    lo = ntuple(d -> FPPREC(coords[d] / dims[d]), 3)
    hi = ntuple(d -> FPPREC((coords[d] + 1) / dims[d]), 3)
    nbr = ntuple(d -> Int.(MPI.Cart_shift(cart, d - 1, 1)), 3)
    # nglobal feeds DOF counts (temperature scale) — a rank-inconsistent
    # value would corrupt the thermostat silently, so verify it here (the
    # constructor is collective anyway).
    ng = Int(nglobal)
    MPI.Allreduce(ng, min, cart) == MPI.Allreduce(ng, max, cart) ||
        error("MPIDomain: nglobal differs across ranks — pass the global atom " *
              "count, not a rank-local one")
    return MPIDomain(cart, ntuple(d -> dims[d], 3), coords, lo, hi, nbr, ng)
end

"""
    domain(atoms::Atoms) -> AbstractDomain

The domain this frame lives on: `atoms.info[:domain]` if installed,
`SerialDomain()` otherwise. `info` is shared between a frame and its
slices, so the domain propagates through the `CellList` re-sort and
migration automatically. `change_arrtyp` keeps the entry host-side (see
`_conv_info`).
"""
domain(atoms::Atoms) = get(atoms.info, :domain, SerialDomain())::AbstractDomain

# Explicit host-side pass-through for change_arrtyp: a Domain must never be
# fed to the device-array constructor. (Without this method the generic
# `eltype(::Type) = Any` fallback happens to take the non-isbits branch —
# an accident we refuse to rely on.)
_conv_info(arrtyp, v::AbstractDomain) = v

"""
    wrapfrac(x) -> w in [0, 1)

Canonical periodic wrap of a fractional coordinate. Single source of truth
for ownership tests, migration, and ghost checks — mixing wrap
implementations across those call sites is how atoms get lost or doubly
owned. The FP edge where `x - floor(x)` rounds to exactly `1.0` (tiny
negative `x`) folds to `0.0`, so `0 ≤ w < 1` always holds and the last
brick's `w < hi = 1.0` test stays exclusive.

**Predicate-only**: the wrapped value is for ownership/halo decisions and
must never be written back to `:pos` — the unwrapped fractional positions
carry the polarization gauge (CLAUDE.md ghost-model rules).
"""
@inline function wrapfrac(x::Real)
    w = x - floor(x)
    return ifelse(w >= one(w), zero(w), w)
end

"""
    own_dim(dom, x, dim) -> Bool

Whether the fractional coordinate `x` falls in this rank's brick along
dimension `dim`. `x` may be unwrapped — [`wrapfrac`](@ref) is applied
internally (idempotent for `x ∈ [0,1)`), so passing raw `:pos` values is
safe: an unwrapped coordinate must never make an atom silently unowned.
`SerialDomain` owns everything.
"""
@inline own_dim(::SerialDomain, x::Real, dim::Integer) = true
@inline function own_dim(d::MPIDomain, x::Real, dim::Integer)
    w = wrapfrac(x)
    return d.lo[dim] <= w < d.hi[dim]
end

"""
    owns(dom, x1, x2, x3) -> Bool

All-dimension ownership test on fractional coordinates (wrapped
internally; raw `:pos` values are safe). When calling per-atom in a loop,
hoist `dom = domain(atoms)` through a function barrier first — `domain`
returns the abstract type and per-call dynamic dispatch would dominate.
"""
owns(d::AbstractDomain, x1::Real, x2::Real, x3::Real) =
    own_dim(d, x1, 1) && own_dim(d, x2, 2) && own_dim(d, x3, 3)

# ---------------------------------------------------------------------------
#  Collective reduction helpers.  SerialDomain versions are identities so
#  the same call sites serve both execution modes; see the AbstractDomain
#  docstring for the collective invariant these establish.
# ---------------------------------------------------------------------------

# One-shot advisory when device buffers first hit MPI (see the device
# transport contract in `allreduce_sum!`).
const _DEVICE_MPI_WARNED = Ref(false)

# Device-transport preparation, centralized so call sites cannot forget
# it: MPI takes a raw pointer and reads it immediately, while KA kernel
# launches are asynchronous — the device stream must be drained first.
# Host arrays return without touching MPI.has_cuda() or the stream, so
# the Phase 1 host call sites stay byte-identical.
#
# Assumption: the buffer was written by kernels on THIS task's stream
# (KA's CUDA synchronize drains the task-local stream). Off-loading pack
# kernels to another task would silently void the drain — don't.
function _device_mpi_prepare!(x::AbstractArray)
    backend = KernelAbstractions.get_backend(x)
    backend isa KernelAbstractions.CPU && return nothing
    if !_DEVICE_MPI_WARNED[]
        _DEVICE_MPI_WARNED[] = true
        MPI.has_cuda() ||
            @warn "Passing device buffers to MPI, but MPI.has_cuda() == false. " *
                  "MPICH-ABI CUDA-aware builds cannot be detected, so this may be " *
                  "fine — but if the process crashes inside MPI, the library is " *
                  "not CUDA-aware (GPU domain decomposition requires one; see " *
                  "docs/src/hpc.md). Set JULIA_MPI_HAS_CUDA=true to silence this " *
                  "on a known CUDA-aware build."
    end
    KernelAbstractions.synchronize(backend)
    return nothing
end

"""
    allreduce_sum!(dom, x::DenseArray) -> x

In-place elementwise global sum. The `DenseArray` restriction is
deliberate: `MPI.Allreduce!` takes a raw pointer, so a non-contiguous view
would be reduced silently wrong — pass the parent array (`vec` of an
`Array`/`CuArray` is fine, it stays dense).

# Device transport contract (GPU domain decomposition)

Device arrays are handed to MPI **directly** and require a CUDA-aware MPI
build. There is deliberately no host-staging fallback: it would be a
silent order-of-magnitude slow path that an unsuspecting user could run
production on. The helper drains the device stream first (KA launches are
asynchronous; MPI reads the raw pointer immediately) and warns once per
process if `MPI.has_cuda()` cannot confirm CUDA support. Any new code
that passes device memory to MPI must route through the Domain helpers —
or through `_dd_transfer!` (GhostsDD.jl), which applies the same drain to
the ghost-exchange point-to-point path — so the synchronize cannot be
forgotten. Call
[`probe_device_mpi!`](@ref) at run entry to fail fast — and
self-diagnosingly — on a non-CUDA-aware build.
"""
allreduce_sum!(::SerialDomain, x::DenseArray) = x
function allreduce_sum!(d::MPIDomain, x::DenseArray)
    _device_mpi_prepare!(x)
    MPI.Allreduce!(x, +, d.comm)
    return x
end

"""
    allreduce_sum(dom, x::Number) -> global sum of x
"""
allreduce_sum(::SerialDomain, x::Number) = x
allreduce_sum(d::MPIDomain, x::Number) = MPI.Allreduce(x, +, d.comm)

"""
    allreduce_max(dom, x::Real) -> global max of x
"""
allreduce_max(::SerialDomain, x::Real) = x
allreduce_max(d::MPIDomain, x::Real) = MPI.Allreduce(x, max, d.comm)

"""
    allreduce_and(dom, x::Bool) -> global logical AND of x

The collective rebuild decision (`isreusable`) is the canonical user: one
rank voting `false` makes every rank rebuild together.
"""
allreduce_and(::SerialDomain, x::Bool) = x
allreduce_and(d::MPIDomain, x::Bool) = MPI.Allreduce(x, &, d.comm)

# ---------------------------------------------------------------------------
# Segmented collectives (slab-decomposed PME, Issue #191).  All three follow
# the same contract as `allreduce_sum!`: `DenseArray` only (MPI reads raw
# pointers, a view would be silently wrong), device buffers go to MPI
# directly after a stream drain, `counts` live on the HOST (`Vector{Cint}`,
# MPI reads them from host memory regardless of buffer residency), and
# every rank participates even with a zero count — empty segments must
# still ride the collective (the empty-rank invariant of the MPI tests).
# Send and receive buffers must share one residency (both host or both
# device): only the send side is stream-drained, and a mixed pair would
# also take different MPI code paths per direction.
# Segment r of a buffer is the contiguous range `sum(counts[1:r]) .+
# (1:counts[r+1])` (0-based rank r), which for a z-slab of a column-major
# grid is exactly the slab — no packing needed.

"""
    reduce_scatter_sum!(dom, sendbuf, recvbuf, counts) -> recvbuf

Elementwise global sum of `sendbuf` (length `sum(counts)`) with segment
`r` of the result scattered to rank `r`'s `recvbuf` (first `counts[r+1]`
elements). The fused equivalent of `allreduce_sum!` followed by keeping
only the own segment, at roughly half the traffic. In serial this is a
plain copy of the single segment.

Goes through the raw `MPI.API.MPI_Reduce_scatter` binding: MPI.jl 0.20
has no high-level wrapper for it. The binding carries `@mpichk` (errors
raise) and the ccall converts `Array`/`CuArray` through the same `MPIPtr`
machinery as the wrapped collectives, so the device-transport contract of
[`allreduce_sum!`](@ref) applies unchanged.
"""
function reduce_scatter_sum!(::SerialDomain, sendbuf::DenseArray{T}, recvbuf::DenseArray{T},
                             counts::Vector{Cint}) where {T}
    # A P-rank count table on the serial path would silently return the
    # first segment as a "result" — refuse instead.
    length(counts) == 1 ||
        error("reduce_scatter_sum!: SerialDomain expects a single-segment " *
              "counts table, got length $(length(counts))")
    copyto!(recvbuf, 1, sendbuf, 1, Int(counts[1]))
    return recvbuf
end
function reduce_scatter_sum!(d::MPIDomain, sendbuf::DenseArray{T}, recvbuf::DenseArray{T},
                             counts::Vector{Cint}) where {T}
    # Raw-API call: MPI.jl's high-level asserts do not apply here, so the
    # bounds are checked by hand — an undersized recvbuf would be an
    # out-of-range WRITE (heap/device corruption), not an error.
    P = MPI.Comm_size(d.comm)
    length(counts) == P ||
        error("reduce_scatter_sum!: length(counts) = $(length(counts)) ≠ P = $P")
    length(sendbuf) >= sum(Int, counts) ||
        error("reduce_scatter_sum!: sendbuf ($(length(sendbuf))) shorter than " *
              "sum(counts) ($(sum(Int, counts)))")
    length(recvbuf) >= Int(counts[MPI.Comm_rank(d.comm)+1]) ||
        error("reduce_scatter_sum!: recvbuf ($(length(recvbuf))) shorter than " *
              "the own segment ($(counts[MPI.Comm_rank(d.comm)+1]))")
    _device_mpi_prepare!(sendbuf)
    MPI.API.MPI_Reduce_scatter(sendbuf, recvbuf, counts, MPI.Datatype(T), MPI.SUM, d.comm)
    return recvbuf
end

"""
    allgatherv!(dom, sendbuf, recvbuf, counts) -> recvbuf

Concatenate every rank's `sendbuf` (first `counts[rank+1]` elements) into
`recvbuf` (length `sum(counts)`) on all ranks, segment `r` at the offset
`sum(counts[1:r])`. Inverse of the scatter half of
[`reduce_scatter_sum!`](@ref). In serial this is a plain copy of the
single segment.
"""
function allgatherv!(::SerialDomain, sendbuf::DenseArray{T}, recvbuf::DenseArray{T},
                     counts::Vector{Cint}) where {T}
    length(counts) == 1 ||
        error("allgatherv!: SerialDomain expects a single-segment counts " *
              "table, got length $(length(counts))")
    copyto!(recvbuf, 1, sendbuf, 1, Int(counts[1]))
    return recvbuf
end
function allgatherv!(d::MPIDomain, sendbuf::DenseArray{T}, recvbuf::DenseArray{T},
                     counts::Vector{Cint}) where {T}
    # The explicit-count Buffer below (needed so padding past the true
    # count is not transmitted) skips VBuffer's length assert — restore it
    # by hand. counts/recvbuf sizing is asserted by MPI.jl (VBuffer +
    # Allgatherv!).
    n = Int(counts[MPI.Comm_rank(d.comm)+1])
    length(sendbuf) >= n ||
        error("allgatherv!: sendbuf ($(length(sendbuf))) shorter than the " *
              "own count ($n)")
    _device_mpi_prepare!(sendbuf)
    MPI.Allgatherv!(MPI.Buffer(sendbuf, Cint(n), MPI.Datatype(T)),
                    MPI.VBuffer(recvbuf, counts), d.comm)
    return recvbuf
end

"""
    alltoallv!(dom, sendbuf, scounts, recvbuf, rcounts) -> recvbuf

Personalized exchange: segment `r` of `sendbuf` (layout as in
[`reduce_scatter_sum!`](@ref), counts `scounts`) goes to rank `r`, which
receives it as its segment of `recvbuf` (counts `rcounts`). The transpose
of the distributed FFT. The count tables must be transpose-consistent:
rank `r`'s `scounts[q+1]` must equal rank `q`'s `rcounts[r+1]` — MPI
checks nothing here, a mismatch hangs or silently corrupts. In serial
this is a plain copy of the single segment (`scounts[1] == rcounts[1]`
then, which IS checked).
"""
function alltoallv!(::SerialDomain, sendbuf::DenseArray{T}, scounts::Vector{Cint},
                    recvbuf::DenseArray{T}, rcounts::Vector{Cint}) where {T}
    length(scounts) == 1 && length(rcounts) == 1 ||
        error("alltoallv!: SerialDomain expects single-segment count " *
              "tables, got lengths $(length(scounts))/$(length(rcounts))")
    scounts[1] == rcounts[1] ||
        error("alltoallv!: serial self-exchange with scounts[1] = " *
              "$(scounts[1]) ≠ rcounts[1] = $(rcounts[1])")
    copyto!(recvbuf, 1, sendbuf, 1, Int(scounts[1]))
    return recvbuf
end
function alltoallv!(d::MPIDomain, sendbuf::DenseArray{T}, scounts::Vector{Cint},
                    recvbuf::DenseArray{T}, rcounts::Vector{Cint}) where {T}
    _device_mpi_prepare!(sendbuf)
    MPI.Alltoallv!(MPI.VBuffer(sendbuf, scounts), MPI.VBuffer(recvbuf, rcounts), d.comm)
    return recvbuf
end

"""
    n_global(dom, atoms) -> Int

Global atom count: `length(atoms)` in serial, the frozen `nglobal` under
decomposition (where `length(atoms)` is the rank-local owned count). Pass
the full local frame — a `getindex` slice would silently change the
serial answer while the MPI one ignores the argument.
"""
n_global(::SerialDomain, atoms) = length(atoms)
n_global(d::MPIDomain, _) = d.nglobal

"""
    nranks(dom) -> Int
    rankid(dom) -> Int

Communicator size and 0-based rank of this process (`1` / `0` in serial).
Accessors so that consumers of rank-indexed partition tables (the slab
PME) never need a direct MPI dependency. Rank identity is the **comm
rank** — the Cartesian `coords` are related by a row-major ordering and
must not be used to index rank-count tables.
"""
nranks(::SerialDomain) = 1
nranks(d::MPIDomain) = MPI.Comm_size(d.comm)
rankid(::SerialDomain) = 0
rankid(d::MPIDomain) = MPI.Comm_rank(d.comm)

"""
    isroot(dom) -> Bool

`true` on the designated I/O rank (rank 0; always `true` in serial). For
I/O gating only — never guard a collective call with it (see
[`AbstractDomain`](@ref)).
"""
isroot(::SerialDomain) = true
isroot(d::MPIDomain) = MPI.Comm_rank(d.comm) == 0

"""
    probe_device_mpi!(dom, backend) -> nothing

Fail-fast self-diagnosis of the device MPI transport, to be called once at
GPU domain-decomposition entry. Announces itself on the root rank, then
Allreduces a 1-element device array: on a non-CUDA-aware MPI build the
process crashes right here — immediately after the announcement, which
makes the otherwise inscrutable segfault self-explaining — instead of at
some random exchange mid-run. A transport that corrupts data instead of
crashing is caught by the exact value check (a sum of ones over `P` ranks
is exact in floating point). No-op for `SerialDomain` and for CPU
backends (host MPI needs no probe). Collective — every rank must call it.
"""
function probe_device_mpi!(dom::MPIDomain, backend::KernelAbstractions.Backend)
    backend isa KernelAbstractions.CPU && return nothing
    isroot(dom) &&
        @info "Probing CUDA-aware MPI (a crash inside these collectives means " *
              "the MPI library is not CUDA-aware — GPU domain decomposition " *
              "requires one, see docs/src/hpc.md)" mpi_has_cuda = MPI.has_cuda()
    P = MPI.Comm_size(dom.comm)
    x = KernelAbstractions.ones(backend, FPPREC, 1)
    allreduce_sum!(dom, x)
    got = Array(x)[1]
    expected = FPPREC(P)
    # The segmented collectives (slab-decomposed PME) can be host-staged or
    # broken on device buffers even when Allreduce is CUDA-aware — OpenMPI
    # implements them in separate components — so each is probed with an
    # exact-value transfer. This is a transport probe, not a routing test:
    # segment-routing correctness lives in the "collectives" MPI check.
    onesP  = KernelAbstractions.ones(backend, FPPREC, P)
    recv1  = KernelAbstractions.zeros(backend, FPPREC, 1)
    recvP  = KernelAbstractions.zeros(backend, FPPREC, P)
    counts = ones(Cint, P)
    reduce_scatter_sum!(dom, onesP, recv1, counts)         # own segment = Σ_ranks 1 = P
    ok_rs = Array(recv1)[1] == expected
    alltoallv!(dom, onesP, counts, recvP, counts)          # 1 from each rank
    ok_a2a = all(==(one(FPPREC)), Array(recvP))
    fill!(recvP, 0)   # else at P = 1 a no-op allgatherv would pass on stale data
    allgatherv!(dom, recv1, recvP, counts)                 # everyone's P
    ok_ag = all(==(expected), Array(recvP))
    # Lockstep verdict: if the transport corrupted data on only some ranks,
    # a rank-local error would leave the healthy ranks hanging in their
    # next collective — make every rank fail together instead.
    allreduce_and(dom, got == expected && ok_rs && ok_a2a && ok_ag) ||
        error("probe_device_mpi!: device collectives returned corrupted data " *
              "(allreduce=$got expected=$expected, reduce_scatter=$ok_rs, " *
              "alltoallv=$ok_a2a, allgatherv=$ok_ag)")
    isroot(dom) && @info "CUDA-aware MPI probe passed (incl. reduce_scatter/alltoallv/allgatherv)"
    return nothing
end
probe_device_mpi!(::SerialDomain, backend::KernelAbstractions.Backend) = nothing
