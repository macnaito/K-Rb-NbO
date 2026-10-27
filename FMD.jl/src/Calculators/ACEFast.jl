# ACEFast.jl — production evaluator for ACEParam, split into three stages:
#
#   Kernel A  (features)    one thread per atom: the neighbour loop builds the
#                           density-trick features S/V/M/N in species-sorted,
#                           component-interleaved GEMM layout.
#   Stage B   (contraction) energies and adjoints dE/d{S,V,M,N} as per-species
#                           GEMMs (`mul!` → BLAS / cuBLAS, no CUDA dependency)
#                           plus small atom×channel / atom×pair kernels. This
#                           holds the dominant O(nch0·P) coupling contractions
#                           that the single-kernel predecessor ran serially
#                           inside each atom's thread.
#   Kernel C  (forces)      one thread per atom: the neighbour loop turns the
#                           adjoints into forces/virials, scattered with
#                           Newton's third law (@atomic to atom and neighbour).
#
# The split is mathematically exact w.r.t. the reference implementation in
# ACE.jl — only the floating-point summation order changes (plus the exact
# c/2 symmetrisation described in `_preprocess_ace_coefficients`). Parity is
# enforced by test/calculators/test_ace.jl (CPU backend) and
# test/gpu/test_ace_gpu.jl (CUDA backend).

using KernelAbstractions
using Atomix: @atomic
using LinearAlgebra: mul!

export calculate!, ACEWorkspace

# ------------------------------------------------------------
# Persistent workspace: preprocessed coefficients + buffers
# ------------------------------------------------------------

"""
    ACEWorkspace(backend, acep::ACEParam, Nat) -> ACEWorkspace

Preprocessed coefficients and preallocated buffers for [`calculate!`](@ref).
Sizes depend only on `Nat` and the channel dimensions of `acep`, both fixed
during an MD run, so one workspace is built once (see `build_mlip_workspace`
in NPT.jl) and reused every step.

Coefficients are **copied and preprocessed at construction** (symmetrised
quadratic forms, pair-flattened 4-body tensors) in `FPPREC`, reading only
the consumed triangles. Build from the host-side Float64 `ACEParam` whenever
possible — a `cu`-converted copy has already been downcast to Float32
(Issue #39). If the coefficients of `acep` are mutated afterwards, the
workspace is stale: rebuild it (the 2-/3-arg `calculate!` methods build a
fresh temporary one per call, so the fitting path is always consistent).
A fingerprint check in `calculate!` (Σ|c| over the fully-consumed `cp`/`c0`
fields) catches wholesale mutation such as `setval!` during fitting;
mutating only a higher-order tensor is not detected.

The species-sorted row order is cached and invalidated by **object
identity** of `atoms.arrays[:elemid]` — replacing the array (as the cell
list rebuild in NPT does via `atoms[idx]`) triggers a re-sort; in-place
mutation of the same array is NOT detected.

Deliberately a separate struct: `ACEParam` fields are serialised by
`vec`/`setval!` for the MPI CG fit, so the workspace must never live there.

Memory: ≈ `8·(3·(nch0+3·nch1+6·nch2+10·nch3) + 2·(P1+P2+P3))` bytes/atom
with `P_l = nch_l(nch_l+1)/2` — e.g. ~12 KB/atom (~1.2 GB at 10⁵ atoms) for
`ns=3, nk=(12,6,4,4)`.
"""
mutable struct ACEWorkspace{D <: NamedTuple, C <: NamedTuple, B <: NamedTuple,
                            Vi <: AbstractVector{Int32}}
    const dims::D    # (; Nat, ns, nch0..nch3, P1..P3)
    const rmin::FPPREC
    const rmax::FPPREC
    const coefabs::Float64  # Σ|c| fingerprint of the source coefficients
    const coef::C    # preprocessed device coefficients (FPPREC) + pair tables
    const buf::B     # features/adjoints/intermediates, GEMM layout
    # --- species-sort cache ---
    elemid_ref::Any  # elemid array the cache was built from (=== check)
    const rowrank::Vi        # atom I → row r
    const elemid_sorted::Vi  # species of row r
    const rowlo::Vector{Int}  # species s → first row (rowlo > rowhi: no atoms)
    const rowhi::Vector{Int}
end

# Heuristic staleness fingerprint: Σ|c| over cp and c0 — the two coefficient
# fields whose every element is consumed. The higher-order tensors cannot be
# fingerprinted wholesale: their unused triangles are garbage by contract
# (may even be NaN, see the NaN-injection parity test). Wholesale mutation
# (`setval!` during fitting) always touches cp/c0 and is caught; a manual
# edit of only a higher-order tensor is not — the snapshot contract in the
# ACEWorkspace docstring is authoritative. Tolerant of the Float32 rounding
# of a `cu`-converted copy (rtol 1e-5 at the check site).
_coefabs(acep::ACEParam) = Float64(sum(abs, acep.cp)) + Float64(sum(abs, acep.c0))

_pairindex(hi::Integer, lo::Integer) = (hi - 1) * hi ÷ 2 + lo   # hi ≥ lo ≥ 1

"""
    _ace_pairtables(nch, P) -> (hi, lo)

Packed-pair index tables for the `P = nch(nch+1)÷2` pairs `(ν ≥ λ)`:
`hi[_pairindex(ν, λ)] = ν`, `lo[_pairindex(ν, λ)] = λ`. Shared by the
ACEFast coefficient preprocessing and the blockchol-si preconditioner
assembly (ACEPrecond.jl) so the packing cannot drift between the two.
"""
function _ace_pairtables(nch, P)
    hi = zeros(Int32, P); lo = zeros(Int32, P)
    for ν in 1:nch, λ in 1:ν
        hi[_pairindex(ν, λ)] = ν
        lo[_pairindex(ν, λ)] = λ
    end
    return (hi, lo)
end

function _preprocess_ace_coefficients(acep::ACEParam, dims)
    ns   = dims.ns
    nch0 = dims.nch0

    # E = Σ_{μ≤ν} c[μ,ν]·Xμ⋅Xν  ⟺  E = X·Ĉ·X with Ĉ diagonal = c and
    # off-diagonal = c/2 (a power-of-two scaling: exact). The adjoint then
    # is dE/dX = 2·Ĉ·X, which reproduces the reference's double
    # accumulation on diagonal pairs. Only the upper triangle μ ≤ ν is
    # read — the strict lower triangle is never consumed by the reference
    # and may hold garbage.
    symh(craw, nch) = begin
        c = Array(craw)
        out = zeros(FPPREC, nch, nch, ns)
        for s in 1:ns, ν in 1:nch, μ in 1:ν
            if μ == ν
                out[μ, μ, s] = FPPREC(c[μ, μ, s])
            else
                v = FPPREC(c[μ, ν, s] / 2)
                out[μ, ν, s] = v
                out[ν, μ, s] = v
            end
        end
        out
    end

    # 4-body tensors flattened to (nch0 × P) per species, columns indexed by
    # the pair p(ν≥λ). Only the ν ≥ λ side is read (same convention as the
    # reference; the ν < λ side may hold garbage).
    pairh(craw, nch, P) = begin
        c = Array(craw)
        out = zeros(FPPREC, nch0, P, ns)
        for s in 1:ns, ν in 1:nch, λ in 1:ν, μ in 1:nch0
            out[μ, _pairindex(ν, λ), s] = FPPREC(c[μ, ν, λ, s])
        end
        out
    end

    hi1, lo1 = _ace_pairtables(dims.nch1, dims.P1)
    hi2, lo2 = _ace_pairtables(dims.nch2, dims.P2)
    hi3, lo3 = _ace_pairtables(dims.nch3, dims.P3)

    return (;
        cp = FPPREC.(Array(acep.cp)),
        c0 = FPPREC.(Array(acep.c0)),
        C00h = symh(acep.c00, nch0),
        C11h = symh(acep.c11, dims.nch1),
        C22h = symh(acep.c22, dims.nch2),
        C33h = symh(acep.c33, dims.nch3),
        C011p = pairh(acep.c011, dims.nch1, dims.P1),
        C022p = pairh(acep.c022, dims.nch2, dims.P2),
        C033p = pairh(acep.c033, dims.nch3, dims.P3),
        hi1, lo1, hi2, lo2, hi3, lo3,
        # component weights: e3/e6 are the "no weight" vectors; w2 carries
        # the packed-STF2 trace multiplicities, w3 the STF3 multiplicities.
        e3 = ones(FPPREC, 3),
        e6 = ones(FPPREC, 6),
        w2 = FPPREC[1, 1, 1, 2, 2, 2],
        w3 = FPPREC[stf3_w(FPPREC(k)) for k in 1:10],
    )
end

function ACEWorkspace(backend, acep::ACEParam, Nat::Integer)
    ns   = length(acep.cp)
    nch0 = size(acep.c0, 1)
    nch1 = size(acep.c11, 1)
    nch2 = size(acep.c22, 1)
    nch3 = size(acep.c33, 1)
    dims = (; Nat = Int(Nat), ns, nch0, nch1, nch2, nch3,
            P1 = nch1 * (nch1 + 1) ÷ 2,
            P2 = nch2 * (nch2 + 1) ÷ 2,
            P3 = nch3 * (nch3 + 1) ÷ 2)

    dev(x) = begin
        y = KernelAbstractions.allocate(backend, eltype(x), size(x)...)
        copyto!(y, x)
        y
    end
    coef = map(dev, _preprocess_ace_coefficients(acep, dims))

    z(sz...) = KernelAbstractions.zeros(backend, FPPREC, sz...)
    n = dims.Nat
    buf = (;
        S    = z(n, nch0),   V    = z(3n, nch1),
        M    = z(6n, nch2),  N    = z(10n, nch3),
        dEdS = z(n, nch0),   dEdV = z(3n, nch1),
        dEdM = z(6n, nch2),  dEdN = z(10n, nch3),
        G0   = z(n, nch0),   G1   = z(3n, nch1),
        G2   = z(6n, nch2),  G3   = z(10n, nch3),
        T1 = z(n, dims.P1), B1 = z(n, dims.P1),
        T2 = z(n, dims.P2), B2 = z(n, dims.P2),
        T3 = z(n, dims.P3), B3 = z(n, dims.P3),
        pe = z(n), vir = z(n, 6),
    )

    zi(sz...) = KernelAbstractions.zeros(backend, Int32, sz...)
    return ACEWorkspace(dims, FPPREC(acep.rmin), FPPREC(acep.rmax), _coefabs(acep),
                        coef, buf, nothing, zi(n), zi(n), zeros(Int, ns), zeros(Int, ns))
end

function _check_workspace(ws::ACEWorkspace, acep::ACEParam, atoms::Atoms)
    d = ws.dims
    Nat = length(atoms)
    ok = d.Nat == Nat && d.ns == length(acep.cp) &&
         d.nch0 == size(acep.c0, 1) && d.nch1 == size(acep.c11, 1) &&
         d.nch2 == size(acep.c22, 1) && d.nch3 == size(acep.c33, 1)
    ok || error("ACEWorkspace mismatch: workspace built for " *
                "(Nat=$(d.Nat), ns=$(d.ns), nch=($(d.nch0),$(d.nch1),$(d.nch2),$(d.nch3))), " *
                "got (Nat=$Nat, ns=$(length(acep.cp)), nch=($(size(acep.c0,1))," *
                "$(size(acep.c11,1)),$(size(acep.c22,1)),$(size(acep.c33,1))))")
    (ws.rmin == acep.rmin && ws.rmax == acep.rmax) ||
        error("ACEWorkspace mismatch: workspace built for rmin/rmax = " *
              "$(ws.rmin)/$(ws.rmax), got $(acep.rmin)/$(acep.rmax)")
    # The workspace evaluates with its own preprocessed coefficient snapshot;
    # catch (heuristically) a caller whose acep has since been mutated.
    ca = _coefabs(acep)
    isapprox(ca, ws.coefabs; rtol = 1e-5) ||
        error("ACEWorkspace stale: coefficients changed since construction " *
              "(Σ|cp|+Σ|c0| $(ws.coefabs) → $ca). Rebuild the workspace.")
    wb = get_backend(ws.buf.pe)
    ab = get_backend(atoms)
    wb == ab || error("ACEWorkspace backend mismatch: workspace on $wb, atoms on $ab")
    return nothing
end

function _refresh_species_order!(ws::ACEWorkspace, atoms::Atoms)
    elemid = atoms.arrays[:elemid]
    ws.elemid_ref === elemid && return nothing

    d = ws.dims
    eh = Int.(Array(elemid))
    length(eh) == d.Nat ||
        error("ACEWorkspace: elemid length $(length(eh)) ≠ Nat $(d.Nat)")
    all(s -> 1 <= s <= d.ns, eh) ||
        error("ACEWorkspace: elemid outside 1:$(d.ns)")

    # Correctness does not depend on the sort being stable (rows are
    # independent); stability just keeps within-species order reproducible.
    p = sortperm(eh)
    sorted = eh[p]
    for s in 1:d.ns
        ws.rowlo[s] = searchsortedfirst(sorted, s)
        ws.rowhi[s] = searchsortedlast(sorted, s)
    end
    copyto!(ws.rowrank, Int32.(invperm(p)))
    copyto!(ws.elemid_sorted, Int32.(sorted))
    ws.elemid_ref = elemid
    return nothing
end

# ------------------------------------------------------------
# Entry points
# ------------------------------------------------------------

"""
    calculate!(acep::ACEParam, atoms::Atoms)
    calculate!(acep::ACEParam, atoms::Atoms, nl::NeighborList[, ws::ACEWorkspace])

Three-stage (features → GEMM contraction → forces) evaluator for
[`ACEParam`](@ref). Adds the ACE contribution to
`atoms.info[:potential_energy]`, `atoms.info[:virial]`, and
`atoms.arrays[:forces]`.

The single-`atoms` method builds a fresh neighbour list at `acep.rmax` —
convenient for fitting / parity passes. The `nl` methods take a pre-built
list (typically built once per MD step via [`build_mlip_nlists`](@ref FMD.Apps.build_mlip_nlists)).
The production NPT MD path passes a persistent [`ACEWorkspace`](@ref) as
well; without one, a temporary workspace is built per call.

Parity against the reference implementation in `ACE.jl` is covered by
`test/calculators/test_ace.jl` (CPU backend) and `test/gpu/test_ace_gpu.jl`
(CUDA backend).
"""
function calculate!(acep::ACEParam, atoms::Atoms)
    nl = neighborlist(atoms, acep.rmax, 0.0)  # ★ rmax 一本
    calculate!(acep, atoms, nl)
end

function calculate!(acep::ACEParam, atoms::Atoms, nl::NeighborList)
    ws = ACEWorkspace(get_backend(atoms), acep, length(atoms))
    calculate!(acep, atoms, nl, ws)
end

# Ghost-format method: identical three-stage pipeline, with the geometry
# stubs (IdLattice / ZeroShift — see ACE.jl) so both kernels are reused
# verbatim reading the extended arrays. Kernel C scatters ±f (and the
# split virial) into ghosts.fext / ghosts.vext; the ghost rows are folded
# back onto the owners afterwards — b.vir doubles as the fold target so
# everything downstream (virial sum, :peratom_virial) is unchanged.
function calculate!(acep::ACEParam, atoms::Atoms, nl::NeighborList, ws::ACEWorkspace)
    backend = get_backend(atoms)
    Nat = length(atoms)
    _check_workspace(ws, acep, atoms)
    _refresh_species_order!(ws, atoms)
    sync!(nl, atoms)
    g = nl.ghosts
    g.n_owned == Nat ||
        error("ACEFast ghost calculate!: list has $(g.n_owned) owned rows, atoms has $Nat")

    b = ws.buf
    fill!(b.vir, zero(FPPREC))
    # scatter contract (see Ghosts docstring): the user zeroes fext/vext.
    fill!(g.fext, zero(FPPREC))
    fill!(g.vext, zero(FPPREC))

    _ace_features_launch!(backend, ws, g, nl.list)

    _ace_stageB!(backend, ws)

    _ace_forces_launch!(backend, ws, g, nl.list)

    fold_ghosts!(atoms.arrays[:forces], g.fext, g)
    fold_ghosts!(b.vir, g.vext, g)

    add_reduced_ev!(atoms, sum(b.pe), sum(b.vir, dims = 1)[1, :])

    if haskey(atoms.arrays, :peratom_virial)
        atoms.arrays[:peratom_virial] .+= b.vir
    end
    return nothing
end

# ------------------------------------------------------------
# Neighbour-kernel launch dispatch (Issue #187)
#
# The generic methods below run the KernelAbstractions kernels (one thread
# per atom) on any backend; ACEFastCUDA.jl specializes both on
# `CUDA.CUDABackend` with warp-per-atom kernels whose accumulators live in
# registers / shared memory (~1.6-3× per ACE call at production sizes, exact
# feature parity). Keep the argument surface identical so the two paths stay
# swappable: (backend, workspace, ghosts, neighbour list).
# ------------------------------------------------------------

function _ace_features_launch!(backend, ws::ACEWorkspace, g, list)
    b = ws.buf
    kernel_ace_features!(backend, 256, ws.dims.Nat)(
        b.S, b.V, b.M, b.N, ws.rowrank,
        g.xcart, g.elemid_ext, IdLattice(),
        list, ZeroShift(), ZeroShift(), ZeroShift(),
        ws.rmin, ws.rmax, ws.dims.ns
    )
    return nothing
end

function _ace_forces_launch!(backend, ws::ACEWorkspace, g, list)
    b = ws.buf
    kernel_ace_forces!(backend, 256, ws.dims.Nat)(
        g.fext, g.vext, b.dEdS, b.dEdV, b.dEdM, b.dEdN, ws.rowrank,
        g.xcart, g.elemid_ext, IdLattice(),
        list, ZeroShift(), ZeroShift(), ZeroShift(),
        ws.rmin, ws.rmax, ws.dims.ns
    )
    return nothing
end

# ------------------------------------------------------------
# Kernel A: features in species-sorted GEMM layout
#
# Row layout: atom I lives in row r = rowrank[I]; the component rows are
# V[3(r-1)+a, ·], M[6(r-1)+k, ·], N[10(r-1)+k, ·]. Channel columns keep the
# neighbour-species offsets of the reference implementation. Each row is
# written by exactly one thread in the same neighbour order as the fused
# predecessor, so the features are bit-identical to it — a parity failure
# therefore localises to Stage B.
# ------------------------------------------------------------

@kernel function kernel_ace_features!(
    S, V, M, N, @Const(rowrank),
    @Const(pos), @Const(elemid),
    @Const(lattice), @Const(nlist), @Const(shiftx), @Const(shifty), @Const(shiftz),
    rmin, rmax, ns
)
    I = @index(Global)
    r = rowrank[I]

    ron = rmax - rmax/10

    nch0 = size(S, 2); n0 = nch0 ÷ ns
    nch1 = size(V, 2); n1 = nch1 ÷ ns
    nch2 = size(M, 2); n2 = nch2 ÷ ns
    nch3 = size(N, 2); n3 = nch3 ÷ ns

    rV = 3 * (r - 1)
    rM = 6 * (r - 1)
    rN = 10 * (r - 1)

    # ---- clear own rows ----
    for μ in 1:nch0
        S[r, μ] = zero(eltype(S))
    end
    for μ in 1:nch1
        V[rV+1, μ] = zero(eltype(V)); V[rV+2, μ] = zero(eltype(V)); V[rV+3, μ] = zero(eltype(V))
    end
    for μ in 1:nch2
        for k in 1:6
            M[rM+k, μ] = zero(eltype(M))
        end
    end
    for μ in 1:nch3
        for k in 1:10
            N[rN+k, μ] = zero(eltype(N))
        end
    end

    x_i, y_i, z_i = pos[I,1], pos[I,2], pos[I,3]

    for jlist in axes(nlist,2)
        j = nlist[I,jlist]
        j < 1 && break

        x_ij, y_ij, z_ij, rij = displacement(
            x_i, y_i, z_i,
            pos[j,1], pos[j,2], pos[j,3],
            shiftx[I,jlist], shifty[I,jlist], shiftz[I,jlist],
            lattice
        )
        (rij <= rmin || rij >= rmax) && continue

        sj = elemid[j]

        offset0 = n0*(sj-1)
        offset1 = n1*(sj-1)
        offset2 = n2*(sj-1)
        offset3 = n3*(sj-1)

        invr = one(rij) / rij
        x̂ = x_ij * invr
        ŷ = y_ij * invr
        ẑ = z_ij * invr

        inv3 = one(rij) / 3
        p11 = x̂*x̂ - inv3
        p22 = ŷ*ŷ - inv3
        p33 = ẑ*ẑ - inv3
        p23 = ŷ*ẑ
        p31 = ẑ*x̂
        p12 = x̂*ŷ

        q, _, _, _ = stf3_10_and_grad(x̂, ŷ, ẑ)

        # Chebyshev recursion (t=1..n0)
        x0      = (2*rij - (rmax + rmin)) / (rmax - rmin)
        dscale0 = (2 / (rmax - rmin))

        t0  = one(rij); dt0 = zero(rij)   # T0
        t1  = x0;       dt1 = one(rij)    # T1

        # t = 1
        b1, _ = tapering(t0, dt0*dscale0, rij, ron, rmax)
        S[r, offset0+1] += b1
        if 1 <= n1
            μ1 = offset1 + 1
            V[rV+1,μ1] += b1*x̂; V[rV+2,μ1] += b1*ŷ; V[rV+3,μ1] += b1*ẑ
            if 1 <= n2
                μ2 = offset2 + 1
                M[rM+1,μ2] += b1*p11; M[rM+2,μ2] += b1*p22; M[rM+3,μ2] += b1*p33
                M[rM+4,μ2] += b1*p23; M[rM+5,μ2] += b1*p31; M[rM+6,μ2] += b1*p12
                if 1 <= n3
                    μ3 = offset3 + 1
                    for k in 1:10
                        N[rN+k,μ3] += b1*q[k]
                    end
                end
            end
        end

        if n0 >= 2
            # t = 2
            b2, _ = tapering(t1, dt1*dscale0, rij, ron, rmax)
            S[r, offset0+2] += b2
            if 2 <= n1
                μ1 = offset1 + 2
                V[rV+1,μ1] += b2*x̂; V[rV+2,μ1] += b2*ŷ; V[rV+3,μ1] += b2*ẑ
                if 2 <= n2
                    μ2 = offset2 + 2
                    M[rM+1,μ2] += b2*p11; M[rM+2,μ2] += b2*p22; M[rM+3,μ2] += b2*p33
                    M[rM+4,μ2] += b2*p23; M[rM+5,μ2] += b2*p31; M[rM+6,μ2] += b2*p12
                    if 2 <= n3
                        μ3 = offset3 + 2
                        for k in 1:10
                            N[rN+k,μ3] += b2*q[k]
                        end
                    end
                end
            end

            # t >= 3
            for t in 3:n0
                t2, dt2 = chebs(x0, t0, t1, dt0, dt1)
                bt, _   = tapering(t2, dt2*dscale0, rij, ron, rmax)
                S[r, offset0+t] += bt

                if t <= n1
                    μ1 = offset1 + t
                    V[rV+1,μ1] += bt*x̂; V[rV+2,μ1] += bt*ŷ; V[rV+3,μ1] += bt*ẑ
                    if t <= n2
                        μ2 = offset2 + t
                        M[rM+1,μ2] += bt*p11; M[rM+2,μ2] += bt*p22; M[rM+3,μ2] += bt*p33
                        M[rM+4,μ2] += bt*p23; M[rM+5,μ2] += bt*p31; M[rM+6,μ2] += bt*p12
                        if t <= n3
                            μ3 = offset3 + t
                            for k in 1:10
                                N[rN+k,μ3] += bt*q[k]
                            end
                        end
                    end
                end

                t0, t1, dt0, dt1 = t1, t2, dt1, dt2
            end
        end
    end
end

# ------------------------------------------------------------
# Stage B: energies + adjoints via GEMMs and small kernels
#
# Species-sorted rows make each species a contiguous row block, so the
# species-sliced coefficient contractions become plain `mul!` on
# (UnitRange, Colon) views — strided matrices that dispatch to BLAS/cuBLAS.
# The remaining per-atom products parallelise over atom×pair / atom×channel.
# ------------------------------------------------------------

# B_l[r,p] = Σ_k w[k] · X[k-row of r, hi(p)] · X[k-row of r, lo(p)]
@kernel function kernel_ace_pairdot!(B, @Const(X), @Const(w), @Const(hi_of_p), @Const(lo_of_p))
    idx = @index(Global)
    P = size(B, 2)
    r = (idx - 1) ÷ P + 1
    p = (idx - 1) % P + 1
    hi = hi_of_p[p]
    lo = lo_of_p[p]
    nc = length(w)
    base = nc * (r - 1)
    acc = zero(eltype(B))
    for k in 1:nc
        acc += w[k] * X[base+k, hi] * X[base+k, lo]
    end
    B[r, p] = acc
end

# pe[r] = cp + c0·S + S·G0 + V:G1 + w2∘M:G2 + w3∘N:G3 + Σ_l T_l·B_l
@kernel function kernel_ace_energy!(
    pe, @Const(elemid_sorted), @Const(cp), @Const(c0),
    @Const(S), @Const(G0), @Const(V), @Const(G1),
    @Const(M), @Const(G2), @Const(N), @Const(G3),
    @Const(w2), @Const(w3),
    @Const(T1), @Const(B1), @Const(T2), @Const(B2), @Const(T3), @Const(B3)
)
    r = @index(Global)
    si = elemid_sorted[r]

    E = cp[si]
    for μ in 1:size(S, 2)
        E += c0[μ, si] * S[r, μ] + S[r, μ] * G0[r, μ]
    end
    rV = 3 * (r - 1)
    for ν in 1:size(V, 2)
        E += V[rV+1, ν]*G1[rV+1, ν] + V[rV+2, ν]*G1[rV+2, ν] + V[rV+3, ν]*G1[rV+3, ν]
    end
    rM = 6 * (r - 1)
    for ν in 1:size(M, 2)
        for k in 1:6
            E += w2[k] * M[rM+k, ν] * G2[rM+k, ν]
        end
    end
    rN = 10 * (r - 1)
    for ν in 1:size(N, 2)
        for k in 1:10
            E += w3[k] * N[rN+k, ν] * G3[rN+k, ν]
        end
    end
    for p in 1:size(T1, 2)
        E += T1[r, p] * B1[r, p]
    end
    for p in 1:size(T2, 2)
        E += T2[r, p] * B2[r, p]
    end
    for p in 1:size(T3, 2)
        E += T3[r, p] * B3[r, p]
    end
    pe[r] = E
end

# dEdX[k-row, ν] = wadj[k]·(2·G[k-row, ν] + Σ_λ th(ν,λ)·X[k-row, λ]) with
# th(ν,λ) = (1 + δ_νλ)·T[r, p(max(ν,λ), min(ν,λ))]. The diagonal factor 2
# reproduces the reference's double accumulation on ν == λ pairs.
@kernel function kernel_ace_adjoint!(dEdX, @Const(X), @Const(G), @Const(T), @Const(wadj))
    idx = @index(Global)
    nch = size(X, 2)
    r = (idx - 1) ÷ nch + 1
    ν = (idx - 1) % nch + 1
    nc = length(wadj)
    base = nc * (r - 1)
    for k in 1:nc
        acc = 2 * G[base+k, ν]
        for λ in 1:nch
            t = T[r, _pairindex(max(ν, λ), min(ν, λ))]
            (ν == λ) && (t += t)
            acc += t * X[base+k, λ]
        end
        dEdX[base+k, ν] = wadj[k] * acc
    end
end

# B-1: quadratic-form GEMMs G_l = X_l·Ĉ_ll ; B-2: 4-body GEMMs T_l = S·C0llp.
# Shared by the generic stage B and the CUDA field-fused one (Issue #189) —
# the GEMM stages are backend-agnostic (mul! dispatches to BLAS/cuBLAS).
function _ace_stageB_gemm_fwd!(ws::ACEWorkspace)
    d = ws.dims
    b = ws.buf
    c = ws.coef
    for s in 1:d.ns
        lo, hi = ws.rowlo[s], ws.rowhi[s]
        lo > hi && continue
        R1  = lo:hi
        R3  = (3lo-2):(3hi)
        R6  = (6lo-5):(6hi)
        R10 = (10lo-9):(10hi)
        @views begin
            mul!(b.G0[R1, :],  b.S[R1, :],  c.C00h[:, :, s])
            mul!(b.G1[R3, :],  b.V[R3, :],  c.C11h[:, :, s])
            mul!(b.G2[R6, :],  b.M[R6, :],  c.C22h[:, :, s])
            mul!(b.G3[R10, :], b.N[R10, :], c.C33h[:, :, s])
            mul!(b.T1[R1, :],  b.S[R1, :],  c.C011p[:, :, s])
            mul!(b.T2[R1, :],  b.S[R1, :],  c.C022p[:, :, s])
            mul!(b.T3[R1, :],  b.S[R1, :],  c.C033p[:, :, s])
        end
    end
    return nothing
end

# B-5: dEdS = c0 + 2·G0 + Σ_l B_l·C0llpᵀ (assign first, then accumulate).
# Shared by both stage B variants, like _ace_stageB_gemm_fwd!.
function _ace_stageB_dEdS!(ws::ACEWorkspace)
    d = ws.dims
    b = ws.buf
    c = ws.coef
    one_ = one(FPPREC)
    for s in 1:d.ns
        lo, hi = ws.rowlo[s], ws.rowhi[s]
        lo > hi && continue
        R1 = lo:hi
        @views begin
            b.dEdS[R1, :] .= transpose(c.c0[:, s]) .+ 2 .* b.G0[R1, :]
            mul!(b.dEdS[R1, :], b.B1[R1, :], transpose(c.C011p[:, :, s]), one_, one_)
            mul!(b.dEdS[R1, :], b.B2[R1, :], transpose(c.C022p[:, :, s]), one_, one_)
            mul!(b.dEdS[R1, :], b.B3[R1, :], transpose(c.C033p[:, :, s]), one_, one_)
        end
    end
    return nothing
end

# Generic stage B. `_ace_stageB!` is the dispatch entry: ACEFastCUDA.jl
# specializes it on `CUDA.CUDABackend` with the field-fused kernels
# (Issue #189); every other backend runs this KernelAbstractions pipeline.
# Kept callable under its own name so the GPU test suite can compare the
# two implementations directly.
_ace_stageB!(backend, ws::ACEWorkspace) = _ace_stageB_generic!(backend, ws)

function _ace_stageB_generic!(backend, ws::ACEWorkspace)
    d = ws.dims
    b = ws.buf
    c = ws.coef
    Nat = d.Nat

    _ace_stageB_gemm_fwd!(ws)

    # B-3: pair products (species-independent, atom×pair parallel).
    # The component weights w2/w3 are folded in HERE (and in the energy);
    kernel_ace_pairdot!(backend, 256, Nat * d.P1)(b.B1, b.V, c.e3, c.hi1, c.lo1)
    kernel_ace_pairdot!(backend, 256, Nat * d.P2)(b.B2, b.M, c.w2, c.hi2, c.lo2)
    kernel_ace_pairdot!(backend, 256, Nat * d.P3)(b.B3, b.N, c.w3, c.hi3, c.lo3)

    # B-4: per-atom energies
    kernel_ace_energy!(backend, 256, Nat)(
        b.pe, ws.elemid_sorted, c.cp, c.c0,
        b.S, b.G0, b.V, b.G1, b.M, b.G2, b.N, b.G3,
        c.w2, c.w3,
        b.T1, b.B1, b.T2, b.B2, b.T3, b.B3
    )

    _ace_stageB_dEdS!(ws)

    # B-6: dEdV/dEdM/dEdN. NOTE the reference's weight conventions: dEdV and
    # dEdM carry NO component weight (the trace multiplicities of M are
    # implicit in Kernel C's full-matrix reconstruction), while dEdN carries
    # w3 explicitly — hence e3/e6/w3 as the adjoint weights.
    kernel_ace_adjoint!(backend, 256, Nat * d.nch1)(b.dEdV, b.V, b.G1, b.T1, c.e3)
    kernel_ace_adjoint!(backend, 256, Nat * d.nch2)(b.dEdM, b.M, b.G2, b.T2, c.e6)
    kernel_ace_adjoint!(backend, 256, Nat * d.nch3)(b.dEdN, b.N, b.G3, b.T3, c.w3)
    return nothing
end

# ------------------------------------------------------------
# Kernel C: forces + per-atom virials from the adjoints
# (same expressions as the fused predecessor; only the dEd* read layout
# changed — row r = rowrank[I], component-interleaved rows)
# ------------------------------------------------------------

@kernel function kernel_ace_forces!(
    forces, vir, @Const(dEdS), @Const(dEdV), @Const(dEdM), @Const(dEdN), @Const(rowrank),
    @Const(pos), @Const(elemid),
    @Const(lattice), @Const(nlist), @Const(shiftx), @Const(shifty), @Const(shiftz),
    rmin, rmax, ns
)
    I = @index(Global)
    r = rowrank[I]

    ron = rmax - rmax/10

    n0 = size(dEdS, 2) ÷ ns
    n1 = size(dEdV, 2) ÷ ns
    n2 = size(dEdM, 2) ÷ ns
    n3 = size(dEdN, 2) ÷ ns

    rV = 3 * (r - 1)
    rM = 6 * (r - 1)
    rN = 10 * (r - 1)

    x_i, y_i, z_i = pos[I,1], pos[I,2], pos[I,3]

    for jlist in axes(nlist,2)
        j = nlist[I,jlist]
        j < 1 && break

        x_ij, y_ij, z_ij, rij = displacement(
            x_i, y_i, z_i,
            pos[j,1], pos[j,2], pos[j,3],
            shiftx[I,jlist], shifty[I,jlist], shiftz[I,jlist],
            lattice
        )
        (rij <= rmin || rij >= rmax) && continue

        sj = elemid[j]

        invr = one(rij) / rij
        x̂ = x_ij * invr
        ŷ = y_ij * invr
        ẑ = z_ij * invr

        fx = zero(eltype(vir))
        fy = zero(eltype(vir))
        fz = zero(eltype(vir))

        # need l=3 q & grads once
        q, gx, gy, gz = stf3_10_and_grad(x̂, ŷ, ẑ)

        offset0 = n0*(sj-1)
        offset1 = n1*(sj-1)
        offset2 = n2*(sj-1)
        offset3 = n3*(sj-1)

        x0      = (2*rij - (rmax + rmin)) / (rmax - rmin)
        dscale0 = (2 / (rmax - rmin))

        t0  = one(rij); dt0 = zero(rij)
        t1  = x0;       dt1 = one(rij)

        # t = 1
        b1, db1 = tapering(t0, dt0*dscale0, rij, ron, rmax)
        Gs = dEdS[r, offset0+1]
        fx += -db1*Gs*x̂; fy += -db1*Gs*ŷ; fz += -db1*Gs*ẑ

        if 1 <= n1
            μ1 = offset1 + 1
            G1 = dEdV[rV+1,μ1]; G2 = dEdV[rV+2,μ1]; G3 = dEdV[rV+3,μ1]
            gdot = G1*x̂ + G2*ŷ + G3*ẑ
            fx += -( db1*gdot*x̂ + (b1*invr)*(G1 - gdot*x̂) )
            fy += -( db1*gdot*ŷ + (b1*invr)*(G2 - gdot*ŷ) )
            fz += -( db1*gdot*ẑ + (b1*invr)*(G3 - gdot*ẑ) )

            if 1 <= n2
                μ2 = offset2 + 1
                G11 = dEdM[rM+1,μ2]; G22 = dEdM[rM+2,μ2]; G33 = dEdM[rM+3,μ2]
                G23 = dEdM[rM+4,μ2]; G31 = dEdM[rM+5,μ2]; G12 = dEdM[rM+6,μ2]
                grx = G11*x̂ + G12*ŷ + G31*ẑ
                gry = G12*x̂ + G22*ŷ + G23*ẑ
                grz = G31*x̂ + G23*ŷ + G33*ẑ
                scalar = x̂*grx + ŷ*gry + ẑ*grz
                trG = G11 + G22 + G33
                Tm = scalar - trG/3
                sx = (2*(grx - scalar*x̂)) * invr
                sy = (2*(gry - scalar*ŷ)) * invr
                sz = (2*(grz - scalar*ẑ)) * invr
                fx += -(db1*Tm*x̂ + b1*sx)
                fy += -(db1*Tm*ŷ + b1*sy)
                fz += -(db1*Tm*ẑ + b1*sz)

                if 1 <= n3
                    μ3 = offset3 + 1
                    for k in 1:10
                        Gk = dEdN[rN+k,μ3]
                        gdotk = gx[k]*x̂ + gy[k]*ŷ + gz[k]*ẑ
                        projx = gx[k] - gdotk*x̂
                        projy = gy[k] - gdotk*ŷ
                        projz = gz[k] - gdotk*ẑ
                        fx += -( db1*Gk*q[k]*x̂ + (b1*invr)*Gk*projx )
                        fy += -( db1*Gk*q[k]*ŷ + (b1*invr)*Gk*projy )
                        fz += -( db1*Gk*q[k]*ẑ + (b1*invr)*Gk*projz )
                    end
                end
            end
        end

        if n0 >= 2
            # t = 2
            b2, db2 = tapering(t1, dt1*dscale0, rij, ron, rmax)
            Gs = dEdS[r, offset0+2]
            fx += -db2*Gs*x̂; fy += -db2*Gs*ŷ; fz += -db2*Gs*ẑ

            if 2 <= n1
                μ1 = offset1 + 2
                G1 = dEdV[rV+1,μ1]; G2 = dEdV[rV+2,μ1]; G3 = dEdV[rV+3,μ1]
                gdot = G1*x̂ + G2*ŷ + G3*ẑ
                fx += -( db2*gdot*x̂ + (b2*invr)*(G1 - gdot*x̂) )
                fy += -( db2*gdot*ŷ + (b2*invr)*(G2 - gdot*ŷ) )
                fz += -( db2*gdot*ẑ + (b2*invr)*(G3 - gdot*ẑ) )

                if 2 <= n2
                    μ2 = offset2 + 2
                    G11 = dEdM[rM+1,μ2]; G22 = dEdM[rM+2,μ2]; G33 = dEdM[rM+3,μ2]
                    G23 = dEdM[rM+4,μ2]; G31 = dEdM[rM+5,μ2]; G12 = dEdM[rM+6,μ2]
                    grx = G11*x̂ + G12*ŷ + G31*ẑ
                    gry = G12*x̂ + G22*ŷ + G23*ẑ
                    grz = G31*x̂ + G23*ŷ + G33*ẑ
                    scalar = x̂*grx + ŷ*gry + ẑ*grz
                    trG = G11 + G22 + G33
                    Tm = scalar - trG/3
                    sx = (2*(grx - scalar*x̂)) * invr
                    sy = (2*(gry - scalar*ŷ)) * invr
                    sz = (2*(grz - scalar*ẑ)) * invr
                    fx += -(db2*Tm*x̂ + b2*sx)
                    fy += -(db2*Tm*ŷ + b2*sy)
                    fz += -(db2*Tm*ẑ + b2*sz)

                    if 2 <= n3
                        μ3 = offset3 + 2
                        for k in 1:10
                            Gk = dEdN[rN+k,μ3]
                            gdotk = gx[k]*x̂ + gy[k]*ŷ + gz[k]*ẑ
                            projx = gx[k] - gdotk*x̂
                            projy = gy[k] - gdotk*ŷ
                            projz = gz[k] - gdotk*ẑ
                            fx += -( db2*Gk*q[k]*x̂ + (b2*invr)*Gk*projx )
                            fy += -( db2*Gk*q[k]*ŷ + (b2*invr)*Gk*projy )
                            fz += -( db2*Gk*q[k]*ẑ + (b2*invr)*Gk*projz )
                        end
                    end
                end
            end

            # t >= 3
            for t in 3:n0
                t2, dt2 = chebs(x0, t0, t1, dt0, dt1)
                bt, dbt = tapering(t2, dt2*dscale0, rij, ron, rmax)

                Gs = dEdS[r, offset0+t]
                fx += -dbt*Gs*x̂; fy += -dbt*Gs*ŷ; fz += -dbt*Gs*ẑ

                if t <= n1
                    μ1 = offset1 + t
                    G1 = dEdV[rV+1,μ1]; G2 = dEdV[rV+2,μ1]; G3 = dEdV[rV+3,μ1]
                    gdot = G1*x̂ + G2*ŷ + G3*ẑ
                    fx += -( dbt*gdot*x̂ + (bt*invr)*(G1 - gdot*x̂) )
                    fy += -( dbt*gdot*ŷ + (bt*invr)*(G2 - gdot*ŷ) )
                    fz += -( dbt*gdot*ẑ + (bt*invr)*(G3 - gdot*ẑ) )

                    if t <= n2
                        μ2 = offset2 + t
                        G11 = dEdM[rM+1,μ2]; G22 = dEdM[rM+2,μ2]; G33 = dEdM[rM+3,μ2]
                        G23 = dEdM[rM+4,μ2]; G31 = dEdM[rM+5,μ2]; G12 = dEdM[rM+6,μ2]
                        grx = G11*x̂ + G12*ŷ + G31*ẑ
                        gry = G12*x̂ + G22*ŷ + G23*ẑ
                        grz = G31*x̂ + G23*ŷ + G33*ẑ
                        scalar = x̂*grx + ŷ*gry + ẑ*grz
                        trG = G11 + G22 + G33
                        Tm = scalar - trG/3
                        sx = (2*(grx - scalar*x̂)) * invr
                        sy = (2*(gry - scalar*ŷ)) * invr
                        sz = (2*(grz - scalar*ẑ)) * invr
                        fx += -(dbt*Tm*x̂ + bt*sx)
                        fy += -(dbt*Tm*ŷ + bt*sy)
                        fz += -(dbt*Tm*ẑ + bt*sz)

                        if t <= n3
                            μ3 = offset3 + t
                            for k in 1:10
                                Gk = dEdN[rN+k,μ3]
                                gdotk = gx[k]*x̂ + gy[k]*ŷ + gz[k]*ẑ
                                projx = gx[k] - gdotk*x̂
                                projy = gy[k] - gdotk*ŷ
                                projz = gz[k] - gdotk*ẑ
                                fx += -( dbt*Gk*q[k]*x̂ + (bt*invr)*Gk*projx )
                                fy += -( dbt*Gk*q[k]*ŷ + (bt*invr)*Gk*projy )
                                fz += -( dbt*Gk*q[k]*ẑ + (bt*invr)*Gk*projz )
                            end
                        end
                    end
                end

                t0, t1, dt0, dt1 = t1, t2, dt1, dt2
            end
        end

        # ---- atomic force + per-atom virial split ----
        @atomic forces[I,1] += fx
        @atomic forces[I,2] += fy
        @atomic forces[I,3] += fz
        @atomic forces[j,1] -= fx
        @atomic forces[j,2] -= fy
        @atomic forces[j,3] -= fz

        @atomic vir[I,1] += fx * x_ij / 2
        @atomic vir[I,2] += fy * y_ij / 2
        @atomic vir[I,3] += fz * z_ij / 2
        @atomic vir[I,4] += (fy * z_ij + fz * y_ij) / 4
        @atomic vir[I,5] += (fz * x_ij + fx * z_ij) / 4
        @atomic vir[I,6] += (fx * y_ij + fy * x_ij) / 4

        @atomic vir[j,1] += fx * x_ij / 2
        @atomic vir[j,2] += fy * y_ij / 2
        @atomic vir[j,3] += fz * z_ij / 2
        @atomic vir[j,4] += (fy * z_ij + fz * y_ij) / 4
        @atomic vir[j,5] += (fz * x_ij + fx * z_ij) / 4
        @atomic vir[j,6] += (fx * y_ij + fy * x_ij) / 4
    end
end
