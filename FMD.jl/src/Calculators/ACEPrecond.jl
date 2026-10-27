using LinearAlgebra
using MPI

# ---------------------------------------------------------------------------
# blockchol-si preconditioner for the ACE CG fit (Issue #177)
#
# Preconditions the normal system H = AᵀWA + Γ of `fitting!` with the EXACT
# per-(coefficient-field × trailing-species-index si) diagonal blocks of H,
# Cholesky-factored. The pathology of the fit is radial-channel collinearity
# WITHIN these blocks (diagonal/Jacobi preconditioning is provably useless on
# this operator), so solving the blocks exactly removes it.
#
# Assembly is feature-side: one pass over the local frame shard materializes
# the weighted design-matrix rows (1 energy + 3N force + 6 virial rows per
# frame) restricted to the live columns of the blocks of the current memory
# group, and each block's live×live Gram is accumulated with BLAS syrk.
# Row weights mirror `build_normal_system` exactly (shared `_normal_fit_weights`, same
# `dataset_args`): energy √(w·ωE)/Nat, force √(w·ωF), virial √(w·ωS)/Nat,
# so Σ RᵀR equals the diagonal blocks of AᵀWA bit-for-bit up to summation
# order. The Gram is label- and rigid-independent; Γ is added to the block
# diagonals just before factorization.
#
# MPI: collective. Frames are already sharded by `build_normal_system`;
# every rank accumulates panel-wise partial Grams for its shard, panels are
# `MPI.Reduce!`-d over the node-local communicator into a node-shared window
# (one full Gram accumulator per node, Issue #184), the node leaders reduce
# panel-wise across nodes to each block's deterministic owner rank, the
# factorization lives there, and `ldiv!` performs the owned local solves
# followed by one `Allreduce!` of the solution vector (same message size as
# the per-iteration Allreduce in `Hmul!`). NOTE: the two-level Reduce
# summation order depends on the rank count AND the rank-to-node placement,
# so factors — and hence PCG trajectories — differ at roundoff between
# those; the converged solution does not.
#
# Row-sign convention: each design row enters the Gram quadratically, so the
# global sign of a row is immaterial. Only the RELATIVE signs inside one row
# are load-bearing (center vs neighbour Newton pair, and the virial pairing);
# they follow the reference force loop (ACE.jl `calculate!`). Consequently the
# force/virial rows of R are the −1 multiple of the true design rows — do NOT
# reuse R for RHS construction.
#
# Block structure (Issue #183): only the LIVE columns of each block — the
# storage actually touched by the kernels under the upper-triangle convention
# (μ ≤ ν for the quadratic tensors, ν ≥ λ for the 4-body tensors) — enter the
# dense Gram and its Cholesky factor. The complementary dead columns are
# exact zero columns of A, so their H-block is exactly Γ·I and ldiv! handles
# them as a scalar division. This halves the factored width (Gram/factor
# bytes ×~3.8 down) while representing the SAME M exactly.
#
# Design note (reviewed alternative, rejected): an ε-ridge applied only inside
# the preconditioner factorization would lift the rigid[f] > 0 requirement
# without touching H, but every ldiv! would then amplify rounding noise in the
# structurally dead directions by 1/ε. With the live/dead split the dead
# directions no longer enter the factors, yet the requirement stays: Γ is the
# definition of M⁻¹ = 1/Γ on the dead directions, and Γ > 0 keeps every live
# sub-block SPD independent of the data (a species absent from the local
# frame shards has an exactly zero live Gram). The hard requirement (error
# out, never run with a silently noise-injecting or singular M) is the
# CLAUDE.md-preferred failure mode; note it forces Γ > 0 on fields a user may
# have deliberately left at 0, which changes the regularised solution —
# documented in parameters.md.
# ---------------------------------------------------------------------------

# Assembly budget, worn by two hats (Issue #184): the default `budget_bytes`
# bounds one memory group = ONE node-shared full-Gram window (Σ S·w² per
# group), and the default `panel_bytes` bounds the rank-private transient of
# one sweep of column-panel buffers (Σ S·w·p). A single block larger than
# either budget escapes into a group / single-column panel of its own (the
# max() escapes in `_precond_plan`).
const PRECOND_GROUP_BYTES = 2^31

# Storage-order rank of each packed pair index: the live columns of a 4-body
# si-slab (ν ≥ λ; λ = dim3 has the larger stride) ascend with λ outer /
# ν inner, whereas `_pairindex` packs ν outer / λ inner — q maps the packed
# index to the storage-order live rank.
function _ace_livepair(nch::Int, P::Int)
    q = zeros(Int32, P)
    r = 0
    for λ in 1:nch, ν in λ:nch
        q[_pairindex(ν, λ)] = Int32(r += 1)
    end
    return q
end

"""
    _ace_live_columns(x) -> Vector{Int}

Ascending 1-based storage indices of the live (data-touched) columns of one
si-slab of coefficient tensor `x` under the ACE upper-triangle convention:
all of cp/c0, `μ ≤ ν` (dim1 ≤ dim2) for the quadratic tensors, `ν ≥ λ`
(dim2 ≥ dim3) for the 4-body tensors. Built from the same `_pairindex`
packing the row kernel uses, so the two cannot drift.
"""
function _ace_live_columns(x::AbstractArray)
    if ndims(x) == 1                       # cp: (ns,)
        return [1]
    elseif ndims(x) == 2                   # c0: (nch0, ns)
        return collect(1:size(x, 1))
    elseif ndims(x) == 3                   # c00..c33: (nch, nch, ns), live μ ≤ ν
        nch = size(x, 1)
        live = zeros(Int, nch*(nch+1) ÷ 2)
        for ν in 1:nch, μ in 1:ν
            live[_pairindex(ν, μ)] = (ν-1)*nch + μ
        end
        return live
    elseif ndims(x) == 4                   # c011..c033: (nch0, nchL, nchL, ns), live ν ≥ λ
        nch0, nchL = size(x, 1), size(x, 2)
        P = nchL*(nchL+1) ÷ 2
        hi, lo = _ace_pairtables(nchL, P)
        q = _ace_livepair(nchL, P)
        live = zeros(Int, nch0 * P)
        for p in 1:P
            base = (Int(hi[p])-1)*nch0 + (Int(lo[p])-1)*nch0*nchL
            lbase = (Int(q[p])-1)*nch0
            for μ in 1:nch0
                live[lbase + μ] = base + μ
            end
        end
        return live
    else
        error("_ace_live_columns: unsupported coefficient tensor rank $(ndims(x))")
    end
end

"""
    _ace_block_layout(acep) -> Vector{(; field, si, range, live)}

The (field × si) block decomposition of `vec(acep)`. Uses the same
`fieldnames` loop as `Base.vec(::ACEParam)` / `setval!`; the trailing tensor
index of every coefficient field is the center species, so each block is a
contiguous range of the flat coefficient vector. `live` holds the ascending
in-block storage indices of the data-touched columns (`_ace_live_columns`);
the complement is structurally dead (exact zero columns of A) and its
H-block is handled analytically as Γ·I.
"""
function _ace_block_layout(acep::ACEParam)
    ns = length(acep.cp)
    out = @NamedTuple{field::Symbol, si::Int, range::UnitRange{Int}, live::Vector{Int}}[]
    off = 0
    for name in fieldnames(typeof(acep))
        x = getfield(acep, name)
        x isa AbstractArray || continue
        w = length(x) ÷ ns
        live = _ace_live_columns(x)   # si-independent — aliased across si (read-only)
        for si in 1:ns
            push!(out, (; field = name, si, range = (off + (si-1)*w + 1):(off + si*w), live))
        end
        off += length(x)
    end
    return out
end

_ace_precond_dims(acep::ACEParam) = begin
    ns = length(acep.cp)
    nch0 = size(acep.c0, 1); nch1 = size(acep.c11, 1)
    nch2 = size(acep.c22, 1); nch3 = size(acep.c33, 1)
    (; ns, nch0, nch1, nch2, nch3,
       n0 = nch0 ÷ ns, n1 = nch1 ÷ ns, n2 = nch2 ÷ ns, n3 = nch3 ÷ ns,
       P1 = nch1*(nch1+1) ÷ 2, P2 = nch2*(nch2+1) ÷ 2, P3 = nch3*(nch3+1) ÷ 2)
end

# Standalone per-frame feature buffers (ACEFast layout, CPU, sized for the
# largest local frame; kernels only touch rows of the current frame).
struct ACEPrecondFrameBuf
    S::Matrix{FPPREC};  V::Matrix{FPPREC}
    M::Matrix{FPPREC};  N::Matrix{FPPREC}
    B1::Matrix{FPPREC}; B2::Matrix{FPPREC}; B3::Matrix{FPPREC}
    rowrank::Vector{Int32}
    e3::Vector{FPPREC}; w2::Vector{FPPREC}; w3::Vector{FPPREC}
    hi1::Vector{Int32}; lo1::Vector{Int32}
    hi2::Vector{Int32}; lo2::Vector{Int32}
    hi3::Vector{Int32}; lo3::Vector{Int32}
    q1::Vector{Int32}; q2::Vector{Int32}; q3::Vector{Int32}
    B0::Vector{FPPREC}; dB0::Vector{FPPREC}
end

function ACEPrecondFrameBuf(dims, maxNat::Int)
    hi1, lo1 = _ace_pairtables(dims.nch1, dims.P1)
    hi2, lo2 = _ace_pairtables(dims.nch2, dims.P2)
    hi3, lo3 = _ace_pairtables(dims.nch3, dims.P3)
    n = max(maxNat, 1)
    return ACEPrecondFrameBuf(
        zeros(FPPREC, n, dims.nch0), zeros(FPPREC, 3n, dims.nch1),
        zeros(FPPREC, 6n, dims.nch2), zeros(FPPREC, 10n, dims.nch3),
        zeros(FPPREC, n, dims.P1), zeros(FPPREC, n, dims.P2), zeros(FPPREC, n, dims.P3),
        collect(Int32, 1:n),
        ones(FPPREC, 3), FPPREC[1, 1, 1, 2, 2, 2], FPPREC[stf3_w(FPPREC(k)) for k in 1:10],
        hi1, lo1, hi2, lo2, hi3, lo3,
        _ace_livepair(dims.nch1, dims.P1), _ace_livepair(dims.nch2, dims.P2),
        _ace_livepair(dims.nch3, dims.P3),
        Vector{FPPREC}(undef, dims.n0), Vector{FPPREC}(undef, dims.n0))
end

# Scatter one pair-derivative 3-vector into the force rows of atoms `i`
# (owner-folded `jo`) and the six virial rows. Row bases: `fi = 1 + 3(i-1)`,
# `fj = 1 + 3(jo-1)`, `vb = 1 + 3Nat`.
@inline function _scatter3!(R, col::Int, gx::FPPREC, gy::FPPREC, gz::FPPREC,
                            fi::Int, fj::Int, vb::Int,
                            dx::FPPREC, dy::FPPREC, dz::FPPREC)
    @inbounds begin
        R[fi+1, col] += gx; R[fi+2, col] += gy; R[fi+3, col] += gz
        R[fj+1, col] -= gx; R[fj+2, col] -= gy; R[fj+3, col] -= gz
        R[vb+1, col] += gx*dx
        R[vb+2, col] += gy*dy
        R[vb+3, col] += gz*dz
        R[vb+4, col] += (gy*dz + gz*dy)/2
        R[vb+5, col] += (gz*dx + gx*dz)/2
        R[vb+6, col] += (gx*dy + gy*dx)/2
    end
    return nothing
end

"""
    _ace_frame_rows!(R, acep, frame, arg, buf, dims, offs, ωE, ωF, ωS, params)

Fill the weighted design-row matrix `R` ((3·maxNat+7) × group LIVE columns)
of one frame, restricted to the blocks present in `offs` (per-field vector
of 0-based column offsets by si; -1 = block not in the current group).
Columns are live-compressed: within a block they follow the storage order of
the live coefficients (`layout.live`), so the dead lower-triangle storage is
never materialized. Row 1 is the energy row, rows `1+3(a-1)+α` the force
rows, rows `2+3Nat … 7+3Nat` the virial rows (Voigt xx,yy,zz,yz,zx,xy with
symmetrized off-diagonals).
"""
function _ace_frame_rows!(R::Matrix{FPPREC}, acep::ACEParam, frame, arg,
                          buf::ACEPrecondFrameBuf, dims, offs,
                          ωE, ωF, ωS, params)
    fill!(R, zero(FPPREC))
    g = arg.ghosts
    g.dd === nothing ||
        error("blockchol_si: DD ghosts unsupported — the owner fold below assumes a serial neighbour list (fold_ghosts! would need the MPI exchange)")
    sync_ghosts!(g, arg.atoms)
    nlist = arg.nlist
    Nat = arg.Nat
    backend = KernelAbstractions.get_backend(arg.atoms)

    # ---- forward features (ACEFast kernels, standalone buffers) ----
    # Deliberately the KernelAbstractions kernel, not the
    # _ace_features_launch! dispatch (Issue #187): this fit-time path runs on
    # standalone buffers with its own rowrank, and the warp specialization's
    # feature output is float-identical anyway — keep the single-buffer call
    # explicit here. If the feature layout ever changes, update both sites.
    kernel_ace_features!(backend, 256, Nat)(
        buf.S, buf.V, buf.M, buf.N, buf.rowrank,
        g.xcart, g.elemid_ext, IdLattice(),
        nlist, ZeroShift(), ZeroShift(), ZeroShift(),
        acep.rmin, acep.rmax, dims.ns)
    kernel_ace_pairdot!(backend, 256, Nat * dims.P1)(buf.B1, buf.V, buf.e3, buf.hi1, buf.lo1)
    kernel_ace_pairdot!(backend, 256, Nat * dims.P2)(buf.B2, buf.M, buf.w2, buf.hi2, buf.lo2)
    kernel_ace_pairdot!(backend, 256, Nat * dims.P3)(buf.B3, buf.N, buf.w3, buf.hi3, buf.lo3)
    KernelAbstractions.synchronize(backend)

    (; nch0, nch1, nch2, nch3, n0, n1, n2, n3, P1, P2, P3) = dims
    S = buf.S; V = buf.V; M = buf.M; N = buf.N
    B1 = buf.B1; B2 = buf.B2; B3 = buf.B3
    B0 = buf.B0; dB0 = buf.dB0

    rmin = acep.rmin; rmax = acep.rmax
    ron = rmax - rmax/FPPREC(10)
    vb = 1 + 3*Nat

    o_cp = offs[:cp]; o_c0 = offs[:c0]; o_c00 = offs[:c00]
    o_c11 = offs[:c11]; o_c22 = offs[:c22]; o_c33 = offs[:c33]
    o_c011 = offs[:c011]; o_c022 = offs[:c022]; o_c033 = offs[:c033]

    @inbounds for i in 1:Nat
        si = Int(g.elemid_ext[i])
        ocp = o_cp[si]; oc0 = o_c0[si]; oc00 = o_c00[si]
        oc11 = o_c11[si]; oc22 = o_c22[si]; oc33 = o_c33[si]
        oc011 = o_c011[si]; oc022 = o_c022[si]; oc033 = o_c033[si]
        # skip centers whose si-blocks are all outside the current group
        (ocp >= 0 || oc0 >= 0 || oc00 >= 0 || oc11 >= 0 || oc22 >= 0 ||
         oc33 >= 0 || oc011 >= 0 || oc022 >= 0 || oc033 >= 0) || continue

        rV = 3*(i-1); rM = 6*(i-1); rN = 10*(i-1)

        # ------------------------------------------------------------------
        # energy row (row 1): the invariant features of center i
        # ------------------------------------------------------------------
        if ocp >= 0
            R[1, ocp + 1] += one(FPPREC)
        end
        if oc0 >= 0
            for μ in 1:nch0
                R[1, oc0 + μ] += S[i, μ]
            end
        end
        if oc00 >= 0
            for ν in 1:nch0
                Sν = S[i, ν]
                base = oc00 + (ν-1)*ν ÷ 2   # base + μ = oc00 + _pairindex(ν, μ)
                for μ in 1:ν
                    R[1, base + μ] += S[i, μ] * Sν
                end
            end
        end
        if oc11 >= 0
            for p in 1:P1
                R[1, oc11 + p] += B1[i, p]   # packed pair index IS the live rank
            end
        end
        if oc22 >= 0
            for p in 1:P2
                R[1, oc22 + p] += B2[i, p]
            end
        end
        if oc33 >= 0
            for p in 1:P3
                R[1, oc33 + p] += B3[i, p]
            end
        end
        if oc011 >= 0
            for p in 1:P1
                b = B1[i, p]
                base = oc011 + (Int(buf.q1[p])-1)*nch0
                for μ in 1:nch0
                    R[1, base + μ] += S[i, μ] * b
                end
            end
        end
        if oc022 >= 0
            for p in 1:P2
                b = B2[i, p]
                base = oc022 + (Int(buf.q2[p])-1)*nch0
                for μ in 1:nch0
                    R[1, base + μ] += S[i, μ] * b
                end
            end
        end
        if oc033 >= 0
            for p in 1:P3
                b = B3[i, p]
                base = oc033 + (Int(buf.q3[p])-1)*nch0
                for μ in 1:nch0
                    R[1, base + μ] += S[i, μ] * b
                end
            end
        end

        # any position-dependent block in the group for this center?
        (oc0 >= 0 || oc00 >= 0 || oc11 >= 0 || oc22 >= 0 || oc33 >= 0 ||
         oc011 >= 0 || oc022 >= 0 || oc033 >= 0) || continue

        # ------------------------------------------------------------------
        # force + virial rows: pair loop, feature-derivative products
        # ------------------------------------------------------------------
        fi = 1 + 3*(i-1)
        x_i, y_i, z_i = g.xcart[i,1], g.xcart[i,2], g.xcart[i,3]

        for jlist in axes(nlist, 2)
            j = nlist[i, jlist]
            j < 1 && break

            dx, dy, dz, rij = displacement(
                x_i, y_i, z_i,
                g.xcart[j,1], g.xcart[j,2], g.xcart[j,3],
                ZeroShift()[i,jlist], ZeroShift()[i,jlist], ZeroShift()[i,jlist],
                IdLattice())
            (rij >= rmax || rij <= rmin) && continue

            sj = Int(g.elemid_ext[j])
            jo = j <= Nat ? Int(j) : Int(g.src[j - Nat])
            fj = 1 + 3*(jo-1)

            cheb_taper_basis!(B0, dB0, rij, rmin, rmax, ron)

            invr = one(FPPREC) / rij
            x̂ = dx*invr; ŷ = dy*invr; ẑ = dz*invr

            p6 = (x̂*x̂ - one(FPPREC)/FPPREC(3),
                  ŷ*ŷ - one(FPPREC)/FPPREC(3),
                  ẑ*ẑ - one(FPPREC)/FPPREC(3),
                  ŷ*ẑ, ẑ*x̂, x̂*ŷ)
            # ∇̂p rows (gradient of p_k in û-space) and their û-projections
            gpx = (2x̂, zero(FPPREC), zero(FPPREC), zero(FPPREC), ẑ, ŷ)
            gpy = (zero(FPPREC), 2ŷ, zero(FPPREC), ẑ, zero(FPPREC), x̂)
            gpz = (zero(FPPREC), zero(FPPREC), 2ẑ, ŷ, x̂, zero(FPPREC))
            q, gqx, gqy, gqz = stf3_10_and_grad(x̂, ŷ, ẑ)

            off0 = n0*(sj-1); off1 = n1*(sj-1); off2 = n2*(sj-1); off3 = n3*(sj-1)

            for t in 1:n0
                db = dB0[t]; bb = B0[t]
                binvr = bb*invr

                # ---- l = 0 : dS = db·û ----
                dSx = db*x̂; dSy = db*ŷ; dSz = db*ẑ
                μp = off0 + t

                if oc0 >= 0
                    _scatter3!(R, oc0 + μp, dSx, dSy, dSz, fi, fj, vb, dx, dy, dz)
                end
                if oc00 >= 0
                    for ν in 1:nch0
                        v = S[i, ν] * (ν == μp ? FPPREC(2) : one(FPPREC))
                        a, b = minmax(μp, ν)
                        col = oc00 + _pairindex(b, a)
                        _scatter3!(R, col, dSx*v, dSy*v, dSz*v, fi, fj, vb, dx, dy, dz)
                    end
                end
                if oc011 >= 0
                    for p in 1:P1
                        v = B1[i, p]
                        col = oc011 + μp + (Int(buf.q1[p])-1)*nch0
                        _scatter3!(R, col, dSx*v, dSy*v, dSz*v, fi, fj, vb, dx, dy, dz)
                    end
                end
                if oc022 >= 0
                    for p in 1:P2
                        v = B2[i, p]
                        col = oc022 + μp + (Int(buf.q2[p])-1)*nch0
                        _scatter3!(R, col, dSx*v, dSy*v, dSz*v, fi, fj, vb, dx, dy, dz)
                    end
                end
                if oc033 >= 0
                    for p in 1:P3
                        v = B3[i, p]
                        col = oc033 + μp + (Int(buf.q3[p])-1)*nch0
                        _scatter3!(R, col, dSx*v, dSy*v, dSz*v, fi, fj, vb, dx, dy, dz)
                    end
                end

                # ---- l = 1 : dV_aα = (db−b/r)·û_a·û_α + (b/r)·δ_aα ----
                if t <= n1 && (oc11 >= 0 || oc011 >= 0)
                    νp = off1 + t
                    c1 = db - binvr
                    dv11 = c1*x̂*x̂ + binvr; dv12 = c1*x̂*ŷ;         dv13 = c1*x̂*ẑ
                    dv21 = dv12;            dv22 = c1*ŷ*ŷ + binvr; dv23 = c1*ŷ*ẑ
                    dv31 = dv13;            dv32 = dv23;            dv33 = c1*ẑ*ẑ + binvr

                    for c in 1:nch1
                        V1 = V[rV+1, c]; V2 = V[rV+2, c]; V3 = V[rV+3, c]
                        fac = (c == νp ? FPPREC(2) : one(FPPREC))
                        vx = (dv11*V1 + dv21*V2 + dv31*V3) * fac
                        vy = (dv12*V1 + dv22*V2 + dv32*V3) * fac
                        vz = (dv13*V1 + dv23*V2 + dv33*V3) * fac
                        hi, lo = νp >= c ? (νp, c) : (c, νp)
                        p11 = _pairindex(hi, lo)
                        if oc11 >= 0
                            _scatter3!(R, oc11 + p11, vx, vy, vz, fi, fj, vb, dx, dy, dz)
                        end
                        if oc011 >= 0
                            base = oc011 + (Int(buf.q1[p11])-1)*nch0
                            for μ in 1:nch0
                                s = S[i, μ]
                                _scatter3!(R, base + μ, vx*s, vy*s, vz*s, fi, fj, vb, dx, dy, dz)
                            end
                        end
                    end
                end

                # ---- l = 2 : dM_kα = db·p_k·û_α + (b/r)(∇̂p_k − (∇̂p_k·û)û)_α ----
                if t <= n2 && (oc22 >= 0 || oc022 >= 0)
                    κp = off2 + t
                    dm = ntuple(k -> begin
                        pu = gpx[k]*x̂ + gpy[k]*ŷ + gpz[k]*ẑ
                        (db*p6[k]*x̂ + binvr*(gpx[k] - pu*x̂),
                         db*p6[k]*ŷ + binvr*(gpy[k] - pu*ŷ),
                         db*p6[k]*ẑ + binvr*(gpz[k] - pu*ẑ))
                    end, 6)

                    for c in 1:nch2
                        fac = (c == κp ? FPPREC(2) : one(FPPREC))
                        vx = zero(FPPREC); vy = zero(FPPREC); vz = zero(FPPREC)
                        for k in 1:6
                            wk = buf.w2[k] * M[rM+k, c]
                            vx += wk*dm[k][1]; vy += wk*dm[k][2]; vz += wk*dm[k][3]
                        end
                        vx *= fac; vy *= fac; vz *= fac
                        hi, lo = κp >= c ? (κp, c) : (c, κp)
                        p22 = _pairindex(hi, lo)
                        if oc22 >= 0
                            _scatter3!(R, oc22 + p22, vx, vy, vz, fi, fj, vb, dx, dy, dz)
                        end
                        if oc022 >= 0
                            base = oc022 + (Int(buf.q2[p22])-1)*nch0
                            for μ in 1:nch0
                                s = S[i, μ]
                                _scatter3!(R, base + μ, vx*s, vy*s, vz*s, fi, fj, vb, dx, dy, dz)
                            end
                        end
                    end
                end

                # ---- l = 3 : dN_kα = db·q_k·û_α + (b/r)(g_k − (g_k·û)û)_α ----
                if t <= n3 && (oc33 >= 0 || oc033 >= 0)
                    λp = off3 + t
                    dn = ntuple(k -> begin
                        gd = gqx[k]*x̂ + gqy[k]*ŷ + gqz[k]*ẑ
                        (db*q[k]*x̂ + binvr*(gqx[k] - gd*x̂),
                         db*q[k]*ŷ + binvr*(gqy[k] - gd*ŷ),
                         db*q[k]*ẑ + binvr*(gqz[k] - gd*ẑ))
                    end, 10)

                    for c in 1:nch3
                        fac = (c == λp ? FPPREC(2) : one(FPPREC))
                        vx = zero(FPPREC); vy = zero(FPPREC); vz = zero(FPPREC)
                        for k in 1:10
                            wk = buf.w3[k] * N[rN+k, c]
                            vx += wk*dn[k][1]; vy += wk*dn[k][2]; vz += wk*dn[k][3]
                        end
                        vx *= fac; vy *= fac; vz *= fac
                        hi, lo = λp >= c ? (λp, c) : (c, λp)
                        p33 = _pairindex(hi, lo)
                        if oc33 >= 0
                            _scatter3!(R, oc33 + p33, vx, vy, vz, fi, fj, vb, dx, dy, dz)
                        end
                        if oc033 >= 0
                            base = oc033 + (Int(buf.q3[p33])-1)*nch0
                            for μ in 1:nch0
                                s = S[i, μ]
                                _scatter3!(R, base + μ, vx*s, vy*s, vz*s, fi, fj, vb, dx, dy, dz)
                            end
                        end
                    end
                end
            end
        end
    end

    # ---- row weights (mirror Hmul!: energy w·ωE/Nat², force w·ωF, virial w·ωS/Nat²) ----
    w = params[:method_weight][frame.info[:method][1]]
    sqE = sqrt(w*ωE) / Nat
    sqF = sqrt(w*ωF)
    sqV = sqrt(w*ωS) / Nat
    ncols = size(R, 2)
    @inbounds for col in 1:ncols
        R[1, col] *= sqE
        for row in 2:(1 + 3*Nat)
            R[row, col] *= sqF
        end
        for row in (vb + 1):(vb + 6)
            R[row, col] *= sqV
        end
    end
    return R
end

"""
    _precond_plan(widths; budget_bytes, panel_bytes, S) -> (; order, groups, panels, sweeps)

Pure planning of the Gram assembly over the block live `widths` (no
allocation, no MPI; identical on every rank): the deterministic
width-descending block `order`, the memory `groups` (each backs ONE
node-shared window of Σ S·w² bytes; an oversized block escapes into a group
of its own), the per-block column `panels` of width
`p = clamp(panel_bytes ÷ (S·w), 1, w)`, and the per-group `sweeps` packing
the (block, panel) items under the same escape rule (Σ S·w·|cols| per sweep
≤ max(panel_bytes, one item)). One sweep = one pass over the local frames.
`S·w² ≤ panel_bytes ⟺ the block is a single full-width panel`, in which
case the sweep arithmetic is identical to the unpanelled assembly.
"""
function _precond_plan(widths::Vector{Int}; budget_bytes::Int = PRECOND_GROUP_BYTES,
                       panel_bytes::Int = PRECOND_GROUP_BYTES, S::Int = sizeof(FPPREC))
    # width-descending, index tie-break (== the layout's ascending range order)
    order = sort(collect(eachindex(widths)); by = idx -> (-widths[idx], idx))

    groups = Vector{Vector{Int}}()
    cur = Int[]; cur_bytes = 0
    for idx in order
        wb = S * widths[idx]^2
        if !isempty(cur) && cur_bytes + wb > max(budget_bytes, wb)
            push!(groups, cur); cur = Int[]; cur_bytes = 0
        end
        push!(cur, idx); cur_bytes += wb
    end
    isempty(cur) || push!(groups, cur)

    panels = [begin
        w = widths[idx]
        p = clamp(panel_bytes ÷ (S * w), 1, w)
        [j:min(j + p - 1, w) for j in 1:p:w]
    end for idx in eachindex(widths)]

    Item = @NamedTuple{idx::Int, cols::UnitRange{Int}}
    sweeps = [begin
        gs = Vector{Vector{Item}}()
        curi = Item[]; curi_bytes = 0
        for idx in group, cols in panels[idx]
            ib = S * widths[idx] * length(cols)
            if !isempty(curi) && curi_bytes + ib > max(panel_bytes, ib)
                push!(gs, curi); curi = Item[]; curi_bytes = 0
            end
            push!(curi, (; idx, cols)); curi_bytes += ib
        end
        isempty(curi) || push!(gs, curi)
        gs
    end for group in groups]

    # always-on invariants: the panels tile each block exactly, and no sweep
    # exceeds its budget (a violation would silently blow the rank-private
    # transient the whole scheme exists to bound)
    for idx in eachindex(widths)
        (first(first(panels[idx])) == 1 && last(last(panels[idx])) == widths[idx] &&
         all(first(panels[idx][k]) == last(panels[idx][k-1]) + 1 for k in 2:length(panels[idx]))) ||
            error("_precond_plan: panels do not tile block $idx")
    end
    for gs in sweeps, sweep in gs
        total = sum(S * widths[item.idx] * length(item.cols) for item in sweep)
        cap = max(panel_bytes, S * widths[first(sweep).idx] * length(first(sweep).cols))
        total <= cap ||
            error("_precond_plan: sweep of $(length(sweep)) items totals $total B " *
                  "> cap $cap B (panel_bytes = $panel_bytes)")
    end

    return (; order, groups, panels, sweeps)
end

"""
    _assemble_block_grams(acep, params, ctx; comm, budget_bytes, panel_bytes,
                          node_color) -> (; layout, owner, grams, mem)

Assemble the DATA-ONLY (no Γ) per-(field × si) diagonal blocks of AᵀWA on
their owner ranks, restricted to the LIVE columns (`layout[idx].live`;
the dead columns of a block are exact zero rows/columns of its Gram and are
never materialized). `grams[idx]` is the dense live×live block (upper
triangle valid) on `owner[idx]` and `nothing` elsewhere. Collective over
`comm`.

Memory shape (Issue #184): each block's Gram is accumulated in column
panels — the rank-private transient is one sweep of w×p panel buffers
(bounded by `panel_bytes`, plus one oversized single-column escape) instead
of the full w×w — and reduced hierarchically: finished panels are
`MPI.Reduce!`-d over the node-local communicator into a node-shared window
(`MPI.Win_allocate_shared`, ONE full Gram accumulator per node and group),
and after the group's sweeps only the node leaders reduce panel-wise across
nodes to the owner's node, where the owner copies its blocks out of the
shared window. All arithmetic is exact; only the summation order differs
from a flat reduction, so factors differ at roundoff between rank counts
AND rank-to-node placements.

`node_color` (testing only): `nothing` uses the physical shared-memory split
(`MPI.Comm_split_type(COMM_TYPE_SHARED)`); an integer forces an artificial
"node" split so the inter-node path can be exercised inside one physical
node (colors must not span physical nodes — `Win_allocate_shared` would
fail loudly).

`mem` reports the rank-local transient peaks in bytes for the banner:
`panel_bytes` (largest sweep), `r_bytes` (row buffer — NOTE: bounded by
neither budget; it scales as S·(3·maxNat+7)·Σ_group live width and can
dominate for large frames), `win_bytes` (largest node window).
"""
function _assemble_block_grams(acep::ACEParam, params, ctx;
                               comm = MPI.COMM_WORLD,
                               budget_bytes::Int = PRECOND_GROUP_BYTES,
                               panel_bytes::Int = PRECOND_GROUP_BYTES,
                               node_color::Union{Nothing, Integer} = nothing)
    rank = MPI.Comm_rank(comm)
    nranks = MPI.Comm_size(comm)

    dims = _ace_precond_dims(acep)
    layout = _ace_block_layout(acep)
    nblocks = length(layout)

    # the row kernel enumerates exactly these fields; a future ACEParam field
    # would otherwise get a silently-zero Gram (block = Γ·I) — error instead
    handled = (:cp, :c0, :c00, :c11, :c22, :c33, :c011, :c022, :c033)
    lf = unique([blk.field for blk in layout])
    (length(lf) == length(handled) && all(f -> f in handled, lf)) ||
        error("_assemble_block_grams: the row kernel handles exactly $handled; " *
              "ACEParam has coefficient fields $(Tuple(lf)) — extend _ace_frame_rows! first")

    # the row kernel and the live maps derive their loop bounds from the
    # (c0, c11, c22, c33) axes; a hand-built ACEParam with inconsistent tensor
    # shapes would silently scatter out of column bounds under @inbounds
    (size(acep.c0) == (dims.nch0, dims.ns) &&
     size(acep.c00) == (dims.nch0, dims.nch0, dims.ns) &&
     size(acep.c11) == (dims.nch1, dims.nch1, dims.ns) &&
     size(acep.c22) == (dims.nch2, dims.nch2, dims.ns) &&
     size(acep.c33) == (dims.nch3, dims.nch3, dims.ns) &&
     size(acep.c011) == (dims.nch0, dims.nch1, dims.nch1, dims.ns) &&
     size(acep.c022) == (dims.nch0, dims.nch2, dims.nch2, dims.ns) &&
     size(acep.c033) == (dims.nch0, dims.nch3, dims.nch3, dims.ns)) ||
        error("_assemble_block_grams: inconsistent ACEParam tensor shapes " *
              "(expected square channel axes tied to the (c0, c11, c22, c33) sizes)")

    # rank-symmetric backend guard: an empty local shard must not skip the
    # collective, or a non-CPU rank would error alone and hang the others
    badbackend = 0
    if !isempty(ctx.dataset_args)
        backend = KernelAbstractions.get_backend(ctx.dataset_args[1].atoms)
        badbackend = backend isa KernelAbstractions.CPU ? 0 : 1
    end
    badbackend = MPI.Allreduce(badbackend, MPI.MAX, comm)
    badbackend == 0 ||
        error("blockchol_si: the fitting path is CPU-only (non-CPU backend on at least one rank)")

    S = sizeof(FPPREC)
    plan = _precond_plan([length(blk.live) for blk in layout];
                         budget_bytes, panel_bytes, S)

    # deterministic owner assignment: live-width-descending greedy min-load
    loads = zeros(Float64, nranks)
    owner = zeros(Int, nblocks)
    for idx in plan.order
        r = argmin(loads)
        owner[idx] = r - 1
        loads[r] += Float64(length(layout[idx].live))^2
    end

    # hierarchical-reduce topology: node communicator, leaders' communicator
    # (COMM_NULL on non-leaders), and the global-rank → leader map. The split
    # key is the global rank, so node rank 0 is the smallest global rank on
    # the node and `uleaders` (ascending) matches the roots_comm rank order.
    node_comm = node_color === nothing ?
        MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, rank) :
        MPI.Comm_split(comm, Int(node_color), rank)
    node_rank = MPI.Comm_rank(node_comm)
    roots_comm = MPI.Comm_split(comm, node_rank == 0 ? 0 : nothing, rank)
    leader = MPI.Allreduce(rank, MPI.MIN, node_comm)
    leaders_of = MPI.Allgather(leader, comm)
    uleaders = sort(unique(leaders_of))
    rroot_of(r) = findfirst(==(leaders_of[r + 1]), uleaders) - 1

    maxNat = isempty(ctx.dataset_args) ? 0 :
             maximum(arg.Nat for arg in ctx.dataset_args)
    buf = ACEPrecondFrameBuf(dims, maxNat)
    grams = Vector{Union{Nothing, Matrix{FPPREC}}}(nothing, nblocks)
    mem_panel = 0; mem_r = 0; mem_win = 0

    for (g, group) in enumerate(plan.groups)
        # column map: per field, per si → 0-based column offset in R (-1 absent)
        offs = Dict{Symbol, Vector{Int}}()
        for name in fieldnames(typeof(acep))
            getfield(acep, name) isa AbstractArray || continue
            offs[name] = fill(-1, dims.ns)
        end
        ncols = 0
        for idx in group
            blk = layout[idx]
            offs[blk.field][blk.si] = ncols
            ncols += length(blk.live)
        end
        R = zeros(FPPREC, 3*maxNat + 7, ncols)

        # node-shared window: ONE full-Gram accumulator per node for the
        # group, on the leader's segment. Every element is written by exactly
        # one node-Reduce (the panels tile each block, recvbuf is pure
        # output), so no zero-fill is needed.
        winlen = sum(length(layout[idx].live)^2 for idx in group)
        win, _ = MPI.Win_allocate_shared(Array{FPPREC},
                                         node_rank == 0 ? winlen : 0, node_comm)
        winarr = MPI.Win_shared_query(Array{FPPREC}, win, 0)
        length(winarr) == winlen ||
            error("_assemble_block_grams: shared-window query returned " *
                  "$(length(winarr)) elements, expected $winlen")
        woff = Dict{Int, Int}()   # block idx → 0-based element offset in winarr
        let o = 0
            for idx in group
                woff[idx] = o
                o += length(layout[idx].live)^2
            end
        end
        # passive shared-lock epoch on the leader segment: required for
        # Win_sync in the visibility handshake below
        MPI.Win_lock(win; rank = 0, type = :shared, nocheck = true)

        for sweep in plan.sweeps[g]
            pbufs = [zeros(FPPREC, length(layout[item.idx].live), length(item.cols))
                     for item in sweep]
            for (frame, arg) in zip(ctx.dataset, ctx.dataset_args)
                _ace_frame_rows!(R, acep, frame, arg, buf, dims, offs,
                                 ctx.ωE, ctx.ωF, ctx.ωS, params)
                for (item, pb) in zip(sweep, pbufs)
                    blk = layout[item.idx]
                    c0 = offs[blk.field][blk.si]
                    j0, j1 = first(item.cols), last(item.cols)
                    Rp = view(R, :, (c0 + j0):(c0 + j1))
                    # above-diagonal strip: (columns before the panel)ᵀ × panel
                    j0 > 1 && BLAS.gemm!('T', 'N', one(FPPREC),
                                         view(R, :, (c0 + 1):(c0 + j0 - 1)), Rp,
                                         one(FPPREC), view(pb, 1:(j0-1), :))
                    # diagonal tile (upper triangle); rows j1+1:w stay zero —
                    # they land in the never-read lower triangle of the window
                    BLAS.syrk!('U', 'T', one(FPPREC), Rp,
                               one(FPPREC), view(pb, j0:j1, :))
                end
            end
            # node-local reduce of each finished panel into the shared window
            for (item, pb) in zip(sweep, pbufs)
                w = length(layout[item.idx].live)
                o0 = woff[item.idx] + (first(item.cols) - 1) * w
                strip = view(winarr, (o0 + 1):(o0 + w * length(item.cols)))
                MPI.Reduce!(pb, node_rank == 0 ? strip : nothing, MPI.SUM,
                            node_comm; root = 0)
            end
            mem_panel = max(mem_panel, sum(sizeof, pbufs; init = 0))
        end

        # panel-wise inter-node reduce (leaders only, in place) to the
        # owner's node; message size stays w×p
        if roots_comm != MPI.COMM_NULL
            for idx in group, cols in plan.panels[idx]
                w = length(layout[idx].live)
                o0 = woff[idx] + (first(cols) - 1) * w
                strip = view(winarr, (o0 + 1):(o0 + w * length(cols)))
                MPI.Reduce!(strip, MPI.SUM, roots_comm; root = rroot_of(owner[idx]))
            end
        end
        # make the leader's window writes visible to same-node readers
        MPI.Win_sync(win); MPI.Barrier(node_comm); MPI.Win_sync(win)

        # owners (not necessarily leaders) copy their blocks out of the
        # shared window; Win_free is collective, so no rank frees before all
        # copies are done
        for idx in group
            rank == owner[idx] || continue
            w = length(layout[idx].live)
            G = Matrix{FPPREC}(undef, w, w)
            copyto!(G, 1, winarr, woff[idx] + 1, w * w)
            grams[idx] = G
        end
        MPI.Win_unlock(win; rank = 0)
        MPI.free(win)
        mem_r = max(mem_r, sizeof(R))
        mem_win = max(mem_win, S * winlen)
    end
    MPI.free(node_comm)
    roots_comm == MPI.COMM_NULL || MPI.free(roots_comm)

    return (; layout, owner, grams,
            mem = (; panel_bytes = mem_panel, r_bytes = mem_r, win_bytes = mem_win))
end

"""
    ACEBlockCholPrecond

Left preconditioner for `cg!`: the (field × si) diagonal blocks of
`H = AᵀWA + Γ`, distributed over ranks. Within a block, the live coordinates
are solved with the Cholesky factor of the live×live sub-block; the dead
coordinates (structural zero columns of A under the upper-triangle
convention) see exactly `H = Γ·I` and are solved analytically as `r/Γ`.
Each coordinate belongs to exactly one block, handled on exactly one rank,
so `ldiv!(z, P, r)` performs the owned solves locally and `Allreduce!`s `z`.
Collective: every rank must call `ldiv!` the same number of times — cg!
does, since all ranks run identical iterations.

Factors depend on the two-level `MPI.Reduce!` summation order, so they
differ at roundoff between rank counts and rank-to-node placements; the
preconditioned system and its converged solution do not (beyond the usual
CG roundoff sensitivity).
"""
struct ACEBlockCholPrecond
    comm::MPI.Comm
    owned::Vector{@NamedTuple{range::UnitRange{Int}, live::Vector{Int}, dead::Vector{Int},
                              Γ::FPPREC, chol::Cholesky{FPPREC, Matrix{FPPREC}}}}
    scratch::Vector{FPPREC}   # max owned live width; gather target for the strided live
                              # solve — shared, so ldiv! is neither reentrant nor
                              # thread-safe (cg! calls it sequentially)
    n::Int
end

function LinearAlgebra.ldiv!(z::AbstractVector, P::ACEBlockCholPrecond, r::AbstractVector)
    length(z) == P.n && length(r) == P.n ||
        error("ACEBlockCholPrecond: size mismatch (n = $(P.n), got $(length(z)) / $(length(r)))")
    fill!(z, zero(eltype(z)))
    for b in P.owned
        zv = view(z, b.range)
        rv = view(r, b.range)
        t = view(P.scratch, 1:length(b.live))
        copyto!(t, view(rv, b.live))   # gather: live indices are non-uniform strides
        ldiv!(b.chol, t)
        view(zv, b.live) .= t
        view(zv, b.dead) .= view(rv, b.dead) ./ b.Γ
    end
    MPI.Allreduce!(z, MPI.SUM, P.comm)
    return z
end

"""
    build_blockchol_si_preconditioner(acep, params, ctx) -> ACEBlockCholPrecond

Build the blockchol-si preconditioner for [`fitting!`](@ref). `ctx` is
`build_normal_system(...).precond_ctx`. Requires `params[:rigid][f] > 0` for
every coefficient field: Γ is the definition of `M⁻¹ = 1/Γ` on the
structurally dead directions (zero columns of A under the upper-triangle
convention), and Γ > 0 keeps every live sub-block SPD independent of the
data. Collective over `MPI.COMM_WORLD`; deterministic (no RNG). See
[`ACEBlockCholPrecond`](@ref).
"""
function build_blockchol_si_preconditioner(acep::ACEParam, params, ctx)
    # both guards are rank-symmetric (params/acep identical on all ranks), so
    # every rank errors together — no collective is stranded
    eltype(acep.cp) === FPPREC ||
        error("build_blockchol_si_preconditioner: ACEParam eltype $(eltype(acep.cp)) ≠ FPPREC")
    for name in fieldnames(typeof(acep))
        getfield(acep, name) isa AbstractArray || continue
        (haskey(params[:rigid], name) && params[:rigid][name] > 0) ||
            error("build_blockchol_si_preconditioner: cg_preconditioner = :blockchol_si " *
                  "requires rigid[:$name] > 0 — Γ defines M⁻¹ on the structurally dead " *
                  "lower-triangle coefficients and keeps the live sub-block SPD " *
                  "independent of the data")
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    asm = _assemble_block_grams(acep, params, ctx; comm)

    # factorize owned blocks (live×live only; the dead part of the block is
    # exactly Γ·I and is kept analytic); fold success across ranks BEFORE
    # erroring so a numerically failed Cholesky on one owner cannot hang the
    # collective
    owned = @NamedTuple{range::UnitRange{Int}, live::Vector{Int}, dead::Vector{Int},
                        Γ::FPPREC, chol::Cholesky{FPPREC, Matrix{FPPREC}}}[]
    failed = 0
    failmsg = ""
    for (idx, blk) in enumerate(asm.layout)
        asm.owner[idx] == rank || continue
        failed == 0 || break
        G = asm.grams[idx]::Matrix{FPPREC}
        Γf = FPPREC(params[:rigid][blk.field])
        for d in axes(G, 1)
            G[d, d] += Γf
        end
        try
            push!(owned, (; range = blk.range, live = blk.live,
                          dead = setdiff(1:length(blk.range), blk.live),
                          Γ = Γf, chol = cholesky!(Symmetric(G, :U))))
        catch err
            failed = 1
            failmsg = " — block (:$(blk.field), si = $(blk.si)): $(sprint(showerror, err))"
        end
    end
    failed = MPI.Allreduce(failed, MPI.MAX, comm)
    failed == 0 ||
        error("build_blockchol_si_preconditioner: block Cholesky failed on at least one rank" * failmsg)

    # memory accounting (informational): resident live-column factors are the
    # per-rank steady state (derived from the factors themselves so the banner
    # cannot drift from what is actually resident); the assembly transients
    # are the measured peaks reported by _assemble_block_grams
    owned_bytes = sum((sizeof(FPPREC) * size(b.chol, 1)^2 for b in owned); init = 0)
    max_owned = MPI.Allreduce(owned_bytes, MPI.MAX, comm)
    max_panel = MPI.Allreduce(asm.mem.panel_bytes, MPI.MAX, comm)
    max_r = MPI.Allreduce(asm.mem.r_bytes, MPI.MAX, comm)
    max_win = MPI.Allreduce(asm.mem.win_bytes, MPI.MAX, comm)
    if rank == 0
        gib(x) = round(x / 2^30; digits = 2)
        println("blockchol_si: $(length(asm.layout)) blocks, resident factors " *
                "<= $(gib(max_owned)) GiB per rank, assembly transient " *
                "$(gib(max_panel)) GiB panels + $(gib(max_r)) GiB rows per rank, " *
                "node window $(gib(max_win)) GiB")
    end

    scratch = zeros(FPPREC, maximum((length(b.live) for b in owned); init = 0))
    n = last(asm.layout[end].range)
    return ACEBlockCholPrecond(comm, owned, scratch, n)
end
