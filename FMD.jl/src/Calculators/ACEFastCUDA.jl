# ACEFastCUDA.jl — warp-per-atom CUDA specializations of the ACEFast neighbour
# kernels (Issue #187).
#
# The generic (KernelAbstractions) kernels in ACEFast.jl run one thread per
# atom: the neighbour loop accumulates ~O(nch) feature components per
# neighbour by read-modify-writing global memory — ~0.9 KB of global traffic
# per pair — which makes their per-pair cost ~25× the real-space pair loop
# and is size-independent (measured flat from 20k to 110k atoms). The CUDA
# specializations below assign one warp (32 threads) per atom and keep every
# accumulator in registers:
#
#   features (A): neighbours are staged 32 at a time into shared memory
#     (radial basis values, the 20 geometric factors, species); the
#     `nslot = nch0 + 3·nch1 + 6·nch2 + 10·nch3` feature slots are statically
#     distributed over the lanes (MR = cld(nslot, 32) registers each) and
#     accumulated in registers, written back once at the end. Neighbours are
#     consumed in list order, so the features are bit-identical to the KA
#     kernel's.
#   forces (C): the atom's nslot adjoint values are staged into shared memory
#     once, then each lane processes every 32nd neighbour (thread-per-
#     neighbour): the j-side scatter keeps the KA kernel's @atomic contract,
#     the i-side sums are block-reduced so atom I costs 9 global atomics per
#     block instead of 9 per pair. Only the i-side summation order differs
#     from the KA kernel (float-roundoff level).
#
# Hard-won details (all measured on the Issue #187 prototype — removing any
# of them costs the speedup or the compile):
# - TPB = 32 (exact warp) beats 64: cld(nslot, 32) registers per lane leave
#   almost no dead slots, and the block barriers degenerate to warp syncs.
# - The staging arrays are padded to TPB+1 rows: the accumulate phase reads
#   one staged neighbour across all lanes at different columns, and an
#   unpadded stride of TPB = 32 puts every lane on the same shared-memory
#   bank (32-way conflict — the kernel then LOSES to the KA one at ≥20k
#   atoms).
# - The neighbour count is a parallel count (valid entries are a prefix, so
#   count == prefix length) — a serial thread-0 scan is a dependent global
#   load chain of ~maxneigh latency per block.
# - The accumulator tuple is updated through function barriers
#   (_warp_meta/_warp_accum/_warp_writeout): an `ntuple` closure capturing a
#   REASSIGNED local is boxed by lowering, and Core.Box does not compile to
#   GPU code.
#
# Dims (ns, nk) are generalized through Val{MR}/Val{N0}/Val{CAP} kernel
# specializations — one compile per dims combination, resolved at launch.

using CUDA
using Base.Cartesian: @nexprs

# One warp per atom. The slot→lane distribution and both kernels assume this
# equals the warp size; it is a design constant, not a tuning knob.
const _ACE_WARP_TPB = 32

# Per-species-block slot order (0-based u), shared by every decode below and
# by the adjoint staging in the forces kernel — the single source of truth
# for how a (tensor, component k, channel t) triple maps to a flat slot:
#   u < n0                    : S,  t = u+1                          fidx = 1
#   u < n0+3n1                : V,  v = u-n0,       k = v%3+1,  t = v÷3+1,  fidx = 1+k
#   u < n0+3n1+6n2            : M,  v = u-(n0+3n1), k = v%6+1,  t = v÷6+1,  fidx = 4+k
#   else                      : N,  v = …,          k = v%10+1, t = v÷10+1, fidx = 10+k
# fidx indexes the staged geometric factor row: [1] = 1, [2:4] = x̂ŷẑ,
# [5:10] = p(11,22,33,23,31,12), [11:20] = q(1:10) — the same component
# orders the KA kernels write into V/M/N rows.

# Per-lane slot metadata (t, fidx, species); species 0 marks an unused
# register. Function barrier: the tuple must be built in one shot so the
# lanes' registers stay statically indexed.
@inline function _warp_meta(::Val{MR}, tid::Int, nslot::Int32, per_sp::Int32,
                            n0::Int32, n1::Int32, n2::Int32) where {MR}
    c1 = n0
    c2 = n0 + Int32(3) * n1
    c3 = c2 + Int32(6) * n2
    return ntuple(Val(MR)) do i
        s = Int32((i - 1) * _ACE_WARP_TPB + (tid - 1))
        if s < nslot
            sp = s ÷ per_sp
            u = s - sp * per_sp
            if u < c1
                (u + Int32(1), Int32(1), sp + Int32(1))
            elseif u < c2
                v = u - c1
                (v ÷ Int32(3) + Int32(1), Int32(2) + v % Int32(3), sp + Int32(1))
            elseif u < c3
                v = u - c2
                (v ÷ Int32(6) + Int32(1), Int32(5) + v % Int32(6), sp + Int32(1))
            else
                v = u - c3
                (v ÷ Int32(10) + Int32(1), Int32(11) + v % Int32(10), sp + Int32(1))
            end
        else
            (Int32(1), Int32(1), Int32(0))
        end
    end
end

# Register accumulation over one staged neighbour. `acc` is reassigned by the
# caller — the function barrier keeps the ntuple closure from capturing a
# reassigned local (which would box it and kill GPU compilation).
@inline function _warp_accum(acc::NTuple{MR, T}, meta, spn::Int32,
                             sh_bt, sh_geom, nb2::Int) where {MR, T}
    return ntuple(Val(MR)) do i
        ti, fi, spi = meta[i]
        v = acc[i]
        if spi == spn
            @inbounds v += sh_bt[nb2, Int(ti)] * sh_geom[nb2, Int(fi)]
        end
        v
    end
end

# One assignment per owned slot — full coverage of the atom's rows, so the
# write IS the clear (the KA kernel's explicit zero loops have no analogue).
@inline function _warp_writeout(acc::NTuple{MR, T}, meta, S, V, M, N, r::Int,
                                n0::Int32, n1::Int32, n2::Int32, n3::Int32) where {MR, T}
    rV = 3 * (r - 1)
    rM = 6 * (r - 1)
    rN = 10 * (r - 1)
    ntuple(Val(MR)) do i
        ti, fi, spi = meta[i]
        if spi != Int32(0)
            @inbounds if fi == Int32(1)
                S[r, (Int(spi) - 1) * Int(n0) + Int(ti)] = acc[i]
            elseif fi <= Int32(4)
                V[rV + Int(fi) - 1, (Int(spi) - 1) * Int(n1) + Int(ti)] = acc[i]
            elseif fi <= Int32(10)
                M[rM + Int(fi) - 4, (Int(spi) - 1) * Int(n2) + Int(ti)] = acc[i]
            else
                N[rN + Int(fi) - 10, (Int(spi) - 1) * Int(n3) + Int(ti)] = acc[i]
            end
        end
        nothing
    end
    return nothing
end

function _cuda_ace_features!(::Val{MR}, ::Val{N0}, S, V, M, N, rowrank,
                             pos, elemid, nlist, rmin::T, rmax::T, ns::Int32,
                             n0::Int32, n1::Int32, n2::Int32, n3::Int32) where {MR, N0, T}
    I = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    r = Int(rowrank[I])

    per_sp = n0 + Int32(3) * n1 + Int32(6) * n2 + Int32(10) * n3
    nslot = ns * per_sp
    maxneigh = size(nlist, 2)

    meta = _warp_meta(Val(MR), tid, nslot, per_sp, n0, n1, n2)

    # TPB+1 rows: see the bank-conflict note in the header.
    sh_bt   = CuStaticSharedArray(T, (_ACE_WARP_TPB + 1, N0))
    sh_geom = CuStaticSharedArray(T, (_ACE_WARP_TPB + 1, 20))
    sh_sp   = CuStaticSharedArray(Int32, _ACE_WARP_TPB)
    sh_nn   = CuStaticSharedArray(Int32, 1)

    # Parallel neighbour count (valid entries are a prefix ⇒ count = length).
    if tid == 1
        sh_nn[1] = Int32(0)
    end
    sync_threads()
    cnt = Int32(0)
    p = tid
    while p <= maxneigh
        if nlist[I, p] >= Int32(1)
            cnt += Int32(1)
        end
        p += _ACE_WARP_TPB
    end
    if cnt > Int32(0)
        CUDA.@atomic sh_nn[1] += cnt
    end
    sync_threads()
    nn = Int(sh_nn[1])

    ron = rmax - rmax / T(10)
    x_i = pos[I, 1]; y_i = pos[I, 2]; z_i = pos[I, 3]

    acc = ntuple(_ -> zero(T), Val(MR))

    nchunk = (nn + _ACE_WARP_TPB - 1) ÷ _ACE_WARP_TPB
    c = 1
    while c <= nchunk
        base = (c - 1) * _ACE_WARP_TPB
        nb = base + tid
        sh_sp[tid] = Int32(0)
        # nb ≤ nn implies a valid entry by the prefix contract; the j ≥ 1
        # guard (mirroring the forces kernel) turns a broken list into inert
        # staging instead of an out-of-range device read.
        if nb <= nn && (j = Int(nlist[I, nb]); j >= 1)
            # Ghost format: xcart rows carry their image shift, so the
            # displacement is the plain difference (cf. IdLattice/ZeroShift).
            dx = x_i - pos[j, 1]; dy = y_i - pos[j, 2]; dz = z_i - pos[j, 3]
            rij = sqrt(dx * dx + dy * dy + dz * dz)
            if rij > rmin && rij < rmax
                invr = one(T) / rij
                x̂ = dx * invr; ŷ = dy * invr; ẑ = dz * invr
                inv3 = one(T) / T(3)
                q, _, _, _ = stf3_10_and_grad(x̂, ŷ, ẑ)
                sh_geom[tid, 1] = one(T)
                sh_geom[tid, 2] = x̂; sh_geom[tid, 3] = ŷ; sh_geom[tid, 4] = ẑ
                sh_geom[tid, 5] = x̂ * x̂ - inv3
                sh_geom[tid, 6] = ŷ * ŷ - inv3
                sh_geom[tid, 7] = ẑ * ẑ - inv3
                sh_geom[tid, 8] = ŷ * ẑ
                sh_geom[tid, 9] = ẑ * x̂
                sh_geom[tid, 10] = x̂ * ŷ
                @nexprs 10 k -> (@inbounds sh_geom[tid, 10 + k] = q[k])
                # Tapered Chebyshev values t = 1..n0 — expression for
                # expression the KA kernel's `tapering(T_t, ·)[1]`.
                x0 = (2 * rij - (rmax + rmin)) / (rmax - rmin)
                tap = taper(rij, ron, rmax)
                t0 = one(T); t1 = x0
                sh_bt[tid, 1] = tap * t0
                if n0 >= Int32(2)
                    sh_bt[tid, 2] = tap * t1
                    t = 3
                    while t <= Int(n0)
                        t2 = 2 * x0 * t1 - t0
                        sh_bt[tid, t] = tap * t2
                        t0 = t1; t1 = t2
                        t += 1
                    end
                end
                sh_sp[tid] = elemid[j]
            end
        end
        sync_threads()
        cl = min(_ACE_WARP_TPB, nn - base)
        nb2 = 1
        while nb2 <= cl
            spn = sh_sp[nb2]
            if spn != Int32(0)
                acc = _warp_accum(acc, meta, spn, sh_bt, sh_geom, nb2)
            end
            nb2 += 1
        end
        sync_threads()
        c += 1
    end

    _warp_writeout(acc, meta, S, V, M, N, r, n0, n1, n2, n3)
    return nothing
end

function _cuda_ace_forces!(::Val{CAP}, forces, vir, dEdS, dEdV, dEdM, dEdN,
                           rowrank, pos, elemid, nlist, rmin::T, rmax::T, ns::Int32,
                           n0::Int32, n1::Int32, n2::Int32, n3::Int32) where {CAP, T}
    I = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    r = Int(rowrank[I])

    per_sp = Int(n0) + 3 * Int(n1) + 6 * Int(n2) + 10 * Int(n3)
    nslot = Int(ns) * per_sp
    maxneigh = size(nlist, 2)
    c1 = Int(n0)
    c2 = c1 + 3 * Int(n1)
    c3 = c2 + 6 * Int(n2)

    sh_adj = CuStaticSharedArray(T, CAP)
    sh_red = CuStaticSharedArray(T, 9)
    if tid <= 9
        sh_red[tid] = zero(T)
    end

    # Stage the atom's adjoints (slot order above — the forces loop reads
    # them back through the same index arithmetic).
    rV = 3 * (r - 1); rM = 6 * (r - 1); rN = 10 * (r - 1)
    s = tid - 1
    while s < nslot
        sp = s ÷ per_sp
        u = s - sp * per_sp
        v = zero(T)
        @inbounds if u < c1
            v = dEdS[r, sp * Int(n0) + u + 1]
        elseif u < c2
            w = u - c1
            v = dEdV[rV + w % 3 + 1, sp * Int(n1) + w ÷ 3 + 1]
        elseif u < c3
            w = u - c2
            v = dEdM[rM + w % 6 + 1, sp * Int(n2) + w ÷ 6 + 1]
        else
            w = u - c3
            v = dEdN[rN + w % 10 + 1, sp * Int(n3) + w ÷ 10 + 1]
        end
        sh_adj[s + 1] = v
        s += _ACE_WARP_TPB
    end
    sync_threads()

    ron = rmax - rmax / T(10)
    x_i = pos[I, 1]; y_i = pos[I, 2]; z_i = pos[I, 3]

    fxI = zero(T); fyI = zero(T); fzI = zero(T)
    v1I = zero(T); v2I = zero(T); v3I = zero(T)
    v4I = zero(T); v5I = zero(T); v6I = zero(T)

    nb = tid
    while nb <= maxneigh
        j = Int(nlist[I, nb])
        if j >= 1
            dx = x_i - pos[j, 1]; dy = y_i - pos[j, 2]; dz = z_i - pos[j, 3]
            rij = sqrt(dx * dx + dy * dy + dz * dz)
            if rij > rmin && rij < rmax
                sj = Int(elemid[j])
                invr = one(T) / rij
                x̂ = dx * invr; ŷ = dy * invr; ẑ = dz * invr
                q, gx, gy, gz = stf3_10_and_grad(x̂, ŷ, ẑ)
                off = (sj - 1) * per_sp
                offV = off + c1; offM = off + c2; offN = off + c3
                x0 = (2 * rij - (rmax + rmin)) / (rmax - rmin)
                dscale0 = 2 / (rmax - rmin)
                fx = zero(T); fy = zero(T); fz = zero(T)
                t0 = one(T); dt0 = zero(T)
                t1 = x0;     dt1 = one(T)
                t = 1
                while t <= Int(n0)
                    # Same recursion state walk as the KA kernel's unrolled
                    # t = 1 / t = 2 / t ≥ 3 blocks.
                    if t == 1
                        Tc = one(T); dTc = zero(T)
                    elseif t == 2
                        Tc = x0; dTc = one(T)
                    else
                        t2, dt2 = chebs(x0, t0, t1, dt0, dt1)
                        t0 = t1; t1 = t2; dt0 = dt1; dt1 = dt2
                        Tc = t2; dTc = dt2
                    end
                    bt, dbt = tapering(Tc, dTc * dscale0, rij, ron, rmax)
                    @inbounds Gs = sh_adj[off + t]
                    fx -= dbt * Gs * x̂; fy -= dbt * Gs * ŷ; fz -= dbt * Gs * ẑ
                    if t <= Int(n1)
                        bV = offV + (t - 1) * 3
                        @inbounds begin
                            G1 = sh_adj[bV + 1]; G2 = sh_adj[bV + 2]; G3 = sh_adj[bV + 3]
                        end
                        gdot = G1 * x̂ + G2 * ŷ + G3 * ẑ
                        fx -= dbt * gdot * x̂ + (bt * invr) * (G1 - gdot * x̂)
                        fy -= dbt * gdot * ŷ + (bt * invr) * (G2 - gdot * ŷ)
                        fz -= dbt * gdot * ẑ + (bt * invr) * (G3 - gdot * ẑ)
                        if t <= Int(n2)
                            bM = offM + (t - 1) * 6
                            @inbounds begin
                                G11 = sh_adj[bM + 1]; G22 = sh_adj[bM + 2]; G33 = sh_adj[bM + 3]
                                G23 = sh_adj[bM + 4]; G31 = sh_adj[bM + 5]; G12 = sh_adj[bM + 6]
                            end
                            grx = G11 * x̂ + G12 * ŷ + G31 * ẑ
                            gry = G12 * x̂ + G22 * ŷ + G23 * ẑ
                            grz = G31 * x̂ + G23 * ŷ + G33 * ẑ
                            scalar = x̂ * grx + ŷ * gry + ẑ * grz
                            trG = G11 + G22 + G33
                            Tm = scalar - trG / T(3)
                            sxv = (2 * (grx - scalar * x̂)) * invr
                            syv = (2 * (gry - scalar * ŷ)) * invr
                            szv = (2 * (grz - scalar * ẑ)) * invr
                            fx -= dbt * Tm * x̂ + bt * sxv
                            fy -= dbt * Tm * ŷ + bt * syv
                            fz -= dbt * Tm * ẑ + bt * szv
                            if t <= Int(n3)
                                bN = offN + (t - 1) * 10
                                # @nexprs, not an ntuple closure: fx/fy/fz are
                                # reassigned, and a closure capturing them
                                # would be boxed (no GPU compile).
                                @nexprs 10 k -> begin
                                    Gk = @inbounds sh_adj[bN + k]
                                    gdotk = gx[k] * x̂ + gy[k] * ŷ + gz[k] * ẑ
                                    fx -= dbt * Gk * q[k] * x̂ + (bt * invr) * Gk * (gx[k] - gdotk * x̂)
                                    fy -= dbt * Gk * q[k] * ŷ + (bt * invr) * Gk * (gy[k] - gdotk * ŷ)
                                    fz -= dbt * Gk * q[k] * ẑ + (bt * invr) * Gk * (gz[k] - gdotk * ẑ)
                                end
                            end
                        end
                    end
                    t += 1
                end

                # j-side scatter: the KA kernel's @atomic contract, verbatim.
                nfx = -fx; nfy = -fy; nfz = -fz
                CUDA.@atomic forces[j, 1] += nfx
                CUDA.@atomic forces[j, 2] += nfy
                CUDA.@atomic forces[j, 3] += nfz
                w1 = fx * dx / 2; w2 = fy * dy / 2; w3 = fz * dz / 2
                w4 = (fy * dz + fz * dy) / 4
                w5 = (fz * dx + fx * dz) / 4
                w6 = (fx * dy + fy * dx) / 4
                CUDA.@atomic vir[j, 1] += w1
                CUDA.@atomic vir[j, 2] += w2
                CUDA.@atomic vir[j, 3] += w3
                CUDA.@atomic vir[j, 4] += w4
                CUDA.@atomic vir[j, 5] += w5
                CUDA.@atomic vir[j, 6] += w6
                fxI += fx; fyI += fy; fzI += fz
                v1I += w1; v2I += w2; v3I += w3
                v4I += w4; v5I += w5; v6I += w6
            end
        end
        nb += _ACE_WARP_TPB
    end

    # i-side block reduction: 9 shared atomics per lane, 9 global atomics per
    # atom (the KA kernel pays 9 global atomics per PAIR on the i side).
    CUDA.@atomic sh_red[1] += fxI
    CUDA.@atomic sh_red[2] += fyI
    CUDA.@atomic sh_red[3] += fzI
    CUDA.@atomic sh_red[4] += v1I
    CUDA.@atomic sh_red[5] += v2I
    CUDA.@atomic sh_red[6] += v3I
    CUDA.@atomic sh_red[7] += v4I
    CUDA.@atomic sh_red[8] += v5I
    CUDA.@atomic sh_red[9] += v6I
    sync_threads()
    if tid == 1
        CUDA.@atomic forces[I, 1] += sh_red[1]
        CUDA.@atomic forces[I, 2] += sh_red[2]
        CUDA.@atomic forces[I, 3] += sh_red[3]
        CUDA.@atomic vir[I, 1] += sh_red[4]
        CUDA.@atomic vir[I, 2] += sh_red[5]
        CUDA.@atomic vir[I, 3] += sh_red[6]
        CUDA.@atomic vir[I, 4] += sh_red[7]
        CUDA.@atomic vir[I, 5] += sh_red[8]
        CUDA.@atomic vir[I, 6] += sh_red[9]
    end
    return nothing
end

# ===========================================================================
# Stage B field-fused kernels (Issue #189)
#
# The generic stage B spends ~80% of its time in kernel_ace_pairdot! /
# kernel_ace_energy! / kernel_ace_adjoint!, all of which re-read the atom's
# X row-block (k×nch) and packed pair row T (P entries) from global memory
# inside their pair/λ loops. One block-per-atom kernel per field (V/M/N)
# stages X, G, and T into shared memory once and produces the packed Gram
# B_l, the field's energy contribution (pair + quadratic, block-reduced into
# pe[r] +=), and the adjoint dEdX_l in a single pass. The S-only energy
# (cp + c0·S + S·G0) moves to a light dedicated kernel that INITIALIZES
# pe[r] — launch order (S-energy first, then the three field kernels) is
# what makes the `+=` accumulation correct; all four run stream-ordered on
# the task-local stream. Term-by-term the expressions and accumulation
# orders match the KA kernels, so B_l and dEdX_l are float-identical; only
# pe changes summation order (roundoff).
#
# Prototype-measured tuning (removing either costs ~10-25%):
# - TPB = 64 beats 128/256 for these block-per-atom kernels.
# - The hi/lo pair tables are NOT staged into shared memory: they are
#   identical across all blocks, so L2 serves them, and the smaller shared
#   footprint buys resident blocks (occupancy was the limiter).
# ===========================================================================

const _ACE_FIELD_TPB = 64

# pe[r] = cp[si] + Σ_μ (c0[μ,si] + G0[r,μ])·S[r,μ] — the S terms of
# kernel_ace_energy!, and the initialization of pe (`=`, not `+=`).
function _cuda_ace_senergy!(pe, elemid_sorted, cp, c0, S, G0)
    r = Int((blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x)
    r > length(pe) && return nothing
    si = Int(elemid_sorted[r])
    E = cp[si]
    for μ in 1:size(S, 2)
        E += (c0[μ, si] + G0[r, μ]) * S[r, μ]
    end
    pe[r] = E
    return nothing
end

# One field (X = V/M/N with k = 3/6/10 interleaved component rows):
#   (1) B[r,p] = Σ_k w[k]·X[·,hi(p)]·X[·,lo(p)]      (kernel_ace_pairdot!)
#       + pair energy Σ_p T[r,p]·B[r,p]              (kernel_ace_energy!)
#   (2) quadratic energy Σ w[k]·X∘G                  (kernel_ace_energy!)
#   (3) dEdX[kr,ν] = wadj[k]·(2G + Σ_λ th(ν,λ)·X)    (kernel_ace_adjoint!)
#   (4) pe[r] += block-reduced energy
# w is the pair/energy weight (e3/w2/w3), wadj the adjoint weight (e3/e6/w3)
# — the two differ for M (w2 vs e6), per the reference's weight conventions.
# B_l and dEdX_l match the KA kernels term for term and in accumulation
# order (identical up to compiler FMA fusion — the same caveat as the
# features kernels); only pe changes summation structure.
function _cuda_ace_field!(B, dEdX, pe, X, G, Tm, hi, lo, w, wadj,
                          k::Int, nch::Int, P::Int)
    r = Int(blockIdx().x)
    tid = Int(threadIdx().x)
    T = eltype(B)
    kn = k * nch
    base = k * (r - 1)

    shX = CuDynamicSharedArray(T, kn)
    shG = CuDynamicSharedArray(T, kn, kn * sizeof(T))
    shT = CuDynamicSharedArray(T, P, 2 * kn * sizeof(T))
    shE = CuDynamicSharedArray(T, 1, (2 * kn + P) * sizeof(T))

    i = tid
    while i <= kn
        k_ = (i - 1) % k + 1
        ν = (i - 1) ÷ k + 1
        @inbounds shX[i] = X[base + k_, ν]
        @inbounds shG[i] = G[base + k_, ν]
        i += _ACE_FIELD_TPB
    end
    p = tid
    while p <= P
        @inbounds shT[p] = Tm[r, p]
        p += _ACE_FIELD_TPB
    end
    if tid == 1
        shE[1] = zero(T)
    end
    sync_threads()

    e = zero(T)

    # (1) packed Gram + pair energy
    p = tid
    while p <= P
        h = Int(@inbounds hi[p])
        l = Int(@inbounds lo[p])
        acc = zero(T)
        kk = 1
        while kk <= k
            @inbounds acc += w[kk] * shX[(h-1)*k + kk] * shX[(l-1)*k + kk]
            kk += 1
        end
        @inbounds B[r, p] = acc
        @inbounds e += shT[p] * acc
        p += _ACE_FIELD_TPB
    end

    # (2) quadratic energy
    i = tid
    while i <= kn
        k_ = (i - 1) % k + 1
        @inbounds e += w[k_] * shX[i] * shG[i]
        i += _ACE_FIELD_TPB
    end

    # (3) adjoint — same λ accumulation order as kernel_ace_adjoint!
    i = tid
    while i <= kn
        k_ = (i - 1) % k + 1
        ν = (i - 1) ÷ k + 1
        acc = 2 * @inbounds(shG[i])
        λ = 1
        while λ <= nch
            h2 = ν > λ ? ν : λ
            l2 = ν > λ ? λ : ν
            t = @inbounds shT[(h2 - 1) * h2 ÷ 2 + l2]
            (ν == λ) && (t += t)
            acc += t * @inbounds(shX[(λ-1)*k + k_])
            λ += 1
        end
        @inbounds dEdX[base + k_, ν] = wadj[k_] * acc
        i += _ACE_FIELD_TPB
    end

    # (4) energy block reduction into pe. The shared-atomic order is
    # scheduler-dependent, so pe[r] can wobble at roundoff level between
    # runs — same accepted property as the i-side reduction of
    # _cuda_ace_forces!; pe is only ever consumed through sum(b.pe).
    CUDA.@atomic shE[1] += e
    sync_threads()
    if tid == 1
        @inbounds pe[r] += shE[1]
    end
    return nothing
end

function _ace_field_launch!(B, dEdX, pe, X, G, Tm, hi, lo, w, wadj, k, nch, P, Nat)
    T = eltype(B)
    shmem = (2 * k * nch + P + 1) * sizeof(T)
    @cuda threads = _ACE_FIELD_TPB blocks = Nat shmem = shmem _cuda_ace_field!(
        B, dEdX, pe, X, G, Tm, hi, lo, w, wadj, Int(k), Int(nch), Int(P))
    return nothing
end

function _ace_stageB!(backend::CUDA.CUDABackend, ws::ACEWorkspace)
    d = ws.dims
    d.Nat == 0 && return nothing   # empty DD rank: a 0-block launch is invalid
    b = ws.buf
    c = ws.coef
    T = eltype(b.pe)
    # Dynamic shared memory per field: (2·k·nch + P + 1)·sizeof(T). 48 KB is
    # the opt-in-free limit; name the cause instead of a raw launch failure.
    shmax = max((2 * 3 * d.nch1 + d.P1 + 1), (2 * 6 * d.nch2 + d.P2 + 1),
                (2 * 10 * d.nch3 + d.P3 + 1)) * sizeof(T)
    if shmax > 48 * 1024
        # The generic KA pipeline is present and correct — keep such models
        # runnable instead of erroring (they just miss the fusion speedup).
        @warn "ACE stage B field kernels: dims (ns=$(d.ns), nch=($(d.nch0)," *
              "$(d.nch1),$(d.nch2),$(d.nch3))) need $(shmax) B of dynamic " *
              "shared memory (> 48 KB) — falling back to the generic stage B" maxlog = 1
        return _ace_stageB_generic!(backend, ws)
    end

    _ace_stageB_gemm_fwd!(ws)

    nthreads = 256
    @cuda threads = nthreads blocks = cld(d.Nat, nthreads) _cuda_ace_senergy!(
        b.pe, ws.elemid_sorted, c.cp, c.c0, b.S, b.G0)

    _ace_field_launch!(b.B1, b.dEdV, b.pe, b.V, b.G1, b.T1, c.hi1, c.lo1,
                       c.e3, c.e3, 3, d.nch1, d.P1, d.Nat)
    _ace_field_launch!(b.B2, b.dEdM, b.pe, b.M, b.G2, b.T2, c.hi2, c.lo2,
                       c.w2, c.e6, 6, d.nch2, d.P2, d.Nat)
    _ace_field_launch!(b.B3, b.dEdN, b.pe, b.N, b.G3, b.T3, c.hi3, c.lo3,
                       c.w3, c.w3, 10, d.nch3, d.P3, d.Nat)

    _ace_stageB_dEdS!(ws)
    return nothing
end

# ---- launch specializations (dispatch targets of ACEFast.jl) ---------------

@inline _ace_warp_geom(d) = (Int32(d.nch0 ÷ d.ns), Int32(d.nch1 ÷ d.ns),
                             Int32(d.nch2 ÷ d.ns), Int32(d.nch3 ÷ d.ns),
                             d.nch0 + 3 * d.nch1 + 6 * d.nch2 + 10 * d.nch3)

# The kernels assume the ACEParam-constructor ordering n0 ≥ n1 ≥ n2 ≥ n3: the
# features kernel indexes the staged radial values `sh_bt[·, t]` (n0 columns)
# with V/M/N channel indices t ≤ n1/n2/n3, and the slot-independent decode is
# only equivalent to the KA kernel's nested `t ≤ n_l` truncations under the
# same monotonicity. The constructor asserts it; a hand-built or deserialized
# ACEParam might not — with @inbounds that would be a silent shared-memory
# OOB read, so fail loudly here instead. The shared-memory capacities are
# checked for the same reason (a raw @cuda compile failure would not name the
# cause). 48 KB is the static shared-memory floor of every supported arch.
function _ace_warp_check(d, n0, n1, n2, n3, nslot, T)
    n0 >= n1 >= n2 >= n3 ||
        error("ACE warp kernels require n0 ≥ n1 ≥ n2 ≥ n3 per species " *
              "(got ($n0, $n1, $n2, $n3)) — the ACEParam constructor ordering")
    shmem_A = (_ACE_WARP_TPB + 1) * (Int(n0) + 20) * sizeof(T)
    shmem_C = nslot * sizeof(T)
    max(shmem_A, shmem_C) <= 48 * 1024 ||
        error("ACE warp kernels: dims (ns=$(d.ns), nch=($(d.nch0),$(d.nch1)," *
              "$(d.nch2),$(d.nch3))) exceed the 48 KB static shared-memory " *
              "budget — fall back to the KernelAbstractions kernels for this model")
    return nothing
end

function _ace_features_launch!(::CUDA.CUDABackend, ws::ACEWorkspace, g, list)
    d = ws.dims
    d.Nat == 0 && return nothing   # empty DD rank: a 0-block launch is invalid
    n0, n1, n2, n3, nslot = _ace_warp_geom(d)
    _ace_warp_check(d, n0, n1, n2, n3, nslot, eltype(ws.buf.S))
    b = ws.buf
    # `@cuda` and the KernelAbstractions CUDABackend both launch on the
    # task-local stream (CUDA.stream()), so this kernel, _ace_stageB!'s
    # cuBLAS/KA work, and the forces launch below are stream-ordered without
    # explicit synchronization.
    @cuda threads = _ACE_WARP_TPB blocks = d.Nat _cuda_ace_features!(
        Val(cld(nslot, _ACE_WARP_TPB)), Val(Int(n0)),
        b.S, b.V, b.M, b.N, ws.rowrank, g.xcart, g.elemid_ext, list,
        ws.rmin, ws.rmax, Int32(d.ns), n0, n1, n2, n3)
    return nothing
end

function _ace_forces_launch!(::CUDA.CUDABackend, ws::ACEWorkspace, g, list)
    d = ws.dims
    d.Nat == 0 && return nothing
    n0, n1, n2, n3, nslot = _ace_warp_geom(d)
    _ace_warp_check(d, n0, n1, n2, n3, nslot, eltype(ws.buf.S))
    b = ws.buf
    @cuda threads = _ACE_WARP_TPB blocks = d.Nat _cuda_ace_forces!(
        Val(nslot),
        g.fext, g.vext, b.dEdS, b.dEdV, b.dEdM, b.dEdN,
        ws.rowrank, g.xcart, g.elemid_ext, list,
        ws.rmin, ws.rmax, Int32(d.ns), n0, n1, n2, n3)
    return nothing
end
