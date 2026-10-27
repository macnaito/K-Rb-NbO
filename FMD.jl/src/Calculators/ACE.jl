using Statistics
using LinearAlgebra
using MPI
using IterativeSolvers
using LinearMaps
using StaticArrays
using Base
using Adapt

export fitting!, ACEParam, calculate!

"""
    ACEParam{FP, T0, T1, T2, T3}

Atomic Cluster Expansion (ACE)-style MLIP parameter container. Stored
as a flat collection of coefficient arrays so it can live on either CPU
or GPU; `Adapt.@adapt_structure` handles the migration.

# Fields

- `rmin::FP`, `rmax::FP` — radial domain (atomic units). The basis is
  smoothly tapered to zero outside `[rmin, rmax]`.
- `cp` — `(ns,)` per-element constant shift.
- `c0`        — single-particle (`ns·nc0 × ns`).
- `c00, c11, c22, c33` — pair products at orders 0, 1, 2, 3.
- `c011, c022, c033`   — triple products `(0, k)` mixing.

`ns` = number of element species; `nc0 ≥ nc1 ≥ nc2 ≥ nc3 ≥ 3` are the
Chebyshev orders per body order. Tuple `params[:ace_nk]` controls these
in [`fitting!`](@ref).

Build with the constructor below, then [`fitting!`](@ref) to optimise the
coefficients or [`calculate!`](@ref) to evaluate.
"""
struct ACEParam{FP, T0, T1, T2, T3}
    rmin::FP
    rmax::FP
    cp::T0
    c0::T1
    c00::T2
    c11::T2
    c22::T2
    c33::T2
    c011::T3
    c022::T3
    c033::T3
end

Adapt.@adapt_structure ACEParam

"""
    ACEParam(rmin, rmax, ns, nc0, nc1, nc2, nc3) -> ACEParam

Allocate a fresh `ACEParam` with all coefficients zero. The size hierarchy
`nc0 ≥ nc1 ≥ nc2 ≥ nc3 ≥ 3` reflects the body-order ordering used inside
the fit; `params[:ace_nk] = (nc0, nc1, nc2, nc3)` is the canonical input
shape used by `FMD.app[:FitACE]`.
"""
function ACEParam(rmin::FP, rmax::FP, ns, nc0, nc1, nc2, nc3) where {FP}
    @assert rmax > rmin > zero(rmin)
    @assert nc0 ≥ nc1 ≥ nc2 ≥ nc3 ≥ 3
    cp = zeros(FP, ns)
    c0 = zeros(FP, ns*nc0, ns)
    c00 = zeros(FP, ns*nc0, ns*nc0, ns)
    c11 = zeros(FP, ns*nc1, ns*nc1, ns)
    c22 = zeros(FP, ns*nc2, ns*nc2, ns)
    c33 = zeros(FP, ns*nc3, ns*nc3, ns)
    c011 = zeros(FP, ns*nc0, ns*nc1, ns*nc1, ns)
    c022 = zeros(FP, ns*nc0, ns*nc2, ns*nc2, ns)
    c033 = zeros(FP, ns*nc0, ns*nc3, ns*nc3, ns)
    return ACEParam(rmin, rmax, cp, c0, c00, c11, c22, c33, c011, c022, c033)
end

"""
    vec(acep::ACEParam) -> Vector

Flatten every coefficient field of `acep` into one vector. Inverse of
[`setval!`](@ref). Used by the CG fit to communicate parameters across
MPI ranks.
"""
function Base.vec(acep::ACEParam)
    parts = Vector{Vector{eltype(acep.cp)}}()
    for name in fieldnames(typeof(acep))
        x = getfield(acep, name)
        x isa AbstractArray || continue
        push!(parts, vec(x))
    end
    return isempty(parts) ? eltype(acep.cp)[] : vcat(parts...)
end

"""
    setval!(acep::ACEParam, val::AbstractVector) -> ACEParam
    setval!(acep::ACEParam, val) -> ACEParam

Inverse of [`Base.vec(::ACEParam)`](@ref): copy `val` into every
coefficient field (with shape preserved). The scalar form fills every
coefficient with the same value (used to zero out a parameter set
quickly).
"""
function setval!(acep::ACEParam, val::AbstractVector)
    offset = 0
    for name in fieldnames(typeof(acep))
        x = getfield(acep, name)

        x isa AbstractArray || continue

        n = length(x)
        @views x .= reshape(val[(offset+1):(offset+n)], size(x))
        offset += n
    end

    @assert offset == length(val) "setval!: length(val)=$(length(val)) but consumed offset=$offset (array fields only)."
    return acep
end

function setval!(acep::ACEParam, val)
    for name in fieldnames(typeof(acep))
        x = getfield(acep, name)

        x isa AbstractArray || continue

        x .= val
    end
    return acep
end

# ---- Sum of two parameter sets (Δ-learning merge) ------------------ #

"""
    +(a::ACEParam, b::ACEParam) -> ACEParam

Sum of two ACE parameter sets **as potentials**: the result evaluates to
exactly `E_a + E_b` (and likewise forces / virial) on every configuration,
because the ACE energy is linear in the coefficients and the Chebyshev
radial basis is nested — the `k`-th basis function depends only on
`(rmin, rmax)`, never on the total order.

The Chebyshev orders of the result are the per-level maxima of the two
inputs (`nc0 ≥ nc1 ≥ nc2 ≥ nc3` is preserved by `max`); the smaller set's
coefficients are embedded species-block-wise (row index `n_l·(s−1) + k`)
before the addition. `rmin` / `rmax` and the species count must match
exactly — anything else is a different basis and errors. CPU arrays only.

Intended for Δ-learning: fit a small correction `ACEParam` on the
residuals of a base model, then ship `base + delta` as one potential.

!!! warning "Caller responsibilities the code cannot check"
    `ACEParam` does not carry element labels, so only the species *count*
    is verified — both inputs must have been fitted with the **same
    `params[:elements]` ordering**, or the merge silently mixes species
    blocks. Likewise, a Δ-learning merge is only meaningful when the
    delta was fitted against residuals of the *same* baseline stack
    (shell / long-range settings) the base model uses.
"""
function Base.:+(a::ACEParam, b::ACEParam)
    (a.rmin == b.rmin && a.rmax == b.rmax) ||
        error("ACEParam +: radial domains differ (a: [$(a.rmin), $(a.rmax)], " *
              "b: [$(b.rmin), $(b.rmax)]) — the Chebyshev bases are incompatible")
    ns = length(a.cp)
    length(b.cp) == ns ||
        error("ACEParam +: species counts differ ($ns vs $(length(b.cp)))")
    nk = max.(_ace_nk(a), _ace_nk(b))
    FP = promote_type(eltype(a.cp), eltype(b.cp), typeof(a.rmin), typeof(b.rmin))
    out = ACEParam(FP(a.rmin), FP(a.rmax), ns, nk...)
    _addembed!(out, a)
    _addembed!(out, b)
    return out
end

# Per-level Chebyshev orders (n0, n1, n2, n3) recovered from tensor sizes.
_ace_nk(acep::ACEParam) = (size(acep.c0, 1), size(acep.c11, 1),
                           size(acep.c22, 1), size(acep.c33, 1)) .÷ length(acep.cp)

# Which level's order each non-species dimension of a coefficient field
# indexes (1-based into the (n0, n1, n2, n3) tuple); the trailing species
# dimension is copied 1:1.
const _ACE_FIELD_LEVELS = (;
    cp = (), c0 = (1,), c00 = (1, 1), c11 = (2, 2), c22 = (3, 3), c33 = (4, 4),
    c011 = (1, 2, 2), c022 = (1, 3, 3), c033 = (1, 4, 4),
)

function _addembed!(dst::ACEParam, src::ACEParam)
    # Guard against a future coefficient field being added to ACEParam
    # without extending _ACE_FIELD_LEVELS — skipping one would silently
    # zero it in the merge result.
    arrfields = Tuple(n for n in fieldnames(ACEParam) if getfield(src, n) isa AbstractArray)
    arrfields == keys(_ACE_FIELD_LEVELS) ||
        error("ACEParam +: _ACE_FIELD_LEVELS is out of sync with the ACEParam fields " *
              "($(arrfields) vs $(keys(_ACE_FIELD_LEVELS))) — update the table")
    ns = length(dst.cp)
    nksrc, nkdst = _ace_nk(src), _ace_nk(dst)
    blockidx(l) = [(s - 1)*nkdst[l] + k for s in 1:ns for k in 1:nksrc[l]]
    for name in keys(_ACE_FIELD_LEVELS)
        x = getfield(src, name)
        x isa Array || error("ACEParam +: parameters must be plain CPU `Array`s " *
                             "(GPU-resident? adapt to CPU first)")
        idx = map(blockidx, _ACE_FIELD_LEVELS[name])
        @views getfield(dst, name)[idx..., :] .+= x
    end
    return dst
end

# ---- Math utilities ------------------------------------------------ #

@inline function chebs(x, t0, t1, dt0, dt1)
    t2 = 2*x*t1 - t0
    dt2 = 2*x*dt1 + 2*t1 - dt0
    return t2, dt2
end

@inline function taper(r, ron, rmax)
    x = (r - ron) / (rmax - ron)
    x = clamp(x, zero(x), one(x))
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    x5 = x4 * x
    return 1 - 10*x3 + 15*x4 - 6*x5
end

@inline function dtaper(r, ron, rmax)
    invdr = 1 / (rmax - ron)
    x = (r - ron) * invdr
    x = clamp(x, zero(x), one(x))
    x2 = x * x
    x3 = x2 * x
    x4 = x3 * x
    dTdx = -30*x2 + 60*x3 - 30*x4
    return dTdx * invdr
end

@inline function tapering(B, dB, r, ron, rmax)
    tap = taper(r, ron, rmax)
    dtap = dtaper(r, ron, rmax)

    B̂ = tap .* B
    dB̂ = dtap.*B .+ tap.*dB

    return B̂, dB̂
end

@inline function displacement(x_i, y_i, z_i, x_j, y_j, z_j, shiftx, shifty, shiftz, lattice)
    sx = x_i - x_j + shiftx
    sy = y_i - y_j + shifty
    sz = z_i - z_j + shiftz
    x_ij = lattice[1,1] * sx + lattice[2,1] * sy + lattice[3,1] * sz
    y_ij = lattice[1,2] * sx + lattice[2,2] * sy + lattice[3,2] * sz
    z_ij = lattice[1,3] * sx + lattice[2,3] * sy + lattice[3,3] * sz
    r_ij = sqrt(x_ij*x_ij + y_ij*y_ij + z_ij*z_ij)
    return x_ij, y_ij, z_ij, r_ij
end

# ---------------------------------------------------------------------------
# Ghost-format geometry stubs. In the ghost model every neighbour row of
# `xcart` already carries its image shift in Cartesian coordinates, so the
# shift-form kernels (whose only geometry entry point is `displacement`) can be
# reused verbatim by feeding them a zero shift and an identity lattice:
# `x*1 + y*0 + z*0` and `x + 0` are exact in floating point, so the ghost
# path computes literally `xcart_i - xcart_j`. Both stubs are empty isbits
# structs (deliberately NOT AbstractArrays, so KernelAbstractions' @Const
# conversion leaves them untouched on every backend).
# ---------------------------------------------------------------------------

struct ZeroShift end
@inline Base.getindex(::ZeroShift, ::Integer, ::Integer) = Int32(0)

struct IdLattice end
# Bool entries: `true*x === x` and `false*x === zero(x)` (strong zero), so
# the identity transform is exact and type-preserving for any element type.
@inline Base.getindex(::IdLattice, i::Integer, j::Integer) = i == j

@inline function trmatmul(A, B)
    C1 = A[1]*B[1] + A[6]*B[6] + A[5]*B[5]
    C2 = A[6]*B[6] + A[2]*B[2] + A[4]*B[4]
    C3 = A[5]*B[5] + A[4]*B[4] + A[3]*B[3]
    C4 = (A[6]*B[5] + A[2]*B[4] + A[4]*B[3] + A[5]*B[6] + A[4]*B[2] + A[3]*B[4])/2
    C5 = (A[1]*B[5] + A[6]*B[4] + A[5]*B[3] + A[5]*B[1] + A[4]*B[6] + A[3]*B[5])/2
    C6 = (A[1]*B[6] + A[6]*B[2] + A[5]*B[4] + A[6]*B[1] + A[2]*B[6] + A[4]*B[5])/2

    return (C1, C2, C3, C4, C5, C6)
end

@inline function trmatvec(A, b)
    c1 = A[1]*b[1] + A[6]*b[2] + A[5]*b[3]
    c2 = A[6]*b[1] + A[2]*b[2] + A[4]*b[3]
    c3 = A[5]*b[1] + A[4]*b[2] + A[3]*b[3]

    return (c1, c2, c3)
end

@inline function trmatdot(A, B)
    val = A[1]*B[1] + A[2]*B[2] + A[3]*B[3] + 2*(A[4]*B[4] + A[5]*B[5] + A[6]*B[6])

    return val
end

# ----------------------------------------------------------
# l=3 STF (symmetric trace-free) rank-3 tensor: 10 unique symmetric components on rhat = (x,y,z)
# order (same as ACE.jl):
# (111, 222, 333, 112, 113, 122, 223, 233, 133, 123)
#
# and their gradients w.r.t (x,y,z) in "rhat-space"
# (projection to tangent plane is handled in force formula)
# ----------------------------------------------------------
@inline function stf3_10_and_grad(x::T, y::T, z::T) where {T}
    # l=3 STF (symmetric trace-free) of rhat packed into 10 unique symmetric components.
    # order (same as ACE.jl):
    # (111, 222, 333, 112, 113, 122, 223, 233, 133, 123)
    inv5 = one(T) / T(5)
    threeinv5 = T(3) * inv5
    zT = zero(T)

    # q values (trace-free)
    q1  = x*x*x - threeinv5*x           # 111
    q2  = y*y*y - threeinv5*y           # 222
    q3  = z*z*z - threeinv5*z           # 333

    q4  = x*x*y - inv5*y                # 112
    q5  = x*x*z - inv5*z                # 113
    q6  = x*y*y - inv5*x                # 122
    q7  = y*y*z - inv5*z                # 223
    q8  = y*z*z - inv5*y                # 233
    q9  = x*z*z - inv5*x                # 133
    q10 = x*y*z                         # 123

    # grads: ∂q/∂x, ∂q/∂y, ∂q/∂z  (in rhat-space; tangent projection handled elsewhere)
    gx1,  gy1,  gz1  = T(3)*x*x - threeinv5, zT,                 zT
    gx2,  gy2,  gz2  = zT,                 T(3)*y*y - threeinv5, zT
    gx3,  gy3,  gz3  = zT,                 zT,                 T(3)*z*z - threeinv5

    gx4,  gy4,  gz4  = T(2)*x*y,           x*x - inv5,          zT
    gx5,  gy5,  gz5  = T(2)*x*z,           zT,                 x*x - inv5
    gx6,  gy6,  gz6  = y*y - inv5,         T(2)*x*y,            zT
    gx7,  gy7,  gz7  = zT,                 T(2)*y*z,            y*y - inv5
    gx8,  gy8,  gz8  = zT,                 z*z - inv5,          T(2)*y*z
    gx9,  gy9,  gz9  = z*z - inv5,         zT,                 T(2)*x*z
    gx10, gy10, gz10 = y*z,                x*z,                x*y

    q  = (q1,q2,q3,q4,q5,q6,q7,q8,q9,q10)
    gx = (gx1,gx2,gx3,gx4,gx5,gx6,gx7,gx8,gx9,gx10)
    gy = (gy1,gy2,gy3,gy4,gy5,gy6,gy7,gy8,gy9,gy10)
    gz = (gz1,gz2,gz3,gz4,gz5,gz6,gz7,gz8,gz9,gz10)

    return q, gx, gy, gz
end

# full contraction for rank-3 symmetric tensor using multiplicities
# weights: 111,222,333 -> 1
# 112,113,122,223,233,133 -> 3
# 123 -> 6
@inline function stf3_dot10(A, B)
    return A[1]*B[1] + A[2]*B[2] + A[3]*B[3] +
           3*(A[4]*B[4] + A[5]*B[5] + A[6]*B[6] + A[7]*B[7] + A[8]*B[8] + A[9]*B[9]) +
           6*(A[10]*B[10])
end

@inline function stf3_w(k::T) where {T}
    # multiplicity weight for component k
    return (k ≤ 3) ? one(T) : (k == 10 ? T(6) : T(3))
end

# ---- ACE calculation ------------------------------------------------ #

# ---- radial Chebyshev basis + taper (value & d/dr) ------------------ #

@inline function cheb_taper_basis!(
    B::AbstractVector{T}, dB::AbstractVector{T},
    r::T, rmin::T, rmax::T, ron::T
) where {T}
    @inbounds begin
        n = length(B)
        @assert length(dB) == n
        x = (T(2)*r - (rmax + rmin)) / (rmax - rmin)   # map [rmin,rmax] -> [-1,1]
        dscale = T(2) / (rmax - rmin)                  # dx/dr

        t0  = one(T);  dt0 = zero(T)
        t1  = x;       dt1 = one(T)

        B[1],  dB[1]  = tapering(t0, dt0*dscale, r, ron, rmax)
        if n >= 2
            B[2], dB[2] = tapering(t1, dt1*dscale, r, ron, rmax)
        end

        for k in 3:n
            t2, dt2 = chebs(x, t0, t1, dt0, dt1)
            B[k], dB[k] = tapering(t2, dt2*dscale, r, ron, rmax)
            t0, t1, dt0, dt1 = t1, t2, dt1, dt2
        end
    end
    return nothing
end


# ---- ACE calculation (energy + forces + virial) --------------------- #

@inbounds function calculate!(
    acep::ACEParam{FP},
    forces, virial, pos, elemid, lattice, nlist, shiftx, shifty, shiftz
) where {FP}

    E = zero(FP)

    rmin  = acep.rmin
    rmax = acep.rmax
    ns = length(acep.cp)

    # basis dims per l inferred from parameter tensor sizes
    nch0 = size(acep.c0, 1)
    nch1 = size(acep.c11, 1)
    nch2 = size(acep.c22, 1)
    nch3 = size(acep.c33, 1)

    n0 = nch0 ÷ ns
    n1 = nch1 ÷ ns
    n2 = nch2 ÷ ns
    n3 = nch3 ÷ ns

    # per-atom feature buffers
    S = zeros(FP, nch0)
    V = zeros(FP, 3, nch1)
    M = zeros(FP, 6, nch2)
    N = zeros(FP, 10, nch3)

    # derivatives wrt features (per-atom)
    dEdS = zeros(FP, nch0)
    dEdV = zeros(FP, 3, nch1)
    dEdM = zeros(FP, 6, nch2)
    dEdN = zeros(FP, 10, nch3)

    # neighbor radial basis workspaces
    B0  = Vector{FP}(undef, n0)
    dB0 = Vector{FP}(undef, n0)

    ron = rmax - rmax/FP(10)

    for i in axes(nlist, 1)
        si = elemid[i]
        x_i, y_i, z_i = pos[i,1], pos[i,2], pos[i,3]

        fill!(S, zero(FP))
        fill!(V, zero(FP))
        fill!(M, zero(FP))
        fill!(N, zero(FP))
        fill!(dEdS, zero(FP))
        fill!(dEdV, zero(FP))
        fill!(dEdM, zero(FP))
        fill!(dEdN, zero(FP))

        @inbounds for jlist in axes(nlist,2)
            j = nlist[i,jlist]
            j < 1 && break

            x_ij, y_ij, z_ij, rij = displacement(
                x_i, y_i, z_i,
                pos[j,1], pos[j,2], pos[j,3],
                shiftx[i,jlist], shifty[i,jlist], shiftz[i,jlist],
                lattice
            )

            (rij >= rmax || rij <= rmin) && continue

            sj = elemid[j]
            offset0 = n0 * (sj - 1)
            offset1 = n1 * (sj - 1)
            offset2 = n2 * (sj - 1)
            offset3 = n3 * (sj - 1)

            cheb_taper_basis!(B0, dB0, rij, rmin, rmax, ron)

            invr = one(FP) / rij
            x̂ = x_ij * invr
            ŷ = y_ij * invr
            ẑ = z_ij * invr

            # STF objects on rhat
            o = (x̂, ŷ, ẑ)
            p = (x̂*x̂ - one(FP)/FP(3),
                 ŷ*ŷ - one(FP)/FP(3),
                 ẑ*ẑ - one(FP)/FP(3),
                 ŷ*ẑ, ẑ*x̂, x̂*ŷ)
            q, _, _, _ = stf3_10_and_grad(x̂, ŷ, ẑ)

            for k in 1:n0
                bk = B0[k]
                S[offset0 + k] += bk
                if k <= n1
                    V[1, offset1 + k] += bk * o[1]
                    V[2, offset1 + k] += bk * o[2]
                    V[3, offset1 + k] += bk * o[3]
                    if k <= n2
                        M[1, offset2 + k] += bk * p[1]
                        M[2, offset2 + k] += bk * p[2]
                        M[3, offset2 + k] += bk * p[3]
                        M[4, offset2 + k] += bk * p[4]
                        M[5, offset2 + k] += bk * p[5]
                        M[6, offset2 + k] += bk * p[6]
                        if k <= n3
                            for kk in 1:10
                                N[kk, offset3 + k] += bk * q[kk]
                            end
                        end
                    end
                end
            end
        end

        # --------------------------------------------------------------
        # Energy + dE/d(features)
        # --------------------------------------------------------------
        E += acep.cp[si]

        for μ in 1:nch0
            S_μ = S[μ]
            coeff = acep.c0[μ,si]
            E += coeff * S_μ
            dEdS[μ] += coeff
        end

        @inbounds for μ in 1:nch0
            S_μ = S[μ]
            for ν in μ:nch0
                S_ν = S[ν]
                coeff = acep.c00[μ,ν,si]
                E += coeff * S_μ * S_ν
                dEdS[μ] += coeff * S_ν
                dEdS[ν] += coeff * S_μ
            end
        end

        @inbounds for μ in 1:nch1
            V_μ = (V[1,μ], V[2,μ], V[3,μ])
            for ν in μ:nch1
                V_ν = (V[1,ν], V[2,ν], V[3,ν])
                coeff = acep.c11[μ,ν,si]
                E += coeff * (V_μ[1]*V_ν[1] + V_μ[2]*V_ν[2] + V_μ[3]*V_ν[3])
                dEdV[1,μ] += coeff * V_ν[1]
                dEdV[2,μ] += coeff * V_ν[2]
                dEdV[3,μ] += coeff * V_ν[3]
                dEdV[1,ν] += coeff * V_μ[1]
                dEdV[2,ν] += coeff * V_μ[2]
                dEdV[3,ν] += coeff * V_μ[3]
            end
        end

        @inbounds for μ in 1:nch2
            A = (M[1,μ], M[2,μ], M[3,μ], M[4,μ], M[5,μ], M[6,μ])
            for ν in μ:nch2
                B = (M[1,ν], M[2,ν], M[3,ν], M[4,ν], M[5,ν], M[6,ν])
                coeff = acep.c22[μ,ν,si]
                E += coeff * trmatdot(A, B)

                dEdM[1,μ] += coeff * B[1]
                dEdM[2,μ] += coeff * B[2]
                dEdM[3,μ] += coeff * B[3]
                dEdM[4,μ] += coeff * B[4]
                dEdM[5,μ] += coeff * B[5]
                dEdM[6,μ] += coeff * B[6]

                dEdM[1,ν] += coeff * A[1]
                dEdM[2,ν] += coeff * A[2]
                dEdM[3,ν] += coeff * A[3]
                dEdM[4,ν] += coeff * A[4]
                dEdM[5,ν] += coeff * A[5]
                dEdM[6,ν] += coeff * A[6]
            end
        end

        @inbounds for μ in 1:nch3
            A = (N[1,μ], N[2,μ], N[3,μ], N[4,μ], N[5,μ], N[6,μ], N[7,μ], N[8,μ], N[9,μ], N[10,μ])
            for ν in μ:nch3
                B = (N[1,ν], N[2,ν], N[3,ν], N[4,ν], N[5,ν], N[6,ν], N[7,ν], N[8,ν], N[9,ν], N[10,ν])
                coeff = acep.c33[μ,ν,si]
                E += coeff * stf3_dot10(A, B)

                for k in 1:10
                    w = stf3_w(FP(k))
                    dEdN[k,μ] += coeff*w*B[k]
                    dEdN[k,ν] += coeff*w*A[k]
                end
            end
        end

        @inbounds for λ in 1:nch1
            V_λ = (V[1,λ], V[2,λ], V[3,λ])
            for ν in λ:nch1
                V_ν = (V[1,ν], V[2,ν], V[3,ν])
                dotV = (V_ν[1]*V_λ[1] + V_ν[2]*V_λ[2] + V_ν[3]*V_λ[3])
                t = zero(FP)
                for μ in 1:nch0
                    t += acep.c011[μ, ν, λ, si] * S[μ]
                    dEdS[μ] += acep.c011[μ, ν, λ, si] * dotV
                end
                E += t * dotV

                dEdV[1,ν] += t * V_λ[1]
                dEdV[2,ν] += t * V_λ[2]
                dEdV[3,ν] += t * V_λ[3]
                dEdV[1,λ] += t * V_ν[1]
                dEdV[2,λ] += t * V_ν[2]
                dEdV[3,λ] += t * V_ν[3]
            end
        end

        @inbounds for λ in 1:nch2
            Mλ1, Mλ2, Mλ3, Mλ4, Mλ5, Mλ6 = M[1,λ], M[2,λ], M[3,λ], M[4,λ], M[5,λ], M[6,λ]
            for ν in λ:nch2
                Mν1, Mν2, Mν3, Mν4, Mν5, Mν6 = M[1,ν], M[2,ν], M[3,ν], M[4,ν], M[5,ν], M[6,ν]
                trM2 = Mν1*Mλ1 + Mν2*Mλ2 + Mν3*Mλ3 + 2*(Mν4*Mλ4 + Mν5*Mλ5 + Mν6*Mλ6)

                t = zero(FP)
                @simd for μ in 1:nch0
                    t += acep.c022[μ, ν, λ, si] * S[μ]
                    dEdS[μ] += acep.c022[μ, ν, λ, si] * trM2
                end

                E += trM2 * t

                dEdM[1,ν] += t * Mλ1
                dEdM[2,ν] += t * Mλ2
                dEdM[3,ν] += t * Mλ3
                dEdM[4,ν] += t * Mλ4
                dEdM[5,ν] += t * Mλ5
                dEdM[6,ν] += t * Mλ6
                dEdM[1,λ] += t * Mν1
                dEdM[2,λ] += t * Mν2
                dEdM[3,λ] += t * Mν3
                dEdM[4,λ] += t * Mν4
                dEdM[5,λ] += t * Mν5
                dEdM[6,λ] += t * Mν6
            end
        end

        @inbounds for λ in 1:nch3
            Nλ = (N[1,λ], N[2,λ], N[3,λ], N[4,λ], N[5,λ], N[6,λ], N[7,λ], N[8,λ], N[9,λ], N[10,λ])
            for ν in λ:nch3
                Nν = (N[1,ν], N[2,ν], N[3,ν], N[4,ν], N[5,ν], N[6,ν], N[7,ν], N[8,ν], N[9,ν], N[10,ν])
                dotN2 = stf3_dot10(Nν, Nλ)
                t = zero(FP)
                for μ in 1:nch0
                    t += acep.c033[μ, ν, λ, si] * S[μ]
                    dEdS[μ] += acep.c033[μ, ν, λ, si] * dotN2
                end

                E += t * dotN2

                for k in 1:10
                    w = stf3_w(FP(k))
                    dEdN[k,ν] += w * t * Nλ[k]
                    dEdN[k,λ] += w * t * Nν[k]
                end
            end
        end

        @inbounds for jlist in axes(nlist,2)
            j = nlist[i,jlist]
            j < 1 && break

            x_ij, y_ij, z_ij, rij = displacement(
                x_i, y_i, z_i,
                pos[j,1], pos[j,2], pos[j,3],
                shiftx[i,jlist], shifty[i,jlist], shiftz[i,jlist],
                lattice
            )
            (rij >= rmax || rij <= rmin) && continue

            sj = elemid[j]

            invr = one(FP) / rij
            x̂ = x_ij * invr
            ŷ = y_ij * invr
            ẑ = z_ij * invr
            q, gx, gy, gz = stf3_10_and_grad(x̂, ŷ, ẑ)

            fx = zero(FP); fy = zero(FP); fz = zero(FP)

            cheb_taper_basis!(B0, dB0, rij, rmin, rmax, ron)
            offset0 = n0 * (sj - 1)
            offset1 = n1 * (sj - 1)
            offset2 = n2 * (sj - 1)
            offset3 = n3 * (sj - 1)

            for t in 1:n0
                db = dB0[t]
                bb = B0[t]

                G  = dEdS[offset0 + t]
                db = dB0[t]
                fx -= db * G * x̂
                fy -= db * G * ŷ
                fz -= db * G * ẑ
                if t <= n1
                    G1 = dEdV[1, offset1 + t]
                    G2 = dEdV[2, offset1 + t]
                    G3 = dEdV[3, offset1 + t]
                    gdot = G1*x̂ + G2*ŷ + G3*ẑ

                    fx -= (db*gdot*x̂ + (bb*invr)*(G1 - gdot*x̂))
                    fy -= (db*gdot*ŷ + (bb*invr)*(G2 - gdot*ŷ))
                    fz -= (db*gdot*ẑ + (bb*invr)*(G3 - gdot*ẑ))
                    if t <= n2
                        G11 = dEdM[1, offset2 + t]
                        G22 = dEdM[2, offset2 + t]
                        G33 = dEdM[3, offset2 + t]
                        G23 = dEdM[4, offset2 + t]
                        G31 = dEdM[5, offset2 + t]
                        G12 = dEdM[6, offset2 + t]

                        grx = G11*x̂ + G12*ŷ + G31*ẑ
                        gry = G12*x̂ + G22*ŷ + G23*ẑ
                        grz = G31*x̂ + G23*ŷ + G33*ẑ

                        scalar = x̂*grx + ŷ*gry + ẑ*grz          # u^T G u
                        trG = G11 + G22 + G33
                        T = scalar - trG/3                        # G:Q(u)

                        sx = (2*(grx - scalar*x̂)) * invr
                        sy = (2*(gry - scalar*ŷ)) * invr
                        sz = (2*(grz - scalar*ẑ)) * invr

                        fx -= (db*T*x̂ + bb*sx)
                        fy -= (db*T*ŷ + bb*sy)
                        fz -= (db*T*ẑ + bb*sz)
                        if t <= n3
                            for k in 1:10
                                Gk = dEdN[k, offset3 + t]

                                ghx = gx[k]; ghy = gy[k]; ghz = gz[k]
                                gdot = ghx*x̂ + ghy*ŷ + ghz*ẑ
                                px = ghx - gdot*x̂
                                py = ghy - gdot*ŷ
                                pz = ghz - gdot*ẑ

                                qq = q[k]
                                fx -= (db*Gk*qq*x̂ + (bb*invr)*Gk*px)
                                fy -= (db*Gk*qq*ŷ + (bb*invr)*Gk*py)
                                fz -= (db*Gk*qq*ẑ + (bb*invr)*Gk*pz)
                            end
                        end
                    end
                end
            end

            forces[i,1] += fx;  forces[i,2] += fy;  forces[i,3] += fz
            forces[j,1] -= fx;  forces[j,2] -= fy;  forces[j,3] -= fz

            virial[1] += fx * x_ij
            virial[2] += fy * y_ij
            virial[3] += fz * z_ij
            virial[4] += (fy * z_ij + fz * y_ij) / 2
            virial[5] += (fz * x_ij + fx * z_ij) / 2
            virial[6] += (fx * y_ij + fy * x_ij) / 2
        end
    end

    return E
end

# Ghost-format raw entry points: the reference implementation with the
# geometry stubs. `forces_ext` / `u_F_ext` are (N+G)×3 extended arrays —
# the reference's `forces[j] -= f` scatter lands on ghost rows, which the
# caller folds back onto the owners with `fold_ghosts!`. Taking the Ghosts
# object (not raw arrays) makes the owned-array misuse inexpressible — the
# reference body runs under function-level `@inbounds`, so a wrong-length
# array would corrupt silently instead of erroring.
function calculate!(acep::ACEParam, forces_ext, virial, g::Ghosts, nlist)
    size(forces_ext, 1) == length(g) ||
        error("ACE ghost calculate!: forces_ext has $(size(forces_ext, 1)) rows, expected N+G = $(length(g))")
    size(nlist, 1) == g.n_owned ||
        error("ACE ghost calculate!: nlist has $(size(nlist, 1)) rows, expected N = $(g.n_owned)")
    return calculate!(acep, forces_ext, virial, g.xcart, g.elemid_ext, IdLattice(), nlist,
                      ZeroShift(), ZeroShift(), ZeroShift())
end

function scatteradd_analytic!(aceg::ACEParam, u_E, u_F_ext, u_V, g::Ghosts, nlist)
    size(u_F_ext, 1) == length(g) ||
        error("ACE ghost scatteradd_analytic!: u_F_ext has $(size(u_F_ext, 1)) rows, expected N+G = $(length(g))")
    size(nlist, 1) == g.n_owned ||
        error("ACE ghost scatteradd_analytic!: nlist has $(size(nlist, 1)) rows, expected N = $(g.n_owned)")
    return scatteradd_analytic!(aceg, u_E, u_F_ext, u_V, g.xcart, g.elemid_ext, IdLattice(), nlist,
                                ZeroShift(), ZeroShift(), ZeroShift())
end

function applyA(acep::ACEParam, arg)
    g = arg.ghosts
    sync_ghosts!(g, arg.atoms)     # follow in-place position updates
    fill!(g.fext, zero(FPPREC))    # scatter contract: user zeroes fext
    F = zeros(FPPREC, arg.Nat, 3)
    V = zeros(6)
    E = calculate!(acep, g.fext, V, g, arg.nlist)
    fold_ghosts!(F, g.fext, g)
    return E, F, V
end

function applyAT!(acep::ACEParam, arg, u_E, u_F, u_V)
    g = arg.ghosts
    sync_ghosts!(g, arg.atoms)
    extend_ghosts!(arg.u_F_ext, u_F, g)
    scatteradd_analytic!(acep, u_E, arg.u_F_ext, u_V, g, arg.nlist)
end

function makearg(atoms::Atoms, rmax)
    # The list/ghost SELECTION is frozen at build (fitting frames are
    # static); the extended positions are re-synced from `atoms` at every
    # applyA/applyAT! call, so in-place coordinate updates are followed —
    # same semantics as the old `arg.pos` reference.
    nl = neighborlist(atoms, rmax, 0.0)
    g = nl.ghosts

    arg = (;
        Nat = length(atoms),
        atoms = atoms,
        ghosts = g,
        nlist = nl.list,
        u_F_ext = KernelAbstractions.zeros(KernelAbstractions.get_backend(atoms), FPPREC, length(g), 3),
        rmax = rmax,
    )
    return arg
end

"""
    _normal_fit_weights(dataset, params) -> (; ωE, ωF, ωS)

Variance-normalised E/F/V weights of the normal system. Must be computed on
the FULL (pre-shard) dataset with the RAW reference labels — every rank gets
identical weights without communication. Single source of truth shared by
[`build_normal_system`](@ref) and the blockchol-si preconditioner assembly
(ACEPrecond.jl), so the operator and its preconditioner cannot drift apart.
"""
function _normal_fit_weights(dataset::Vector{Atoms}, params)
    ωE, ωF, ωS = 1.0, params[:weight_force], params[:weight_stress]
    ∑ω = ωE + ωF + ωS
    ωE, ωF, ωS = ωE/∑ω, ωF/∑ω, ωS/∑ω

    # エネルギー / 原子の分散
    arr = [atoms.info[:potential_energy][1]/length(atoms) for atoms in dataset]
    σE = isnan(std(arr)) ? 1.0 : std(arr)*sqrt(length(arr))
    ωE /= σE^2

    # 力の分散
    arr = reduce(vcat, [vec(atoms.arrays[:forces]) for atoms in dataset])
    σF = std(arr)*sqrt(length(arr))
    ωF /= σF^2

    # virial の分散
    arr = reduce(vcat, [vec(atoms.info[:virial])/length(atoms) for atoms in dataset])
    σS = std(arr)*sqrt(length(arr))
    ωS /= σS^2

    return (; ωE, ωF, ωS)
end

"""
    build_normal_system(acep::ACEParam, dataset::Vector{Atoms}, params)
        -> (; Hmul!, y, x0)

Build the MPI-distributed normal system that [`fitting!`](@ref) solves:
`Hmul!(q, v)` applies the regularised normal operator `AᵀA + R` to `v`,
`y` is the RHS `Aᵀ (E_ref, F_ref, V_ref)`, and `x0 = vec(acep)` is the
CG start vector. Frame partial sums are sharded across ranks and
Allreduced inside, so the returned system is identical on every rank.

Single source of truth for the fit system — `fitting!` is exactly `cg!`
applied to this. Unexported; also called by the MPI tests to assert
1-rank vs 2-rank parity of the partial sums at machine precision without
the CG solve on top (Issue #23).

Collective over `MPI.COMM_WORLD`: every rank must call this — and apply
the returned `Hmul!` the same number of times — or the internal
`Allreduce!` deadlocks. Never guard a call with `if rank == 0`.

Mutates `acep`: each `Hmul!` application overwrites the coefficients
with `v`. `fitting!` hands `x0` to `cg!` as the in-place solution
buffer, so after a solve it holds the solution, not the start vector.
"""
function build_normal_system(acep::ACEParam, dataset::Vector{Atoms}, params)
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpisize = MPI.Comm_size(MPI.COMM_WORLD)
    dataset = deepcopy(dataset)
    (; ωE, ωF, ωS) = _normal_fit_weights(dataset, params)

    dataset = dataset[(mpirank+1):mpisize:length(dataset)]

    # ---- baseline subtraction (core-shell [+ NLH]) ----
    subtract_baseline!(dataset, params)

    dataset_args = [makearg(atoms, acep.rmax) for atoms in dataset]

    ufi_grad = deepcopy(acep)
    x0 = vec(acep)
    y = zeros(length(x0))

    # > Calculate LHS
    function Hmul!(q, v)
        setval!(acep, v)
        setval!(ufi_grad, zero(eltype(v)))

        for idata in eachindex(dataset)
            Nat = length(dataset[idata])
            w = params[:method_weight][dataset[idata].info[:method][1]]
            E, F, V = applyA(acep, dataset_args[idata])
            applyAT!(ufi_grad, dataset_args[idata], w*ωE*E/Nat^2, w*ωF*F, w*ωS*V/Nat^2)
        end

        if mpirank == 0
            for key in keys(params[:rigid])
                getfield(ufi_grad,key) .+= params[:rigid][key] .* getfield(acep, key)
            end
        end

        q .= vec(ufi_grad)

        MPI.Allreduce!(q, MPI.SUM, MPI.COMM_WORLD)

        return q
    end

    # ---- Calculate RHS ---- #
    setval!(ufi_grad, zero(eltype(x0)))
    for idata in eachindex(dataset)
        Nat = length(dataset[idata])
        w = params[:method_weight][dataset[idata].info[:method][1]]
        applyAT!(ufi_grad, dataset_args[idata], w*ωE*(dataset[idata].info[:potential_energy][1])/Nat^2, w*ωF*(dataset[idata].arrays[:forces]), w*ωS*(dataset[idata].info[:virial])/Nat^2)
    end
    y .= vec(ufi_grad)
    MPI.Allreduce!(y, MPI.SUM, MPI.COMM_WORLD)

    # precond_ctx: the LOCAL frame shard and its makearg geometry, plus the
    # shared weights — everything the blockchol-si preconditioner assembly
    # (ACEPrecond.jl) needs to build the exact diagonal blocks of the same
    # operator Hmul! applies. Sharing the very same `dataset_args` guarantees
    # identical neighbour lists / ghosts.
    return (; Hmul!, y, x0, precond_ctx = (; dataset, dataset_args, ωE, ωF, ωS))
end

"""
    fitting!(acep::ACEParam, dataset::Vector{Atoms}, params) -> ACEParam

Run the CG Δ-learning fit for the ACE coefficients. The pipeline mirrors
[`fitting!`](@ref): for each frame, subtract the
core-shell + Coulomb + ExternalField + Spring contribution from the
reference forces / energy / virial; then solve the regularised normal
equations for the residual.

Required `params`: `:nsteps`, `:weight_force`, `:weight_stress`,
`:method_weight`, `:rigid` (smoothness regulariser dictionary keyed by
coefficient field name, e.g. `c0`, `c00`, `c11`, …).

Optional `params[:cg_preconditioner]`: `:none` (default; byte-identical to
the historical unpreconditioned path) or `:blockchol_si` — precondition CG
with the exact per-(coefficient-field × species) diagonal blocks of the
normal operator, Cholesky-factored
([`build_blockchol_si_preconditioner`](@ref)). Removes the radial-channel
collinearity that stalls plain CG; requires `rigid[f] > 0` for every
coefficient field.
"""
function fitting!(acep::ACEParam, dataset::Vector{Atoms}, params)
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)

    sys = build_normal_system(acep, dataset, params)
    x = sys.x0

    # ---- optional preconditioner ---- #
    # :none must stay byte-identical to the historical path: cg! dispatches
    # to the unpreconditioned CGIterable iff Pl isa Identity.
    mode = get(params, :cg_preconditioner, :none)
    Pl = if mode === :blockchol_si
        build_blockchol_si_preconditioner(acep, params, sys.precond_ctx)
    elseif mode === :none
        Identity()
    else
        error("fitting!: unknown cg_preconditioner $(repr(mode)) — expected :none or :blockchol_si")
    end

    # ---- Conjugate Gradient ---- #
    Aop = LinearMap(sys.Hmul!, length(x); ismutating = true, issymmetric = true, isposdef = true)
    cg!(x, Aop, sys.y; Pl = Pl, reltol = 1e-12, maxiter = params[:nsteps], verbose = (mpirank==0))

    # ---- ベクトル → c1, c2, c3 ---- #
    setval!(acep, x)

    if mpirank == 0
        @show mean(abs, acep.cp) mean(abs, acep.c0) mean(abs, acep.c00) mean(abs, acep.c11) mean(abs, acep.c22) mean(abs, acep.c33) mean(abs, acep.c011) mean(abs, acep.c022) mean(abs, acep.c033)
    end

    return acep
end

@inbounds function scatteradd_analytic!(
    aceg::ACEParam{FP,T0,T1,T2,T3},
    u_E, u_F, u_V,
    pos, elemid, lattice, nlist, shiftx, shifty, shiftz
) where {FP,T0,T1,T2,T3}

    # ---- cutoffs / sizes (match current calculate!) ----
    rmin = aceg.rmin
    rmax = aceg.rmax
    ron  = rmax - rmax/FP(10)

    ns   = length(aceg.cp)
    nch0 = size(aceg.c0,  1)
    nch1 = size(aceg.c11, 1)
    nch2 = size(aceg.c22, 1)
    nch3 = size(aceg.c33, 1)

    n0 = nch0 ÷ ns
    n1 = nch1 ÷ ns
    n2 = nch2 ÷ ns
    n3 = nch3 ÷ ns

    half = FP(0.5)

    # virial adjoints (packed)
    uV1, uV2, uV3, uV4, uV5, uV6 = u_V[1], u_V[2], u_V[3], u_V[4], u_V[5], u_V[6]

    # ---- per-i work arrays (same shapes as calculate) ----
    S     = zeros(FP, nch0)
    Vfeat = zeros(FP, 3, nch1)
    Mfeat = zeros(FP, 6, nch2)
    Nfeat = zeros(FP, 10, nch3)

    dEdS_bar = zeros(FP, nch0)
    dEdV_bar = zeros(FP, 3, nch1)
    dEdM_bar = zeros(FP, 6, nch2)
    dEdN_bar = zeros(FP, 10, nch3)

    # radial basis workspaces (single basis for all l)
    B0  = Vector{FP}(undef, n0)
    dB0 = Vector{FP}(undef, n0)

    for i in axes(nlist, 1)
        si = elemid[i]
        x_i, y_i, z_i = pos[i,1], pos[i,2], pos[i,3]

        fill!(S, zero(FP))
        fill!(Vfeat, zero(FP))
        fill!(Mfeat, zero(FP))
        fill!(Nfeat, zero(FP))
        fill!(dEdS_bar, zero(FP))
        fill!(dEdV_bar, zero(FP))
        fill!(dEdM_bar, zero(FP))
        fill!(dEdN_bar, zero(FP))

        # ------------------------------------------------------------
        # forward: build S/V/M/N (same as current calculate!)
        # ------------------------------------------------------------
        for jlist in axes(nlist, 2)
            j = nlist[i, jlist]
            j < 1 && break

            x_ij, y_ij, z_ij, rij = displacement(
                x_i, y_i, z_i,
                pos[j,1], pos[j,2], pos[j,3],
                shiftx[i,jlist], shifty[i,jlist], shiftz[i,jlist],
                lattice
            )
            (rij <= rmin || rij >= rmax) && continue

            sj = elemid[j]
            offset0 = n0 * (sj - 1)
            offset1 = n1 * (sj - 1)
            offset2 = n2 * (sj - 1)
            offset3 = n3 * (sj - 1)

            cheb_taper_basis!(B0, dB0, rij, rmin, rmax, ron)

            invr = one(FP) / rij
            x̂ = x_ij * invr
            ŷ = y_ij * invr
            ẑ = z_ij * invr

            o = (x̂, ŷ, ẑ)
            p = (x̂*x̂ - one(FP)/FP(3),
                 ŷ*ŷ - one(FP)/FP(3),
                 ẑ*ẑ - one(FP)/FP(3),
                 ŷ*ẑ, ẑ*x̂, x̂*ŷ)
            q, _, _, _ = stf3_10_and_grad(x̂, ŷ, ẑ)

            @inbounds for k in 1:n0
                bk = B0[k]
                S[offset0 + k] += bk

                if k <= n1
                    Vfeat[1, offset1 + k] += bk * o[1]
                    Vfeat[2, offset1 + k] += bk * o[2]
                    Vfeat[3, offset1 + k] += bk * o[3]

                    if k <= n2
                        Mfeat[1, offset2 + k] += bk * p[1]
                        Mfeat[2, offset2 + k] += bk * p[2]
                        Mfeat[3, offset2 + k] += bk * p[3]
                        Mfeat[4, offset2 + k] += bk * p[4]
                        Mfeat[5, offset2 + k] += bk * p[5]
                        Mfeat[6, offset2 + k] += bk * p[6]

                        if k <= n3
                            for kk in 1:10
                                Nfeat[kk, offset3 + k] += bk * q[kk]
                            end
                        end
                    end
                end
            end
        end

        # ------------------------------------------------------------
        # reverse (A): (u_F,u_V) -> dEd*_bar
        # ------------------------------------------------------------
        for jlist in axes(nlist, 2)
            j = nlist[i, jlist]
            j < 1 && break

            x_ij, y_ij, z_ij, rij = displacement(
                x_i, y_i, z_i,
                pos[j,1], pos[j,2], pos[j,3],
                shiftx[i,jlist], shifty[i,jlist], shiftz[i,jlist],
                lattice
            )
            (rij <= rmin || rij >= rmax) && continue

            sj = elemid[j]
            offset0 = n0 * (sj - 1)
            offset1 = n1 * (sj - 1)
            offset2 = n2 * (sj - 1)
            offset3 = n3 * (sj - 1)

            invr = one(FP) / rij
            x̂ = x_ij * invr
            ŷ = y_ij * invr
            ẑ = z_ij * invr

            # pair-force adjoint from F + virial (packed symmetric)
            fbx = u_F[i,1] - u_F[j,1]
            fby = u_F[i,2] - u_F[j,2]
            fbz = u_F[i,3] - u_F[j,3]

            fbx += uV1*x_ij + uV5*(z_ij*half) + uV6*(y_ij*half)
            fby += uV2*y_ij + uV4*(z_ij*half) + uV6*(x_ij*half)
            fbz += uV3*z_ij + uV4*(y_ij*half) + uV5*(x_ij*half)

            rdot_fb = x̂*fbx + ŷ*fby + ẑ*fbz

            cheb_taper_basis!(B0, dB0, rij, rmin, rmax, ron)

            # l=3 STF (q and grads in rhat-space) needed only if t<=n3
            q, gx, gy, gz = stf3_10_and_grad(x̂, ŷ, ẑ)

            @inbounds for t in 1:n0
                # l=0
                dEdS_bar[offset0 + t] += -dB0[t] * rdot_fb

                # l>0 (same basis, truncated)
                if t <= n1
                    bb = B0[t]
                    db = dB0[t]
                    a  = bb * invr
                    k  = db - a

                    μ1 = offset1 + t
                    dEdV_bar[1,μ1] += -(a*fbx + k*x̂*rdot_fb)
                    dEdV_bar[2,μ1] += -(a*fby + k*ŷ*rdot_fb)
                    dEdV_bar[3,μ1] += -(a*fbz + k*ẑ*rdot_fb)

                    if t <= n2
                        μ2 = offset2 + t
                        α  = db
                        γ  = FP(2) * bb * invr
                        k1 = (γ - α)
                        k2 = (α / FP(3))
                        s1 = rdot_fb

                        H11 = s1*(k1*x̂*x̂ + k2) - γ*fbx*x̂
                        H12 = s1*(k1*x̂*ŷ)      - γ*fbx*ŷ
                        H13 = s1*(k1*x̂*ẑ)      - γ*fbx*ẑ
                        H21 = s1*(k1*ŷ*x̂)      - γ*fby*x̂
                        H22 = s1*(k1*ŷ*ŷ + k2) - γ*fby*ŷ
                        H23 = s1*(k1*ŷ*ẑ)      - γ*fby*ẑ
                        H31 = s1*(k1*ẑ*x̂)      - γ*fbz*x̂
                        H32 = s1*(k1*ẑ*ŷ)      - γ*fbz*ŷ
                        H33 = s1*(k1*ẑ*ẑ + k2) - γ*fbz*ẑ

                        dEdM_bar[1,μ2] += H11
                        dEdM_bar[2,μ2] += H22
                        dEdM_bar[3,μ2] += H33
                        dEdM_bar[6,μ2] += (H12 + H21)
                        dEdM_bar[4,μ2] += (H23 + H32)
                        dEdM_bar[5,μ2] += (H31 + H13)

                        if t <= n3
                            μ3 = offset3 + t
                            for kcomp in 1:10
                                qk  = q[kcomp]
                                ghx = gx[kcomp]; ghy = gy[kcomp]; ghz = gz[kcomp]
                                gdot = ghx*x̂ + ghy*ŷ + ghz*ẑ

                                projx = ghx - gdot*x̂
                                projy = ghy - gdot*ŷ
                                projz = ghz - gdot*ẑ

                                fbproj = fbx*projx + fby*projy + fbz*projz
                                dEdN_bar[kcomp,μ3] += -( db*qk*rdot_fb + (bb*invr)*fbproj )
                            end
                        end
                    end
                end
            end
        end

        # ------------------------------------------------------------
        # reverse (B): scatter to coefficients (match corrected calculate!)
        # ------------------------------------------------------------

        aceg.cp[si] += u_E

        # c0
        @inbounds for μ in 1:nch0
            aceg.c0[μ,si] += u_E*S[μ] + dEdS_bar[μ]
        end

        # c00
        @inbounds for μ in 1:nch0
            Sμ = S[μ]
            for ν in μ:nch0
                Sν = S[ν]
                aceg.c00[μ,ν,si] += u_E*(Sμ*Sν) + dEdS_bar[μ]*Sν + dEdS_bar[ν]*Sμ
            end
        end

        # c11
        @inbounds for μ in 1:nch1
            Vμ1,Vμ2,Vμ3 = Vfeat[1,μ], Vfeat[2,μ], Vfeat[3,μ]
            for ν in μ:nch1
                Vν1,Vν2,Vν3 = Vfeat[1,ν], Vfeat[2,ν], Vfeat[3,ν]
                dotV = Vμ1*Vν1 + Vμ2*Vν2 + Vμ3*Vν3
                bμ = dEdV_bar[1,μ]*Vν1 + dEdV_bar[2,μ]*Vν2 + dEdV_bar[3,μ]*Vν3
                bν = dEdV_bar[1,ν]*Vμ1 + dEdV_bar[2,ν]*Vμ2 + dEdV_bar[3,ν]*Vμ3
                aceg.c11[μ,ν,si] += u_E*dotV + bμ + bν
            end
        end

        # c22
        @inbounds for μ in 1:nch2
            A = (Mfeat[1,μ], Mfeat[2,μ], Mfeat[3,μ], Mfeat[4,μ], Mfeat[5,μ], Mfeat[6,μ])
            for ν in μ:nch2
                B = (Mfeat[1,ν], Mfeat[2,ν], Mfeat[3,ν], Mfeat[4,ν], Mfeat[5,ν], Mfeat[6,ν])
                trM2 = trmatdot(A, B)
                barμ = dEdM_bar[1,μ]*B[1] + dEdM_bar[2,μ]*B[2] + dEdM_bar[3,μ]*B[3] +
                       dEdM_bar[4,μ]*B[4] + dEdM_bar[5,μ]*B[5] + dEdM_bar[6,μ]*B[6]
                barν = dEdM_bar[1,ν]*A[1] + dEdM_bar[2,ν]*A[2] + dEdM_bar[3,ν]*A[3] +
                       dEdM_bar[4,ν]*A[4] + dEdM_bar[5,ν]*A[5] + dEdM_bar[6,ν]*A[6]
                aceg.c22[μ,ν,si] += u_E*trM2 + barμ + barν
            end
        end

        # c33
        @inbounds for μ in 1:nch3
            for ν in μ:nch3
                trN2 = stf3_dot10(@view(Nfeat[:,μ]), @view(Nfeat[:,ν]))
                barμ = stf3_dot10(@view(dEdN_bar[:,μ]), @view(Nfeat[:,ν]))
                barν = stf3_dot10(@view(dEdN_bar[:,ν]), @view(Nfeat[:,μ]))
                aceg.c33[μ,ν,si] += u_E*trN2 + barμ + barν
            end
        end

        # c011  (same ν>=λ convention as calculate)
        @inbounds for λ in 1:nch1
            Vλ1,Vλ2,Vλ3 = Vfeat[1,λ], Vfeat[2,λ], Vfeat[3,λ]
            for ν in λ:nch1
                Vν1,Vν2,Vν3 = Vfeat[1,ν], Vfeat[2,ν], Vfeat[3,ν]
                dotV = Vν1*Vλ1 + Vν2*Vλ2 + Vν3*Vλ3

                # bν = dEdV_bar[:,ν] · Vλ , bλ = dEdV_bar[:,λ] · Vν
                bν = dEdV_bar[1,ν]*Vλ1 + dEdV_bar[2,ν]*Vλ2 + dEdV_bar[3,ν]*Vλ3
                bλ = dEdV_bar[1,λ]*Vν1 + dEdV_bar[2,λ]*Vν2 + dEdV_bar[3,λ]*Vν3

                for μ in 1:nch0
                    Sμ = S[μ]
                    coeff = aceg.c011[μ,ν,λ,si] # not used, just for readability

                    aceg.c011[μ,ν,λ,si] +=
                        u_E*(Sμ*dotV) +
                        dEdS_bar[μ]*dotV +
                        Sμ*(bν + bλ)
                end
            end
        end

        # c022  (same ν>=λ convention)
        @inbounds for λ in 1:nch2
            A = (Mfeat[1,λ], Mfeat[2,λ], Mfeat[3,λ], Mfeat[4,λ], Mfeat[5,λ], Mfeat[6,λ])
            for ν in λ:nch2
                B = (Mfeat[1,ν], Mfeat[2,ν], Mfeat[3,ν], Mfeat[4,ν], Mfeat[5,ν], Mfeat[6,ν])
                trM2 = trmatdot(A, B)

                barν = dEdM_bar[1,ν]*A[1] + dEdM_bar[2,ν]*A[2] + dEdM_bar[3,ν]*A[3] +
                       dEdM_bar[4,ν]*A[4] + dEdM_bar[5,ν]*A[5] + dEdM_bar[6,ν]*A[6]
                barλ = dEdM_bar[1,λ]*B[1] + dEdM_bar[2,λ]*B[2] + dEdM_bar[3,λ]*B[3] +
                       dEdM_bar[4,λ]*B[4] + dEdM_bar[5,λ]*B[5] + dEdM_bar[6,λ]*B[6]

                for μ in 1:nch0
                    Sμ = S[μ]
                    aceg.c022[μ,ν,λ,si] +=
                        u_E*(Sμ*trM2) +
                        dEdS_bar[μ]*trM2 +
                        Sμ*(barν + barλ)
                end
            end
        end

        # c033  (same ν>=λ convention)
        @inbounds for λ in 1:nch3
            for ν in λ:nch3
                dotN2 = stf3_dot10(@view(Nfeat[:,ν]), @view(Nfeat[:,λ]))

                barν = stf3_dot10(@view(dEdN_bar[:,ν]), @view(Nfeat[:,λ]))
                barλ = stf3_dot10(@view(dEdN_bar[:,λ]), @view(Nfeat[:,ν]))

                for μ in 1:nch0
                    Sμ = S[μ]
                    aceg.c033[μ,ν,λ,si] +=
                        u_E*(Sμ*dotN2) +
                        dEdS_bar[μ]*dotN2 +
                        Sμ*(barν + barλ)
                end
            end
        end
    end

    return aceg
end
