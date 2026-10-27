using LinearAlgebra
import AbstractFFTs as AFFTs
using FFTW
using StaticArrays
import KernelAbstractions as KA
import AcceleratedKernels as AK
# 電荷スプレッドの格子への足し込みに使う（Atomix は何も export しないので
# `using Atomix` ではマクロが入らない）．Base.@atomic は添字形も構文としては
# 解釈するが，modifyindex_atomic! が GenericMemory 専用で Array / device array の
# メソッドを持たないため使えない．Issue #127 まではこの import が無く，先に
# include される Spring.jl / Ewald.jl の（本文では未使用の）import に依存していた．
using Atomix: @atomic
using FMD: FPPREC

export PMEInfo, calculate!, cellupdate!, pme_fftsize

const BSPORDER = 4

@inline @fastmath function spline(s, n)
    x0 = max(s - n, zero(s))^3
    x1 = max(s - n - 1, zero(s))^3
    x2 = max(s - n - 2, zero(s))^3
    x3 = max(s - n - 3, zero(s))^3
    x4 = max(s - n - 4, zero(s))^3
    return (x0 - 4*x1 + 6*x2 - 4*x3 + x4) / 6
end

@inline @fastmath function dspline(s, n)
    x0 = max(s - n, zero(s))^2
    x1 = max(s - n - 1, zero(s))^2
    x2 = max(s - n - 2, zero(s))^2
    x3 = max(s - n - 3, zero(s))^2
    x4 = max(s - n - 4, zero(s))^2
    return (x0 - 4*x1 + 6*x2 - 4*x3 + x4)/2
end

# Second derivative of the order-4 B-spline basis (analytic HVP,
# Issue #147): piecewise linear (C⁰) — one order less continuous than the
# force interpolation, which is why the HVP local term is the sole probe
# of the BSPORDER=4 accuracy question (theory note §5).
@inline @fastmath function d2spline(s, n)
    x0 = max(s - n, zero(s))
    x1 = max(s - n - 1, zero(s))
    x2 = max(s - n - 2, zero(s))
    x3 = max(s - n - 3, zero(s))
    x4 = max(s - n - 4, zero(s))
    return x0 - 4*x1 + 6*x2 - 4*x3 + x4
end

function bsp_weight(u, order)
    if u <= 0 || order <= u
        return zero(FPPREC)
    elseif order == 2
        return one(FPPREC) - abs(u - 1)
    else
        return  (u / (order - 1)) * bsp_weight(u,order-1) + ((order - u)/(order - 1)) * bsp_weight(u - 1, order-1)
    end
end

"""
    pme_fftsize(a, b, c, pme_meshsize) -> NTuple{3,Int}

FFT grid dimensions for cell edge lengths `a, b, c` (atomic units) at the
requested mesh spacing: `ceil(side / pme_meshsize)`, rounded **up** to the
next 2·3·5·7-smooth integer, floored at `BSPORDER`.

The smooth rounding is a performance guard, not an accuracy knob: FFTW and
CUFFT fall off a cliff on sizes with large prime factors (measured on this
project's grids: a 61³ rFFT+irFFT pair is 1.7× slower than 64³ despite 17%
fewer points), and `ceil` alone lands on primes for perfectly ordinary
cells — e.g. a 16³ BaTiO₃ supercell at `pme_meshsize = 2.0` gives 61.
Rounding up only ever refines the grid, so the documented contract (each
side has at least `side / pme_meshsize` cells) is unchanged.

Single-sourced on purpose: [`PMEInfo`](@ref) sizes its grid here and the
α auto-determination ([`pme_optimal_alpha`](@ref)) estimates the
reciprocal-space error for the same grid — a drift between the two would
optimize α against a mesh the solver does not use.

The `BSPORDER` floor (not `BSPORDER - 1`) is wrap safety for the B-spline
spread: the smallest wrapped index is `fftsize - (BSPORDER-1)`, which at
`fftsize == BSPORDER - 1` is an out-of-bounds 0 hidden by
`inbounds = true` (Issue #53).
"""
function pme_fftsize(a, b, c, pme_meshsize)
    raw = ceil.(Int, (a, b, c) ./ pme_meshsize)
    # smooth-round THEN floor: the order only preserves smoothness because
    # BSPORDER itself is 2357-smooth — assert that so a future BSPORDER
    # change cannot silently reintroduce a non-smooth grid
    @assert ispow2(BSPORDER) "pme_fftsize assumes a power-of-two BSPORDER"
    return max.(nextprod.(Ref((2, 3, 5, 7)), raw), BSPORDER)
end

# (_devcopy — host → compute-backend copy — lives in Calculators.jl,
# shared with the DD reduction helper.)

"""
    PMEInfo <: CoulombInfo

Particle-mesh Ewald accelerator. Caches the rFFT plans, the modified
B-spline shape factor `bsp2factor`, the Gaussian-screened convolution
factor `expfactor`, and the per-cell pressure-tensor factor `presfactor`.

Construct with `PMEInfo(cell, params)`; call `cellupdate!` after a cell
change. Required `params`: `:alpha` (Ewald α), `:pme_meshsize` (target
real-space mesh spacing in atomic units — the FFT grid is sized by
[`pme_fftsize`](@ref): each cell side gets at least `(side / pme_meshsize)`
cells, rounded up to a 2·3·5·7-smooth count for FFT speed and floored at
`BSPORDER` cells per side for wrap safety of the B-spline spread, so a
cell smaller than `BSPORDER*pme_meshsize` per side gets a finer grid than
requested).

Prefer `PMEInfo` over [`EwaldInfo`](@ref) for production NPT MD; the
analytic Ewald is dominated by O(N²) k-space summation while PME scales
as O(N log N). The two share the [`calculate!`](@ref) interface so
swapping them is a single-line change.
"""
struct PMEInfo <: CoulombInfo
    rplan
    irplan
    fftsize::NTuple{3,Int}
    rfftsize::NTuple{3,Int}
    expfactor
    bsp2factor
    presfactor
    # z-slab distributed solver state (PMESlab.jl), `nothing` on the
    # replicated path. Untyped like the other fields; the replicated
    # plans/factors above are kept even when `dd` is set — the HVP path
    # and Ewald-style callers stay on them.
    dd
end

"""
    PMEInfo(cell, params) -> PMEInfo

Build a particle-mesh Ewald accelerator sized for `cell` with mesh spacing
`params[:pme_meshsize]` (atomic units). Allocates the backend buffers
(CPU or GPU, from the array type of `cell`), plans the rFFT, and caches
the B-spline shape factor (cell-independent).
Calls [`cellupdate!`](@ref) once before returning so the resulting object
is ready to feed [`calculate!`](@ref).
"""
function PMEInfo(cell, params)
    backend = get_backend(cell)

    cell_cpu = Array(cell)
    a, b, c = norm(cell_cpu[1,:]), norm(cell_cpu[2,:]), norm(cell_cpu[3,:])
    # ceil → 2357-smooth rounding → BSPORDER floor; see pme_fftsize for
    # the rationale of each stage (Issues #53 and #216).
    fftsize = pme_fftsize(a, b, c, params[:pme_meshsize])
    rfftsize = (fftsize[1]÷2+1, fftsize[2], fftsize[3])

    plan_rfft = AFFTs.plan_rfft(KA.zeros(backend, FPPREC, fftsize))
    plan_irfft = AFFTs.plan_brfft(KA.zeros(backend, Complex{FPPREC}, rfftsize), fftsize[1])

    expfactor = KA.zeros(backend, FPPREC, rfftsize...)
    bsp2factor = zeros(rfftsize...)
    presfactor = KA.zeros(backend, FPPREC, rfftsize..., 6)

    bspfactor1d = [zeros(Complex, rfftsize[i]) for i in 1:3]

    for d in 1:3, i in 0:rfftsize[d]-1
        ∑Mexp = 0.0 + 0.0im
        for j in 0:BSPORDER-2
            ∑Mexp += bsp_weight(j+1, BSPORDER)*exp(2*pi*1im*i*j/fftsize[d])
        end
        bspfactor1d[d][i+1] = exp(2*pi*1im*(BSPORDER - 1)*i/fftsize[d])/∑Mexp
    end

    for k in axes(bsp2factor,3), j in axes(bsp2factor,2), i in axes(bsp2factor,1)
        bsp2factor[i,j,k] = abs(bspfactor1d[1][i])^2 * abs(bspfactor1d[2][j])^2 * abs(bspfactor1d[3][k])^2
    end

    bsp2factor = _devcopy(backend, FPPREC.(bsp2factor))

    pmeinfo = PMEInfo(plan_rfft, plan_irfft, fftsize, rfftsize, expfactor, bsp2factor,
                      presfactor, nothing)
    cellupdate!(pmeinfo, cell, params)
    return pmeinfo
end

# `dims` is the local shape of the (possibly ky-slab) factor arrays and
# `yoff` the global ky offset of their second dimension: the replicated
# path launches with dims = rfftsize, yoff = 0, the slab path
# (PMESlab.jl) with dims = (nx2, nyloc, nz) and its rank's offset. One
# kernel for both keeps the factor formulas single-sourced.
@kernel inbounds = true function kernel_cellupdate!(expfactor, presfactor, rcell, volume, alpha, dims, yoff, fftsize)
    index = @index(Global)

    i = (index - 1) % dims[1]
    jl = ((index - 1) ÷ dims[1]) % dims[2]
    k = ((index - 1) ÷ (dims[1]*dims[2])) % dims[3]
    j = jl + yoff                                  # global ky index (0-based)
    mi = ifelse(i > fftsize[1]/2, i-fftsize[1], i)
    mj = ifelse(j > fftsize[2]/2, j-fftsize[2], j)
    mk = ifelse(k > fftsize[3]/2, k-fftsize[3], k)

    if i == 0 && j == 0 && k == 0
        expfactor[i+1,jl+1,k+1] = zero(eltype(expfactor))
        presfactor[i+1,jl+1,k+1,:] .= zero(eltype(presfactor))
    else
        rvx = mi*rcell[1,1] + mj*rcell[2,1] + mk*rcell[3,1]
        rvy = mi*rcell[1,2] + mj*rcell[2,2] + mk*rcell[3,2]
        rvz = mi*rcell[1,3] + mj*rcell[2,3] + mk*rcell[3,3]
        m² = rvx*rvx + rvy*rvy + rvz*rvz
        expfactor[i+1,jl+1,k+1] = 1 / (volume*FPPREC(pi)) * exp(-m²*(FPPREC(pi)/alpha)^2) / m²
        factor = ifelse(i == 0 || (i == fftsize[1]/2 && iseven(fftsize[1])), FPPREC(1.0), FPPREC(2.0))
        presfactor[i+1,jl+1,k+1,1] = factor * expfactor[i+1,jl+1,k+1] /2 * (1 - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvx*rvx)/m²)
        presfactor[i+1,jl+1,k+1,2] = factor * expfactor[i+1,jl+1,k+1] /2 * (1 - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvy*rvy)/m²)
        presfactor[i+1,jl+1,k+1,3] = factor * expfactor[i+1,jl+1,k+1] /2 * (1 - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvz*rvz)/m²)
        presfactor[i+1,jl+1,k+1,4] = factor * expfactor[i+1,jl+1,k+1] /2 * (  - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvy*rvz)/m²)
        presfactor[i+1,jl+1,k+1,5] = factor * expfactor[i+1,jl+1,k+1] /2 * (  - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvz*rvx)/m²)
        presfactor[i+1,jl+1,k+1,6] = factor * expfactor[i+1,jl+1,k+1] /2 * (  - 2*(1 + (FPPREC(pi)/alpha)^2 * m²)*(rvx*rvy)/m²)
    end
end

"""
    cellupdate!(pmeinfo::PMEInfo, cell, params) -> PMEInfo

Recompute `expfactor` and `presfactor` for the new `cell` (called once per
NPT MD step). Mutates and returns `pmeinfo`. The B-spline shape factor
`bsp2factor` is cell-independent and is not touched.
"""
function cellupdate!(pmeinfo::PMEInfo, cell, params)
    backend = get_backend(cell)

    (; fftsize, rfftsize, expfactor, presfactor) = pmeinfo
    cell = Array(cell)
    volume = dot(cell[1,:], cross(cell[2,:], cell[3,:]))
    rcell = (cell \ I)'

    kernel_cellupdate!(backend, 512, prod(rfftsize))(
        expfactor, presfactor,
        SMatrix{3,3,FPPREC}(rcell), FPPREC(volume),
        FPPREC(params[:alpha]), rfftsize, 0, fftsize
    )
    # Slab-decomposed factors ride along (once per MD step, ~1/P of the
    # full-grid work — negligible next to the launch above). Passes this
    # call's rcell/volume so the cell geometry has a single derivation.
    pmeinfo.dd === nothing ||
        _cellupdate_slab!(pmeinfo.dd, SMatrix{3,3,FPPREC}(rcell), FPPREC(volume),
                          FPPREC(params[:alpha]))
    return pmeinfo
end

@kernel inbounds = true function kernel_calcweights!(weights, dweights, fractionals, grididx, fftsize)
    index = @index(Global)

    s1 = fractionals[index,1] * fftsize[1]
    s2 = fractionals[index,2] * fftsize[2]
    s3 = fractionals[index,3] * fftsize[3]

    grididx[index,1] = floor(Int, s1)
    grididx[index,2] = floor(Int, s2)
    grididx[index,3] = floor(Int, s3)

    u1 = s1 - grididx[index,1]
    u2 = s2 - grididx[index,2]
    u3 = s3 - grididx[index,3]
    for l in 1:BSPORDER
        n = -BSPORDER + l
        weights[index,1,l]  = spline(u1, n)
        weights[index,2,l]  = spline(u2, n)
        weights[index,3,l]  = spline(u3, n)
        dweights[index,1,l] = dspline(u1, n)
        dweights[index,2,l] = dspline(u2, n)
        dweights[index,3,l] = dspline(u3, n)
    end
end

function interpolation_weights(pmeinfo::PMEInfo, fractionals)
    backend = get_backend(fractionals)

    weights = KA.zeros(backend, FPPREC, (size(fractionals)..., BSPORDER))
    dweights = KA.zeros(backend, FPPREC, (size(fractionals)..., BSPORDER))
    grididx = KA.zeros(backend, Int32, size(fractionals))

    # Empty frames occur on DD ranks that own no atoms; they still have to
    # reach the grid reduction downstream, so return the (empty) tables
    # rather than relying on a zero-ndrange launch being legal on every
    # backend.
    size(weights, 1) > 0 &&
        kernel_calcweights!(backend, 512, size(weights,1))(weights, dweights, fractionals, grididx, pmeinfo.fftsize)

    return grididx, weights, dweights
end

# Analytic-HVP variant (Issue #147): same node bookkeeping as
# kernel_calcweights!, plus the second-derivative weights.
@kernel inbounds = true function kernel_calcweights2!(weights, dweights, d2weights, fractionals, grididx, fftsize)
    index = @index(Global)

    s1 = fractionals[index,1] * fftsize[1]
    s2 = fractionals[index,2] * fftsize[2]
    s3 = fractionals[index,3] * fftsize[3]

    grididx[index,1] = floor(Int, s1)
    grididx[index,2] = floor(Int, s2)
    grididx[index,3] = floor(Int, s3)

    u1 = s1 - grididx[index,1]
    u2 = s2 - grididx[index,2]
    u3 = s3 - grididx[index,3]
    for l in 1:BSPORDER
        n = -BSPORDER + l
        weights[index,1,l]  = spline(u1, n)
        weights[index,2,l]  = spline(u2, n)
        weights[index,3,l]  = spline(u3, n)
        dweights[index,1,l] = dspline(u1, n)
        dweights[index,2,l] = dspline(u2, n)
        dweights[index,3,l] = dspline(u3, n)
        d2weights[index,1,l] = d2spline(u1, n)
        d2weights[index,2,l] = d2spline(u2, n)
        d2weights[index,3,l] = d2spline(u3, n)
    end
end

function interpolation_weights2(pmeinfo::PMEInfo, fractionals)
    backend = get_backend(fractionals)

    weights = KA.zeros(backend, FPPREC, (size(fractionals)..., BSPORDER))
    dweights = KA.zeros(backend, FPPREC, (size(fractionals)..., BSPORDER))
    d2weights = KA.zeros(backend, FPPREC, (size(fractionals)..., BSPORDER))
    grididx = KA.zeros(backend, Int32, size(fractionals))

    size(weights, 1) > 0 &&
        kernel_calcweights2!(backend, 512, size(weights,1))(weights, dweights, d2weights, fractionals, grididx, pmeinfo.fftsize)

    return grididx, weights, dweights, d2weights
end

@kernel inbounds = true function kernel_spread!(gridcharges, grididx, charges, weights, fftsize)
    index = @index(Global)
    qᵢ = charges[index]
    g₁ = grididx[index,1]
    g₂ = grididx[index,2]
    g₃ = grididx[index,3]
    for l3 in 1:BSPORDER, l2 in 1:BSPORDER, l1 in 1:BSPORDER
        gx = g₁ - BSPORDER + l1
        gx += ifelse(gx < 1, fftsize[1], zero(eltype(fftsize)))
        gx -= ifelse(gx > fftsize[1], fftsize[1], zero(eltype(fftsize)))
        gy = g₂ - BSPORDER + l2
        gy += ifelse(gy < 1, fftsize[2], zero(eltype(fftsize)))
        gy -= ifelse(gy > fftsize[2], fftsize[2], zero(eltype(fftsize)))
        gz = g₃ - BSPORDER + l3
        gz += ifelse(gz < 1, fftsize[3], zero(eltype(fftsize)))
        gz -= ifelse(gz > fftsize[3], fftsize[3], zero(eltype(fftsize)))

        @atomic gridcharges[gx,gy,gz] += qᵢ * weights[index,1,l1] * weights[index,2,l2] * weights[index,3,l3]
    end
end

function spread!(chargesᵍ, pmeinfo::PMEInfo, grididx, charges, weights)
    backend = get_backend(charges)
    kernel_spread!(backend, 512, length(charges))(chargesᵍ, grididx, charges, weights, Int32.(pmeinfo.fftsize))
end

@kernel inbounds = true function kernel_gather!(U, forces, grididx, q, G, fftsize, weights, dweights, icell)
    index = @index(Global)
    Uᵢ = U[index]
    fx = zero(FPPREC)
    fy = zero(FPPREC)
    fz = zero(FPPREC)
    qᵢ = q[index]
    g₁ = grididx[index,1]
    g₂ = grididx[index,2]
    g₃ = grididx[index,3]
    for l3 in 1:BSPORDER, l2 in 1:BSPORDER, l1 in 1:BSPORDER
        gx = g₁ - BSPORDER + l1
        gx += ifelse(gx < 1, fftsize[1], zero(eltype(fftsize)))
        gx -= ifelse(gx > fftsize[1], fftsize[1], zero(eltype(fftsize)))
        gy = g₂ - BSPORDER + l2
        gy += ifelse(gy < 1, fftsize[2], zero(eltype(fftsize)))
        gy -= ifelse(gy > fftsize[2], fftsize[2], zero(eltype(fftsize)))
        gz = g₃ - BSPORDER + l3
        gz += ifelse(gz < 1, fftsize[3], zero(eltype(fftsize)))
        gz -= ifelse(gz > fftsize[3], fftsize[3], zero(eltype(fftsize)))

        Gq = G[gx,gy,gz] * qᵢ
        Uᵢ += FPPREC(0.5)*Gq * weights[index,1,l1]*weights[index,2,l2]*weights[index,3,l3]
        fx -= Gq * weights[index,2,l2] * weights[index,3,l3] * dweights[index,1,l1] * fftsize[1]
        fy -= Gq * weights[index,1,l1] * weights[index,3,l3] * dweights[index,2,l2] * fftsize[2]
        fz -= Gq * weights[index,1,l1] * weights[index,2,l2] * dweights[index,3,l3] * fftsize[3]
    end
    U[index] = Uᵢ
    forces[index,1] += fx * icell[1,1] + fy * icell[1,2] + fz * icell[1,3]
    forces[index,2] += fx * icell[2,1] + fy * icell[2,2] + fz * icell[2,3]
    forces[index,3] += fx * icell[3,1] + fy * icell[3,2] + fz * icell[3,3]
end

function gather!(U, forces, pmeinfo::PMEInfo, grididx, charges, cell, gridenergy, weights, dweights)
    backend = get_backend(charges)
    icell = SMatrix{3,3,FPPREC}(inv(Array(cell)))
    kernel_gather!(backend, 512, length(charges))(U, forces, grididx, charges, gridenergy, Int32.(pmeinfo.fftsize), weights, dweights, icell)
end

"""
    _pme_convolve(pmeinfo::PMEInfo, grid) -> potential grid

Reciprocal-space convolution `G ∗ grid` (forward rFFT, multiply by
`expfactor·bsp2factor`, inverse rFFT) **without** the virial: the extra
solve of the analytic HVP (Issue #147), where the dipole grid of the
direction vector rides through the identical operator as the charge grid,
and the shell-force-only pass of the CG inner loop (Issue #157), which
convolves the ordinary charge grid but discards the virial.

!!! warning "Keep this identical to [`_pme_solve`](@ref)"
    The convolution below and the `gridenergy` line of `_pme_solve` are
    the same expression written twice, and since Issue #157 both are on
    the force path — the CG inner loop takes this one and the closing
    core evaluation takes that one. Any drift between them (factor order,
    fusion, a different temporary) changes the MD trajectory. The CPU
    bit-identity assertion in `test/calculators/test_pme.jl`
    ("OnlyShellForce reciprocal path") is what pins them together.
"""
function _pme_convolve(pmeinfo::PMEInfo, grid)
    return _pme_convolve_hat(pmeinfo, pmeinfo.rplan * grid)
end

"""
    _pme_convolve_hat(pmeinfo::PMEInfo, gridf) -> potential grid

Second half of [`_pme_convolve`](@ref), taking the forward transform
directly. Callers that also need `gridf` — the analytic HVP, whose virial
derivative is the cross term between the charge and dipole transforms
(Issue #151) — take it once and pass it here; the broadcast temporary
protects `gridf` from the destructive c2r plan, as in [`_pme_solve`](@ref).
"""
function _pme_convolve_hat(pmeinfo::PMEInfo, gridf)
    return pmeinfo.irplan * (gridf .* pmeinfo.expfactor .* pmeinfo.bsp2factor)
end

"""
    _pme_presfactor_contract(pmeinfo::PMEInfo, w) -> Vector{FPPREC} (Voigt-6)

Contract a real reciprocal-grid weight `w` with the six components of
`pmeinfo.presfactor`, returning the result as a host Voigt-6 vector. This
is the shape of every reciprocal-space stress quantity: [`_pme_solve`](@ref)
passes `-bsp2factor·|Q̂|²` for the virial itself, `hvp_coulomb!` passes
`bsp2factor·2Re(conj(Q̂)·Q̂_v)` for its directional derivative (Issue #151).

Written as one gemv against a `(M, 6)` reshape of `presfactor` rather than
six reductions. The obvious spelling,

    [sum(@. w * pmeinfo.presfactor[:, :, :, d]) for d in 1:6]

is ~9× slower on the GPU and not because of arithmetic: `@.` does not
broadcast an index expression, so each component materializes a grid-sized
copy of the slice, and each `sum` ends in its own device→host readback —
six grid allocations and six synchronization points where one gemv and one
readback do. `presfactor` is contiguous with the component index slowest,
so both reshapes are free.

`w` must be a materialized dense grid of `rfftsize`, not a view: the
reshape wants contiguity, and a strided argument would fall back to a
generic (on the GPU, scalar-indexing) path. The return type is asserted
because `PMEInfo`'s fields are untyped, so inference cannot see it.
"""
function _pme_presfactor_contract(pmeinfo::PMEInfo, w)
    M = prod(pmeinfo.rfftsize)
    return Array(reshape(pmeinfo.presfactor, M, 6)' * reshape(w, M))::Vector{FPPREC}
end

"""
    _pme_solve(pmeinfo::PMEInfo, gridcharge) -> (gridenergy, virial6)

Reciprocal-space solve on the (full) charge grid: forward rFFT, convolution
with `expfactor·bsp2factor`, inverse rFFT back to the real-space energy
grid, plus the 6-component reciprocal virial as a host vector.

This is the grid-global stage of the PME pipeline and the boundary at
which a distributed (e.g. slab-decomposed) solver can be substituted
later. Note that `spread!`/`gather!` are local in the *atom* index only —
the grid points an atom touches are global, so a distributed
implementation still needs a charge-grid reduction (halo exchange or
Allreduce) before this stage and an energy-grid distribution after it.

The `gridenergy` line must stay identical to [`_pme_convolve`](@ref) —
see the warning there.
"""
function _pme_solve(pmeinfo::PMEInfo, gridcharge)
    gridchargeᶠ = pmeinfo.rplan * gridcharge
    # brfft may destroy its input (FFTW c2r); pass the broadcast temporary,
    # never gridchargeᶠ itself — it is still needed for the virial below.
    gridenergy = pmeinfo.irplan * (gridchargeᶠ .* pmeinfo.expfactor .* pmeinfo.bsp2factor)
    virial6 = _pme_presfactor_contract(pmeinfo,
                                       @. -pmeinfo.bsp2factor * abs2(gridchargeᶠ))
    return gridenergy, virial6
end

"""
    _pme_site_charges(atoms::Atoms, params) -> (cpos_r, spos_r, qc, qs)

Preamble shared by every reciprocal-space pass over the frame: the wrapped
fractional coordinates of both site families and the per-atom core / shell
charge arrays on the compute backend.

Wrapped fractionals, because `kernel_spread!`'s ±1 `ifelse` corrects a
single period only: unwrapped `:pos` drifting beyond one box would index
the grid out of range (latent even in serial, over a long NPT run). The
charge assignment is periodic, so the wrap is physically exact and `:pos`
itself is left alone (gauge). The shell fractional is wrapped
independently for the same reason — on the grid a shell is just another
point charge — which keeps it inside the ±1 budget too; the intra-atom
correction reads `:shelldisp` directly and is unaffected.

Empty domain-decomposition ranks get correctly-shaped empty arrays: the
`AK.map!` calls are guarded, the broadcasts are not (they are no-ops).
"""
function _pme_site_charges(atoms::Atoms, params)
    backend = get_backend(atoms.info[:lattice])
    cpos_r = Cores.wrapfrac.(atoms.arrays[:pos])
    spos_r = Cores.wrapfrac.(cpos_r .+ atoms.arrays[:shelldisp] * _devcopy(backend, FPPREC.(Array(atoms.info[:lattice]) \ I)))
    prm_qc = params[:qc]
    prm_qs = params[:qs]
    qc = similar(atoms.arrays[:elemid], FPPREC)
    qs = similar(atoms.arrays[:elemid], FPPREC)
    if length(atoms) > 0
        AK.map!(x -> prm_qc[x], qc, atoms.arrays[:elemid])
        AK.map!(x -> prm_qs[x], qs, atoms.arrays[:elemid])
    end
    return cpos_r, spos_r, qc, qs
end

"""
    calculate!(pmeinfo::PMEInfo, atoms::Atoms, params)

Add the PME reciprocal-space + self-energy contributions to
`atoms.info[:potential_energy]`, `atoms.info[:virial]`,
`atoms.arrays[:forces]`, and `atoms.arrays[:shellforces]`. The real-space
short-range part is handled separately by [`CoulombBuckingham`](@ref).

For a non-neutral cell the charged-cell correction is included via
[`chargedcell_correction!`](@ref) (defined in Ewald.jl), matching
[`EwaldInfo`](@ref) (Issue #15).

Required `params`: `:alpha`, `:qc`, `:qs`. The grid sizes and B-spline
shape factor live inside `pmeinfo` and were captured at construction time
from `params[:pme_meshsize]`.
"""
function calculate!(pmeinfo::PMEInfo, atoms::Atoms, params)
    backend = get_backend(atoms.info[:lattice])
    dom = Cores.domain(atoms)
    N = length(atoms)
    gridcharge = KA.zeros(backend, FPPREC, pmeinfo.fftsize)

    cpos_r, spos_r, qc, qs = _pme_site_charges(atoms, params)

    # NOTE: this block and the gather block below share the same `N > 0`
    # guard — the gridindex/weights locals bound here are read there.
    # Removing one guard without the other turns a compile-time property
    # into a runtime UndefVarError.
    if N > 0
        gridindex_c, weights_c, dweights_c = interpolation_weights(pmeinfo, cpos_r)
        gridindex_s, weights_s, dweights_s = interpolation_weights(pmeinfo, spos_r)
        spread!(gridcharge, pmeinfo, gridindex_c, qc, weights_c)
        spread!(gridcharge, pmeinfo, gridindex_s, qs, weights_s)
    end

    if pmeinfo.dd === nothing
        # Replicated-grid reduction: the grid points an atom touches are
        # global, so the owned-atom spreads are summed across ranks here;
        # the FFT solve below is then replicated on every rank. An empty
        # rank contributes a zero grid but must ride the collective.
        if dom isa Cores.MPIDomain
            # Device grids go to MPI directly (CUDA-aware transport; the
            # stream drain lives inside allreduce_sum! — Domain.jl contract).
            Cores.allreduce_sum!(dom, vec(gridcharge))
        end
        gridenergy, vsum = _pme_solve(pmeinfo, gridcharge)
    else
        # Slab path (PMESlab.jl): the reduce_scatter inside the forward
        # transform performs the cross-rank sum, the FFT work is divided
        # over ranks, and vsum comes back already Allreduced — the same
        # replicated quantity as above, so the virial line is shared.
        # NOTE: unlike _pme_solve, gridenergy here is pmeinfo-owned
        # persistent storage (dd.gridenergy) — consumed by the gathers
        # below, never to be held across another solve.
        gridenergy, vsum = _pme_solve_dd!(pmeinfo, dom, gridcharge)
    end
    # Grid virial: replicated once the grid is global (identical FFT on
    # every rank, or an explicit Allreduce on the slab path) — added
    # directly per the reduction regime.
    atoms.info[:virial] .-= _devcopy(backend, FPPREC.(vsum))

    potential_energies = KA.zeros(backend, FPPREC, length(qc))

    if N > 0
        gather!(potential_energies, atoms.arrays[:forces], pmeinfo, gridindex_c, qc, atoms.info[:lattice], gridenergy, weights_c, dweights_c)
        gather!(potential_energies, atoms.arrays[:shellforces], pmeinfo, gridindex_s, qs, atoms.info[:lattice], gridenergy, weights_s, dweights_s)

        arrays = (
            elemid = atoms.arrays[:elemid],
            forces = atoms.arrays[:forces],
            shellforces = atoms.arrays[:shellforces],
            shelldisp = atoms.arrays[:shelldisp],
        )

        virials = KA.zeros(backend, FPPREC, length(qc), 6)
        kernel_subtractintraatom!(backend)(arrays, potential_energies, virials, params[:qc], params[:qs], params[:alpha]; ndrange = length(qc))
        add_reduced_ev!(atoms, AK.sum(potential_energies), AK.sum(virials, dims = 1)[1, :])
    else
        add_reduced_ev!(atoms, zero(FPPREC), zeros(FPPREC, 6))
    end

    # Charged-cell correction (Issue #15) — see chargedcell_correction!
    # (Ewald.jl). qc/qs are the per-atom core/shell charge arrays (on the
    # compute backend) built above; both count toward the cell total. Q is
    # an owned partial sum; the correction itself is replicated.
    Qlocal = N == 0 ? zero(FPPREC) : FPPREC(AK.sum(qc) + AK.sum(qs))
    Q = Cores.allreduce_sum(dom, Qlocal)
    chargedcell_correction!(atoms, Q, params[:alpha])
end

"""
    calculate!(pmeinfo::PMEInfo, ::OnlyShellForce, atoms::Atoms, params)

Shell-force-only variant of [`calculate!`](@ref) for the shell-relaxation
inner loop (Issue #157). Only `atoms.arrays[:shellforces]` is meaningful
on return: the energy and the virial are not written at all, and
`atoms.arrays[:forces]` receives the intra-atom half of the correction
alone — a partial sum, not a core force. That is the `OnlyShellForce`
contract, the same one `calculate!(::CoulombBuckingham, ::OnlyShellForce,
…)` answers to, and the reason [`shell_forces!`](@ref) documents its own
`info` / `:forces` outputs as unspecified. (The Buckingham variant happens
to leave `:forces` untouched; this one cannot, because
`kernel_subtractintraatom!` writes both force arrays from one `factor` and
splitting it would mean a second kernel for no gain. Do not read `:forces`
after either.)

The shell forces come out bit-identical to the full method: the charge
grid is built from both site families exactly as there (the mesh does not
know which sites the caller cares about), the convolution is the same
expression, and the shell `gather!` / `kernel_subtractintraatom!` calls
are the same calls with the same arguments. What is skipped is only work
the CG loop discards — the reciprocal virial (42% of a full call at
production settings), the core gather, the energy/virial reductions
(under domain decomposition, one 7-double `Allreduce` per iteration), and
the charged-cell correction, which touches `info` alone.
"""
function calculate!(pmeinfo::PMEInfo, ::OnlyShellForce, atoms::Atoms, params)
    backend = get_backend(atoms.info[:lattice])
    dom = Cores.domain(atoms)
    N = length(atoms)
    gridcharge = KA.zeros(backend, FPPREC, pmeinfo.fftsize)

    cpos_r, spos_r, qc, qs = _pme_site_charges(atoms, params)

    # As in the full method, this block and the gather block below share
    # the `N > 0` guard: the shell gridindex/weights locals bound here are
    # read there.
    if N > 0
        gridindex_c, weights_c, _ = interpolation_weights(pmeinfo, cpos_r)
        gridindex_s, weights_s, dweights_s = interpolation_weights(pmeinfo, spos_r)
        spread!(gridcharge, pmeinfo, gridindex_c, qc, weights_c)
        spread!(gridcharge, pmeinfo, gridindex_s, qs, weights_s)
    end

    if pmeinfo.dd === nothing
        # The grid reduction is NOT optional: the grid points an atom
        # touches are global, so an unreduced grid would make every rank
        # convolve its own partial charge — silently wrong rather than an
        # error. An empty rank contributes zeros but must ride the
        # collective.
        if dom isa Cores.MPIDomain
            Cores.allreduce_sum!(dom, vec(gridcharge))
        end
        # _pme_convolve is _pme_solve without the virial, expression for
        # expression — hence the bit-identical gridenergy.
        gridenergy = _pme_convolve(pmeinfo, gridcharge)
    else
        # Slab path: _pme_solve_dd! calls this same function, so the
        # full/shell bit-identity holds by structural sharing.
        gridenergy = _pme_convolve_dd!(pmeinfo, dom, gridcharge)
    end

    if N > 0
        # kernel_subtractintraatom! writes the intra-atom correction into
        # BOTH force arrays and the energy/virial scratch; only its
        # shellforces half is meaningful here, and the scratch is dropped.
        potential_energies = KA.zeros(backend, FPPREC, length(qc))
        virials = KA.zeros(backend, FPPREC, length(qc), 6)
        gather!(potential_energies, atoms.arrays[:shellforces], pmeinfo, gridindex_s, qs, atoms.info[:lattice], gridenergy, weights_s, dweights_s)

        arrays = (
            elemid = atoms.arrays[:elemid],
            forces = atoms.arrays[:forces],
            shellforces = atoms.arrays[:shellforces],
            shelldisp = atoms.arrays[:shelldisp],
        )
        kernel_subtractintraatom!(backend)(arrays, potential_energies, virials, params[:qc], params[:qs], params[:alpha]; ndrange = length(qc))
    end
    return nothing
end
