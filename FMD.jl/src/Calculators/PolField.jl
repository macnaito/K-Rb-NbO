import AcceleratedKernels as AK
import KernelAbstractions as KA
import AbstractFFTs: plan_rfft, plan_brfft
using LinearAlgebra: mul!
using CUDA.CUFFT  # registers CUFFT plans for CuArrays
using FFTW        # registers FFTW plans for Array (CPU)

# ---------------------------------------------------------------------------
#  Shared machinery for polarisation-field biases
#
#  Every bias in this family restrains the same object — the Gaussian-smoothed,
#  gauge-referenced per-group polarisation field
#
#      P_g  = P_g^raw − M_frac_ref · L(t) / V_g(t)
#      P̃_g  = G_σ * P_g                     (periodic convolution via rFFT)
#
#  — and differs only in how it turns P̃ into an energy.  The pieces below are
#  the parts that do not depend on that choice: the FFT plans and buffers, the
#  reference reconstruction, the DD reduction, the flat-groupid decode, and the
#  smoothing itself (which, being real and even, is its own adjoint — the
#  gradient pass reuses the same routine).
#
#  Grid convention (ASE-natural, shared by every consumer):
#      flat groupid = ix·Ny·Nz + iy·Nz + iz + 1
#  i.e. iz cycles fastest, ix slowest.
#
#  Not exported: this is an internal contract between the bias types in this
#  directory, not a public API.
#
#  Deliberately NOT extracted (yet):
#
#  * a three-component driver around the forward pass.  The per-component loop
#    body differs between biases (PolarizationUmbrella subtracts the LS-fit
#    modulation while writing the residual), and hoisting it would mean
#    passing a closure into a device loop for no reduction in real duplication.
#  * a `_gather_column!` mirroring `_scatter_column!`.  The gather back from the
#    grid is fused with each bias's own chain-rule term, so splitting it would
#    add a pass over the group table rather than remove one.  A bias with no
#    chain term (a fixed target map) can gather with a plain AK.foraxes loop.
#
#  Both become worth revisiting once a second consumer shows the same shape;
#  the point of this layer is to share what is genuinely identical, not to
#  guess at an abstraction before the second case exists.
# ---------------------------------------------------------------------------

"""
    _decode_g(i, Ny, Nz) -> (gx, gy, gz)

Flat group index (1-based) → grid indices (0-based), ASE-natural order (iz
fastest). Inlined into GPU closures, so it stays branch-free and allocation
-free.
"""
@inline function _decode_g(i::Integer, Ny::Integer, Nz::Integer)
    i0 = i - 1
    gz = i0 % Nz
    gxy = i0 ÷ Nz
    gy = gxy % Ny
    gx = gxy ÷ Ny
    return gx, gy, gz
end

"""
    _build_ghat(backend, gridsize, sigma) -> Array

rFFT-space Gaussian kernel `Ĝ(k) = exp(-σ²k²/2)` on the `(rNx, Ny, Nz)`
half-spectrum grid, with `σ` in unit-cell units. `sigma = 0` gives the
all-ones kernel, i.e. the identity smoothing that recovers the raw
per-group field.
"""
function _build_ghat(backend, gridsize::NTuple{3,Int}, sigma::Real)
    Nx, Ny, Nz = gridsize
    rNx = Nx ÷ 2 + 1
    Ghat_cpu = zeros(FPPREC, rNx, Ny, Nz)
    σ = FPPREC(sigma)
    for kz in 0:Nz-1, ky in 0:Ny-1, kx in 0:rNx-1
        nx = kx
        ny = ky <= Ny ÷ 2 ? ky : ky - Ny
        nz = kz <= Nz ÷ 2 ? kz : kz - Nz
        k2 = (2π * nx / Nx)^2 + (2π * ny / Ny)^2 + (2π * nz / Nz)^2
        Ghat_cpu[kx + 1, ky + 1, kz + 1] = exp(-FPPREC(0.5) * σ^2 * k2)
    end
    Ghat_dev = KA.zeros(backend, FPPREC, rNx, Ny, Nz)
    copyto!(Ghat_dev, Ghat_cpu)
    return Ghat_dev
end

"""
    _polfield_plans(backend, gridsize) -> (plan_rfft, plan_brfft)

Forward / backward real FFT plans for the group lattice, allocated once at
construction. `backend` picks the implementation (FFTW on the host, CUFFT on
`CUDA.CUDABackend()`).
"""
function _polfield_plans(backend, gridsize::NTuple{3,Int})
    Nx, Ny, Nz = gridsize
    rNx = Nx ÷ 2 + 1
    return (plan_rfft(KA.zeros(backend, FPPREC, Nx, Ny, Nz)),
            plan_brfft(KA.zeros(backend, Complex{FPPREC}, rNx, Ny, Nz), Nx))
end

"""
    _polfield_workspace(backend, gridsize) -> NamedTuple

Preallocated buffers shared by every polarisation-field bias: the real and
half-spectrum grids used by the smoothing pass, the three residual
components, and the `(ng, 3)` gradient table. A bias with extra state
(a cosine modulation, a target map) merges its own entries onto this.
"""
function _polfield_workspace(backend, gridsize::NTuple{3,Int})
    Nx, Ny, Nz = gridsize
    rNx = Nx ÷ 2 + 1
    ng = Nx * Ny * Nz
    return (
        Pgrid    = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        Phat     = KA.zeros(backend, Complex{FPPREC}, rNx, Ny, Nz),
        Ptilde_d = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        rgrid_x  = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        rgrid_y  = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        rgrid_z  = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        rmag     = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        ggrid    = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        rhat     = KA.zeros(backend, Complex{FPPREC}, rNx, Ny, Nz),
        dUdPg_d  = KA.zeros(backend, FPPREC, Nx, Ny, Nz),
        dUdPg    = KA.zeros(backend, FPPREC, ng, 3),
    )
end

# Pick the residual buffer for Cartesian component `d` — the three are
# separate arrays rather than one (Nx,Ny,Nz,3) so the FFT plans stay 3D.
@inline _rgrid_for(ws, d::Int) =
    d == 1 ? ws.rgrid_x : (d == 2 ? ws.rgrid_y : ws.rgrid_z)

"""
    _polfield_pg(atoms, M_frac_ref, qc_table, qs_table, ng) -> (Pg, Pg_ref)

Gauge-referenced per-group polarisation of the current frame,

```
Pg_ref(t) = M_frac_ref · L(t) / V_g(t),   Pg = Pg_raw − Pg_ref(t)
```

with `Pg_ref` returned as well because the adjoint needs it for the cell
derivative ([`grad_group_p!`](@ref)).

The fractional reference co-transforms with the cell, so pure cell
deformation with static atoms leaves `Pg` at zero. Under domain
decomposition `Pg_raw` arrives as an owned partial sum and is Allreduced
here — from this point on every derived quantity is replicated on every
rank, which is what lets the callers add their group-lattice terms directly.
"""
function _polfield_pg(atoms, M_frac_ref, qc_table, qs_table, ng::Int)
    Vg = volume(atoms) / FPPREC(ng)
    L_dev = similar(M_frac_ref, FPPREC, 3, 3)
    copyto!(L_dev, FPPREC.(Array(atoms.info[:lattice])))
    # Fractional → Cartesian row-wise: M_frac * lattice (cf. group_p).
    Pg_ref = (M_frac_ref * L_dev) ./ FPPREC(Vg)

    # Passing `ng` makes group_p size its table from the grid instead of a
    # collective maximum, which is what keeps this off the per-force-evaluation
    # collective path.  It also means the returned table is `ng` rows BY
    # CONSTRUCTION — a size check here would be vacuous.  The real guard against
    # a frame whose groups do not fit the grid is inside group_p
    # ("groupid exceeds the group count"), plus the constructor's assert that
    # the grid matches the frame's group count in the first place.
    Pg_raw = group_p(atoms, qc_table, qs_table; ng = ng)
    dom = Cores.domain(atoms)
    if dom isa Cores.MPIDomain
        Cores.allreduce_sum!(dom, vec(Pg_raw))
    end
    return Pg_raw .- Pg_ref, Pg_ref
end

"""
    _scatter_column!(grid, Pg, d, Ny, Nz)

Copy Cartesian component `d` of the flat `(ng, 3)` group table onto the
`(Nx, Ny, Nz)` grid the FFT operates on.
"""
function _scatter_column!(grid, Pg, d::Int, Ny::Int, Nz::Int)
    AK.foraxes(Pg, 1) do i
        gx, gy, gz = _decode_g(i, Ny, Nz)
        grid[gx + 1, gy + 1, gz + 1] = Pg[i, d]
    end
    return grid
end

"""
    _gauss_smooth!(dst, src, Ghat, khat, plan_r, plan_b, invN)

Periodic Gaussian convolution of one real grid, `dst = G_σ * src`, via the
half-spectrum FFT (`khat` is the complex scratch). Real and even, so the
operator is its own adjoint — the gradient pass calls this with the
residual in place of the field, and no separate transpose is needed.
"""
function _gauss_smooth!(dst, src, Ghat, khat, plan_r, plan_b, invN)
    mul!(khat, plan_r, src)
    khat .*= Ghat
    mul!(dst, plan_b, khat)
    dst .*= invN
    return dst
end
