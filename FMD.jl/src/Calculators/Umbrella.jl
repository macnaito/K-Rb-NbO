import AcceleratedKernels as AK
import KernelAbstractions as KA
using Atomix
using CUDA
import MPI
# The FFT plans, buffers and the smoothing itself live in PolField.jl — this
# file only holds the bias built on top of them.

export umbrella_p!, PolarizationUmbrella, group_p, group_mfrac, grad_group_p!

# ---------------------------------------------------------------------------
#  PolarizationUmbrella  –  Gaussian-smoothed polarization umbrella sampling
#
#  Bias on a coarse-grained polarization field:
#      P̃_g = G_σ * P_g           (periodic Gaussian convolution via rFFT)
#  with two restraints:
#      U_par  = (k_par/2) (A - A_target)²              (amplitude bias)
#      U_homo = (k_h/2) Σ_g [max(0, |r_g| - r_th)]²   (homogeneity bias)
#  where
#      A      = Σ_g (P_g · ê) cos(θ_g) / Σ_g cos²(θ_g)   (LS-fit amplitude)
#      r_g    = P̃_g - A·ê·cos(θ_g)
#      θ_g    = 2π(m·g/N) + phase                       (mode wavevector m)
#
#  Special cases:
#      σ = 0  : G_σ → identity → recovers the legacy box-filter umbrella
#               (raw P_g penalised, sub-group modes have null space).
#      σ > 0  : Gaussian smoothing closes the null space.  Long modes
#               (λ > σ) are penalised with weight ~k_h, short modes
#               (λ < σ) are exponentially relaxed.
#      r_th=0 : original quadratic homogeneity penalty.
#      r_th>0 : flat-bottom (shifted-harmonic) homogeneity penalty —
#               natural fluctuations |r_g| ≤ r_th are unpenalised.
#               Requires extra chain-rule term that vanishes for r_th=0.
#
#  Backend: explicit `backend=KA.CPU()` or `backend=CUDA.CUDABackend()`.
#  FFT plans are allocated at construction.
#
#  Grid convention (ASE-natural):
#      flat groupid = ix·Ny·Nz + iy·Nz + iz + 1
#  i.e. iz cycles fastest, ix slowest.  This matches what
#  `unit_cell * (Nx, Ny, Nz)` produces in ASE when the per-atom
#  groupid is written as `i ÷ nu + 1`.  `m = (mx, my, mz)` therefore
#  refers directly to (real_x, real_y, real_z) Fourier modes; no axis
#  transposition or NSUP cell-replication trick is needed.
# ---------------------------------------------------------------------------

"""
    PolarizationUmbrella{T1,T2,T3,T4,T5,T6,T7,T8}

Polarisation-bias term for umbrella sampling. Adds two restraints to the
energy / forces of [`npt!`](@ref FMD.Apps.npt!):

```
U_par  = (k_par/2)  · (A − A_target)²                    # amplitude bias
U_homo = (k_homo/2) · Σ_g [max(0, |r_g| − r_th)]²        # homogeneity bias
```

where `A` is the least-squares amplitude of the per-group polarisation
along direction `e` projected onto a Fourier mode `m`, `r_g` is the
residual after subtracting the LS-fit modulation, and the per-group
polarisation is Gaussian-smoothed with width `sigma` (in unit-cell units;
`sigma = 0` recovers the legacy box-filter umbrella).

# Fields

- `k_par`, `k_homo`, `amplitude` — bias strengths and target amplitude.
- `e::Vector{FP}` — unit polarisation direction.
- `gridsize::NTuple{3,Int}` — `(Nx, Ny, Nz)` group lattice (must match
  `atoms.arrays[:groupid]`).
- `m::NTuple{3,Int}` — bias mode wavevector. `(0, 0, 0)` is the uniform
  bias; non-zero `m` couples to a Fourier mode.
- `phase::FP` — phase offset of the cosine modulation.
- `M_frac_ref` — gauge-fixing fractional dipole moment per group
  ([`group_mfrac`](@ref)), captured at construction.
- `results::Dict{String,Float}` — populated each step with `cv` and
  `bias` for the `Bias.log` output.
- `sigma`, `r_th`, `ghat_at_m`, `Ghat` — Gaussian smoothing kernel and
  flat-bottom threshold.
- `plan_rfft`, `plan_brfft`, `workspace` — pre-allocated FFT plans /
  buffers (CUDA or FFTW depending on `backend`).
- `sumr2_log` — optional path for per-step Σ|r|² logging.

Build via the constructor; pass it as `params[:polarization_umbrella]`
to `FMD.app[:NPTMD]`.
"""
struct PolarizationUmbrella{T1,T2,T3,T4,T5,T6,T7,T8}
    k_par::T1
    k_homo::T1
    amplitude::T1
    e::T4              # polarisation direction (unit vector, length 3)
    gridsize::NTuple{3,Int}
    m::NTuple{3,Int}   # wavevector (integer mode indices)
    phase::T1
    denom_inv::T1      # 1 / Σ_g cos²(θ_g)
    M_frac_ref::T2     # gauge-fixing fractional moment per group, constant
    results::T3
    sigma::T1          # smoothing width (unit-cell units)
    r_th::T1           # flat-bottom threshold (0 = pure quadratic)
    ghat_at_m::T1      # Ĝ(m) at the bias mode wavevector (=1 for m=0)
    Ghat::T5           # rFFT'd Gaussian kernel (rNx, Ny, Nz)
    plan_rfft::T6
    plan_brfft::T7
    workspace::T8      # NamedTuple of preallocated buffers
    sumr2_log::Union{Nothing,String}   # optional per-step Σ|r|² log
end

"""
    PolarizationUmbrella(atoms::Atoms, k_par, k_homo, amplitude, e_dir,
                         gridsize, qc_table, qs_table;
                         gauge, sigma = 0.0, m = (0, 0, 0), phase = 0.0,
                         backend = KA.CPU(), r_th = 0.0,
                         sumr2_log = nothing) -> PolarizationUmbrella

Build a polarisation umbrella with the given bias strengths
(`k_par`, `k_homo`), target amplitude, and direction `e_dir`. `gridsize`
is the `(Nx, Ny, Nz)` group lattice and must agree with
`atoms.arrays[:groupid]`. `qc_table`, `qs_table` are the per-element
charge tables.

`gauge` is **required** and fixes the zero-point of the polarisation: pass
a [`PolarizationGauge`](@ref) (normally loaded with [`load_gauge`](@ref) so
that every window of a sampling run shares one file), or an `Atoms` holding
the reference frame, which is built into a gauge on the spot. There is no
default — the choice of "which structure has P = 0" is physics, and the
previous implicit answer (the construction frame) made `amplitude` mean
"on top of whatever the starting frame happened to be polarised to". To
reproduce that old behaviour, write `gauge = atoms` explicitly.

`backend` selects the FFT implementation: `KA.CPU()` uses FFTW,
`CUDA.CUDABackend()` uses CUFFT and operates on `CuArray` data.

Under domain decomposition, construct the umbrella from the **full**
replicated frame *before* `distribute` (the constructor asserts and the
gauge reference `M_frac_ref` assume the whole system); the runtime
evaluation (`umbrella_p!`) is DD-aware and Allreduces the per-group
polarisation.

For uniform bias use `m = (0, 0, 0)`; for an inhomogeneous (Fourier-mode)
bias set `m` to the desired integer wavevector. `sigma > 0` Gaussian-
smooths the per-group polarisation field; `r_th > 0` switches the
homogeneity penalty from quadratic to flat-bottom (residuals smaller than
`r_th` are unpenalised).
"""
function PolarizationUmbrella(
    atoms::Atoms, k_par, k_homo, amplitude, e_dir, gridsize,
    qc_table, qs_table;
    gauge::Union{PolarizationGauge,Atoms},
    sigma=0.0, m=(0, 0, 0), phase=0.0,
    backend=KA.CPU(), r_th=0.0,
    sumr2_log::Union{Nothing,String}=nothing,
)
    # The gauge reference M_frac_ref and the asserts below are whole-system
    # sums/extrema — a distributed frame would freeze a rank-local gauge
    # and every rank would then bias against a different reference
    # (rank-dependent forces with no error anywhere downstream).
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("PolarizationUmbrella: construct from the full replicated frame " *
              "BEFORE distribute (the gauge reference and the group asserts " *
              "assume the whole system)")
    @assert minimum(atoms.arrays[:groupid]) == 1 "groupid should start from 1"

    Nx, Ny, Nz = Tuple(gridsize)
    ng = Nx * Ny * Nz
    @assert ng == maximum(atoms.arrays[:groupid]) "gridsize must match #groups"

    # Gauge-fixing reference is stored in fractional space so that the
    # per-step Cartesian Pg_ref(t) = M_frac_ref·L(t) / V_g(t) co-transforms
    # with the cell.  This keeps Pg = Pg_raw - Pg_ref(t) at zero when atoms
    # are static and only the cell deforms (volume-preserving shear,
    # isotropic expansion under NPT, etc.).
    #
    # Which frame that reference comes from is the caller's declared choice
    # (Issue #122), and validate_gauge is what makes a mismatch between the
    # gauge and this frame loud instead of a silently shifted zero-point.
    pg = _as_gauge(gauge, (Nx, Ny, Nz), qc_table, qs_table)
    pg.gridsize == (Nx, Ny, Nz) ||
        error("PolarizationUmbrella: gauge is for grid $(pg.gridsize) but this " *
              "umbrella is being built on $((Nx, Ny, Nz))")
    validate_gauge(pg, atoms)
    M_frac_ref = KA.zeros(backend, FPPREC, ng, 3)
    copyto!(M_frac_ref, pg.M_frac_ref)

    e_vec_cpu = FPPREC.(collect(e_dir))
    e_vec_cpu ./= sqrt(sum(e_vec_cpu .^ 2))
    e_vec = KA.zeros(backend, FPPREC, 3)
    copyto!(e_vec, e_vec_cpu)

    mx = mod(Int(m[1]), max(Nx, 1))
    my = mod(Int(m[2]), max(Ny, 1))
    mz = mod(Int(m[3]), max(Nz, 1))
    ph = FPPREC(phase)

    # Constructor enumerates the grid in (ix slow, iz fast) order, matching
    # the runtime decode below.
    denom = zero(FPPREC)
    for gx in 0:Nx-1, gy in 0:Ny-1, gz in 0:Nz-1
        θ = 2π * (mx * gx / Nx + my * gy / Ny + mz * gz / Nz) + ph
        denom += cos(θ)^2
    end
    denom_inv = one(FPPREC) / denom

    # Ĝ at the specific bias mode wavevector m.  Used to convert the raw LS
    # projection of Pg onto cos(θ) into the smoothed projection of P̃_g, which
    # equals Ĝ(m) · A_raw exactly because cos(θ_g) is an eigenfunction of the
    # Gaussian-smoothing operator: G_σ * cos(θ_g) = Ĝ(m) · cos(θ_g).
    # For m = 0 this is 1, so the raw and smoothed projections coincide.
    k2_at_m = (2π * mx / Nx)^2 + (2π * my / Ny)^2 + (2π * mz / Nz)^2
    ghat_at_m = exp(-FPPREC(0.5) * FPPREC(sigma)^2 * FPPREC(k2_at_m))

    Ghat = _build_ghat(backend, (Nx, Ny, Nz), sigma)
    rfft_plan, brfft_plan = _polfield_plans(backend, (Nx, Ny, Nz))

    # Mode-specific state on top of the shared buffers: cos(θ_g) on the grid,
    # used by the flat-bottom chain-rule term.
    cos_grid_cpu = zeros(FPPREC, Nx, Ny, Nz)
    for gx in 0:Nx-1, gy in 0:Ny-1, gz in 0:Nz-1
        θ = 2π * (mx * gx / Nx + my * gy / Ny + mz * gz / Nz) + ph
        cos_grid_cpu[gx + 1, gy + 1, gz + 1] = cos(θ)
    end
    cos_grid = KA.zeros(backend, FPPREC, Nx, Ny, Nz)
    copyto!(cos_grid, cos_grid_cpu)

    workspace = merge(_polfield_workspace(backend, (Nx, Ny, Nz)),
                      (cos_grid = cos_grid,))

    if sumr2_log !== nothing && (!MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0)
        # Pre-distribute every rank holds the replicated frame and calls
        # this constructor — only one process may truncate the log.
        open(sumr2_log, "w") do io
            println(io, "sum_r2,cv,U_homo")
        end
    end

    return PolarizationUmbrella(
        FPPREC(k_par), FPPREC(k_homo), FPPREC(amplitude),
        e_vec, (Nx, Ny, Nz), (mx, my, mz), ph, denom_inv,
        M_frac_ref, Dict{String,Any}(),
        FPPREC(sigma), FPPREC(r_th), FPPREC(ghat_at_m),
        Ghat, rfft_plan, brfft_plan, workspace,
        sumr2_log,
    )
end

# ---------------------------------------------------------------------------
#  group_p  –  raw group polarisation (flat groupid)
# ---------------------------------------------------------------------------

# Whole-system group count: under domain decomposition the rank-local
# groupid maximum is not the global one (a rank may own no atoms of the
# highest groups) — the group tables must be sized identically on every
# rank for the Pg Allreduce. Serial: identity.
function _global_ngroups(atoms)
    gl = length(atoms) == 0 ? 0 : Int(maximum(atoms.arrays[:groupid]))
    return Int(Cores.allreduce_max(Cores.domain(atoms), gl))
end

"""
    group_p(atoms, qc_table, qs_table; ng = nothing) -> Matrix

Per-group Cartesian polarisation **density**

```
P_g = (1/V_g) Σ_{i ∈ g} (qc[s_i] · r_i + qs[s_i] · (r_i + sd_i)),
V_g = V / ng
```

reduced over atoms whose `groupid` is `g`. Dividing by the group volume
`V_g` (the cell volume split evenly over the `ng` groups) is part of the
definition, so the result is a *density* — in atomic units e/Bohr², not
the e·Bohr of a dipole moment. This is `P_g^raw` of
`docs/src/theory/polarization-bias-virial.md`; subtracting the gauge
reference ([`group_p_gauged`](@ref)) gives the `P_g` that every bias in
the polarisation family restrains.

Returns an `(ng, 3)` matrix on the same backend as `atoms`. `qc_table`,
`qs_table` are the per-element core / shell charge vectors (typically
`_shell.qc`, `_shell.qs` from a `ShellParam.jld2`). `ng` is the group count;
it defaults to the whole-system `groupid` maximum (a collective under domain
decomposition), so callers that already know the group lattice pass it
explicitly to stay off the collective path.
"""
function group_p(atoms, qc_table, qs_table; ng::Union{Nothing,Int} = nothing)
    L = Array(atoms.info[:lattice])
    sx = atoms.arrays[:pos][:,1]
    sy = atoms.arrays[:pos][:,2]
    sz = atoms.arrays[:pos][:,3]
    # r = pos * lattice: the ROWS of `lattice` are the cell vectors, the
    # convention used by Cores/NeighborList.jl, the XYZ I/O and every other
    # calculator.  Indexing L the other way round agrees only while the cell
    # is orthorhombic and then silently disagrees once it is sheared.
    rx = L[1,1]*sx .+ L[2,1]*sy .+ L[3,1]*sz
    ry = L[1,2]*sx .+ L[2,2]*sy .+ L[3,2]*sz
    rz = L[1,3]*sx .+ L[2,3]*sy .+ L[3,3]*sz
    rshellx = rx .+ atoms.arrays[:shelldisp][:,1]
    rshelly = ry .+ atoms.arrays[:shelldisp][:,2]
    rshellz = rz .+ atoms.arrays[:shelldisp][:,3]

    elemid = atoms.arrays[:elemid]
    qc = similar(elemid, FPPREC)
    qs = similar(elemid, FPPREC)
    AK.map!(e -> qc_table[e], qc, elemid)
    AK.map!(e -> qs_table[e], qs, elemid)

    # ng (the group-grid size) is a build-time constant when called from
    # umbrella_p! — passing it avoids a collective per force evaluation;
    # the standalone path falls back to the collective count.
    gmax = ng === nothing ? _global_ngroups(atoms) : ng
    length(atoms) == 0 || Int(maximum(atoms.arrays[:groupid])) <= gmax ||
        error("group_p: groupid exceeds the group count $gmax")
    Pg = similar(qc, gmax, 3)
    fill!(Pg, zero(eltype(Pg)))
    groupid = atoms.arrays[:groupid]

    Vg = volume(atoms) / gmax

    if length(atoms) > 0   # empty ranks return the zero partial (global size)
        AK.foreachindex(qc) do i
            Atomix.@atomic Pg[groupid[i],1] += (qc[i] * rx[i] + qs[i] * rshellx[i]) / Vg
            Atomix.@atomic Pg[groupid[i],2] += (qc[i] * ry[i] + qs[i] * rshelly[i]) / Vg
            Atomix.@atomic Pg[groupid[i],3] += (qc[i] * rz[i] + qs[i] * rshellz[i]) / Vg
        end
    end

    return Pg
end

# ---------------------------------------------------------------------------
#  group_mfrac  –  fractional gauge-fixing moment per group (constant)
#
#  Returns Σ_{i∈g} [ qc·s + qs·(s + sd·L^(-1)) ] for each group g,
#  i.e. the moment in fractional coordinates.  Computed once at construction
#  time; later evaluations of Pg_ref(t) = M_frac_ref · L(t) / V_g(t)
#  reproduce a gauge-covariant reference that follows the cell.
# ---------------------------------------------------------------------------

"""
    group_mfrac(atoms, qc_table, qs_table) -> Matrix

Per-group dipole moment in **fractional** coordinates,

```
M_g^frac = Σ_{i ∈ g} (qc[s_i] · s_i + qs[s_i] · (s_i + sd_i · L⁻¹))
```

used as the gauge-fixing reference for [`PolarizationUmbrella`](@ref).
Unlike [`group_p`](@ref) this is a *moment* and carries no `1/V_g` (atomic
units: e, fractional coordinates being dimensionless). The two meet at MD
time, where the row-wise product with `L(t)` and the division by `V_g(t)`
turn it into a density directly comparable to [`group_p`](@ref):

```
P_g^ref(t) = M_g^frac · L(t) / V_g(t)
```

Carrying the reference in fractional form is what makes it
cell-deformation-covariant, keeping the bias zero under homogeneous cell
deformation.
"""
function group_mfrac(atoms, qc_table, qs_table)
    L_inv = inv(Array(atoms.info[:lattice]))
    sx = atoms.arrays[:pos][:,1]
    sy = atoms.arrays[:pos][:,2]
    sz = atoms.arrays[:pos][:,3]

    sd = atoms.arrays[:shelldisp]
    # Cartesian → fractional is the row-wise inverse of r = pos * lattice.
    sd_frac = sd * L_inv
    sshellx = sx .+ sd_frac[:,1]
    sshelly = sy .+ sd_frac[:,2]
    sshellz = sz .+ sd_frac[:,3]

    elemid = atoms.arrays[:elemid]
    qc = similar(elemid, FPPREC)
    qs = similar(elemid, FPPREC)
    AK.map!(e -> qc_table[e], qc, elemid)
    AK.map!(e -> qs_table[e], qs, elemid)

    gmax = maximum(atoms.arrays[:groupid])
    Mfr = similar(qc, gmax, 3)
    fill!(Mfr, zero(eltype(Mfr)))
    groupid = atoms.arrays[:groupid]

    AK.foreachindex(qc) do i
        Atomix.@atomic Mfr[groupid[i],1] += qc[i] * sx[i] + qs[i] * sshellx[i]
        Atomix.@atomic Mfr[groupid[i],2] += qc[i] * sy[i] + qs[i] * sshelly[i]
        Atomix.@atomic Mfr[groupid[i],3] += qc[i] * sz[i] + qs[i] * sshellz[i]
    end

    return Mfr
end

# ---------------------------------------------------------------------------
#  grad_group_p!  –  back-propagate dU/dPg to forces and virial
# ---------------------------------------------------------------------------

"""
    grad_group_p!(atoms, Pg, dUdPg, qc_table, qs_table, Pg_ref=nothing)

Adjoint of [`group_p`](@ref): scatter the per-group polarisation-density
gradient `dUdPg` (`(ng, 3)`) into per-atom forces and shell forces — the
`1/V_g` of [`group_p`](@ref) therefore reappears in the scattered forces.
Used inside [`umbrella_p!`](@ref) to translate bias-energy derivatives into
MD forces. `Pg_ref` is the cell-covariant reference subtracted from `Pg`
before computing the residual; pass `nothing` (default) when the bias is on
the raw per-group polarisation. Under domain decomposition `Pg` and `dUdPg`
must already be whole-system (post-Allreduce) group tables — the
atom-loop virial is the only owned partial sum here.
"""
function grad_group_p!(atoms, Pg, dUdPg, qc_table, qs_table, Pg_ref=nothing;
                       ng::Union{Nothing,Int} = nothing)
    L = Array(atoms.info[:lattice])
    sx = atoms.arrays[:pos][:,1]
    sy = atoms.arrays[:pos][:,2]
    sz = atoms.arrays[:pos][:,3]
    # r = pos * lattice, same convention as group_p (see the note there).
    rx = L[1,1]*sx .+ L[2,1]*sy .+ L[3,1]*sz
    ry = L[1,2]*sx .+ L[2,2]*sy .+ L[3,2]*sz
    rz = L[1,3]*sx .+ L[2,3]*sy .+ L[3,3]*sz
    rshellx = rx .+ atoms.arrays[:shelldisp][:,1]
    rshelly = ry .+ atoms.arrays[:shelldisp][:,2]
    rshellz = rz .+ atoms.arrays[:shelldisp][:,3]

    elemid = atoms.arrays[:elemid]
    qc = similar(elemid, FPPREC)
    qs = similar(elemid, FPPREC)
    AK.map!(e -> qc_table[e], qc, elemid)
    AK.map!(e -> qs_table[e], qs, elemid)

    gmax = ng === nothing ? _global_ngroups(atoms) : ng
    length(atoms) == 0 || Int(maximum(atoms.arrays[:groupid])) <= gmax ||
        error("grad_group_p!: groupid exceeds the group count $gmax")
    groupid = atoms.arrays[:groupid]

    Vg = volume(atoms) / gmax

    fcore = atoms.arrays[:forces]
    fshell = atoms.arrays[:shellforces]
    virial = similar(fcore, length(atoms), 6); fill!(virial, zero(eltype(virial)))

    if length(atoms) > 0   # empty ranks still ride the collective below
        AK.foreachindex(qc) do i
        fx = -dUdPg[groupid[i],1] * qc[i] / Vg
        fy = -dUdPg[groupid[i],2] * qc[i] / Vg
        fz = -dUdPg[groupid[i],3] * qc[i] / Vg
        fcore[i,1] += fx
        fcore[i,2] += fy
        fcore[i,3] += fz

        fsx = -dUdPg[groupid[i],1] * qs[i] / Vg
        fsy = -dUdPg[groupid[i],2] * qs[i] / Vg
        fsz = -dUdPg[groupid[i],3] * qs[i] / Vg
        fshell[i,1] += fsx
        fshell[i,2] += fsy
        fshell[i,3] += fsz
        virial[i,1] += rx[i] * fx + rshellx[i] * fsx
        virial[i,2] += ry[i] * fy + rshelly[i] * fsy
        virial[i,3] += rz[i] * fz + rshellz[i] * fsz
        virial[i,4] += (ry[i] * fz + rz[i] * fy) / 2 +
                       (rshelly[i] * fsz + rshellz[i] * fsy) / 2
        virial[i,5] += (rx[i] * fz + rz[i] * fx) / 2 +
                       (rshellx[i] * fsz + rshellz[i] * fsx) / 2
        virial[i,6] += (rx[i] * fy + ry[i] * fx) / 2 +
                       (rshellx[i] * fsy + rshelly[i] * fsx) / 2
        end
        # The atom-loop virial is an owned partial sum (reduction regime);
        # the group-lattice virials below are replicated (global Pg/dUdPg)
        # and are added directly.
        add_reduced_ev!(atoms, zero(FPPREC), AK.sum(virial, dims = 1)[1, :])
    else
        add_reduced_ev!(atoms, zero(FPPREC), zeros(FPPREC, 6))
    end

    # Pg carries an explicit 1/V_g(t), and ∂(1/V_g)/∂ε_αβ = -δ_αβ / V_g, so
    # this piece of -∂U/∂ε is ISOTROPIC: the full contraction Σ_γ dUdPg_γ Pg_γ
    # enters each of xx, yy, zz.  A per-component product would drop the cross
    # terms (dUdPg_y·Pg_y never reaching xx), and the constant n_g inside
    # V_g = V/n_g cancels out of the logarithmic derivative — it must not
    # appear here.  Derivation: docs/src/theory/polarization-bias-virial.md.
    dUdPg_dot_Pg = similar(fcore, size(Pg,1))
    AK.foraxes(Pg,1) do g
        dUdPg_dot_Pg[g] = dUdPg[g,1] * Pg[g,1] + dUdPg[g,2] * Pg[g,2] +
                          dUdPg[g,3] * Pg[g,3]
    end
    view(atoms.info[:virial], 1:3) .+= AK.sum(dUdPg_dot_Pg)

    # Pg_ref(t) depends on h(t) explicitly; its cell-derivative contributes a
    # symmetric tensor +(dUdPg)_α (Pg_ref)_β (sym) to the virial.  Without
    # this term the Cartesian r·F sum (which uses real atomic positions and
    # thus picks up Pg_raw, not Pg_raw - Pg_ref) leaves the missing piece
    # uncompensated and the barostat sees a residual stress.
    if Pg_ref !== nothing
        virial_ref = similar(fcore, size(Pg,1), 6); fill!(virial_ref, zero(eltype(virial_ref)))
        AK.foraxes(Pg,1) do g
            virial_ref[g,1] = dUdPg[g,1] * Pg_ref[g,1]
            virial_ref[g,2] = dUdPg[g,2] * Pg_ref[g,2]
            virial_ref[g,3] = dUdPg[g,3] * Pg_ref[g,3]
            virial_ref[g,4] = (dUdPg[g,2] * Pg_ref[g,3] + dUdPg[g,3] * Pg_ref[g,2]) / 2
            virial_ref[g,5] = (dUdPg[g,1] * Pg_ref[g,3] + dUdPg[g,3] * Pg_ref[g,1]) / 2
            virial_ref[g,6] = (dUdPg[g,1] * Pg_ref[g,2] + dUdPg[g,2] * Pg_ref[g,1]) / 2
        end
        atoms.info[:virial] .+= AK.sum(virial_ref, dims = 1)[1,:]
    end

    nothing
end

# ---------------------------------------------------------------------------
#  umbrella_p!  –  Gaussian-smoothed homogeneity restraint with optional
#                  flat-bottom (shifted-harmonic) shape.
#
#  Pass 1: 3 components of P̃_g via rFFT, store r_g^d in rgrid_{x,y,z}.
#  Pass 2: chain back through G to get dU/dP_g, scatter to atomic forces.
#  Chain-rule correction (X) for shifted form: A depends on raw P, so r_g
#  also depends on A; for r_th=0 the correction vanishes by LS orthogonality,
#  for r_th>0 it does not.
# ---------------------------------------------------------------------------

"""
    umbrella_p!(atoms::Atoms, pu::PolarizationUmbrella, prms)

Apply the polarisation umbrella defined by `pu` to `atoms`: compute the
per-group polarisation ([`group_p`](@ref)), the LS-fit amplitude `A` along
`pu.e`, the homogeneity residual, the (Gaussian-smoothed) bias energy, and
scatter the gradient back into `atoms.arrays[:forces]` and
`atoms.arrays[:shellforces]`.

Populates `pu.results["cv"]` (collective variable, the LS-fit amplitude)
and `pu.results["bias"]` (current bias energy) for the per-step
`Bias.log` output written by [`npt!`](@ref FMD.Apps.npt!). Called from
[`shell_forces!`](@ref) / [`core_forces!`](@ref) when
`prms[:polarization_umbrella]` is present.
"""
function umbrella_p!(atoms::Atoms, pu::PolarizationUmbrella, prms)
    Nx, Ny, Nz = pu.gridsize
    ng = Nx * Ny * Nz

    # Gauge-referenced group polarisation (and the reference itself, which the
    # adjoint needs for the cell derivative).  Everything from here on is
    # replicated on every rank — see _polfield_pg.
    Pg, Pg_ref_now = _polfield_pg(atoms, pu.M_frac_ref, prms[:qc], prms[:qs], ng)
    mx, my, mz = pu.m
    ph = pu.phase
    e = pu.e
    denom_inv = pu.denom_inv
    invN = one(FPPREC) / FPPREC(ng)
    r_th = pu.r_th
    # Local-bind ghat_at_m so the GPU closure does NOT capture the entire pu
    # struct (which contains a non-bitstype sumr2_log::Union{Nothing,String}).
    ghat_at_m = pu.ghat_at_m

    # 2. Amplitude A from SMOOTHED P̃ via LS projection.
    #    Use the analytic identity G * cos(θ) = Ĝ(m) cos(θ): the smoothed
    #    projection equals Ĝ(m) times the raw projection, no extra FFT needed.
    #    For m = (0,0,0) this is Ĝ(0)=1, so behaviour is identical to a raw
    #    projection.  For m ≠ 0 this enforces internal consistency with the
    #    homogeneity penalty (residual r_g orthogonal to ê·cos(θ)).
    #
    #    Index decode: gz fastest, gx slowest — matches ASE's
    #    `atoms * (Nx, Ny, Nz)` layout when groupid = i÷nu + 1.
    # Deterministic reduction (no atomics): with threads the atomic
    # accumulation order is schedule-dependent, and a replicated quantity
    # that differs by one ulp across ranks would slowly split the cell
    # degrees of freedom under DD.
    proj = similar(Pg, size(Pg, 1))
    AK.foraxes(Pg, 1) do i
        gx, gy, gz = _decode_g(i, Ny, Nz)
        θ = 2π * (mx * gx / Nx + my * gy / Ny + mz * gz / Nz) + ph
        s = cos(θ)
        proj[i] = (Pg[i, 1] * e[1] + Pg[i, 2] * e[2] + Pg[i, 3] * e[3]) * s
    end
    amp_now = sum(proj) * denom_inv * ghat_at_m

    amp_target = pu.amplitude
    U_par  = pu.k_par / 2 * (amp_now - amp_target)^2
    dUdamp = pu.k_par * (amp_now - amp_target)

    # 3. Pass 1: smoothed residual r_g^d via rFFT
    Pgrid     = pu.workspace.Pgrid
    Phat      = pu.workspace.Phat
    Ptilde_d  = pu.workspace.Ptilde_d
    rmag      = pu.workspace.rmag
    ggrid     = pu.workspace.ggrid
    rhat      = pu.workspace.rhat
    dUdPg_d   = pu.workspace.dUdPg_d
    dUdPg     = pu.workspace.dUdPg
    Ghat      = pu.Ghat
    k_homo    = pu.k_homo

    for d in 1:3
        rgrid_d = _rgrid_for(pu.workspace, d)

        _scatter_column!(Pgrid, Pg, d, Ny, Nz)
        _gauss_smooth!(Ptilde_d, Pgrid, Ghat, Phat, pu.plan_rfft, pu.plan_brfft, invN)

        e_d = e[d:d]
        AK.foraxes(Pg, 1) do i
            gx, gy, gz = _decode_g(i, Ny, Nz)
            θ = 2π * (mx * gx / Nx + my * gy / Ny + mz * gz / Nz) + ph
            s = cos(θ)
            rgrid_d[gx + 1, gy + 1, gz + 1] =
                Ptilde_d[gx + 1, gy + 1, gz + 1] - amp_now * e_d[1] * s
        end
    end

    rgrid_x = pu.workspace.rgrid_x
    rgrid_y = pu.workspace.rgrid_y
    rgrid_z = pu.workspace.rgrid_z

    # 4. |r_g| and U_homo (shifted) + diagnostic Σ|r|²
    @. rmag = sqrt(rgrid_x*rgrid_x + rgrid_y*rgrid_y + rgrid_z*rgrid_z)
    @. ggrid = max(zero(FPPREC), rmag - r_th)             # m_g (scratch)
    U_homo_acc = FPPREC(0.5) * k_homo * sum(abs2, ggrid)
    sum_r2 = sum(abs2, rgrid_x) + sum(abs2, rgrid_y) + sum(abs2, rgrid_z)

    # Chain-rule correction for shifted form
    cos_grid = pu.workspace.cos_grid
    e_host = Array(e)
    e1 = e_host[1]; e2 = e_host[2]; e3 = e_host[3]
    @. ggrid = ifelse(
        rmag > r_th,
        cos_grid * (rmag - r_th) * (rgrid_x*e1 + rgrid_y*e2 + rgrid_z*e3) / max(rmag, eps(FPPREC)),
        zero(FPPREC),
    )
    X = sum(ggrid)
    chain_factor = dUdamp - k_homo * X

    # 5. Pass 2: per-component gradient via self-adjoint Gaussian
    for d in 1:3
        rgrid_d = _rgrid_for(pu.workspace, d)
        e_d = e[d:d]

        @. ggrid = ifelse(
            rmag > r_th,
            k_homo * (rmag - r_th) * rgrid_d / max(rmag, eps(FPPREC)),
            zero(FPPREC),
        )

        # Self-adjoint: the same convolution carries dU/dP̃ back to dU/dP.
        _gauss_smooth!(dUdPg_d, ggrid, Ghat, rhat, pu.plan_rfft, pu.plan_brfft, invN)

        AK.foraxes(Pg, 1) do i
            gx, gy, gz = _decode_g(i, Ny, Nz)
            θ = 2π * (mx * gx / Nx + my * gy / Ny + mz * gz / Nz) + ph
            s = cos(θ)
            # Chain rule via A_smooth: dA_smooth/dP_h^β = (Ĝ(m)/denom) ê_β cos(θ_h).
            # Extra ghat_at_m vs the raw-projection version (=1 for m=0).
            chain = chain_factor * s * denom_inv * ghat_at_m
            dUdPg[i, d] = dUdPg_d[gx + 1, gy + 1, gz + 1] + chain * e_d[1]
        end
    end

    # 6. Energy and atomic forces
    atoms.info[:potential_energy] .+= U_par + U_homo_acc
    grad_group_p!(atoms, Pg, dUdPg, prms[:qc], prms[:qs], Pg_ref_now; ng = ng)

    pu.results["cv"]      = amp_now
    pu.results["bias"]    = U_par
    pu.results["U_homo"]  = U_homo_acc
    pu.results["sum_r2"]  = sum_r2

    if pu.sumr2_log !== nothing && Cores.isroot(Cores.domain(atoms))
        open(pu.sumr2_log, "a") do io
            println(io, "$(sum_r2),$(amp_now),$(U_homo_acc)")
        end
    end

    nothing
end
