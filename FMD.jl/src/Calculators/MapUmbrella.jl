export PolarizationMapUmbrella, save_target_map, load_target_map

# ---------------------------------------------------------------------------
#  PolarizationMapUmbrella  –  restrain the local polarisation to an arbitrary
#  per-group target map
#
#      U     = (k/2) Σ_g w_g [max(0, |r_g| − r_th)]²
#      r_g   = P̃_g − P_target_g
#      P̃     = G_σ * (P_raw − P_ref(t))
#
#  Where PolarizationUmbrella biases a single Fourier mode through a
#  least-squares amplitude, this one names the whole field: it is what you
#  reach for when the configuration you want is a *shape* (a domain wall, a
#  vortex, a seeded nucleus) rather than an amplitude.
#
#  Because the target is a constant, this is strictly SIMPLER than the mode
#  umbrella, not an extension of it: the LS amplitude A depended on P, so the
#  flat-bottom form needed a chain-rule correction through dA/dP.  With a fixed
#  map that term is identically zero and the gradient is exactly
#  k·w·max(0,|r|−r_th)·r̂ carried back through the (self-adjoint) smoothing.
#
#  Typical use is preparation rather than sampling: hold the wanted state
#  while the system equilibrates at temperature, then decay k (or widen r_th)
#  and let it go.  Because the bias is an explicit potential, its energy,
#  forces and virial are all defined, so the release can run under the same
#  barostat as the production run.
# ---------------------------------------------------------------------------

"""
    PolarizationMapUmbrella{T1,T2,T3,T4,T5,T6,T7}

Bias that restrains the Gaussian-smoothed per-group polarisation towards a
fixed per-group target map:

```
U = (k/2) Σ_g w_g [max(0, |P̃_g − P_target_g| − r_th)]²
```

# Fields

- `k` — bias strength.
- `P_target_x/y/z` — the target map, held as three `(Nx, Ny, Nz)` grids
  (the constructor takes it as an `(ng, 3)` table in flat-groupid order).
- `wgrid` — per-group weight on the same grid; `wsum` is its sum, the
  denominator of the reported collective variable.
- `gridsize`, `sigma`, `r_th` — group lattice, smoothing width, flat-bottom
  threshold. Same meaning as [`PolarizationUmbrella`](@ref).
- `M_frac_ref` — gauge reference from the declared [`PolarizationGauge`](@ref).
- `Ghat`, `plan_rfft`, `plan_brfft`, `workspace` — shared smoothing machinery.
- `results` — `cv` (weighted RMS deviation from the target), `bias` (`U`) and
  `sum_r2` each step, for `Bias.log`.

Pass it as `params[:polarization_umbrella]`, the same key the mode umbrella
uses — [`umbrella_p!`](@ref) dispatches on the type.
"""
struct PolarizationMapUmbrella{T1,T2,T3,T4,T5,T6,T7}
    k::T1
    P_target_x::T2
    P_target_y::T2
    P_target_z::T2
    wgrid::T2
    wsum::T1
    gridsize::NTuple{3,Int}
    sigma::T1
    r_th::T1
    M_frac_ref::T3     # name fixed: src/Apps/NPT.jl reads it for dipole_phys
    Ghat::T4           # name fixed: src/Apps/ParamSpec.jl checks its array type
    plan_rfft::T5
    plan_brfft::T6
    workspace::T7
    results::Dict{String,Any}
    sumr2_log::Union{Nothing,String}
end

"""
    PolarizationMapUmbrella(atoms::Atoms, k, P_target, gridsize,
                            qc_table, qs_table;
                            gauge, weight = nothing, sigma = 0.0, r_th = 0.0,
                            backend = KA.CPU(), sumr2_log = nothing)

Build a map umbrella. `P_target` is an `(ng, 3)` Cartesian table in the same
units as [`group_p`](@ref) (dipole per volume, atomic units) and in flat
groupid order (`ix·Ny·Nz + iy·Nz + iz + 1`, iz fastest). Measure it with
[`group_p_gauged`](@ref) against the *same* `gauge` you pass here — that is
what makes the target and the run comparable.

`weight` (default all ones) scales the restraint per group. Zeroing a band
is the point for domain-wall work: pin the domains and leave the wall free,
so its width and internal structure are found by the system rather than
imposed. **The weight multiplies the smoothed residual**, so with `sigma > 0`
a `w = 0` group still feels force leaking in from weighted neighbours within
the kernel radius — that is the intended behaviour (the constraint is on the
smoothed field), not an approximation.

`r_th > 0` gives the restraint a flat bottom: deviations below `r_th` are
free, so thermal fluctuation is not fought. Widening `r_th` is an
alternative to decaying `k` when releasing the bias.

Under domain decomposition, construct from the **full** replicated frame
before `distribute`, as with every gauge consumer.
"""
function PolarizationMapUmbrella(
    atoms::Atoms, k, P_target, gridsize, qc_table, qs_table;
    gauge::Union{PolarizationGauge,Atoms},
    weight = nothing, sigma = 0.0, r_th = 0.0,
    backend = KA.CPU(), sumr2_log::Union{Nothing,String} = nothing,
)
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("PolarizationMapUmbrella: construct from the full replicated " *
              "frame BEFORE distribute (the gauge reference and the group " *
              "asserts assume the whole system)")
    @assert minimum(atoms.arrays[:groupid]) == 1 "groupid should start from 1"

    Nx, Ny, Nz = Tuple(Int.(gridsize))
    ng = Nx * Ny * Nz
    @assert ng == maximum(atoms.arrays[:groupid]) "gridsize must match #groups"

    Pt = Array(P_target)
    size(Pt) == (ng, 3) ||
        error("PolarizationMapUmbrella: P_target is $(size(Pt)), expected " *
              "($ng, 3) for gridsize $((Nx, Ny, Nz))")
    all(isfinite, Pt) ||
        error("PolarizationMapUmbrella: P_target contains non-finite entries")

    w = weight === nothing ? ones(FPPREC, ng) : Vector{FPPREC}(Array(weight))
    length(w) == ng ||
        error("PolarizationMapUmbrella: weight has $(length(w)) entries, " *
              "expected $ng")
    all(>=(0), w) ||
        error("PolarizationMapUmbrella: weight must be non-negative (a " *
              "negative weight would push the field AWAY from the target)")
    wsum = sum(w)
    wsum > 0 ||
        error("PolarizationMapUmbrella: all weights are zero — the bias would " *
              "do nothing and the reported CV would divide by zero")

    pg = _as_gauge(gauge, (Nx, Ny, Nz), qc_table, qs_table)
    pg.gridsize == (Nx, Ny, Nz) ||
        error("PolarizationMapUmbrella: gauge is for grid $(pg.gridsize) but " *
              "this umbrella is being built on $((Nx, Ny, Nz))")
    validate_gauge(pg, atoms)

    M_frac_ref = KA.zeros(backend, FPPREC, ng, 3)
    copyto!(M_frac_ref, pg.M_frac_ref)

    # Target and weight live on the grid, not the flat table: every use of
    # them downstream is elementwise against a smoothed grid, so converting
    # once here keeps the per-step path free of index arithmetic.
    tgt = ntuple(d -> _to_grid(backend, Pt[:, d], (Nx, Ny, Nz)), 3)
    wgrid = _to_grid(backend, w, (Nx, Ny, Nz))

    Ghat = _build_ghat(backend, (Nx, Ny, Nz), sigma)
    rfft_plan, brfft_plan = _polfield_plans(backend, (Nx, Ny, Nz))
    workspace = _polfield_workspace(backend, (Nx, Ny, Nz))

    if sumr2_log !== nothing && (!MPI.Initialized() || MPI.Comm_rank(MPI.COMM_WORLD) == 0)
        open(sumr2_log, "w") do io
            println(io, "sum_r2,cv,bias")
        end
    end

    return PolarizationMapUmbrella(
        FPPREC(k), tgt[1], tgt[2], tgt[3], wgrid, FPPREC(wsum),
        (Nx, Ny, Nz), FPPREC(sigma), FPPREC(r_th),
        M_frac_ref, Ghat, rfft_plan, brfft_plan, workspace,
        Dict{String,Any}(), sumr2_log)
end

# Flat group-indexed vector → (Nx, Ny, Nz) grid, ASE-natural order.
function _to_grid(backend, v::AbstractVector, gridsize::NTuple{3,Int})
    Nx, Ny, Nz = gridsize
    host = zeros(FPPREC, Nx, Ny, Nz)
    for i in eachindex(v)
        gx, gy, gz = _decode_g(i, Ny, Nz)
        host[gx + 1, gy + 1, gz + 1] = FPPREC(v[i])
    end
    dev = KA.zeros(backend, FPPREC, Nx, Ny, Nz)
    copyto!(dev, host)
    return dev
end

@inline _target_for(pu::PolarizationMapUmbrella, d::Int) =
    d == 1 ? pu.P_target_x : (d == 2 ? pu.P_target_y : pu.P_target_z)

"""
    umbrella_p!(atoms::Atoms, pu::PolarizationMapUmbrella, prms)

Apply the map umbrella: smooth the gauge-referenced per-group polarisation,
measure the residual against the target map, add the bias energy, and
scatter the gradient into `atoms.arrays[:forces]` /
`atoms.arrays[:shellforces]` (and the virial).

Populates `pu.results["cv"]` — the weighted RMS deviation
`sqrt(Σ w|r|² / Σ w)`, which goes to zero as the configuration reaches the
target — plus `"bias"` and `"sum_r2"`.
"""
function umbrella_p!(atoms::Atoms, pu::PolarizationMapUmbrella, prms)
    Nx, Ny, Nz = pu.gridsize
    ng = Nx * Ny * Nz
    invN = one(FPPREC) / FPPREC(ng)
    r_th = pu.r_th
    k = pu.k

    Pg, Pg_ref_now = _polfield_pg(atoms, pu.M_frac_ref, prms[:qc], prms[:qs], ng)

    Pgrid    = pu.workspace.Pgrid
    Phat     = pu.workspace.Phat
    Ptilde_d = pu.workspace.Ptilde_d
    rmag     = pu.workspace.rmag
    ggrid    = pu.workspace.ggrid
    rhat     = pu.workspace.rhat
    dUdPg_d  = pu.workspace.dUdPg_d
    dUdPg    = pu.workspace.dUdPg
    wgrid    = pu.wgrid
    Ghat     = pu.Ghat

    # Pass 1: smoothed field, then the residual against the fixed target.
    for d in 1:3
        rgrid_d = _rgrid_for(pu.workspace, d)
        tgt_d = _target_for(pu, d)
        _scatter_column!(Pgrid, Pg, d, Ny, Nz)
        _gauss_smooth!(Ptilde_d, Pgrid, Ghat, Phat, pu.plan_rfft, pu.plan_brfft, invN)
        @. rgrid_d = Ptilde_d - tgt_d
    end

    rgrid_x = pu.workspace.rgrid_x
    rgrid_y = pu.workspace.rgrid_y
    rgrid_z = pu.workspace.rgrid_z

    @. rmag = sqrt(rgrid_x * rgrid_x + rgrid_y * rgrid_y + rgrid_z * rgrid_z)
    @. ggrid = max(zero(FPPREC), rmag - r_th)          # m_g (scratch)
    # mapreduce rather than broadcast-then-sum: the latter would allocate a
    # whole group grid twice per force evaluation (a device allocation and an
    # extra kernel on the GPU).
    U = FPPREC(0.5) * k * mapreduce((w, m) -> w * m * m, +, wgrid, ggrid)
    sum_r2 = sum(abs2, rgrid_x) + sum(abs2, rgrid_y) + sum(abs2, rgrid_z)
    # Weighted RMS deviation: the monitor that says how close the run is to
    # the target, on the same footing as the restrained quantity.
    cv = sqrt(mapreduce((w, r) -> w * r * r, +, wgrid, rmag) / pu.wsum)

    # Pass 2: dU/dP̃ = k·w·m·r̂, carried back through the self-adjoint
    # smoothing.  No chain-rule term exists here — unlike the LS-amplitude
    # umbrella, the target does not depend on P.
    for d in 1:3
        rgrid_d = _rgrid_for(pu.workspace, d)
        @. ggrid = ifelse(
            rmag > r_th,
            k * wgrid * (rmag - r_th) * rgrid_d / max(rmag, eps(FPPREC)),
            zero(FPPREC),
        )
        _gauss_smooth!(dUdPg_d, ggrid, Ghat, rhat, pu.plan_rfft, pu.plan_brfft, invN)
        AK.foraxes(Pg, 1) do i
            gx, gy, gz = _decode_g(i, Ny, Nz)
            dUdPg[i, d] = dUdPg_d[gx + 1, gy + 1, gz + 1]
        end
    end

    atoms.info[:potential_energy] .+= U
    grad_group_p!(atoms, Pg, dUdPg, prms[:qc], prms[:qs], Pg_ref_now; ng = ng)

    pu.results["cv"]     = cv
    pu.results["bias"]   = U
    pu.results["sum_r2"] = sum_r2

    if pu.sumr2_log !== nothing && Cores.isroot(Cores.domain(atoms))
        open(pu.sumr2_log, "a") do io
            println(io, "$(sum_r2),$(cv),$(U)")
        end
    end

    nothing
end

# ---------------------------------------------------------------------------
#  Target-map persistence
#
#  A target map is only meaningful together with the gauge it was measured
#  against, and the two are produced by different scripts at different times.
#  Storing the gauge's identity inside the map file is what makes the
#  mismatch — the exact failure PolarizationGauge exists to prevent — loud
#  instead of silent.
# ---------------------------------------------------------------------------

const _TARGET_FORMAT = 1

"""
    save_target_map(path, P_target, gauge::PolarizationGauge)
    load_target_map(path, gauge::PolarizationGauge) -> Matrix{FPPREC}

Persist / restore a target map together with the identity of the gauge it
was measured against. `load_target_map` refuses a file whose gauge does not
match the one handed to it.

The stored identity includes the reference **structure** (`pos_ref`, `id_ref`),
not only the topology (`gridsize`, `elemid`, `groupid`). Two gauges built from
the same composition and grouping but different reference frames — an ideal
cell at a slightly different lattice constant, a thermally rattled snapshot —
agree on every topological field while placing `P = 0` somewhere else entirely,
and a map measured against one would then be silently offset when used with the
other. That is the exact failure [`PolarizationGauge`](@ref) exists to prevent,
so the check has to reach the structure. Everything is stored outright rather
than hashed, so it does not depend on hash stability across Julia versions.
"""
function save_target_map(path::AbstractString, P_target, gauge::PolarizationGauge)
    Pt = Matrix{FPPREC}(Array(P_target))
    ng = prod(gauge.gridsize)
    size(Pt) == (ng, 3) ||
        error("save_target_map: P_target is $(size(Pt)), expected ($ng, 3) for " *
              "the gauge's grid $(gauge.gridsize)")
    jldsave(path;
            format = _TARGET_FORMAT,
            P_target = Pt,
            gridsize = collect(gauge.gridsize),
            elemid = gauge.elemid,
            groupid = gauge.groupid,
            id_ref = gauge.id_ref,
            pos_ref = gauge.pos_ref)
    return path
end

@doc (@doc save_target_map)
function load_target_map(path::AbstractString, gauge::PolarizationGauge)
    d = JLD2.load(path)
    haskey(d, "format") ||
        error("load_target_map: $path is not a target-map file (no format tag)")
    d["format"] == _TARGET_FORMAT ||
        error("load_target_map: $path has format $(d["format"]), this build " *
              "writes $(_TARGET_FORMAT)")
    Tuple(Int.(d["gridsize"])) == gauge.gridsize &&
        Vector{Int}(d["elemid"]) == gauge.elemid &&
        Vector{Int}(d["groupid"]) == gauge.groupid &&
        Vector{Int}(d["id_ref"]) == gauge.id_ref &&
        Matrix{FPPREC}(d["pos_ref"]) == gauge.pos_ref ||
        error("load_target_map: $path was measured against a different gauge.  " *
              "A target map is only meaningful with the gauge it was measured " *
              "against — the zero-point moves with the reference structure, so " *
              "the map would be silently offset.  Reload that gauge (the same " *
              "save_gauge file), or re-measure the map against this one with " *
              "group_p_gauged")
    return Matrix{FPPREC}(d["P_target"])
end
