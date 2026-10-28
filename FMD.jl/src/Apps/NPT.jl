using Random
using LinearAlgebra
using FMD
using FMD.Cores
using FMD.Calculators
using Printf
using MPI

# Thermostat/barostat machinery (structs, update!, scale_cell!,
# conserved_energy, voigt2matrix, _baro_velocity_exponent) lives in
# MDControl.jl (Issue #136).
#
# The driver skeleton shared with relax! — backend/DD entry
# (_setup_frame), the MLIP dispatch helpers, and the neighbour-list family
# (ForceStack, build_force_stack, rebuild_lists!, evaluate_forces!) —
# lives in Driver.jl (Issue #207). What remains here is MD-specific: the
# initial velocities, the shell-history predictor, the COM removal, and
# the integration loop.

"""
    _init_shist!(atoms::Atoms) -> Int

Register the 3-term shell-displacement extrapolation history
`atoms.arrays[:shist1..3]` (initialised to copies of the current
`:shelldisp`) and return the initial rotation count `0`. Living in
`atoms.arrays` means the `CellList` re-sort (`atoms[idx]`) permutes the
history together with every other per-atom array — no manual re-ordering
by the caller.
"""
function _init_shist!(atoms::Atoms)
    atoms.arrays[:shist1] = copy(atoms.arrays[:shelldisp])
    atoms.arrays[:shist2] = copy(atoms.arrays[:shelldisp])
    atoms.arrays[:shist3] = copy(atoms.arrays[:shelldisp])
    return 0
end

"""
    _extrapolate_shelldisp!(atoms::Atoms, shist_count::Int)

Write the 3-point predictor `3·s(t) − 3·s(t−1) + s(t−2)` into `:shelldisp`
as the shell-CG initial guess. No-op during warmup (`shist_count < 3`),
where the history slots still hold initialisation copies that must never
be read.
"""
function _extrapolate_shelldisp!(atoms::Atoms, shist_count::Int)
    if shist_count >= 3
        @. atoms.arrays[:shelldisp] = 3*atoms.arrays[:shist1] - 3*atoms.arrays[:shist2] + atoms.arrays[:shist3]
    end
    return nothing
end

"""
    _remove_com!(atoms, dom)

Subtract the whole-system mass-weighted mean velocity. The serial branch
performs bit-identically the historical in-loop expression; under DD the
four scalars (Σmv, Σm) are fused into one Allreduce so the subtraction is
rank-uniform. Empty ranks contribute zeros but ride the collective. (The
initial-velocity block in `npt!` inlines the same reduction on its host
buffer.)

Note: with the total momentum pinned to zero the mass-weighted fractional
COM is a constant of motion, so the sampled configurational measure is
V^(N−1) and the barostats target the Gamma(N) ideal-gas volume marginal
(docs/src/theory/scr-barostat.md). The NoseHoover/CSVR targets (3N−3)
must stay consistent with this call; the SCR measure correction in
`_scr_mu` is independent of the DOF count by construction, but the
Gamma(N) test expectations in test_scr.jl do assume it.
"""
function _remove_com!(atoms::Atoms, dom)
    if dom isa Cores.MPIDomain
        combuf = zeros(FPPREC, 4)
        if length(atoms) > 0
            # Array(): device reduction → host buffer (identity on CPU).
            combuf[1:3] .= Array(vec(sum(atoms.arrays[:velo] .* atoms.arrays[:masses], dims = 1)))
            combuf[4] = sum(atoms.arrays[:masses])
        end
        Cores.allreduce_sum!(dom, combuf)
        # The subtrahend must live on the compute backend for the broadcast.
        comrow = similar(atoms.arrays[:velo], 1, 3)
        copyto!(comrow, reshape(combuf[1:3] ./ combuf[4], 1, 3))
        atoms.arrays[:velo] .-= comrow
    else
        com_vel = sum(atoms.arrays[:velo] .* atoms.arrays[:masses], dims = 1) / sum(atoms.arrays[:masses])
        @. atoms.arrays[:velo] -= com_vel
    end
    return nothing
end

"""
    _rotate_shist!(atoms::Atoms, shist_count::Int) -> Int

Shift the history by one step (`:shist1` becomes the current `:shelldisp`)
and return the incremented rotation count.
"""
function _rotate_shist!(atoms::Atoms, shist_count::Int)
    atoms.arrays[:shist3] .= atoms.arrays[:shist2]
    atoms.arrays[:shist2] .= atoms.arrays[:shist1]
    atoms.arrays[:shist1] .= atoms.arrays[:shelldisp]
    return shist_count + 1
end

"""
    _report_temperature(atoms, dom) -> Float64

Instantaneous kinetic temperature (Kelvin) reported in `log.txt`, over
3N-3 whole-system DOF: COM momentum is removed every step
(`_remove_com!`), and the thermostat targets in [`update!`](@ref) use
the same count — dividing by 3N would systematically under-report by
(N-1)/N (Issue #93). Collective under DD: `kinetic_energy` and
`n_global` are whole-system values on every rank.
"""
_report_temperature(atoms, dom) =
    2 * kinetic_energy(atoms) * UNITS[:Kelvin] / (3 * Cores.n_global(dom, atoms) - 3)

"""
    npt!(atoms::Atoms, params) -> Atoms

MD with selectable temperature/pressure control (Issue #136) using a
core-shell potential plus an optional fitted MLIP. The default is NPT with
the MTTK barostat and a Nosé–Hoover particle thermostat; a fixed-cell
NVT/NVE run is `barostat = :none` (± `thermostat = :none`). Runs on the
GPU by default; pass `params[:backend] = :cpu` for host execution (the
array type is resolved once by [`_arrtyp`](@ref) and used for atoms,
params, and the MLIP alike).

# `params` keys

- `:acep` ([`ACEParam`](@ref)) or `:splp` ([`SplineParam`](@ref)) — the
  MLIP (optional). Pass at most one; omit both to run the pure shell
  model (Coulomb + Buckingham + Spring, no MLIP term — Issue #133).
- Shell parameters required by [`CoreShell`](@ref): `:qc`, `:qs`, `:k2`,
  `:k4`, `:eta_c`, `:eta_s`, `:a`, `:rho`, `:b`, `:c`, `:e`.
- `:rc`, `:skin` — Coulomb real-space cutoff and neighbor-list skin.
- `:pme_meshsize` — PME mesh spacing (the reciprocal-space part always
  runs through [`PMEInfo`](@ref); the grid is 2·3·5·7-smooth rounded, see
  [`pme_fftsize`](@ref)). The Ewald α is auto-determined from
  (`:rc`, `:pme_meshsize`) by error-estimate minimization and the
  estimated RMS force error is logged; pass `:alpha` to override
  (Issue #216 — `:ewald_tolerance` is a fitting-app key and is ignored
  here with a warning).
- `:cgtol`, `:cgmaxiter` — shell relaxation tolerance per MD step.
- `:shell_dynamics` (optional) — `:scf` (default; per-step CG relaxation
  from the 3-point predictor) or `:xlbomd` (shadow XL-BOMD, Issue #149:
  an auxiliary shell field propagated by its own time-reversible Verlet,
  with the dynamics running on the exact gradient of the shadow energy —
  no SCF iterations, ~3 force evaluations per step). `:xlbomd` rejects
  `:polarization_umbrella` and is tuned by `:xl_kappa`, `:xl_warmup`,
  `:xl_max_deviation`, `:xl_kernel`, `:xl_kernel_order`,
  `:xl_kernel_spectrum`. It integrates a *shadow* surface, so its
  trajectories differ from the `:scf` ones, and the reported virial is
  that surface's strain derivative (Issue #151) — which is what makes a
  barostat well-defined here. See `docs/src/theory/xlbomd-shell.md`.
- `:initial_temperature`, `:target_temperature`, optionally
  `:final_temperature` (linear ramp).
- `:seed` — RNG seed for the initial velocities (used when
  `initial_temperature > 0`). The draw is keyed by `:id` from a dedicated
  host RNG (the ambient global RNG is never consulted, Issue #34), so it
  is invariant to the row order of the input frame. The default `seed = 0`
  makes runs reproducible — identical input gives an identical trajectory;
  pass a distinct seed per run for statistically independent replicas
  (e.g. umbrella windows).
- `:thermostat` — `:nh` (Nosé–Hoover, default), `:csvr` ([`CSVR`](@ref)
  stochastic velocity rescaling; requires `:seed` and a positive target
  temperature) or `:none`.
- `:barostat` — `:mttk` ([`MTTK`](@ref), default), `:scr` ([`SCR`](@ref)
  stochastic cell rescaling — first-order, no cell ringing; requires
  `:compressibility`, `:seed` and a thermostat) or `:none` (fixed cell).
- `:tau_t`, `:tau_p` — thermostat / barostat time constants (a.u.);
  required by the respective non-`:none` selection.
- `:compressibility` — isothermal-compressibility estimate κ_T for
  `:scr` (inverse-pressure a.u., `UNITS[:GPa]/B_GPa`); only sets the
  effective relaxation time, never the sampled ensemble.
- `:target_stress` — Voigt-6 stress; required unless `barostat = :none`.
- `:timestep`, `:nsteps`, `:output_interval`, `:output_dir`.
- `:efield` — external field (atomic units).
- `:backend` (optional) — `:cuda` (default) or `:cpu`.
- `:mpi` (optional) — `true` runs domain-decomposed under mpiexec, on
  either backend (`:cuda` requires a CUDA-aware MPI build — see
  docs/src/hpc.md; explicit opt-in, never inferred from
  `MPI.Initialized()`).
- `:polarization_umbrella` (optional) — a [`PolarizationUmbrella`](@ref) to
  bias the macroscopic polarisation; when set, `Bias.log` is also written.
- `:surface_term` (optional) — a [`SurfaceTerm`](@ref) adding the
  depolarization field of an anisotropic external medium (generalised
  Ewald surface term). Its gauge also fixes the `dipole_phys` zero-point
  when neither `:polarization_gauge` nor an umbrella supplies one.
  Combines with `shell_dynamics = :xlbomd` (Issue #181): its analytic HVP
  rides the shadow step, and `xl_init!` refuses an `:xl_kappa` /
  `:xl_kernel_spectrum` that provably cannot cover the spectrum the
  depolarization block adds.
- `:polmap_interval` (optional) — write the local polarization map
  (gauged per-group polarization, [`write_polmap`](@ref)) every this many
  steps as `polmap_<n>.txt`. Requires `:polarization_gauge`: unlike
  `dipole_phys` the map has **no** starting-frame fallback (Issue #220).
- `:polarization_gauge` (optional) — a [`PolarizationGauge`](@ref) fixing the
  zero-point of the reported `dipole_phys`. Takes precedence over the
  umbrella's own gauge, which takes precedence over the surface term's
  (normally all the same object — validate_params cross-checks them). With
  none of the three, the zero-point falls back to **this run's starting
  frame**, which makes `dipole_phys` a change-since-restart rather than a
  polarisation — set this key whenever the absolute value matters or runs
  are to be concatenated.

# Output (under `params[:output_dir]`)

- `log.txt` — per-step `temp` (kinetic temperature over 3N-3 DOF, see
  [`_report_temperature`](@ref)), full lattice (9), `dipole` (3),
  gauge-fixed `dipole_phys` (3), `volume`, `conserved` (diagnostic H′ of
  the active control combination, see [`conserved_energy`](@ref)),
  `shell_rms` (rms of the shell-force components left by this step's CG
  relaxation, Ha/Bohr like `cgtol`; the stopping test bounds only their
  maximum. NaN on a `shell_dynamics = :xlbomd` shadow step, which runs no
  CG),
  `ncg` (dimensionless count: shell-relaxation force evaluations inside
  the CG loop, the [`calculate!`](@ref) return value, Issue #144. The
  initial shell evaluation and the core evaluation are not counted, so
  the per-step force-evaluation cost is `ncg + 1` shell + 1 core).
  Units: K, Å, Å³, Hartree, Ha/Bohr.
- `step_<n>.xyz` — extxyz snapshots every `output_interval` steps.
- `polmap_<n>.txt` — when `polmap_interval` is set: the local polarization
  map every `polmap_interval` steps ([`write_polmap`](@ref) format; e/Å²,
  flat groupid order, `P = 0` at the gauge frame).
- `Bias.log` — when umbrella sampling is active: `CV,Bias` per step.

Returns the final `Atoms` (on the selected backend).

See also: [`update!`](@ref), [`MTTK`](@ref), [`NoseHoover`](@ref),
[`scale_cell!`](@ref), [`conserved_energy`](@ref), [`apply_mlip!`](@ref),
[`build_mlip_nlists`](@ref), [`_resolve_mlip`](@ref).
"""
function npt!(atoms::Atoms, params)
    # Input-boundary checks that need nothing built: keep them ahead of
    # _setup_frame, so a malformed direct call (which bypasses
    # validate_params) fails on its own message rather than on whatever
    # solver parameter the setup happens to touch first (the α resolution
    # reads :pme_meshsize since Issue #216). `get`, not indexing: a direct
    # call missing :initial_temperature must keep failing on whatever the
    # setup reports, not on a FieldError here.
    if get(params, :initial_temperature, 0.0) > 0.0
        haskey(params, :seed) ||
            error("params[:seed] is required when initial_temperature > 0 " *
                  "(default_params(Val(:NPTMD), ...) supplies seed = 0)")
        params[:seed] isa Integer ||
            error("params[:seed] must be an Integer (got $(typeof(params[:seed])))")
    end

    # Backend / DD entry, params conversion, gauge validation and the
    # frame split are the shared driver skeleton (Driver.jl). Comm_size == 1
    # still runs the DD code path, so `mpiexec -n 1` is comparable to serial.
    atoms, params, arrtyp, dom = _setup_frame(atoms, params, "npt!")
    # Local polarization map output (Issue #220): nothing unless
    # polmap_interval is set; checked here, before the force stack is
    # built, so a missing gauge fails at entry.
    pm = _polmap_init(atoms, params)

    # Initialize velocities before the first CellList build. The draw is
    # keyed by :id — atom id k always receives stream positions
    # 3(k-1)+1 .. 3k of Xoshiro(params[:seed]) — so the assignment is
    # invariant to the row order of the input frame (restart frames read
    # back cell-sorted rows) and, in Phase 1, to the domain decomposition
    # (a rank draws a prefix of the stream covering its local id set). A
    # dedicated host RNG keeps the ambient global RNG state out of the
    # trajectory (Issue #34); the host buffer also leaves the caller's
    # frame untouched. The COM removal is a whole-system sum (to be
    # collectivized together with the DD driver).
    if params[:initial_temperature] > 0.0
        # (presence/type of :seed already checked at the entry above)
        rng = Random.Xoshiro(params[:seed])
        ids = Array(atoms.arrays[:id])
        nglob = Cores.n_global(dom, atoms)
        maximum(ids; init = 0) <= nglob ||
            error("npt!: atoms.arrays[:id] exceeds the global atom count — " *
                  "pass the full frame (ids must lie in 1:N)")
        stream = randn(rng, FPPREC, 3, nglob)          # column k = draws for id k
        velo_h = permutedims(stream[:, ids])
        masses_h = Array(atoms.arrays[:masses])
        @. velo_h *= sqrt(params[:initial_temperature] / UNITS[:Kelvin] / masses_h)
        # COM removal: whole-system sum, fused 4-scalar collective (the
        # in-loop counterpart is _remove_com!).
        combuf = zeros(FPPREC, 4)
        if length(atoms) > 0
            combuf[1:3] .= vec(sum(velo_h .* masses_h, dims = 1))
            combuf[4] = sum(masses_h)
        end
        Cores.allreduce_sum!(dom, combuf)
        velo_h .-= reshape(combuf[1:3] ./ combuf[4], 1, 3)
        copyto!(atoms.arrays[:velo], velo_h)
    end

    th = _build_thermostat(params)
    baro = _build_barostat(params)
    # Shadow XL-BOMD restrictions (Issue #149). validate_params reports the
    # same conditions for app-level runs; this covers direct npt! callers.
    # A barostat is supported since Issue #151 — xl_shadow_forces! reports
    # the shadow virial, so the cell is driven by the strain derivative of
    # the surface the dynamics runs on — and :surface_term since Issue
    # #181 gave the depolarization field its analytic HVP.
    if get(params, :shell_dynamics, :scf) === :xlbomd
        haskey(params, :polarization_umbrella) &&
            error("shell_dynamics = :xlbomd does not support the polarization " *
                  "umbrella (its bias has no analytic Hessian-vector product yet)")
    end
    # Run constants of the control context: the whole-system atom count
    # (rank-local under DD would silently rescale the thermostat masses per
    # rank) and the host-side target stress tensor. :target_stress is
    # conditionally required (a fixed-cell run has no use for it —
    # _build_barostat demands it for :mttk); the zeros fallback is never
    # read by the NoBarostat path.
    nglob_ctrl = Cores.n_global(dom, atoms)
    tstress_h = haskey(params, :target_stress) ?
        voigt2matrix(Array(params[:target_stress])) : zeros(3, 3)
    # One-time barostat setup against the starting frame (SCR freezes its
    # strain-coupling reference cell here; no-op otherwise).
    _init_barostat!(baro, atoms, tstress_h)

    fs, atoms = build_force_stack(atoms, params, arrtyp, dom)
    evaluate_forces!(atoms, fs, params)

    # Gauge-fixing fractional moment used downstream to report a physical
    # (cell-covariant) dipole_phys = M_abs(t) - Lᵀ(t)·M_frac_ref_total.
    # Precedence: an explicitly declared gauge wins over the umbrella's own
    # (they are the same object in a normal umbrella run), and both win over
    # the implicit "this run's starting frame" fallback.
    mfr_total = if haskey(params, :polarization_gauge)
        # Already validated against the whole frame at entry, before the
        # domain split.
        total_mfrac_ref(params[:polarization_gauge])
    elseif haskey(params, :polarization_umbrella)
        Vector(FPPREC.(Array(sum(params[:polarization_umbrella].M_frac_ref, dims = 1))[1, :]))
    elseif haskey(params, :surface_term)
        # The surface term's zero-point IS a total fractional dipole —
        # reporting dipole_phys against the same gauge makes log.txt
        # directly comparable with the δM the term biases.
        Vector{FPPREC}(params[:surface_term].M_frac_ref)
    else
        total_mfrac_ref(atoms, params)
    end

    shist_count = _init_shist!(atoms)
    xl = xl_init!(atoms, params)   # nothing unless shell_dynamics = :xlbomd

    for step in 1:params[:nsteps]
        c = _control_context(params, step, nglob_ctrl, tstress_h)
        update!(atoms, th, baro, c, arrtyp)
        @. atoms.arrays[:velo] += atoms.arrays[:forces]/atoms.arrays[:masses] * params[:timestep]/2

        _remove_com!(atoms, dom)

        scale_cell!(baro, atoms, c, arrtyp)
        atoms.arrays[:pos] .+= ((atoms.arrays[:velo] * params[:timestep]) * (atoms.info[:lattice] \ I))
        # The XL path owns its own shell predictor (the auxiliary field), so
        # the polynomial extrapolation is skipped there.
        xl === nothing && _extrapolate_shelldisp!(atoms, shist_count)

        atoms = rebuild_lists!(fs, atoms, params, arrtyp, dom)

        ncg = evaluate_forces!(atoms, fs, params; xl = xl)
        shist_count = _rotate_shist!(atoms, shist_count)

        calc_dipole_phys!(atoms, params, mfr_total)

        @. atoms.arrays[:velo] += atoms.arrays[:forces]/atoms.arrays[:masses] * params[:timestep]/2
        update!(atoms, th, baro, c, arrtyp)

        # IO -----------------------------------------------------------
        # kinetic_energy / dipole / conserved_energy are collective
        # (whole-system values on every rank); only the writes are
        # root-gated.
        temp = _report_temperature(atoms, dom)
        # Sampled at the end of the step, after the closing half-kick and
        # update!, so velocities, positions, cell and the control
        # variables all sit at the same time t+dt; see conserved_energy
        # for when it is an actual invariant.
        conserved = conserved_energy(th, baro, atoms, c)

        if Cores.isroot(dom)
            open(joinpath(params[:output_dir], "log.txt"), ifelse(step == 1, "w", "a")) do io
                # Columns: temp | lattice(9) | dipole(3) | dipole_phys(3) | volume | conserved | shell_rms | ncg
                @printf(io, "%15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %6d \n",
                    temp,
                    (Matrix(atoms.info[:lattice]) .* UNITS[:Å])...,
                    (Vector(atoms.info[:dipole] .* UNITS[:Å]))...,
                    (Vector(atoms.info[:dipole_phys] .* UNITS[:Å]))...,
                    volume(atoms) * UNITS[:Å]^3,
                    conserved,
                    Array(atoms.info[:shell_residual_rms])[1],
                    ncg)
            end
        end

        # cv/bias are replicated on every rank (umbrella_p! Allreduces the
        # group polarisation, #95); the isroot gate only picks the writer.
        if haskey(params, :polarization_umbrella) && Cores.isroot(dom)
            if step == 1
                #> Write header
                open(joinpath(params[:output_dir], "Bias.log"), "w") do io
                    println(io, "CV,Bias")
                end
            end
            open(joinpath(params[:output_dir], "Bias.log"), "a") do io
                println(io, "$(params[:polarization_umbrella].results["cv"]),$(params[:polarization_umbrella].results["bias"])")
            end
        end

        if step % params[:output_interval] == 0
            # gather_frame is collective (every rank participates); the
            # root receives the full frame in the original :id order and
            # writes it alone.
            frame = Cores.gather_frame(atoms, dom)
            if Cores.isroot(dom)
                open(joinpath(params[:output_dir], "step_$(step).xyz"), "w") do io
                    writexyz(io, frame, params)
                end
            end
        end

        if pm !== nothing && step % pm.every == 0
            # Collective: the group table is Allreduced inside, so every
            # rank holds the whole-system map and only the write is
            # root-gated (no per-atom gather — the map is per group).
            # Kept right after the snapshot with nothing touching pos /
            # lattice in between, so polmap_<n>.txt IS the map of
            # step_<n>.xyz (app[:PolMap] on the snapshot reproduces it —
            # pinned by test_polmap_app.jl).
            Pg = group_p_gauged(atoms, pm.M_frac_ref, params[:qc], params[:qs])
            if Cores.isroot(dom)
                write_polmap(joinpath(params[:output_dir], "polmap_$(step).txt"),
                             Pg, pm.gridsize, atoms.info[:lattice]; step = step)
            end
        end
    end

    # A shadow XL-BOMD run that kept falling back was paying for the
    # relaxation it was meant to avoid — report the rate rather than let it
    # hide in the ncg column.
    if xl !== nothing && Cores.isroot(dom)
        @info "shadow XL-BOMD: $(xl.nreset) fallback relaxation(s) in $(params[:nsteps]) steps"
    end

    return atoms
end

"""
    _polmap_init(atoms, params) -> Union{Nothing,NamedTuple}

Set up the local polarization map output (Issue #220): `nothing` unless
`params[:polmap_interval]` is present, otherwise `(; every, M_frac_ref,
gridsize)` with the gauge's fractional reference table copied onto the
compute backend once, so the per-step call is a plain
[`group_p_gauged`](@ref) with no host round trip.

Covers direct `npt!` callers (validate_params reports the same conditions
for app-level runs): the interval must be a positive Integer, and
`:polarization_gauge` must be declared — the map has no starting-frame
fallback, because a map measured from wherever a run happened to start is
not a polarization and every chained run would carry a different zero.
The gauge itself was validated against the whole frame in `_setup_frame`.
"""
function _polmap_init(atoms::Atoms, params)
    haskey(params, :polmap_interval) || return nothing
    every = params[:polmap_interval]
    every isa Integer && every >= 1 ||
        error("npt!: polmap_interval must be a positive Integer (got $(repr(every)))")
    haskey(params, :polarization_gauge) ||
        error("npt!: polmap_interval requires params[:polarization_gauge] — the local " *
              "polarization map is only defined against a declared zero-point " *
              "(no starting-frame fallback)")
    gauge = params[:polarization_gauge]
    M_frac_ref = similar(atoms.arrays[:pos], FPPREC, size(gauge.M_frac_ref)...)
    copyto!(M_frac_ref, gauge.M_frac_ref)
    return (; every = Int(every), M_frac_ref = M_frac_ref, gridsize = gauge.gridsize)
end

"""
    default_params(::Val{:NPTMD}, params) -> NamedTuple

Default params for `FMD.app[:NPTMD]`. The shell-model placeholders
are intentionally zero — production runs are expected to splat values
loaded from `SplineP.jld2` or `ACEP.jld2` (the
`pot = FMD.loadpotential(...)`; `pot...` pattern in
`example/NPT-MD/input.jl`) on top of these defaults.

# Defaults

| Key | Default | Meaning |
|---|---|---|
| `nsteps` | `1000` | MD steps |
| `thermostat` | `:nh` | Particle thermostat (`:nh`, `:csvr` or `:none`) |
| `barostat` | `:mttk` | Barostat (`:mttk`, `:scr` or `:none` = fixed cell) |
| `seed` | `0` | Initial-velocity RNG seed |
| `cgtol`, `cgmaxiter` | `2e-5`, `30` | Per-step shell relaxation |
| `efield` | `[0,0,0]` | External field |
| `skin` | `0.1` | Neighbor list skin |
| `rc` | `9/Å` | Coulomb cutoff |
| `e` | `1e5` | Buckingham cap |
| `qc`, `qs`, `k2`, `k4` | zeros | Shell-model placeholders |
| `eta_c`, `eta_s` | `1e-5/Å` | Gaussian widths |
| `a`, `rho`, `b`, `c` | zeros / 1e-10 | Buckingham per pair |

# Required (caller-provided)

- `:input` — extxyz file with the starting configuration (`input.xyz`).
- `:output_dir` — output directory; created with `mkpath`.
- `:elements` — element-symbol → species id.
- `:acep` **or** `:splp` — fitted MLIP (typically loaded from
  `ACEP.jld2` / `SplineP.jld2`). Optional: omit both to run the pure
  shell model (the Buckingham block then carries all short-range physics).
- Shell parameters from `ShellParam.jld2` / the fit JLD2 (qref, qc, qs,
  k2, k4, eta_c, eta_s).
- Time control: `:timestep`, `:tau_t`, `:tau_p`,
  `:initial_temperature`, `:target_temperature`, `:target_stress`.
  (`:tau_t` / `:tau_p` / `:target_stress` are required only by the
  non-`:none` thermostat / barostat selections — a fixed-cell NVE run
  needs none of them.)

See `docs/src/parameters.md` for the full reference.
"""
function default_params(::Val{:NPTMD}, params)
    _assert_elements(params)
    nelem = length(params[:elements])

    a = (
        nsteps = 1000,
        thermostat = :nh,
        barostat = :mttk,
        seed = 0,
        cgtol = 2e-5,
        cgmaxiter = 30,
        efield = [0.0, 0.0, 0.0],
        skin = 0.1,

        rc = 9.0/(UNITS[:Å]),
        e = 1e5,

        qc = zeros(nelem),
        qs = zeros(nelem),
        k2 = zeros(nelem),
        k4 = zeros(nelem),
        eta_c = fill(1e-5/(UNITS[:Å]), nelem),
        eta_s = fill(1e-5/(UNITS[:Å]), nelem),
        a = zeros(nelem,nelem),
        rho = fill(1e-10, (nelem,nelem)),
        b = zeros(nelem,nelem),
        c = zeros(nelem,nelem),
    )
end

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:NPTMD](params)`
#
# Production MD (NPT by default; fixed-cell NVT/NVE via
# params[:thermostat]/[:barostat]) on the selected backend (GPU by
# default, params[:backend] = :cpu for host execution) using a core-shell
# potential plus an optional fitted MLIP.
# Reads the first frame of `params[:input]` (extxyz), tags species using
# `params[:elements]`, creates `params[:output_dir]` if needed, and runs
# [`npt!`](@ref).
#
# Required params: see `default_params(Val(:NPTMD), params)` and `npt!`.
# Output:          `<output_dir>/log.txt`, `step_<n>.xyz`, optionally
#                  `Bias.log` (when `:polarization_umbrella` is set).
# See also:        [`npt!`](@ref), [`PolarizationUmbrella`](@ref),
#                  [`default_params`](@ref).
# --------------------------------------------------------------------------
app[:NPTMD] = function (params)
    validate_params(:NPTMD, params)

    atoms = readxyz(open(params[:input], "r"))[1]

    mkpath(params[:output_dir])

    setelemid!(atoms, params[:elements])

    npt!(atoms, params)

    return 0
end
