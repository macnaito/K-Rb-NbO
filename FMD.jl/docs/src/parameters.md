# Parameters reference

```@meta
CurrentModule = FMD
```

Every workflow entry point in `FMD.app` takes a single `NamedTuple` of
parameters. The canonical pattern in input scripts is

```julia
params = (; dataset = "dfpt.xyz", elements = Dict("K" => 1, ...), ...)
params = (; FMD.default_params(Val(:FitShellModel), params)..., params...)
FMD.app[:FitShellModel](params)
```

so caller-provided keys override the per-job defaults. This page is the
reference for **every** key; use it together with the docstrings on
[`default_params`](@ref) for the exact code paths.

## Validation

Every `FMD.app[:Job]` entry point validates the merged `params` against
the schema in `src/Apps/ParamSpec.jl` before doing any work (file IO,
`MPI.Init`, GPU allocation). Unknown keys (typos) are rejected with a
"did you mean" hint, missing required keys are reported together, and
cross-field constraints (array shapes vs `nelem`, `acep`/`splp`
exclusivity, MLIP cutoff ≤ `rc`, …) are checked up front. All
violations are collected into a single `ArgumentError`. Call
[`validate_params`](@ref)`(:Job, params)` directly to pre-check an
input script without running the job.

## Units

Internally all numeric quantities are in **atomic units** (Hartree,
Bohr, electron mass). Convert via `UNITS` from
[`FMD.Cores.UNITS`](@ref):

```julia
rc        = 9.0 / UNITS[:Å]        # 9 Å in Bohr
T_au      = 300.0 / UNITS[:Kelvin] # 300 K in Hartree
E_in_eV   = E_au * UNITS[:eV]      # back to eV for output
timestep  = 1.0 / UNITS[:fs]       # 1 fs in a.u.
```

`UNITS` keys: `eV`, `Å`, `dalton`, `Kelvin`, `Dirac`, `Hartree`,
`second`, `fs`. See [`FMD.Cores.UNITS`](@ref).

## Common keys (most apps)

| Key | Default | Unit | Meaning |
|---|---|---|---|
| `nsteps`            | varies | — | Iterations (CMA-ES generations / CG steps / MD steps) |
| `dataset_idx`       | `:` | — | Frames to use from the input dataset |
| `cgtol`             | `1e-8` (fit), `2e-5` (MD) | a.u. | Shell-relaxation residual tolerance |
| `cgmaxiter`         | `100` (fit), `30` (MD) | — | Shell-relaxation max iterations |
| `ewald_tolerance`   | `1e-6`–`1e-8` | — | **Fitting apps only** (Ewald path): error budget driving α *and* kcut together. The PME apps ignore it with a warning — see below |
| `efield`            | `[0,0,0]` | a.u. | External electric field |
| `skin`              | `0.1` (fit), `0.2` (MD examples) | — | Neighbor-list skin (fraction of cutoff) |
| `rc`                | `9–11/Å` | a.u. | Coulomb / pair cutoff |
| `alpha`             | auto | a.u. | Ewald α. PME apps (`:NPTMD` / `:Relax`): auto-determined, pass a value only to override (external-code comparison). Fitting apps: resolved from `ewald_tolerance` via [`ewaldcoeff`](@ref) |
| `e`                 | `1e5` | a.u. | Buckingham smooth cap |
| `qc`, `qs`          | per nelem | a.u. | Core / shell charges |
| `k2`, `k4`          | per nelem | a.u. | Core-shell harmonic / quartic spring constants |
| `eta_c`, `eta_s`    | `1e-5–1e-10/Å` | a.u. | Gaussian widths of cores / shells |
| `a`, `rho`, `b`, `c`| per nelem×nelem | a.u. | Buckingham `(A, ρ, b, c)` per pair |
| `elements`          | required | — | `Dict{String,Int}` element-symbol → species id |

`pme_meshsize` (used by [`PMEInfo`](@ref) — production NPT MD always
evaluates reciprocal-space Coulomb via PME) is in atomic-unit length and
controls FFT grid density (`fft_dim ≈ side / pme_meshsize`, rounded up to
a 2·3·5·7-smooth size — FFT libraries are several times slower on sizes
with large prime factors, and plain `ceil` lands on primes for ordinary
cells). Typical NPT-MD value: `2.0`.

### How the PME apps choose α (Issue #216)

On the PME path the two **cost** knobs are `rc` (real-space work) and
`pme_meshsize` (grid work); α is a splitting parameter with no physical
meaning, and the reciprocal-space error is governed by α·h (h = mesh
spacing). The old `ewald_tolerance → α` resolution saw only the
real-space budget, so at a fixed mesh, *tightening* the tolerance grew α
and could inflate the reciprocal error by orders of magnitude at
unchanged cost (measured in Issue #211: 1e-4 → 1e-8 made the
symmetry-breaking mesh artifact 1000× worse). `:NPTMD` and `:Relax`
therefore auto-determine α by minimizing an a-priori RMS force error
estimate (Kolafa–Perram real space + the exact aliasing error of the
implemented reciprocal operator) at your fixed (`rc`, `pme_meshsize`),
and log the estimate. Consequences:

- to buy accuracy, tighten `pme_meshsize` (or grow `rc`) — every knob is
  now monotone: more cost, more accuracy;
- the logged estimate is a **site-RMS for disordered (thermal)
  configurations**: it overestimates on correlated crystals (safe side),
  says nothing about `max|ΔF|` (≈ 4–5× the RMS), and cannot see the
  symmetric-lattice cancellations of Issue #211;
- α is chosen once per run from the starting cell and held fixed —
  re-optimizing mid-run would make the Hamiltonian time-dependent;
- fitting apps are untouched: their Coulomb is the analytic Ewald
  (precise-limit labels), where `ewald_tolerance` remains the honest
  accuracy knob.

## `:FitShellModel`

Built by [`FMD.default_params`](@ref) with `Val(:FitShellModel)`. Source:
`src/Apps/FitShellModel.jl`. Inputs: `dfpt.xyz`. Output: `ShellParam.jld2`.

| Key | Default | Notes |
|---|---|---|
| `nsteps` | `100` | CMA-ES generations |
| `sigma0` | `1.0` | Initial CMA-ES step size |
| `popsize` | `1` (auto-bumped to `≥ 4 + 3·log(nvars)`) | CMA-ES population |
| `data_workers` | `1` | MPI ranks per candidate (data-parallel inner loop) |
| `opt_weights` | required | `(bec = α, eps = β)` — divided by per-component dataset variance |
| `formal_charges` | required | Vector indexed by element id; used in `qs = -(qc - qref·formal)` |
| `qref` | required (initial) | Global charge rescaling, optimised |
| `lb`, `ub` | required | `NamedTuple` of optimisation bounds; keys = optimised vars |

The keys that appear in `lb` (and `ub`) are exactly the variables CMA-ES
optimises. Anything not in `lb` is held fixed at its caller-provided
value.

## `:FitSpline`

Source: `src/Apps/FitSpline.jl`. Inputs: `synth.xyz` + shell parameters
(typically loaded from `ShellParam.jld2`). Output: `SplineP.jld2`.

| Key | Default | Notes |
|---|---|---|
| `nsteps` | `10000` | CG iterations inside `fitting!` |
| `traindata_ratio` | required | Fraction of dataset used for training (rest is test) |
| `weight_force` | `1.0` | Loss weight on per-atom forces |
| `weight_stress` | `1.0` | Loss weight on virial |
| `method_weight` | required | `NamedTuple` weighting per `method` tag in the dataset (`(PBEsol = 1.0, MACE = 0.5)` is the convention used by `synth.xyz`) |
| `rigid` | `(c1=1e-8, c2=1e0, c3=1e0)` | Smoothness regularisers per body order |
| `spline_rmax2` | `6/Å` | 2-body spline cutoff |
| `spline_rmax3` | `5/Å` | 3-body spline cutoff |
| `spline_nbasis2` | `12` | Number of cubic B-spline basis fns per pair |
| `spline_nbasis3` | `6` | Per triplet — cost grows as `nbasis3³ × nelem³`, keep small |
| `splp` | optional | Existing [`SplineParam`](@ref) to warm-start |
| `nlh` | optional | [`NLHParam`](@ref) short-range repulsion baseline, subtracted from the reference E/F/V before the fit and added back at prediction time. Built with `NLHParam(elements, rc_nlh)`; requires `rc_nlh <= rc`. Saved into `SplineP.jld2` so MD re-applies exactly what was subtracted |

## `:FitACE`

Source: `src/Apps/FitACE.jl`. Inputs: `synth.xyz` + shell parameters.
Output: `ACEP.jld2`.

| Key | Default | Notes |
|---|---|---|
| `nsteps` | `10000` | CG iterations |
| `cg_preconditioner` | `:none` | `:blockchol_si` enables the per-(coefficient-field × species) block-Cholesky PCG — removes the radial-channel collinearity that stalls plain CG. Only the live (upper-triangle-convention) coefficients are factored; the structurally dead lower-triangle coefficients are handled analytically as `Γ·I`. Requires `rigid[f] > 0` for **every** coefficient field (Γ defines the preconditioner on the dead coefficients and keeps every live sub-block SPD independent of the data). Assembly memory is bounded per rank (column-panel accumulation) with one shared full-Gram window per node, so rank density is not limited by the Gram size. The hierarchical Reduce makes the factors — and hence `nsteps`-truncated coefficients — reproducible only for a fixed rank count **and** rank-to-node placement (roundoff-level differences otherwise); the converged solution is unaffected. NOTE: if you previously ran with some `rigid[f] = 0`, satisfying this requirement changes the regularisation and therefore the fitted solution — that is a modelling change, not a solver detail. Under `nsteps` truncation the preconditioner also changes (improves) the truncated coefficients even at identical `rigid` |
| `traindata_ratio`, `method_weight`, `weight_force`, `weight_stress` | as in FitSpline | |
| `rigid` | required | `NamedTuple` regularisers, one per coefficient field (`c0`, `c00`, `c11`, `c22`, `c33`, `c011`, `c022`, `c033`, `cp`) |
| `ace_rmin` | `1/Å` | Inner cutoff |
| `ace_rmax` | required (unless `acep` given) | ACE radial cutoff — deliberately no default; a physical choice the caller must make |
| `ace_nk` | `(12,6,4,4)` | Chebyshev orders `(n0, n1, n2, n3)` — must satisfy `n0 ≥ n1 ≥ n2 ≥ n3 ≥ 3` |
| `acep` | optional | Existing [`ACEParam`](@ref) to warm-start |
| `nlh` | optional | [`NLHParam`](@ref) short-range repulsion baseline — same semantics as in FitSpline, saved into `ACEP.jld2` |

## `:Relax`

Source: `src/Apps/Relax.jl`. Inputs: extxyz starting configuration +
fitted potential. Outputs: `relaxed.xyz` and `log.txt` under
`params[:output_dir]`. `FMD.app[:Relax]` returns `0` when the relaxation
converged and `1` when it did not — a non-converged run is a legitimate
outcome to branch on, not a crash.

Structure relaxation with FIRE 2.0 (Bitzek et al., *Phys. Rev. Lett.* **97**,
170201 (2006); Guénolé et al., *Comput. Mater. Sci.* **175**, 109584 (2020)),
optionally relaxing the cell together with the atoms in one joint
minimisation.

FIRE rather than CG or L-BFGS because the core-shell force evaluation
relaxes the shells to `cgtol` internally before returning core forces: from
the outer optimiser's point of view energy and force are mutually
inconsistent at O(`cgtol`), and line-search methods — which compare energies
across trial steps — reject steps and hysterese on that noise. FIRE uses
forces only.

| Key | Default | Unit | Notes |
|---|---|---|---|
| `input` | required | — | Extxyz file with starting cell + atoms |
| `output_dir` | required | — | Created with `mkpath` |
| `acep` **or** `splp` | optional | — | At most one; omit both for the pure shell model |
| `nlh` | optional | — | [`NLHParam`](@ref) baseline, as in `:NPTMD` |
| `ftol` | `1e-4` | Ha/Bohr | Convergence threshold on `max|F|`. `1e-4 Ha/Bohr ≈ 5e-3 eV/Å` |
| `maxsteps` | `2000` | — | Iteration cap. Reaching it is reported as **not converged** (with the partially relaxed structure still written), never raised as an error |
| `dt0`, `dt_max` | `1/UNITS[:fs]`, `10/UNITS[:fs]` | a.u. | FIRE step size. The pseudo-dynamics is mass-weighted (as LAMMPS `min_style fire`), so these mean what `timestep` means in MD and the MD intuition transfers |
| `dmax` | `0.1/UNITS[:Å]` | Bohr | Per-step displacement cap. Under `mpi = true` this is load-bearing: `migrate` routes atoms through a ±1-brick exchange and errors on a larger jump |
| `cgtol`, `cgmaxiter` | `2e-6`, `30` | — | Per-step shell relaxation. Tighter than the `:NPTMD` default by 10×; see the `ftol` note below |
| `cell_dof` | `:none` | — | Which cell degrees of freedom relax: `:none` (fixed), `:iso` (hydrostatic only), `:diag` (three axes independently, no shear), `:full` (all six). See below |
| `target_stress` | conditional | a.u. (Voigt-6) | `[σxx, σyy, σzz, σyz, σxz, σxy]`, **tension positive** as in `:NPTMD`. Required when `cell_dof != :none`. A hydrostatic pressure `P` enters as `[-P,-P,-P,0,0,0]` |
| `stol` | conditional | a.u. | Tolerance on `max|P(σ − σ_target)|`. Required when `cell_dof != :none` |
| `cell_factor` | `N` | — | Scale of the strain degree of freedom (ASE `UnitCellFilter` convention). A work-preserving change of coordinates `q = cell_factor·ε`, not a mass |
| `emax` | `1e-3` | — | Per-step strain cap. Early in a cell relaxation this, rather than `dt`, is usually the binding step-size control |
| `pme_rebuild_ratio` | `1.05` | — | Edge-length ratio at which the PME accelerator is rebuilt; see below |
| `verify_cgtol_factor`, `verify_mesh_factor` | `0.1`, `0.5` | — | Tightening factors of the post-convergence re-check |
| `ftol_cgtol_margin` | `10` | — | Entry-guard factor between `ftol` and `cgtol` |
| `output_interval` | `50` | steps | Snapshot frequency for `step_<n>.xyz` |
| `pme_meshsize` | required | a.u. | PME mesh spacing |
| `backend`, `mpi`, `pme_dd` | as `:NPTMD` | — | Same semantics |
| `surface_term` | optional | — | Supported (it is a physical potential). `polarization_umbrella` and `shell_dynamics = :xlbomd` are **refused** — a bias is not the equilibrium structure, and the shadow field is a dynamics device |

### `ftol` has to clear the solver noise, not just be small

The relaxation is driven by forces that carry the error of two numerical
solvers, and `ftol` below either of them is unreachable in principle:

- The inner shell CG leaves an O(`cgtol`) **non-variational** error in the
  core forces (the Hellmann–Feynman cancellation `∂E/∂s = 0` holds only at
  exact relaxation). An entry guard refuses `ftol < ftol_cgtol_margin·cgtol`
  rather than letting the run spin to `maxsteps`.
- The PME mesh error is **not** small-and-smooth on a symmetric structure:
  it breaks the symmetry that makes the exact forces vanish. Measured on an
  ideal NaCl supercell whose analytic-Ewald forces are zero to 1e-17, PME
  reports `max|F|` of 1.8e-5 on one mesh spacing and 1.7e-17 on the next —
  the error tracks the grid's commensurability with the crystal, not the
  nominal spacing.

So a converged verdict is **re-checked** with both tightened
(`verify_cgtol_factor`, `verify_mesh_factor`). If the criterion does not
survive, the run does not accept it *at that point* — and does not stop
either: it adopts the tightened solvers and keeps relaxing, once, then
judges again. A run that escalates can therefore still finish `converged`
(and return `0`); the log says that it had to tighten mid-flight. Seeing that message means the `cgtol` / `pme_meshsize` you asked for
were too loose for your `ftol` — set them tighter from the start.

Consequence for `log.txt`: the energy and stress columns take a small step
at the escalation, because they are evaluated on a finer mesh from there on.
That is a change of quadrature, not the relaxation going uphill. The final
line is written from inside the loop, i.e. *before* the re-check, so the
reported `max|F|` can differ from that column by the solver error.

### `cell_dof` is a projection, not a per-component mask

| `cell_dof` | strain subspace | `ndof` | equivalent |
|---|---|---|---|
| `:none` | `0` | 0 | fixed cell |
| `:iso` | `tr(ε)/3·I` | 1 | LAMMPS `fix box/relax ... couple xyz` |
| `:diag` | `Diagonal(ε)` | 3 | axes free, no shear |
| `:full` | `ε` | 6 | fully flexible |

`:iso` is **not** "the three diagonal components are free" — that is
`:diag`, where `a`, `b` and `c` relax independently and a cubic crystal
under an anisotropic target stops being cubic. A boolean per-component mask
cannot express `:iso` at all, which is why this is a projection.

The convergence residual is measured *after* projection: the components the
constraint freezes carry a residual stress that is the constraint reaction
force, and requiring it to vanish would be requiring the constraint to do
nothing. Under `:iso`, a tetragonal cell keeps a deviatoric residual by
construction and only `|tr(Δσ)/3|` is the run's business.

### A rigid rotation of the cell is not relaxed away

The strain is parameterised as a **symmetric** tensor, which is what keeps
the three rigid-rotation degrees of freedom of a general 3×3 out of the cell
force. A consequence: a cell handed in with a rotational component keeps it.
A sheared input comes back cubic in its *metric* `L·Lᵀ` while the lattice
matrix stays visibly non-diagonal — the same structure in a rotated frame.
(`lat[2,1] += γ·lat[2,2]` is half symmetric strain and half rigid rotation,
so exactly half of such a shear survives the relaxation.)

Related: the product of symmetric increments is not itself symmetric, so an
O(dε²) rotation accumulates over a run. Harmless for a rotationally
invariant Hamiltonian; not harmless for `surface_term` with an anisotropic
response, which is one reason that combination warns.

### A non-zero `efield` cannot be combined with a moving cell

[`ExternalField`](@ref) contributes force but **neither energy nor virial**.
For the atoms that is complete — the force is the gradient of the field term,
so `F = 0` is a genuine stationary point. For the cell it is not: the field's
strain derivative `−(M⊗E + E⊗M)/2` is simply absent from the stress, and a
cell relaxed to `σ = σ_target` would sit at the wrong cell with nothing in
the output to say so. The combination is refused — a *non-zero* field only,
so the `efield = [0,0,0]` that `default_params` supplies is never in the
way. With `cell_dof = :none` a field is fine.

### The PME rebuild triggers on edge lengths, not volume

`fftsize` is fixed at construction and depends on each cell side separately,
so an isochoric shear degrades the mesh spacing along an axis while leaving
the volume untouched. A volume-based trigger would sleep through it.

### High-symmetry starting structures

A high-symmetry structure is a stationary point by symmetry, so the run can
stop at step 0 having moved nothing — and a stationary point may be a
**saddle**, not a minimum. The run warns when the input already satisfies
the criteria. To check, re-relax from a slightly randomised copy and see
whether it returns to the same structure. There is no automatic saddle
detection; this is a documented operating procedure.

## `:NPTMD`

Source: `src/Apps/NPT.jl`. Inputs: extxyz starting configuration +
fitted potential. Outputs: trajectory and `log.txt` under
`params[:output_dir]`.

| Key | Default | Unit | Notes |
|---|---|---|---|
| `input` | required | — | Extxyz file with starting cell + atoms |
| `output_dir` | required | — | Created with `mkpath` |
| `acep` **or** `splp` | optional | — | At most one of [`ACEParam`](@ref) / [`SplineParam`](@ref); omit both to run the pure shell model (no MLIP term) |
| `nlh` | optional | — | [`NLHParam`](@ref) short-range repulsion baseline. Normally arrives via [`loadpotential`](@ref), which rebuilds it from the coefficients the fit subtracted, so a potential fitted with a baseline always runs with it. Requires `nlh.rc <= rc` |
| `nsteps` | `1000` | — | MD steps |
| `seed` | `0` | — | Initial-velocity RNG seed (id-keyed draw). Same input + seed → same trajectory; vary per run for independent replicas |
| `backend` | `:cuda` | — | `:cuda` (GPU) or `:cpu` (host execution) |
| `mpi` | `false` | — | `true` runs domain-decomposed under mpiexec (explicit opt-in; works with both backends — `backend = :cuda` requires a CUDA-aware MPI build, see the HPC page) |
| `pme_dd` | `true` | — | Under `mpi = true` with more than one rank, run the PME reciprocal solve z-slab distributed instead of replicated per rank (Issue #195). `false` forces the replicated solve — the escape hatch for MPI builds whose segmented collectives are not CUDA-aware. Ignored in serial. Must be the same on every rank. See the HPC page |
| `timestep` | required | a.u. | One MD step. `1 fs = 1.0/UNITS[:fs]` (≈ 41.341 a.u.) |
| `thermostat` | `:nh` | — | Particle thermostat: `:nh` (Nosé–Hoover), `:csvr` (stochastic velocity rescaling, Bussi et al. 2007 — exact canonical KE sampling; the rescaling noise is drawn from a `seed`-derived stream, so `seed` also selects the trajectory) or `:none` |
| `barostat` | `:mttk` | — | Barostat: `:mttk` (flexible cell, 2nd order), `:scr` (anisotropic stochastic cell rescaling, Bernetti–Bussi 2020 / Del Tatto 2022 — 1st order/overdamped, no cell ringing, correct NST ensemble; requires `compressibility`, `seed`, and a thermostat; anisotropic targets couple via a Parrinello–Rahman strain term referenced to the initial cell) or `:none` (fixed cell — NVT/NVE). `:mttk` always carries an internal Nosé–Hoover thermostat on the cell momenta, so `thermostat = :none, barostat = :mttk` is **not** a textbook NPH ensemble |
| `tau_t` | conditional | a.u. | Thermostat time constant; required unless `thermostat = :none`. Typical: `20·timestep` for equilibration. For `:csvr` this is the kinetic-energy relaxation time itself — sampling is canonical for any `tau_t`, but dynamical observables (e.g. dipole correlations for ε(ω)) are damped on this timescale, so use a long `tau_t` (ps order) for production dynamics |
| `tau_p` | conditional | a.u. | Barostat time constant; required unless `barostat = :none`. Typical: `500·timestep` |
| `compressibility` | conditional | a.u. (inverse pressure) | κ_T estimate; required by (and only legal with) `barostat = :scr`. From a bulk modulus `B` in GPa: `UNITS[:GPa] / B`. Only sets the effective relaxation time `τ_eff = tau_p·κ_true/κ_input` — the sampled ensemble is exact for any value while the per-step rescaling stays perturbative. `npt!` refuses at `max|μ−I| ≥ 0.25`, which catches the too-large unit mistake `B·UNITS[:GPa]`; the too-small mistake `1/B` (bare GPa⁻¹, ~3×10⁴× under) is **not detectable** — it leaves the ensemble correct but stretches `τ_eff` by the same factor, so the cell barely moves. If the cell columns of `log.txt` look frozen, check this unit first |
| `target_temperature` | required | K | Run target (divided by `UNITS[:Kelvin]` internally) |
| `initial_temperature`| required | K | Used for randn-velocity init when `> 0` |
| `final_temperature`  | optional | K | If given, ramp linearly from `target` to `final` over `nsteps` |
| `target_stress`      | conditional | a.u. (Voigt-6) | `[σxx, σyy, σzz, σyz, σxz, σxy]`; required unless `barostat = :none` (a fixed-cell run has no use for it) |
| `output_interval`    | required | steps | Snapshot frequency for `step_<n>.xyz` |
| `shell_dynamics` | `:scf` | — | How the shells follow the cores each step. `:scf` (default) relaxes them by conjugate gradient from a 3-point predictor. `:xlbomd` runs **shadow XL-BOMD** (Issue #149): an auxiliary shell field is propagated by its own time-reversible Verlet and the dynamics uses the exact gradient of a shadow energy built at that field — zero SCF iterations, a fixed ~3 force evaluations per step, and a conserved quantity that holds by construction rather than by tight convergence. Incompatible with `polarization_umbrella`; `surface_term` **is** supported (Issue #181 — but see its row: it raises the auxiliary spectrum, and `xl_init!` refuses a `xl_kappa` / `xl_kernel_spectrum` that provably cannot cover it). Barostats are supported (the reported virial is the shadow virial, Issue #151). **This is a different potential-energy surface: results can differ from `:scf`** — measured on BaTiO₃: pointwise force deviation from the relaxed surface ~6% (median max-norm), conserved-quantity drift as good as a `cgtol = 1e-8` relaxation, GPU throughput 2.0–2.6× the default. The **stress is the shadow stress**: exact for the surface the dynamics runs on, and offset from the relaxed one by the same first-order tracking lag as the forces — measured 1.4% of `max|W|` (0.86% with `:chebyshev`), 0.012 GPa in the pressure. See `docs/src/theory/xlbomd-shell.md` §4.1, §7, §9 |
| `xl_kappa` | `0.5` (`:spring`) / `1.8` (`:chebyshev`) | — | `:xlbomd` only. Auxiliary coupling `κ = ω²Δt²`. The stability bound is `κ·λmax(spring⁻¹H) < 4`; with the default `:spring` kernel that spectrum is material-dependent (measured `λmax ≈ 2.8` for BaTiO₃, so `κ < 1.28`), which is why the `κ = 1.8` of the XL literature diverges here. `:chebyshev` flattens the spectrum into `1 ± ρ`, `ρ < 1`, so `κ = 1.8` is stable by construction and becomes the default there (theory note §6, §6.1). Dimensionless: keep it fixed when changing `timestep` |
| `xl_kernel` | `:spring` | — | `:xlbomd` only. Operator applied to the residual shell force before it drives the auxiliary Verlet. `:spring` (default) is the closed-form spring inverse implicit in `s₁ − ŝ` — no extra cost, but the effective coupling varies by a factor ~6 across modes, and the worst-tracked one is the ferroelectric soft mode that carries ε(ω). `:chebyshev` corrects it towards the full shell Hessian with a fixed minimax polynomial, at `xl_kernel_order − 1` extra Hessian-vector products per step. Measured on BaTiO₃ (2×2×2, 300 steps, defaults): deviation from the relaxed surface 5.8e-2 → 4.1e-2 (force, max-norm), 5.4e-2 → 4.0e-2 (force, rms), **5.7e-2 → 2.1e-2 (polarization)** — the soft mode is where the gain is. Conserved-quantity drift is unchanged. **It is not free**: at 2 extra HVPs the CPU step drops to 10.3 steps/s against 17.3 for `:scf` (the XL step does not yet share its PME intermediates with the force pass, Issue #153), so on CPU this trades wall-clock for accuracy. On GPU — the production backend — it stays ahead of `:scf`: 1.6× at 135 atoms, 1.16× at 2560 (theory note §6.1, §7) |
| `xl_kernel_order` | `3` | — | `:chebyshev` only. Degree of the minimax polynomial + 1; costs `order − 1` Hessian-vector products per step. `2` is the cheapest useful value (one extra HVP) |
| `xl_kernel_spectrum` | `(0.4, 6.0)` | — | `:chebyshev` only. Interval `(a, b)` bounding the spectrum of `spring⁻¹·H_ss`. **One-sided**: too small an `a` merely weakens the tracking of the softest modes, but `b` *must* exceed the true `λmax` or the auxiliary field destabilizes — widen it when in doubt. The default is the BaTiO₃ measurement (`λmax ≈ 2.8`) with a 2× margin; measure it for another potential with `bench/check_xl_spectrum.jl`. A violation surfaces as the `xl_max_deviation` fallback, never as a silently wrong answer |
| `xl_warmup` | `3` | steps | `:xlbomd` only. Relaxed steps before the shadow dynamics engages, and again after every fallback |
| `xl_max_deviation` | `0.05` | Bohr | `:xlbomd` only. Fallback threshold on `max|s₁ − ŝ|`; exceeding it (or a NaN) triggers a full relaxation, reseeds the auxiliary field, and warns. The guard against the auxiliary field drifting off the physical surface |
| `pme_meshsize`       | required | a.u. | PME mesh spacing ([`PMEInfo`](@ref) is built unconditionally) |
| `polarization_umbrella` | optional | — | A [`PolarizationUmbrella`](@ref) (mode amplitude + homogeneity) or a [`PolarizationMapUmbrella`](@ref) (arbitrary per-group target map); enables the bias and `Bias.log`. Both take their gauge as a required constructor argument (Issue #122) |
| `polarization_gauge` | optional | — | A [`PolarizationGauge`](@ref) fixing the zero-point of the reported `dipole_phys`. Wins over the umbrella's gauge, which wins over the surface term's (normally all the same object; `validate_params` cross-checks them). **Without any of the three the zero-point is this run's starting frame**, so `dipole_phys` measures change-since-restart rather than polarisation — set it whenever the absolute value matters or runs are chained. Build it from the paraelectric prototype and share one [`save_gauge`](@ref) file across windows and restarts |
| `polmap_interval` | optional | steps | Write the **local polarization map** — the gauged per-group polarization the biases restrain, `(ng, 3)` in flat groupid order (`ix·Ny·Nz + iy·Nz + iz + 1`, iz fastest) — every this many steps as `polmap_<n>.txt` ([`write_polmap`](@ref) format: plain text, one `ix iy iz Px Py Pz` row per group, P in e/Å², `np.loadtxt`-readable). **Requires `polarization_gauge`**: unlike `dipole_phys` there is no starting-frame fallback, because a map measured from wherever a run happened to start is not a polarization and chained runs would each carry a different zero. The map is per unit cell, **not** smoothed (the bias's `sigma` is a bias detail). Cost: one `group_p` plus a `3·ng` Allreduce on the output steps only, no per-atom gather; the file is ~55 bytes per group (16³: 0.2 MB, 64³: 15 MB), so size the interval for large grids. The map is only comparable across frames while the atoms stay within half a group-lattice period of the gauge frame: `:pos` is never wrapped, so a run is continuous, but the gauge is validated at entry only, and a system that has drifted rigidly by more than that cannot be re-matched to the gauge (the same limit as `dipole_phys`). Existing snapshots can be post-processed into the same files with `:PolMap` (below) |
| `surface_term` | optional | — | A [`SurfaceTerm`](@ref): depolarization field of a finite anisotropic external permittivity — the generalised Ewald surface term `U = (2π/V)·δM·T_Ω·δM` (isotropic sphere: `2π/((2ε_out+1)V)·δM²`). Omitting it is the conductor (tin-foil) boundary the Coulomb solvers already implement. Takes its gauge as a required constructor argument, normally the same [`PolarizationGauge`](@ref) file as the umbrella. **`eps_out` and `shape` are lab-frame constants**: the external medium does not co-rotate or co-deform with the cell (∂T/∂ε = 0), so under a barostat the embedding stays fixed in space rather than glued to the material — the stress carries no shape-relaxation term. Combines with `shell_dynamics = :xlbomd` since Issue #181 gave it an analytic HVP; the depolarization block raises `λmax(spring⁻¹·H_ss)` by `Δλ = (4π/V)(Σ q_s²/k2)·λmax(T)`, so `xl_init!` refuses the two declarations that are then wrong with certainty: an `xl_kernel_spectrum` upper bound below `Δλ` (`:chebyshev`), and an `xl_kappa` with `κ·Δλ ≥ 4` (`:spring` only — the Chebyshev kernel flattens the coupling, so switching to it with a wide enough interval is the escape from that one). Passing them is not a certificate — measure with `bench/check_xl_spectrum.jl`. Its virial and virial derivative are whole-system, so — like the reciprocal-space part — they do not appear in the `peratomstress` columns. `linear_response!` refuses it: the dielectric tensor it reports is the conductor-boundary one. See `docs/src/theory/surface-term.md` |

`nsteps`, `target_temperature`, `final_temperature` can be reset and
`app[:NPTMD]` re-invoked on the same configuration to chain
equilibration → production phases (the umbrella-sampling input scripts
do this).

`log.txt` columns are `temp | lattice(9) | dipole(3) | dipole_phys(3) |
volume | conserved | ncg` (units: K, Å, Å³, Hartree; `ncg` is a
dimensionless count — the shell-relaxation force evaluations inside the
CG loop, as returned by `calculate!`. The initial shell evaluation and
the core evaluation are not counted, so the per-step cost is `ncg + 1`
shell + 1 core force evaluations — the SCF cost to watch when tuning
`cgtol`. A shell evaluation is the cheaper of the two: since Issue #157
it asks every term for shell forces only, which on the `PMEInfo` path
skips the reciprocal virial, the core gather and the energy/virial
reductions — measured 2.0× cheaper per reciprocal call, and ~1.13× on the
whole step, on the GPU in single precision, on top of the 1.7× Issue #153
took off the full call). `conserved` is the
diagnostic conserved quantity H′ of the active thermostat/barostat
combination — monitor its drift to judge integration quality, not its
absolute value (it is sampled mid-step at a consistent phase, and its
relative drift grows roughly linearly with step count — ~5e-9/step on
the 64-atom test fixture at 1 fs — from the O(dt³)-per-step bookkeeping
of the thermostat work accumulators; a jump or change of slope is the
signal to look for). It is an actual invariant only while the target
temperature and stress are fixed. With `barostat = :scr` the column is
the enthalpy-like `K + U + P₀V` (+ thermostat terms) instead — the
Euler-integrated SCR has no discrete invariant — and the health signal
is stationarity (no trend), not constancy; with an anisotropic target
stress the untracked deviatoric strain energy adds a trend of its own
during shape relaxation, so judge stationarity only after the cell shape
has equilibrated.

Two `:scr` fine points: anisotropic targets couple through a
Parrinello–Rahman strain term referenced to the cell at `npt!` entry, so
a restarted run re-freezes that reference (hydrostatic targets are
unaffected); and `:mttk` and `:scr` sample the **same** volume marginal
(the ideal-gas `Gamma(N, kT/P₀)` of the frozen-COM convention — verified
against each other in `test/apps/test_scr.jl`), so switching barostats
does not shift equilibrium volumes. The derivation, including why the
measure correction differs from the SCR papers' 9-DOF value, is in
[Theory notes](theory/scr-barostat.md).

## `:PolMap`

Source: `src/Apps/PolMap.jl`. Post-process: measure the local polarization
map of extxyz frames against a declared gauge and write it in the same
`polmap_<n>.txt` format `:NPTMD` writes at `polmap_interval`. Host only —
one `group_p` per frame, far cheaper than reading the frame.

| Key | Default | Unit | Notes |
|---|---|---|---|
| `input` | required | — | An extxyz path or a `Vector` of paths; every frame of every file is processed |
| `output_dir` | required | — | Created with `mkpath` |
| `polarization_gauge` | required | — | The [`PolarizationGauge`](@ref) the run used ([`load_gauge`](@ref) of the shared file). Every frame is validated against it (`groupid` / `elemid` / `id` must match the gauge frame) — a re-grouped or differently ordered file is refused, not silently mapped |
| `elements`, `qc`, `qs` | required | — | Species ids and core / shell charges. The `pot...` splat of [`loadpotential`](@ref) supplies them; its other fields (`k2`, `rc`, the MLIP, …) are accepted and ignored |
| `overwrite` | `false` | — | An existing `polmap_<n>.txt` in `output_dir` is refused unless `true` — pointing `output_dir` at a run's own directory would otherwise silently replace the in-loop maps with ones measured against whatever gauge this call was given |

Output naming mirrors the driver so a post-processed trajectory lines up
with an in-loop one file for file: `step_<n>.xyz` → `polmap_<n>.txt` with
`step=<n>` in the header; any other single-frame file → `polmap_<stem>.txt`;
a multi-frame file → `polmap_<stem>_<k>.txt`. Two inputs that would collide
on an output name are refused.

Reading the files: in Python `np.loadtxt("polmap_500.txt")` gives an
`(ng, 6)` array of `ix iy iz Px Py Pz` with P in e/Å² (1 e/Å² =
1602.18 μC/cm²); `.reshape(Nx, Ny, Nz, 6)` recovers the grid because iz
cycles fastest. In Julia [`read_polmap`](@ref) returns the map in atomic
units, so it can go straight into [`save_target_map`](@ref) — the way to turn
a time-averaged single-domain run into a target map. The xyz snapshots round
positions to 8 significant digits, so a post-processed map agrees with the
in-loop one to ~1e-6 relative, not to machine precision.

## Parameter source-of-truth

When a default disagrees with the example scripts, the example wins —
the defaults are conservative starting points. In particular, the
shell-model placeholders inside `default_params` for `:FitSpline`,
`:FitACE`, and `:NPTMD` are zero, expecting to be splatted over by
values loaded from `ShellParam.jld2` / `SplineP.jld2` / `ACEP.jld2`.
The pattern in `example/NPT-MD/input.jl` is

```julia
pot = FMD.loadpotential("./ACEP.jld2")   # schema-validated, flat NamedTuple
params = (;
    pot...,                  # qref, qc, qs, k2, k4, eta_c, eta_s, acep, elements, rc
    # … runtime knobs …
)
params = (; FMD.default_params(Val(:NPTMD), params)..., params...)
FMD.app[:NPTMD](params)
```

[`loadpotential`](@ref) checks the file against the three known schemas
(`ShellParam.jld2` / `ACEP.jld2` / `SplineP.jld2`), reports every
violation at once, and never returns `provenance` (so the splat cannot
inject unknown keys into `params`).

The double splat `(; default..., params...)` lets the JLD2-loaded
shell values silently override the zero placeholders.
