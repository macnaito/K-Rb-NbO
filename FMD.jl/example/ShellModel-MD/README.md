# BaTiO₃ pure shell-model MD + frequency-dependent permittivity

Runs classical core-shell MD of BaTiO₃ with literature parameters — no
MLIP, `FMD.app[:NPTMD]` without `acep`/`splp` — and extracts the
frequency-dependent dielectric permittivity ε(ω) from the equilibrium
dipole fluctuations. Companion to `example/NPT-MD` (shell model + MLIP).

## Files

| file | role |
|---|---|
| `makecell.py` | pre-processing: cubic BaTiO₃ 16³ supercell → `input.xyz` (stdlib only) |
| `potential.jl` | the shell-model parameter set (single source, shared by both inputs) |
| `input_equil.jl` | stage 1: NPT 400 K, 20 ps → `out_equil/` |
| `input_prod.jl` | stage 2: NVT 400 K at the NPT-averaged cell ⟨h⟩, 100 ps → `out_prod/` |
| `dielectric.py` | post-processing: `out_prod/log.txt` → ε′(ω), ε″(ω) (needs numpy; matplotlib optional) |

## Quickstart

```bash
cd example/ShellModel-MD
python3 makecell.py                      # input.xyz  (20,480 atoms)
julia --project=../.. input_equil.jl     # out_equil/
julia --project=../.. input_prod.jl      # input_prod.xyz, out_prod/
python3 dielectric.py                    # dielectric.csv, dielectric.png
```

Runs on the GPU by default; uncomment `backend = :cpu` in both input
files for CPU-only machines (and expect to shrink `-n` / `nsteps`
accordingly — see "Knobs").

Wall-time orientation (measured, 16³ at Float32 on an RTX 5070 Ti):
~32 steps/s — ≈10 min for the 20 ps equilibration, ≈1 h for the 100 ps
production. On consumer GPUs run Float32 (`precision = "Float32"` under
`[FMD]` in `LocalPreferences.toml`, see the `FMD` module docstring) —
their FP64 rate is 1/64 and Float64 there is ~14× slower; datacenter
GPUs run Float64 at full rate. Note the test suite asserts Float64 —
Float32 is untested there, so cross-check e.g. ε(0) against a Float64
run before quoting production numbers. A 4³ cell runs at ~6 steps/s on
a single CPU core, so a shortened pipeline (`-n 4`, a few thousand
steps) is a tens-of-minutes CPU job.

## What happens, and why two MD stages

1. **NPT 400 K (equilibration).** 400 K is comfortably above this
   potential's Curie point: the cell stays paraelectric cubic, so there
   are no ferroelectric domains to worry about. Besides thermalizing,
   this run measures the equilibrium lattice under zero stress.
2. **NVT 400 K (production).** `input_prod.jl` averages the lattice over
   the second half of the NPT log (the *instantaneous* final cell is a
   random draw from the cell distribution — using it would bias the
   strain), affinely remaps the final snapshot onto that ⟨h⟩, and runs
   with `barostat = :none`. The fluctuation formula below is derived at
   fixed cell, and this defines the **clamped** permittivity at the
   NPT(400 K, σ=0) average cell; a barostat would also contaminate the
   dipole dynamics with cell-fluctuation ringing.

Velocities are redrawn from `params[:seed]` at every `app[:NPTMD]` start
(whenever `initial_temperature > 0`), which is why the two stages use
different seeds and `dielectric.py` discards the first 10 % of the
production run as re-thermalization.

## The potential

BaTiO₃ core-shell + Buckingham set fitted to r²SCAN-D4 reference data
with the Ewald solver — H. Azuma *et al.*, Acta Mater. **296**, 121216
(2025), [doi:10.1016/j.actamat.2025.121216](https://doi.org/10.1016/j.actamat.2025.121216)
— defined in `potential.jl`, the single source shared by both input
scripts.

- **Units**: parameters are written in their published eV/Å values and
  divided by `UNITS[...]` factors at input time; everything below the
  params boundary is atomic units. (`docs/src/parameters.md` has the
  full parameter reference.)
- **Function form**: published tables using `(A + B·r)exp(−r/ρ) − C/r⁶`
  map onto FMD.jl's `a(1 + b·r)exp(−r/ρ) − c/r⁶` with `a = A`,
  `b = B/A` (this set has `B ≡ 0`). `ρ = 0` entries marking
  non-interacting pairs become small positive placeholders (`ρ` is a
  divisor in the kernel).
- The masses come from FMD's built-in `ATOMIC_MASSES`; `eta_c`/`eta_s`
  (core/shell Gaussian widths) keep their near-point-charge defaults.

## ε(ω): definition and conventions

`dielectric.py` evaluates, per axis and then isotropically averaged,

$$\varepsilon(\omega) = 1 + \frac{\langle|\delta\mathbf M|^2\rangle
  - i\omega \int_0^{t_{\max}} \langle\delta\mathbf M(0)\!\cdot\!\delta\mathbf M(t)\rangle
  e^{-i\omega t}\,dt}{3\,\varepsilon_0 V k_B T},
  \qquad \delta\mathbf M = \mathbf M - \langle\mathbf M\rangle$$

with the ε(ω) = ε′ − iε″ sign convention (ε″ ≥ 0 is loss). Conventions
baked into the pipeline:

- **M(t)** is the gauge-fixed physical dipole `dipole_phys` from
  `log.txt` (written every step, e·Å).
- **Tin-foil boundary**: the formula matches FMD.jl's Ewald/PME
  convention (no surface/depolarization term).
- **Clamped**: fixed cell at the NPT ⟨h⟩, see above. The free
  (constant-stress) permittivity would additionally contain the
  piezoelectric/strain response.
- **ε_∞ is ignored**: shells follow the cores adiabatically, so the
  equilibrium fluctuations of M contain only the electronic response
  slaved to ionic motion; the frozen-core electronic ε_∞ − 1 would have
  to be added separately. The ionic GHz–THz features targeted here are
  unaffected, but the high-frequency plateau is off by it: a harmonic
  Clausius–Mossotti estimate from the shell polarizabilities
  (α_i = q_{s,i}²/k2_i) gives ε_∞ ≈ 3.4 for this set, so above the
  phonon response the computed ε′ falls to 1 where the physical value
  is ≈ 3.4. If you need ε_∞, compute it by linear response: apply a
  small `efield`, relax the shells at frozen cores, and take ΔM/E.

With the default 10 ps ACF window (`--tmax-ps`) a 100 ps production run
gives well-converged spectra across ~0.1–50 THz — the soft-mode /
relaxational response of paraelectric BaTiO₃ sits in the 0.1–20 THz
range. Resolving slower dynamics needs both a longer ACF window and a
longer run.

## Knobs

- `makecell.py -n 4` — a 4³ cell (320 atoms) runs the full pipeline in
  minutes on CPU; expect much noisier ε(ω) and stronger finite-size
  effects.
- `nsteps` in `input_prod.jl` — statistics and the lowest resolvable
  frequency both scale with the production length; extend for smoother
  low-frequency ε(ω).
- `dielectric.py --tmax-ps / --discard / --dt-fs` — ACF truncation,
  discarded initial fraction, and the timestep (must match
  `input_prod.jl` if you change it there).
- `thermostat = :none` in `input_prod.jl` — NVE production, the purist's
  choice for correlation functions; watch the `conserved` column of
  `log.txt` for drift.
- `shell_dynamics = :xlbomd` in either input — shadow XL-BOMD instead of
  the per-step shell relaxation: no SCF iterations and a fixed ~3 force
  evaluations per step (the `ncg` column drops to 0). It works with the
  barostat of `input_equil.jl` too — the reported stress is the strain
  derivative of the shadow surface, not of the relaxed one. The dynamics
  runs on that shadow surface, so ε(ω) (and the equilibrium cell) from an
  XL run should be cross-checked against a `:scf` segment before it is
  quoted — see `docs/src/theory/xlbomd-shell.md`.
- `xl_kernel = :chebyshev` alongside it — spends one or two extra
  Hessian-vector products per step to equalize how well the auxiliary
  field tracks the stiff and soft shell modes. On this potential it
  roughly halves the deviation of the polarization from the relaxed
  surface, which is the quantity ε(ω) is built from; the defaults
  (`xl_kernel_order = 3`, `xl_kernel_spectrum = (0.4, 6.0)`) are measured
  for exactly this parameter set by `bench/check_xl_spectrum.jl`.
