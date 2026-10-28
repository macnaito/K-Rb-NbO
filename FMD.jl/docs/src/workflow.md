# Workflow

```@meta
CurrentModule = FMD
```

FMD.jl is built around a four-stage pipeline (the relaxation stage is
optional). Each stage produces
artefacts that the next stage consumes, so a complete run looks like

```
init.xyz ──► [Stage 1: dataset] ──► synth.xyz, dfpt.xyz
                                     │
                                     ▼
              [Stage 2: fit]    ──► ShellParam.jld2
                                     │
                                     ▼
                                   SplineP.jld2  (or ACEP.jld2)
                                     │
                                     ▼
              [Stage 3: relax]  ──► out/relaxed.xyz  (optional)
                                     │
                                     ▼
              [Stage 4: MD]     ──► out/log.txt, out/step_*.xyz
```

This page is the map. For per-app params see
[Parameters](parameters.md), for HPC setup see [HPC setup](hpc.md), and
for the dataset-generation Python pipeline see [Dataset](dataset.md).

## Stage 1 — Dataset

Generate two extxyz datasets: `synth.xyz` (DFT + fine-tuned MACE labels)
and `dfpt.xyz` (Born effective charges + dielectric tensor labels).

| | Subdirectory | Output | Used by |
|---|---|---|---|
| DFT  | `dataset/dft/kno/`  | `synth.xyz` | `FitSpline` / `FitACE` |
| DFPT | `dataset/dfpt/kno/` | `dfpt.xyz`  | `FitShellModel` |

Both subdirectories run a Python pipeline (MACE-MP NPT-MD → relaxation →
VASP single-point or DFPT → MACE fine-tune & re-evaluation). The full
recipe and the reasoning behind every filter / VASP flag is in
[Dataset](dataset.md).

## Stage 2 — Fitting

Two sub-stages, both in `example/training/`.

!!! note
    `example/training/` は現在，手元で完結する BaTiO₃ の教材になっている：
    ラベルは VASP ではなく MACE 単点評価（`train.xyz` を同梱），shell
    パラメータは文献値を使い 2a は optional．本節のファイル名
    （`synth.xyz` / `ShellParam.jld2`）は `dataset/` の本番パイプラインの
    命名で，example の実ファイルとコマンドは
    `example/training/README.md` に従うこと．

### 2a. Shell model — `FMD.app[:FitShellModel]`

Fits per-element charges `qc, qs`, spring constants `k2, k4`, and Gaussian
widths `eta_c, eta_s` against `dfpt.xyz` using CMA-ES. The loss is a
weighted sum of BEC and dielectric-tensor squared error (see
[`FMD.Apps.opt!`](@ref)).

```bash
mpiexec -n 72 julia --project=../.. input_fitsm.jl   # HPC: popsize = 72
```

The number of MPI ranks is `popsize × data_workers`（current example
input: `popsize = 16`）. Output:
`ShellParam.jld2`, `parity_bec.csv`, `parity_eps.csv`.

### 2b. Short-range MLIP — `FMD.app[:FitSpline]` / `FMD.app[:FitACE]`

Δ-learns the residual after subtracting the core-shell evaluator from
DFT forces / energies / virials. Pick **one** representation:

- **Spline**: `input_fitspline.jl` → 2-/3-body cubic B-splines.
- **ACE**: `input_fitace.jl` → ACE-style radial basis.

```bash
mpiexec -n 8 julia --project=../.. input_fitspline.jl
# or
mpiexec -n 8 julia --project=../.. input_fitace.jl
```

Each script loads the shell parameters（`ShellParam.jld2`，または現
example のように文献値）and the training dataset（本番: `synth.xyz` /
example: `train.xyz`）, then writes `SplineP.jld2` (or `ACEP.jld2`) plus
parity CSVs. The output JLD2 contains the MLIP coefficients and the
shell parameters, so the production-MD script usually only needs to
load one file — **except the baseline Buckingham terms**: if the fit
used them（現 example がそう），they are *not* stored in the JLD2 and
the MD input must re-supply the same values
（`example/training/input_md.jl` 参照，恒久対応は Issue #201）.

## Stage 3 — Structure relaxation (optional but recommended)

Before running production MD, relax the starting configuration into the
*fitted potential's own* equilibrium with `FMD.app[:Relax]`:

```bash
julia --project example/Relax/input.jl
```

Otherwise the MD begins at whatever geometry the training data happened to
contain, and the first picoseconds are spent shedding a stress the force
field never agreed with. Relaxing the cell as well (`cell_dof = :iso`) also
gives the potential's equilibrium lattice constant, which is the natural
thing to compare against experiment or DFT before trusting any dynamics.

The driver is [`FMD.Apps.relax!`](@ref); see the `:Relax` section of the
[parameters reference](parameters.md) for the full key list and, in
particular, for why `ftol` has to clear both the shell-CG tolerance and the
PME mesh error. Output under `params[:output_dir]`:

- `relaxed.xyz` — the final structure, written whether or not it converged.
- `log.txt` — per step: `step`, `max|F|`, energy, stress Voigt-6, volume,
  `dt`, `α`, `ncg`, lattice (9 cols). Units: Ha/Bohr, Ha, GPa, Å³, Å.

`FMD.app[:Relax]` returns `0` on convergence and `1` otherwise, so a batch
driver can branch on it rather than parsing the log.

## Stage 4 — Production MD

Run NPT MD with `FMD.app[:NPTMD]` using a fitted potential and a
starting configuration (`relaxed.xyz` from Stage 3, or an extxyz file):

```bash
julia --project example/NPT-MD/input.jl
```

The driver is described in [`FMD.Apps.npt!`](@ref). Output goes to
`params[:output_dir]`:

- `log.txt` — per-step kinetic temperature (3N-3 DOF, Issue #93), full
  lattice (9 cols), `dipole` (3 cols), gauge-fixed `dipole_phys`
  (3 cols), `volume`, `conserved` (diagnostic H′, Issue #136),
  `shell_rms` (rms shell-force residual left by the step's CG, Ha/Bohr;
  NaN on an XL-BOMD shadow step), `ncg` (shell-relaxation force
  evaluations, a dimensionless count, Issue #144). Units: K, Å, Å³,
  Hartree, Ha/Bohr.
- `step_<n>.xyz` — extxyz snapshots every `params[:output_interval]`
  steps.
- `Bias.log` (optional) — populated when
  `params[:polarization_umbrella]` is set, one `CV,Bias` line per step.
- `polmap_<n>.txt` (optional) — the local polarization map every
  `params[:polmap_interval]` steps (needs `params[:polarization_gauge]`);
  see below.

### Umbrella sampling

`example/UmbrellaSampling/` (uniform) and
`example/UmbrellaSampling_Inhomo/` (cosine-modulated) sweep `cv` values
through `parameters.json` windows. Each window is a regular `:NPTMD`
run with an extra `params[:polarization_umbrella]` (a
[`PolarizationUmbrella`](@ref FMD.Calculators.PolarizationUmbrella)). Post-process with the bundled
`wham.py` to recover the free-energy surface; `FitLandau.py` /
`tdgl.py` fit Landau coefficients on top.

`example/UmbrellaSampling_Map/` uses the other bias in the family, a
[`PolarizationMapUmbrella`](@ref FMD.Calculators.PolarizationMapUmbrella),
which restrains the *whole* per-group polarization field to a target map
rather than a single amplitude. That is the tool for preparing a named
configuration — a domain wall, a seeded nucleus — and then releasing it: the
example forms the pattern cold, equilibrates while held, decays the restraint,
and finishes with unbiased MD. Zero the per-group weight in a band around the
wall so the domains are pinned while the wall picks its own width.

Both biases take a [`PolarizationGauge`](@ref FMD.Calculators.PolarizationGauge)
as a required argument: polarization is defined only relative to a declared
zero-point structure, and every window (and every target map) must share one
so their numbers are comparable. Persist it with
[`save_gauge`](@ref FMD.Calculators.save_gauge) and load the same file
everywhere.

### Local polarization maps

The per-unit-cell polarization — the field the biases restrain — can be
written during the run: set `params[:polmap_interval]` together with a
`polarization_gauge`, and `:NPTMD` writes `polmap_<n>.txt` (plain text,
`ix iy iz Px Py Pz` per group, e/Å²; [`write_polmap`](@ref FMD.Calculators.write_polmap))
next to the snapshots. A trajectory that was run without it, or one whose
gauge is to be changed afterwards, gets the same files from its
`step_<n>.xyz` through `FMD.app[:PolMap]`. Both are the `group_p_gauged`
of the frame against the declared gauge; the map is never written against
the starting frame, so maps from chained runs share one zero. See the
`:PolMap` section of [Parameters](parameters.md).

## End-to-end sequence

```text
1. dataset/dft/kno  : md_umlip → relax_umlip → gen_vasp + VASP →
                      finetune_mace        ──► synth.xyz
2. dataset/dfpt/kno : md_umlip → relax_umlip → gen_vasp + VASP DFPT
                                            ──► dfpt.xyz
3. example/training : input_fitsm.jl       ──► ShellParam.jld2 (optional)
                      input_fitspline.jl   ──► SplineP.jld2  (Δ-learn)
                      input_fitace.jl      ──► ACEP.jld2     (Δ-learn)
   （example は MACE ラベルの BaTiO₃ 版で 1–2 を代替する — README 参照）
4. example/Relax    : input.jl             ──► out/relaxed.xyz  (optional)
5. example/NPT-MD   : input.jl             ──► out/log.txt, out/step_*.xyz
```

Stages 1 and 2 do not need to be repeated for every production run —
once `SplineP.jld2` / `ACEP.jld2` exists, only steps 4–5 change between
production runs (different cells, temperatures, biases, …).

## Where to look next

- [Parameters](parameters.md) — every `params` key per app, with units
  and defaults.
- [HPC setup](hpc.md) — MPI / CUDA / Preferences configuration and how
  to translate the bundled Fugaku scripts to a generic SLURM site.
- [Dataset](dataset.md) — Python pipeline: VASP flags, MACE config,
  filtering rules.
- [API reference](api/apps.md) — auto-generated from docstrings.
