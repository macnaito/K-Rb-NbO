# Dataset generation

The Python pipeline in `dataset/` produces the two extxyz files consumed
by the fitting stage:

- `synth.xyz` — DFT energy / forces / stress + a fine-tuned MACE
  re-evaluation, used by `:FitSpline` and `:FitACE`.
- `dfpt.xyz` — VASP DFPT-derived Born effective charges + dielectric
  tensor, used by `:FitShellModel`.

The two pipelines share the first two stages (NPT-MD with a MACE
foundation model + relaxation) and diverge at the VASP step. This page
gives the recipe; the canonical files are
`dataset/dft/kno/{md_umlip,relax_umlip,gen_vasp,finetune_mace}.py` and
`dataset/dfpt/kno/{md_umlip,relax_umlip,gen_vasp}.py`. (The `kno/`
sub-directory holds the KNbO₃ example; copy and rename for other
materials.)

## Prerequisites

```bash
pip install ase mace-torch natsort
export VASP_PP_PATH=/path/to/vasp/pseudopotentials   # any standard layout
```

A working VASP build is required for the DFT and DFPT single-points.
The MACE-MP foundation model `medium-omat-0` is downloaded automatically
on first use.

## DFT pipeline → `synth.xyz`

### 1. NPT-MD sampling — `md_umlip.py`

Samples ~500 frames per seed from a high-temperature NPT-MD trajectory
under MACE-MP.

```bash
python md_umlip.py --seed 0
python md_umlip.py --seed 1
python md_umlip.py --seed 2
python md_umlip.py --seed 3
```

| Setting | Value | Meaning |
|---|---|---|
| `MODEL` | `medium-omat-0` | MACE-MP foundation model |
| `DEVICE` | `cuda` | GPU required |
| `T_K` | `1500.0` | Sampling temperature, well above `T_c` |
| `DT_FS` | `1.0` | Timestep |
| `NSTEPS` | `10000` | 10 ps total |
| `LOGINTERVAL` | `20` | → 500 frames per seed |
| `TTIME_FS` | `25.0` | Thermostat time constant |
| `PTIME_FS` | `75.0` | Barostat time constant |
| `B_EVA3` | `0.6` | Bulk modulus estimate (eV/Å³) |

Output: `mdtraj_seed{0..N}.xyz`.

### 2. Cell-constrained relaxation — `relax_umlip.py`

Random-samples frames from the MD trajectories and runs FIRE relaxation
under MACE.

```bash
python relax_umlip.py --nsamples 200 --seed 0
```

| Setting | Value | Meaning |
|---|---|---|
| `MAX_STEPS` | `20` | FIRE iterations cap |
| `FMAX` | `0.05` eV/Å | Convergence criterion |

Output: `relax.xyz` — full relaxation trajectories (every step), so the
DFPT pipeline can re-filter by `mace_forces` / `mace_stress`.

### 3. VASP single-point — `gen_vasp.py`

Two-phase usage:

```bash
# Phase 1 — write VASP input directories
python gen_vasp.py --nsamples 100 --seed 0
# (submit each workdir/<i>/ to your queue)

# Phase 2 — collect VASP outputs into dft.xyz
python gen_vasp.py --collect --output dft.xyz
```

| VASP flag | Value | Reason |
|---|---|---|
| `xc` | `PBEsol` | functional |
| `prec` | `Accurate` | |
| `kspacing` | `0.3` | k-point density |
| `encut` | `600` | plane-wave cutoff (eV) |
| `ediff` | `1e-6` | SCF tolerance |
| `nsw, ibrion, isif` | `0, -1, 2` | static: MACE already relaxed; ISIF=2 → stress without cell relax |
| `kpar, ncore` | `2, 4` | parallelisation |

### 4. MACE fine-tune + assembly — `finetune_mace.py`

```bash
# Phase 1 — fine-tune MACE on dft.xyz
python finetune_mace.py --train dft.xyz --device cuda

# Phase 2 — re-evaluate the MD/relax frames with the fine-tuned model
#           and merge with DFT into synth.xyz
python finetune_mace.py --collect --output synth.xyz
```

`synth.xyz` carries a `method=` tag per frame (`PBEsol` or `MACE`); the
default `method_weight = (PBEsol = 1.0, MACE = 0.5)` in the fitting
inputs gives DFT twice the weight of the augmented MACE frames.

### Output schema — `synth.xyz`

| Field | Where | Notes |
|---|---|---|
| `potential_energy` | per-frame info | eV |
| `virial`           | per-frame info | Voigt-6, eV |
| `method`           | per-frame info | `PBEsol` or `MACE` |
| `forces`           | per-atom array | eV/Å |
| `species`, `pos`, … | per-atom | usual extxyz columns |

## DFPT pipeline → `dfpt.xyz`

Stages 1–2 are identical to the DFT pipeline (re-use the same
`relax.xyz` if convenient — `dataset/dfpt/kno/` ships its own copies of
`md_umlip.py` and `relax_umlip.py` for self-containedness).

### 3. VASP DFPT — `gen_vasp.py` (DFPT version)

```bash
python gen_vasp.py --nsamples 30 --seed 0
# (submit, then …)
python gen_vasp.py --collect
```

Differences vs. the DFT version:

- **Pre-filtering** — frames whose `mace_forces` / `mace_stress` exceed
  `MAX_FORCE_EVA = 5.0 eV/Å` / `MAX_STRESS = 0.05 eV/Å³` are dropped.
  DFPT does not converge well on poorly relaxed configurations.
- **VASP flags**:

  | flag | DFT | DFPT | reason |
  |---|---|---|---|
  | `kspacing` | 0.3 | **0.25** | denser k-grid for response functions |
  | `ediff` | 1e-6 | **1e-8** | tight SCF for ε / BEC |
  | `lepsilon` | — | **True** | enable dielectric tensor calc |
  | `lpead` | — | **True** | self-consistent BEC |
  | `kpar, ncore` | 2, 4 | 1, 1 | DFPT does not parallelise the same way |

### Output schema — `dfpt.xyz`

| Field | Where | Notes |
|---|---|---|
| `dielectric_tensor` | per-frame info | Voigt-6 (electronic part) |
| `becex`, `becey`, `becez` | per-atom arrays | each `(N, 3)` — components of the BEC tensor for the x/y/z applied-field response |

## Re-using dataset scripts for other materials

Copy `dataset/dft/kno/` (or `dataset/dfpt/kno/`) to a new subdirectory
and adjust:

- `init.xyz` — the unit cell.
- `MODEL` — MACE foundation model (or a smaller / cheaper one).
- `T_K` — sample well above the expected `T_c`.
- `MAX_FORCE_EVA` / `MAX_STRESS` — system-specific tolerances for the
  DFPT filter.

The Julia fitting stage does not assume KNbO₃; it discovers the species
via `params[:elements]` and works for any perovskite or composition.
