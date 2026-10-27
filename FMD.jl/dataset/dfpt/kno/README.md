# KNbO3 DFPT dataset generation

Produces `dfpt.xyz` for FMD shell-model fitting (FitShellModel).

## Prerequisites

```
pip install ase mace-torch natsort
```

VASP pseudopotentials: set `$VASP_PP_PATH`.

## Workflow

### 1. NPT-MD with MACE foundation model

```bash
python md_umlip.py --seed 0
python md_umlip.py --seed 1
python md_umlip.py --seed 2
python md_umlip.py --seed 3
```

Output: `mdtraj_seed{0..3}.xyz`

---

### 2. Structure relaxation with MACE

```bash
python relax_umlip.py --nsamples 200
```

Output: `relax.xyz`  (stores `mace_forces` / `mace_stress` for downstream filtering)

---

### 3. VASP DFPT (LEPSILON + LPEAD)

Structures are filtered by max force/stress before sampling to avoid DFPT convergence failure.

```bash
# write VASP inputs
python gen_vasp.py --nsamples 30

# submit jobs, then collect BEC + dielectric tensor
python gen_vasp.py --collect
```

Output: `workdir/{0000..0029}/`, `dfpt.xyz`

---

## Output format

`dfpt.xyz` is FMD extxyz with per-frame:

| key | shape | description |
|---|---|---|
| `becex` | (N, 3) | `becex[i, α] = ∂P_x/∂u_{i,α}` |
| `becey` | (N, 3) | `becey[i, α] = ∂P_y/∂u_{i,α}` |
| `becez` | (N, 3) | `becez[i, α] = ∂P_z/∂u_{i,α}` |
| `dielectric_tensor` | Voigt-6 | Electronic dielectric tensor |

The `bec*` convention matches FMD `linear_response!`
(`becex[i, α] = ∂F_α/∂E_x = ∂P_x/∂u_{i,α}` by the Maxwell relation), which
is the **transpose** of VASP/ASE's native row/column ordering — the slicing
in `gen_vasp.py --collect` performs this swap.

Pass to FMD shell-model fitting via `dataset = "dfpt.xyz"`.

> **Note** — `dfpt.xyz` files generated before the transpose fix (Issue #5)
> store the BEC components with `α↔β` swapped. Re-run `python gen_vasp.py
> --collect` against the existing `workdir/` to regenerate; VASP itself does
> not need to be re-run.
