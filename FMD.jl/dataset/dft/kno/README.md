# KNbO3 dataset generation

Produces `synth.xyz` for FMD shell-model fitting (FitSpline / FitACE).

## Prerequisites

```
pip install ase mace-torch
```

VASP pseudopotentials: set `$VASP_PP_PATH`.

## Workflow

### 1. NPT-MD with MACE foundation model

```bash
# one job per seed (run in parallel as HPC array)
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

Output: `relax.xyz`

---

### 3. VASP single-point DFT

```bash
# write VASP input directories
python gen_vasp.py --nsamples 100

# submit jobs (edit submit script as needed), then collect outputs
python gen_vasp.py --collect
```

Output: `workdir/{0000..0099}/`, `dft.xyz`

---

### 4. MACE fine-tuning + dataset assembly

```bash
# fine-tune on DFT labels
python finetune_mace.py

# re-evaluate MD/relax frames with the fine-tuned model and merge with DFT
python finetune_mace.py --collect
```

Output: `ft_compiled.model`, `synth.xyz`

---

## Output format

`synth.xyz` is FMD extxyz with per-frame tags:

| tag | frames | weight in FitSpline/FitACE |
|---|---|---|
| `method=PBEsol` | DFT single-point | 1.0 |
| `method=MACE` | fine-tuned MACE | 0.5 |

Pass to FMD fitting via `dataset = "synth.xyz"` and `method_weight = { PBEsol = 1.0, MACE = 0.5 }`.
