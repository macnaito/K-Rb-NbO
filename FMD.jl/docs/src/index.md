# FMD.jl

Fast Molecular Dynamics for ferroelectric perovskites — a Julia
implementation of a core-shell model + short-range MLIP (Spline or ACE),
designed for HPC with MPI + CUDA.

## Pipeline

1. **Dataset** (Python + VASP + MACE-MP) — see [Dataset](dataset.md).
2. **Fitting** (Julia + MPI) — `FitShellModel` → `FitSpline` / `FitACE`.
3. **Production MD** (Julia + MPI + CUDA) — `NPTMD`, optionally with
   polarisation umbrella sampling.

The end-to-end story is [Workflow](workflow.md).

## Quick start

Use FMD as a dependency in another Julia project:

```julia
using Pkg
Pkg.add(path = "/path/to/FMD.jl")
```

Or run the bundled examples directly:

```bash
git clone <repo> && cd FMD.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
julia --project example/NPT-MD/input.jl
```

For MPI / CUDA configuration on HPC, see [HPC setup](hpc.md).

## Building these docs locally

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
open docs/build/index.html
```

## Map

- [Workflow](workflow.md) — what each stage does and how they chain.
- [Dataset](dataset.md) — Python pipeline (MACE-MP NPT-MD → VASP →
  fine-tune) producing `synth.xyz` and `dfpt.xyz`.
- [Parameters](parameters.md) — every `params` key for every workflow
  app, with units and defaults.
- [HPC setup](hpc.md) — MPI / CUDA / Preferences and how to translate
  the Fugaku examples to a generic SLURM site.
- API reference: [Apps](api/apps.md), [Cores](api/cores.md),
  [Calculators](api/calculators.md).
