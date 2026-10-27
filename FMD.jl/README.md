# Fast Molecular Dynamics for Ferroelectrics

Julia implementation of a core-shell model + short-range MLIP (Spline or ACE)
for ferroelectric perovskites.  Designed for HPC with MPI + CUDA.

## Workflow

1. **Dataset creation** (`dataset/`)
   - NPT-MD with a universal MLIP (MACE) to sample configurations
   - DFT single-point calculations (VASP) for energy / forces / stress labels
   - MACE fine-tuning on DFT data; re-evaluate sampled structures with the fine-tuned model and merge with DFT frames to augment the dataset
   - DFPT calculations (VASP) for Born effective charges and dielectric tensors

2. **Fitting** (`example/training/`)
   - Shell model: CMA-ES fit to BEC + dielectric tensor (`FitShellModel`)
   - Short-range MLIP: CG fit to the Δ-learning dataset on top of the shell model (`FitSpline` / `FitACE`)

3. **Production MD** (`example/NPT-MD/`)
   - NPT-MD using the fitted potential (`NPTMD`)

## Installation

**To use FMD.jl as a dependency in another project:**

```julia
using Pkg
Pkg.add(path = "/path/to/FMD.jl")   # local
# or
Pkg.add(url = "https://github.com/.../FMD.jl")  # from git
```

**To run the examples in this repository directly:**

```bash
git clone <repo> && cd FMD.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## MPI configuration

Required for multi-rank fitting and MD on HPC:

```bash
julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
```

## CUDA configuration

On HPC systems that require a specific CUDA version, set in `LocalPreferences.toml`:

```toml
[CUDA_Runtime_jll]
version = "12.0"
local = "true"
```

and add to `Project.toml` under `[extras]`:

```toml
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
```

## Usage

Input scripts follow the pattern in `example/`:

```julia
# example/training/input_fitsm.jl
params = (; dataset = "dfpt.xyz", ...)
params = (; FMD.default_params(Val(:FitShellModel), params)..., params...)
FMD.app[:FitShellModel](params)
```

See `example/training/` for fitting and `example/NPT-MD/` for production MD.

## Documentation

Full reference + workflow / parameters / HPC guides build locally with
[Documenter.jl](https://documenter.juliadocs.org/):

```bash
julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=docs docs/make.jl
open docs/build/index.html
```

The narrative pages under `docs/src/` are also readable as plain
Markdown on GitHub:

- [`docs/src/workflow.md`](docs/src/workflow.md) — three-stage pipeline.
- [`docs/src/dataset.md`](docs/src/dataset.md) — Python / VASP recipe.
- [`docs/src/parameters.md`](docs/src/parameters.md) — `params` key reference.
- [`docs/src/hpc.md`](docs/src/hpc.md) — MPI / CUDA / Preferences setup.
