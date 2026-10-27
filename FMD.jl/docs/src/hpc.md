# HPC setup

```@meta
CurrentModule = FMD
```

FMD.jl needs three things on every HPC system: an MPI implementation,
a CUDA toolchain (for the production-MD GPU path), and the runtime
choice between `Float32` / `Float64`. All three are configured through
Julia Preferences in `LocalPreferences.toml` at the project root.

## Preferences (one file controls everything)

```toml
# LocalPreferences.toml — at the FMD.jl repo root

[FMD]
precision = "Float64"

[CUDA_Runtime_jll]
version = "12.2"
local = "true"

[MPIPreferences]
abi = "MPICH"
binary = "system"
libmpi = "libmpi"
mpiexec = "mpiexec"
```

Sections:

- `[FMD]` — `precision = "Float32" | "Float64"` (resolves
  [`FMD.FPPREC`](@ref)).
- `[CUDA_Runtime_jll]` — pin the CUDA runtime to a system-installed
  version (`local = "true"`) when the cluster requires it. Drop the
  section entirely to let `CUDA.jl` use its bundled runtime.
- `[MPIPreferences]` — switches `MPI.jl` to the system MPI. Generated
  by running

  ```julia
  julia --project -e 'using MPIPreferences; MPIPreferences.use_system_binary()'
  ```

  on the head node; fields like `abi`, `libmpi`, `mpiexec` are detected
  automatically.

If `CUDA_Runtime_jll` is set, also list it under `[extras]` in
`Project.toml` so `Pkg.instantiate()` does not drop it:

```toml
[extras]
CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
```

### Precompile workload

Precompiling FMD executes a small serial-CPU MD workload
(`src/precompile.jl`, Issue #111) so that fresh Julia processes skip the
~30 s first-call JIT of the `npt!` stack. This adds ~40 s to `Pkg.precompile`
(once per source change) and requires a **writable temp directory** on the
node doing the precompilation. If either is a problem (e.g. a restricted
compute node), disable the workload without touching the code:

```toml
[FMD]
precompile_workload = false
```

## MPI rank layout

| Job | Sensible `-n` | Why |
|---|---|---|
| `:FitShellModel` | `popsize × data_workers` | CMA-ES groups × per-candidate data ranks; see [`FMD.Apps.opt!`](@ref) |
| `:FitSpline` / `:FitACE` | `8` (per training set) | Pure data parallelism inside [`fitting!`](@ref) |
| `:NPTMD` | `1` per replica (or `nodes` for ensemble) | Single-replica MD; ensembles run independently |

Set the BLAS / OpenMP thread count to `1` whenever launching multiple
ranks per node — Julia's `BLAS.set_num_threads(1)` is already called in
the fitting drivers, but the env knobs below cover the dependent
libraries:

```bash
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## PME reciprocal-space decomposition (`pme_dd`)

Under MPI domain decomposition with more than one rank, the PME
reciprocal solve runs **z-slab distributed** by default (Issue #195): the
charge grid is reduce-scattered into contiguous z-slabs, the FFT work is
divided over ranks (2D x/y transform + one all-to-all transpose + 1D z
transform), and the energy grid is allgathered back for the unchanged
B-spline gather. This removes the replicated-FFT floor that otherwise
caps multi-rank throughput (~105 ms/step at a 61³ grid regardless of
rank count).

- `pme_dd = false` forces the replicated solve — the escape hatch for
  MPI builds whose *segmented* collectives (`MPI_Reduce_scatter`,
  `MPI_Alltoallv`) are not CUDA-aware even though `Allreduce` is;
  `probe_device_mpi!` tests exactly this at GPU-DD entry and fails fast.
- The slab count equals the rank count, so ranks beyond the grid's z
  extent own zero planes; that is supported but wasted — prefer rank
  counts ≤ the FFT grid side (61 at production 16³ cells), with threads
  (`-t 8`) providing the remaining node-level parallelism.
- The HVP / XL-BOMD path stays on the replicated solve in this phase.
- Reciprocal energies/forces are not bit-reproducible across different
  rank counts (the FFT summation order changes; deviations are at
  rounding level, ~1e-14 relative). This was already true of the
  replicated solve through its grid Allreduce — compare runs at
  different P with tolerances, never bitwise.

## Bundled batch scripts

Two examples live in `script/`. Both are written for Fugaku / `pjsub`
(`#PJM`) but are easy to translate:

### Shell-model fit (`script/flow.sh`, CPU)

```bash
#!/bin/bash
#PJM -L elapse=8:00:00
#PJM -L rscgrp=cl-small
#PJM -L node=2
#PJM --mpi proc=160
#PJM -j

module load oneapi
export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

mpiexec -n 160 julia -t1 --project ../../script/fitshellmodel.jl
```

### Production MD (`script/flow_typeII.sh`, GPU)

```bash
#!/bin/bash
#PJM -L elapse=0:10:00
#PJM -L rscgrp=cx-small
#PJM -L node=2
#PJM --mpi proc=80
#PJM -j

module load hpc_sdk
export JULIA_NUM_THREADS=1
# … (same env knobs as above) …

mpiexec -machinefile $PJM_O_NODEINF -n $PJM_MPI_PROC julia --project input.jl
```

### Translating to SLURM

Replace the `#PJM` block with `#SBATCH` directives:

```bash
#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --partition=<your-cpu-partition>
#SBATCH --nodes=2
#SBATCH --ntasks=160
#SBATCH --cpus-per-task=1

# load your site's MPI / CUDA modules instead of `oneapi`/`hpc_sdk`
module load openmpi/4.1
module load cuda/12.2

export JULIA_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun julia -t1 --project ../../script/fitshellmodel.jl
```

For GPU production MD use `--gres=gpu:N` and replace `srun` with
`mpiexec -n $SLURM_NTASKS julia ...` if your site's `srun` does not
launch the right MPI rank topology.

## GPU domain decomposition (`mpi = true`, `backend = :cuda`)

Run exactly like the CPU decomposition — `mpiexec -n P julia -t1 script.jl`
with `MPI.Init()` before `using FMD` and `mpi = true` in the params — but
with `backend = :cuda` (the default). Per rank:

- npt! binds the rank to a GPU by node-local rank (shared-memory split
  modulo the visible device count) **before its first CUDA allocation**;
  aim for one rank per GPU in production (ranks sharing a GPU work but
  contend for it). **Caveat**: device buffers your script allocates
  *before* calling npt! — a GPU-built `PolarizationUmbrella` in
  particular — are not rebound. On multi-GPU nodes either pin one GPU
  per rank at launch (`CUDA_VISIBLE_DEVICES` per rank, the robust
  option) or construct such objects only after selecting the device.
- The frame is split on the host, then moved to the device; the per-step
  ghost exchange and the grid/Pg reductions run on **device buffers
  passed directly to MPI** (see the transport section below). Neighbor
  rebuilds round-trip through the host (migrate + cell build are
  host-side) — a per-rebuild cost, not a per-step one.
- Snapshots gather to the root through a device-aware `gather_frame`.

## CUDA-aware MPI (GPU × MPI transport)

FMD's distributed-GPU code path passes **device buffers directly to MPI**
(halo exchange, grid reductions). This requires an MPI build with CUDA
support ("CUDA-aware MPI"). There is deliberately no host-staging
fallback — it would be a silent order-of-magnitude slow path that an
unsuspecting user could run production on; a non-aware build fails
loudly instead:

- At GPU-decomposition entry FMD probes the transport: the root rank
  prints `Probing CUDA-aware MPI …` and Allreduces a 1-element device
  array. **If the run crashes right after that line, your MPI build is
  not CUDA-aware.** Point `LocalPreferences.toml` at a CUDA-enabled MPI,
  or run CPU decomposition (`backend = :cpu`) / a single-GPU run instead.
- `MPI.has_cuda()` detects CUDA support only on OpenMPI; MPICH-ABI
  builds — including genuinely CUDA-aware ones — report `false`, so FMD
  treats it as advisory: one warning per run, silenced with
  `JULIA_MPI_HAS_CUDA=true` when you know your build is CUDA-aware.
  (On OpenMPI 5 the value is meaningful only after `MPI.Init()`.)
- OpenMPI note: some distributions (e.g. conda-forge) ship with runtime
  CUDA support disabled — set `OMPI_MCA_opal_cuda_support=1` or the
  build reports `has_cuda() == false` *and* refuses device buffers.
- The GPU+MPI test tier (`FMD_TEST_GPU=1 FMD_TEST_MPI=1`) exercises this
  transport end-to-end (probe + device Allreduce value checks) and turns
  red on a non-aware build.

## Common pitfalls

- **`MPIPreferences` mismatch** — if MPI seems to hang or all ranks
  print rank `0`, regenerate `LocalPreferences.toml` on the actual
  compute node (not the login node) — some sites ship a different
  `libmpi.so` for compute vs. login.
- **`CUDA_Runtime_jll.version`** must match the system CUDA driver
  major version. Check with `nvidia-smi`.
- **`Float32` vs. `Float64`** — Float32 is fine for MD trajectories
  but breaks shell-model fitting at tight tolerances. Switch to
  `Float64` in `LocalPreferences.toml` and rebuild before running
  `:FitShellModel`.
- **`-t1`** is mandatory under MPI: extra Julia threads contend with
  GC and BLAS and slow the fit by 2–3×.
