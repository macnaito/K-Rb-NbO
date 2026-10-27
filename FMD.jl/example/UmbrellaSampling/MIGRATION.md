# Umbrella convention change

The `UniformPolarizationUmbrella` and `CosPolarizationUmbrella` structs
were unified into a single `PolarizationUmbrella` (FFT-smoothed,
3D-mode-aware).  See `src/Calculators/Umbrella.jl`.

The flat `groupid` index decode was changed to ASE-natural order
(iz fastest, ix slowest):

    flat groupid = ix · Ny · Nz + iy · Nz + iz + 1

Previously `gx` cycled fastest, which forced cell builders to use a
two-stage NSUP trick (`unit_cell * (Nx, Ny, 1) * (1, 1, Nz)`) plus an
explicit axis transposition in `gridsize` / `m`.  Both are no longer
needed.

To migrate an old run:

* Regenerate `cell.xyz` with `makecell.py` in this directory.  The straightforward
  `unit_cell * (Nx, Ny, Nz)` is correct.
* Update `parameters.json` so that `gridsize = [Nx, Ny, Nz]` matches the
  cell's replication along (real x, real y, real z) — no transposition.
* For the cosine example: `m = [mx, my, mz]` directly addresses real
  Fourier modes (e.g. `[0, 0, 1]` = wave along real z).
