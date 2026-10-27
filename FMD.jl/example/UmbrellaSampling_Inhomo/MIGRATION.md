# Umbrella convention change

The `CosPolarizationUmbrella` struct (z-axis-only cosine modulation) was
unified into a single `PolarizationUmbrella` (FFT-smoothed,
3D-mode-aware).  See `src/Calculators/Umbrella.jl`.

The flat `groupid` index decode was changed to ASE-natural order
(iz fastest, ix slowest):

    flat groupid = ix · Ny · Nz + iy · Nz + iz + 1

Previously `gx` cycled fastest, which forced cell builders to use a
two-stage NSUP trick (`unit_cell * (Nx, Ny, 1) * (1, 1, Nz)`) plus an
explicit axis transposition in `gridsize` / `m`.  Both are no longer
needed.

`parameters.json` field changes:

* `d` (single-axis index) → `u` (3-vector polarization direction).
* `m` (single integer along z) → `m` (length-3 array of integer
  Fourier-mode indices, e.g. `[0, 0, 1]` for a wave along real z).
* New optional `sigma` (Gaussian smoothing width in unit-cell units;
  `0.0` recovers the legacy box-filter behaviour).

To migrate an old run:

* Regenerate `cell.xyz` with `makecell.py` in this directory — drop any NSUP loops.
* Replace `"d": <i>` with `"u": [<unit-vector>]` and `"m": <integer>`
  with `"m": [<mx>, <my>, <mz>]`.
