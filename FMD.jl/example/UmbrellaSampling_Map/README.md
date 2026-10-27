# Domain-wall preparation with a target polarization map

`PolarizationMapUmbrella` restrains the local (per-unit-cell) polarization
towards a map you supply. Where the uniform and cosine umbrellas bias a single
number — a macroscopic amplitude — this one names the whole field, so the
configuration you want can be *stated* rather than coaxed out of an applied
field.

That distinction matters for domain walls. Driving a wall into place with an
external field means guessing a field distribution that produces the pattern
through the material's own nonlinear, hysteretic response, and the amplitude
you end up with is field-enhanced rather than spontaneous. Restraining the
polarization directly gives you the pattern you asked for, a number that says
how close you are (the `CV` column of `Bias.log`), and a well-defined energy —
so forces, virial and the barostat all stay consistent, and the restraint can
be released smoothly.

## Files

| | |
|---|---|
| `makecell.py` | writes `prototype.xyz` (the P = 0 reference), `cell.xyz` (the seeded domain pattern) and `target.json` (target map + weight mask) |
| `input.jl` | the four-stage run: form → hold → release → free |

## Running

```
python makecell.py
julia --project input.jl
```

`makecell.py` needs a trained potential's charges only indirectly — the target
map is expressed in polarization units, so measure the spontaneous
polarization of your own potential first (the file says how) and put it in
`P_SPONT_Z`.

## The two things that are easy to get wrong

**The gauge.** Polarization is only defined relative to a declared zero. The
target map and the run must use the *same* `PolarizationGauge`, or they refer
to different origins and nothing will tell you. `input.jl` writes `gauge.jld2`
once and loads it everywhere, which is the point of persisting it.

**The weight mask.** Set `weight = 0` in a band around each wall. The domains
are then held while the wall is free to choose its own width and internal
structure — which is usually the thing you wanted to measure. Restraining the
wall region too would impose the answer.

Note that the weight multiplies the *smoothed* residual, so with `sigma > 0` a
released group still feels some force leaking in from its restrained
neighbours within the smoothing radius. That is the intended meaning (the
restraint is on the coarse-grained field), not an approximation.

## Watching the map

Every `polmap_interval` steps the run writes `polmap_<n>.txt`: the same
per-unit-cell polarization the bias restrains, measured against `gauge.jld2`,
one `ix iy iz Px Py Pz` row per unit cell in e/Å² (1 e/Å² = 1602 μC/cm²).
It is plain text, so the wall profile is a few lines of numpy:

```python
import json
import numpy as np
Nx, Ny, Nz = json.load(open("domain.json"))["gridsize"]
m = np.loadtxt("out_free/polmap_100000.txt")        # (ng, 6): ix iy iz Px Py Pz
Pz = m[:, 5].reshape(Nx, Ny, Nz)                     # iz cycles fastest
profile = Pz.mean(axis=(1, 2))                       # P_z along x, the wall normal
```

Average the profile over the `out_free` snapshots to get the wall width
without the restraint. For a run that was made without `polmap_interval`,
`FMD.app[:PolMap]` produces the same files from `step_<n>.xyz` afterwards:

```julia
frames = filter(f -> endswith(f, ".xyz"), readdir("out_free"; join = true))
p = (; pot..., input = frames, output_dir = "out_free/polmap",
     polarization_gauge = gauge)
FMD.app[:PolMap](p)
```

## Releasing

Two knobs, and they do different things:

- decaying `k` weakens the restraint everywhere;
- widening `r_th` (the flat-bottom width) leaves the restraint strong but
  stops it fighting fluctuations smaller than `r_th`.

`input.jl` decays `k`. Widening `r_th` instead is often gentler when the
pattern is already correct and you only want thermal freedom back.

After stage 4 the run is unbiased: if the wall survives with `k = 0`, it is a
real (meta)stable state of the potential, not something the restraint was
holding up.

## Reference

Shell-model parameters and the BaTiO3 soft-mode displacements used to seed the
pattern: H. Azuma et al., *Acta Mater.* **296**, 121216 (2025),
doi:10.1016/j.actamat.2025.121216.
