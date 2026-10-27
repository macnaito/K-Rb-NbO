# MTTK cell thermostat: 6 vs 9 degrees of freedom

**Status: normative.** This note answers Issue #132: should the MTTK
cell thermostat count the cell momenta as 9 DOF (as in MTK's papers) or
6? Answer: **6** for this codebase, because the integrator constrains
the cell strain rate to the symmetric subspace. Pins the `6kT` cell-NH
target and `Q_cell = 6kT·τ_p²` in `_mttk_chain!`/`_mass_cell`
(src/Apps/MDControl.jl). Settled across PRs #131, #137, #141.

## The literature vs this integrator

Martyna–Tobias–Klein (J. Chem. Phys. 101, 4177 (1994), Eq. 2.13) writes
the flexible-cell conserved quantity with `(N_f + d²)·kT·ξ`: the cell
momentum `p_g` is a **full d×d matrix** (d² = 9 thermostatted DOF).
That is correct for their equations of motion, where all nine components
of `p_g` are dynamical (MTK note that `p_g` stays symmetric if
initialized symmetric and the forces are symmetric — but the *ensemble*
they analyse thermostats all nine).

FMD's integrator (like LAMMPS' `fix npt` with its upper-triangular cell,
and like the restricted-cell-DOF analysis of K. Shinohara, "Conserved
quantities and ensemble measure for Martyna–Tobias–Klein barostats with
restricted cell degrees of freedom", arXiv:2603.24061 — which derives
exactly the `d² → n_c` replacement used here) does not
give the antisymmetric components any dynamics at all: `zeta_b` starts
at zero, and every increment — stress tensors, the isotropic MTK kinetic
term, scalar dampings — is symmetric. The rotational 3 DOF are
constrained out identically, not merely initially. A thermostat target
must count the DOF that actually exchange energy: **6**.

Equipartition under the constraint: the symmetric `ζ_b` has 6
independent components; the cell kinetic term is the weighted form
`(Q_b/2)·Σₐᵦζₐᵦ² = (Q_b/2)·(Σᵢζᵢᵢ² + 2Σᵢ<ⱼζᵢⱼ²)`. Each independent
quadratic mode carries `kT/2` **regardless of its weight**, so

```
⟨Q_b·Σₐᵦ ζ_b,ₐᵦ²⟩ = 6·kT     (not 9·kT).
```

The same counting fixes the cell-NH mass `Q_cell = 6kT·τ_p²` (one NH on
a 6-DOF bath) and, together with the SCR derivation's symmetric-noise
Itô term ([SCR note](scr-barostat.md) §4), makes the whole control layer
consistent on the 6-DOF strain space.

## What the wrong count does (measured), and what pins what

Injecting the 9-DOF count (`ckT = 9kT`, `Q_cell = 9kT·τ_p²`) into the
ideal-gas driver of `test/apps/test_scr.jl`:

- `⟨Q_b·Σζ_b²⟩/6kT = 1.54/1.52` (seeds 11/7) — the 6 physical cell
  modes are overheated by exactly 3/2.
- Volume fluctuations inflate only mildly (`var/Gamma(N)` = 1.15/1.17,
  inside the 20% test band) and the mean barely moves — so ensemble-level
  V statistics alone would **miss** this fault at the test's statistics.

Division of labour (an honest accounting per the
[theory-notes house rules](index.md)):

- The **equipartition assertion** `⟨Q_b·Σζ_b²⟩ = 6kT ± 15%` is a pin of
  the *implemented* constant: the cell NH enforces its own target
  (`⟨G_cell⟩ = 0` in stationarity), so it detects a revert (fault sits
  3.4× beyond the band) but is tautological about whether 6 is
  physically right. Conservation (H′) tests are equally blind — H′ is
  conserved for any self-consistent constant, which is why Issue #132
  called for equilibrium statistics in the first place.
- The **physics** of the count rests on the constrained-dynamics
  argument above, and empirically on the cross-integrator agreement:
  SCR — whose −2kT/V constant is derived independently on the same
  6-DOF space ([SCR note](scr-barostat.md)) — measures
  `var/Gamma(N)` = 0.994/0.978 vs MTTK's 1.035/0.992, all within 3.5%
  of 1, while the 9-DOF fault predicts +15–17%. The current 20% test
  band (chosen for seed robustness) does not enforce that margin; a
  higher-statistics V-variance pin is the natural upgrade if this ever
  needs to be load-bearing.

## History

- Pre-#137: a single shared Nosé–Hoover saw particles + cell with a
  `9kT` cell share (and the cell was effectively unthermostatted, its
  weight in the shared friction being ~`6kT/3NkT`).
- PR #131: fixed the MTK velocity-exponent divisor (`3N+3 → 3N−3`);
  its self-review surfaced this DOF question as Issue #132.
- PR #137: orthogonalized the control layer; the MTTK barostat gained
  its own cell NH targeting `6kT` with `Q_cell = 6kT·τ_p²`.
- PR #141: this note, plus the equipartition + absolute-`Gamma(N)`
  ensemble tests that verify the count and close Issue #132.

## Pinned by this note

- `_mttk_chain!`: `ckT = 6·c.kT`; `_mass_cell`: `6·c.kT·b.tau²`.
- Test: "MTTK ideal-gas Gamma(N) + cell equipartition (Issue #132)" in
  test/apps/test_scr.jl.
