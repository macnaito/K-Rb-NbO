# SCR barostat: 6-DOF derivation of the measure correction

**Status: normative.** This note derives, from Fokker–Planck stationarity
on the symmetric 6-DOF strain space, the measure-correction constant of
the anisotropic stochastic-cell-rescaling (SCR) barostat as implemented
in `_scr_mu` (src/Apps/MDControl.jl). Result: **−2kT/V**, not the
−kT/V of the SCR papers. The discrepancy is a convention difference:
the papers' value pairs with their *unsymmetrized* 9-DOF update (their
implementations remove cell rotations by QR **after** the update —
thesis §2.6 strategy 2 — which leaves the det statistics of the 9-DOF
noise untouched). The papers do not re-derive the correction for the
symmetrized-noise variant (strategy 1) used here; pairing strategy-1
noise with the verbatim −kT/V is precisely the miscalibration this
derivation replaces. Settled in Issues #140/#132, PR #141.

References: Bernetti & Bussi, J. Chem. Phys. 153, 114107 (2020)
(isotropic SCR, the FP framework used here); Del Tatto, Raiteri,
Bernetti & Bussi, Appl. Sci. 12, 1139 (2022) and the Del Tatto thesis
(arXiv:2111.06402) (anisotropic SCR, 9-DOF formulation).

## 1. Conventions

- Box matrix `h`: columns = cell vectors. FMD's `lattice` `L` has rows =
  cell vectors, so `h = Lᵀ`; the update `h ← μh` is `L ← L·μ` (μ
  symmetric).
- Stress: tension-positive; hydrostatic compression at pressure `P` is
  `S = −P·I`. `P₀ = −tr(S_ext)/3`. `stress(atoms, true)` includes the
  instantaneous kinetic term `S_kin = −(1/V)·Σᵢ mᵢvᵢvᵢᵀ`.
- One Euler (Itô) step of the cell:

  ```
  μ  = I + dε,      dε = −λ′·B + amp·ξ
  λ′   = κ_T·Δt/(3τ_p)                      (dimensionless mobility × Δt)
  amp² = 2λ′·kT/V
  B    = (P₀ − c·kT/V)·I + S_int + h·Σ·hᵀ/V    (c: the constant at issue)
  ```

  applied as `h ← μh`, fractional positions untouched (`x ← μx`),
  velocities `v ← μ⁻¹v` (physical phase-space volume `dx·dp` preserving).
- Noise `ξ`: symmetric, GOE-normalized — `E[ξᵢⱼξₖₗ] =
  ½(δᵢₖδⱼₗ + δᵢₗδⱼₖ)`, i.e. `Var = 1` on the diagonal, `Var = ½`
  off-diagonal (`_sym_randn`). Consequences used below:
  `E[ξ²] = 2I` and `tr ξ ~ N(0, 3)`.

## 2. State space: gauge invariance and the metric G

Products of symmetric matrices are not symmetric, so `h` itself does not
stay on a 6-dimensional slice — but physics depends only on the metric
`G = hᵀh` (the rotational part of `h` is gauge). Under `h ← (I+dε)h`:

```
dG            = 2·hᵀdε h + hᵀdε² h
E[dG]         = 2·hᵀ⟨dε⟩h + 2·amp²·G                  (uses E[ξ²] = 2I)
Cov(dGₐᵦ,dG꜀ᵈ) = 2·amp²·(Gₐ꜀Gᵦᵈ + GₐᵈGᵦ꜀)
```

The diffusion tensor is the inverse metric of the **affine-invariant**
Riemannian structure on SPD(3). Under it, the `ln V` direction and the
shape directions (`det Ĝ = 1`) are orthogonal, and the GOE noise splits
accordingly (`tr ξ` independent of the traceless part). Stationarity
therefore decomposes exactly into a trace sector (§4), a deviatoric
sector (§5), and a shape-measure convention (§6).

## 3. Target ensemble: an explicit convention choice

In scaled coordinates the position Jacobian is `∫dx^N = V^N ds^N`. FMD
pins the total momentum to zero every step (`_remove_com!`), which makes
the mass-weighted fractional COM a **constant of motion**: only `N−1`
configurational coordinates are free, so the sampled configurational
measure is `V^(N−1)`, and consistently `⟨2K⟩ = (3N−3)kT`, i.e.
`⟨S_kin⟩ = −((N−1)kT/V)·I`.

| Convention | config. V-power | kinetic pressure | ideal-gas P(V) |
|---|---|---|---|
| SCR papers (free COM, `⟨2K⟩ = 3NkT`) | `V^N` | `NkT/V` | `Gamma(N+1)` |
| **FMD (frozen COM, `⟨2K⟩ = (3N−3)kT`)** | **`V^(N−1)`** | **`(N−1)kT/V`** | **`Gamma(N)`** |

Key identity (thermodynamic consistency of the two columns): with
`q(G) = ∫ds e^(−βU)`,

```
V·⟨S_int⟩ₐᵦ = −kT·∂ ln(V^(N−1)·q)/∂εₐᵦ .
```

The V-power and the kinetic-pressure estimator are **not independent
conventions** — keeping the COM treatment consistent makes the same
`(N−1)` appear on both sides of the stationarity condition and cancel,
so the correction constant `c` is independent of `N` and of the
thermostat's DOF bookkeeping. FMD's MTTK integrator samples `Gamma(N)`
on the ideal gas (measured, and expected: its MTK kinetic force term is
`2K/(3N−3)·I`, the same frozen-COM count; see `test/apps/test_scr.jl`),
so this choice also makes SCR and MTTK sample the **same** volume
marginal.

## 4. Trace sector: c = 2

With `x = ln V` and `ln det μ = tr dε − ½·tr(dε²) + O(dε³)`:

```
E[Δx]   = −λ′·tr B − ½·amp²·E[tr ξ²] = −λ′·(tr B + 6kT/V)
Var(Δx) = amp²·Var(tr ξ) = 3·amp² = 6λ′·kT/V     ⟹  D = 3λ′·kT/V
```

The `−3·amp²` term is the **Itô drift of the multiplicative noise**; for
the 9-DOF iid noise of the papers `E[tr ξ²] = 3` and it is half as
large. The zero-current condition `b = D·∂ₓ ln ρₓ + ∂ₓD` with the target
`ρₓ ∝ V^N e^(−βP₀V)` (the `x`-representation of `Gamma(N)`) and
`∂ₓD = −3λ′kT/V` gives

```
required:  b = 3λ′·[(N−1)kT/V − P₀]
scheme:    b = −3λ′·[P₀ − ⟨P_int⟩ − c·kT/V + 2kT/V],   ⟨P_int⟩ = (N−1)kT/V
                                            ⟹  c = 2 .
```

The `(N−1)` cancels between the kinetic-pressure estimator and the
`V^(N−1)` Jacobian + `∂ₓD` terms (§3 identity — for interacting systems
it holds as a conditional average at fixed cell, the same adiabatic
assumption as the SCR papers); the **only** job of `c` is cancelling
the Itô drift. Caveat on coordinates: treating a single diagonal
component as a standalone coordinate appears to give a different `c` —
that bookkeeping drops the `ln det` contribution of the shear noise;
the global `ln V` coordinate used here is the correct reduction. Hence:

| noise structure | Itô drift in ln V | correction |
|---|---|---|
| symmetric GOE 6-DOF (this code) | `−3·amp²` | `−2kT/V` |
| iid 9-DOF (SCR papers) | `−1.5·amp²` | `−kT/V` |

The papers' value is correct in the papers' convention (9-DOF noise,
free COM ⇒ `Gamma(N+1)`); ours is correct in ours. Verification table
(the map `P(V) = Gamma(N−3+c, kT/P₀)` under FMD conventions):

| c | predicted | measured |
|---|---|---|
| 1 (papers' value, transplanted) | `Gamma(N−1)` | mean/`Gamma(N)` ≈ 7/8 (real driver, PR #141 review) |
| **2 (this note)** | **`Gamma(N)`** | mean/var = 1.007/0.994, 1.010/0.978 (real driver, seeds 7/11) |
| 3 (an earlier miscalibration to `Gamma(N+1)`) | `Gamma(N+1)` | mean/`Gamma(N)` ≈ 9/8 (real driver) |

plus a standalone MC (no FMD code; `N ∈ {4, 8}`, persistent- and
resampled-velocity variants) reproducing `n = N−3+c` at all three `c` to
1–2%.

## 5. Deviatoric sector: FDT closes without correction

For a shear coordinate `u = ε₁₂` the noise variance is `amp²/2`
(`D_u = λ′kT/(2V)`) while the target's gradient **double-counts** the
symmetric pair (`∂/∂ε₁₂ + ∂/∂ε₂₁`): `dΨ/du = −2βV·[⟨S₁₂⟩ + ∂Φ_ext]`.
The product reproduces the scheme drift `−λ′·[⟨S₁₂⟩ + (hΣhᵀ)₁₂/V]`
exactly — the ½ variance and the ×2 gradient cancel. This is *why* the
GOE normalization (off-diagonal `Var = ½`) is the correct one. Diagonal
deviatoric components: full variance, single-counted gradient — same
result. The Parrinello–Rahman coupling `Φ_ext = ½·tr(Σ·G)` gives
`∂Φ_ext/∂ε = hΣhᵀ`, matching the `hΣhᵀ/V` bracket term (exact at the
reference cell `h₀`, the standard PR caveat). No `c`-like correction
appears in this sector.

## 6. Shape measure

The shape (det-1) directions sample the invariant volume induced by the
GOE noise + affine geometry; MTTK's phase-space measure may differ from
it by an O(1) *convention* on shape fluctuations — with real shear
stiffness those are `O(kT/(C·V))`-small, so the difference is
higher-order and irrelevant at production sizes (see the parameters page
note). On an ideal gas the shape performs an unbounded random walk (no
restoring force) — physical, shared by MTTK, and exactly orthogonal to
the volume marginal used by the tests.

## 7. Velocity rescaling and the thermostat requirement

`v ← μ⁻¹v` preserves physical phase-space volume. Integrating the joint
FP over `v` kills all `v`-cross terms by parts, so the G-marginal
condition (§4–5) involves only conditional mean stresses. The cell move
does perturb the `v`-marginal at `O(Δt)`; the thermostat relaxes it —
the theoretical reason `barostat = :scr` **requires** a thermostat
(enforced in validation). CSVR/NH use `N_f = 3N−3`, consistent with §3.

## 8. Discretization error

Euler–Maruyama is weak order 1: the stationary bias is `O(Δt)`
(`∝ λ′`), and there is no discrete invariant (Bernetti–Bussi §2.4) —
hence the enthalpy-like `conserved` column for `:scr`. The
`max|μ−I| < 0.25` guard in `scale_cell!` keeps the runtime inside the
perturbative regime the derivation assumes.

## Pinned by this note

- `_scr_mu` (src/Apps/MDControl.jl): the `(b.P0 - 2kT/V)` bracket term;
  `_sym_randn`: the GOE normalization (diag 1, off-diag ½).
- Tests (test/apps/test_scr.jl): Itô-drift MC
  (`E[ln det μ − tr(μ−I)] = −3·amp²`), ideal-gas `Gamma(N)` mean/var for
  SCR **and** MTTK, SCR↔MTTK mean equivalence, E[μ] triclinic pin,
  kernel covariance.
- Changing the noise normalization, the COM handling (`_remove_com!`),
  or the thermostat DOF count invalidates parts of this chain — start
  from §3.
