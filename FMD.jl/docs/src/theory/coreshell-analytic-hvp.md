# Analytic Hessian-vector products for the core-shell model

**Status: normative.** This note derives the analytic shell-displacement
Hessian-vector products (HVP) implemented in `src/Calculators/HVP.jl`
(Issue #147, part of the SCF-free program #117). It pins the variable
convention, the per-term Hessian blocks (each the *exact* derivative of
the implemented force expression, approximations included), the Ewald and
PME reciprocal-space HVP algorithms, the BSPORDER = 4 retention decision,
and the CG breakdown-guard convention. Replaces the finite-difference
directional derivatives (`jvp!` / `dF_du_dirderiv!`, pre-#147
`CoreShell.jl`), which survive as test oracles `fd_jvp!` /
`fd_dFdu_dirderiv!` in `test/helpers.jl`.

References: Toukmaji, Sagui, Board, Darden, JCP 113, 10913 (2000)
(dipole-PME); Essmann et al., JCP 103, 8577 (1995) (smooth PME);
Niklasson, JCP 147, 054103 (2017) (the shadow-dynamics consumer, #117 P4).

## 1. Conventions

- Lattice rows are the cell vectors (`Atoms.jl`); `pos` is fractional,
  Cartesian core positions are `x_c = pos·L`, fractional coordinates of a
  Cartesian point are `frac = r·inv(L)`.
- `shelldisp` `u` is the **Cartesian** shell offset; the shell site is
  `x_s = x_c + u`.
- The independent variables of the force arrays are the two **positions**
  `(x_c, x_s)`, not `(x_c, u)`:

  ```
  forces      = -∂E/∂x_c  at fixed x_s
  shellforces = -∂E/∂x_s  at fixed x_c
  ```

  Pinned by the Spring term: `U_spring` depends on `x_s - x_c` alone yet
  `kernel_spring!` writes equal-and-opposite contributions into both
  arrays — impossible under an `(x_c, u)` reading, where `∂U_spring/∂x_c`
  at fixed `u` would vanish.
- The two blocks evaluated by `hvp!(Jv, Kv, v, atoms, ws, prms)` for a
  direction `v` in shell-displacement space:

  ```
  Jv = (∂F_s/∂u)·v = -(∂²E/∂x_s∂x_s)·v     symmetric
  Kv = (∂F_c/∂u)·v = -(∂²E/∂x_c∂x_s)·v     mixed
  ```

  `Jv` is what the linear-response CG inverts; `Kv` is the BEC relaxation
  term and the future shadow-force mixed term (#117 P4).
- Translation identity (exactness probe): with E translation invariant,
  `Σ_i (J + K)_ij = 0` column by column. Measured ≤ 2e-15 (max-norm, all
  terms, random directions) — the BEC acoustic sum rule then reduces
  *exactly* to the summed CG residual (§7).

## 2. Derive on the implemented expressions

House rule from PR #141, applied throughout: every second derivative is
the exact `r`-derivative of the **implemented** first-derivative kernel —
never of the underlying mathematical function.

- `dϕC` (CoulombBuckingham.jl) mixes `erfc_approx` (a 7-coefficient
  rational fit, error ~1.5e-7) for the *value* with the analytic
  `erfc′ = -2/√π·e^{-x²}` for the *derivative term*. `d2ϕC`
  differentiates that mixed expression as written, which requires
  `derfc_approx` — the exact derivative of the rational fit itself. Its
  polynomial-derivative coefficients are formed in FPPREC: forming them
  in the Float32 of the stored literals bakes in a ~1e-7 coefficient
  rounding the value path never sees (measured as a 2.4e-8 FD-agreement
  floor before the fix).
- Shifted-force scheme: `ϕ_eff′(r) = ϕ′(r) − ϕ′(rc)` implies
  `ϕ_eff″ = ϕ″` — the rc constants drop out of every second derivative;
  no new rc precomputation exists anywhere in the HVP.
- The cutoff branch is the force kernels' `ifelse(r > rc, 0, 1)`
  verbatim. `ϕ″` jumps at rc (shifted-force smooths the value and first
  derivative only); the FD oracle carries the same jump, so the
  comparison is consistent — but fixtures keep pair distances away
  from rc.

## 3. Per-term Hessian blocks

Pair notation: `B(r) = (ϕ″ − ϕ_eff′/r)·r̂⊗r̂ + (ϕ_eff′/r)·I`; with the
unnormalized pair vector `rvec`, `B·v = (ϕ″ − ϕ_eff′/r)/r²·(rvec·v)·rvec
+ (ϕ_eff′/r)·v`.

| term | Jv row i | Kv row i |
|---|---|---|
| Spring | `-(k2 + k4·d²/6)·v_i - (k4/3)(u·v)·u_i` | −(same) |
| intra correction | `t_intra(u_i, v_i)` (below) | −(same) |
| short-range shell(i)–core(j) | `-B_sc·v_i` | — |
| short-range shell(i)–shell(j) | `-B_ss·(v_i - v_j)` | — |
| short-range core(i)–shell(j) | — | `+B_cs·v_j` |
| Ewald reciprocal | §4 | §4 |
| PME reciprocal | §5 | §5 |
| external field / self / charged-cell | 0 | 0 |

Only the three pair geometries that appear in the corresponding force
arrays contribute to each block (the sc/ss pairs never touch `forces`,
the cs pairs never touch `shellforces` — the `(x_c, x_s)` convention
of §1 doing its work). Ghost format: `v` is replicated to ghost rows by
`Cores.extend_ghosts!` (the same primitive the fitting adjoints use), and
the kernels are pure gather, matching the force kernels.

Intra correction (`kernel_subtractintraatom!`, both branches
differentiated as implemented, `w = qc·qs`, `d = |u|`):

```
erf branch (d ≥ 0.1/α):  factor = w·g′(d)/d,  g(d) = erf(αd)/d
    g″(d) = -2α/√π·e^{-(αd)²}·(2α² + 2/d²) + 2·erf(αd)/d³
    t_intra = factor·v + (w·g″ − factor)·(u·v)/d²·u
Taylor branch (d < 0.1/α): factor = w·(-4α³/(3√π) + 4α⁵/(5√π)·d²)
    t_intra = factor·v + w·(8α⁵/(5√π))·(u·v)·u
```

The library `erf` is exact, so its analytic derivative is too (unlike
`erfc_approx` above).

## 4. Ewald reciprocal HVP (CPU reference)

`E_rec = Σ_K ½·kf·(∑cos² + ∑sin²)` over the 2N core+shell point charges;
the force pass stores per-site phases `kqrcos/kqrsin` (cores rows 1:N,
shells N+1:2N) and the structure factor `S(k) = (∑cos, ∑sin)`.
Differentiating the implemented force expression
`F_a = Σ_K kf·k·(kqrsin_a·∑cos − kqrcos_a·∑sin)` along `v` with the
per-k dipole factors of the direction (shell sums only — core phases do
not move with u):

```
T_c(k) = Σ_j q_sj·cos(k·x_sj)·(k·v_j)
T_s(k) = Σ_j q_sj·sin(k·x_sj)·(k·v_j)

Jv_i += Σ_K kf·k·[ (k·v_i)·(kqrcos_i^s·∑cos + kqrsin_i^s·∑sin)
                   − (kqrsin_i^s·T_s + kqrcos_i^s·T_c) ]
Kv_i += Σ_K kf·k·[ − (kqrsin_i^c·T_s + kqrcos_i^c·T_c) ]
```

`EwaldHvpCache` reuses `_ewald_workspace` / `_ewald_fill_kqr!` so the
phases are bit-identical to the force pass, and caches `S(k)` once per
snapshot. One HVP costs one force-pass-shaped K loop. CPU-only, like the
Ewald force pass; this is the discretization-free oracle for §5 and the
production HVP of the (Ewald-based) fitting path.

## 5. PME reciprocal HVP and the BSPORDER decision

Implemented force: `F_a = -q_a·Σ_g ∇W_a(g)·φ(g)`, `φ = G ∗ Q`
(`gridenergy`), derivative chain `(v·∇) = Σ_d vsc_d·∂/∂s_d` with
`vsc = (v·inv(L)) .* fftsize`. Exact derivative along `v`:

```
Q_v(g) = Σ_j q_sj·(v_j·∇)W_j^s(g)          dipole spread (kernel_spread_dipole!)
φ_v    = G ∗ Q_v                            _pme_convolve (same operator, no virial)
Jv_i   = -q_si·[ ∇W_i^s·φ_v + (v_i·∇)∇W_i^s·φ ]
Kv_i   = -q_ci·[ ∇W_i^c·φ_v ]
```

One HVP = one dipole spread + one FFT round trip + first-derivative
gathers at both site families + the second-derivative gather of `φ` at
the shell sites (`kernel_gather_hvp!`). Only that last local term reads
`d2weights`: the order-4 B-spline second derivative is piecewise linear
(C⁰), one order less continuous than the force interpolation — the whole
BSPORDER question lives in this single term. `PMEHvpCache` freezes the
wrapped fractionals, weights, and `φ` at the snapshot; the charge/force
path (`interpolation_weights`, `_pme_solve`) is untouched.

Measured discretization error against §4 (max-norm relative, worst of 4
random directions; `bench/check_hvp_bsporder.jl`):

BaTiO₃ 2×2×2, production parameters (α = 0.1448, example/ShellModel-MD):

```
mesh [a.u.]  grid          relJ      relK      relF(force)
2.0 (prod)   (8,8,8)       5.3e-3    5.0e-3    3.2e-3
1.0          (16,16,16)    5.2e-4    3.8e-4    3.4e-4
0.5          (31,31,31)    1.4e-4    6.7e-5    3.8e-5
```

NaCl-8, α = 0.30 (harder screening, test fixture):

```
mesh   relJ      relK      relF
1.0    1.6e-2    2.5e-3    3.6e-3
0.5    3.3e-3    1.2e-4    4.1e-4
0.25   4.6e-4    1.1e-5    5.1e-5
```

**Decision: BSPORDER stays 4.** The meaningful criterion is the ratio
relJ/relF at the production mesh — at a coarse mesh the generic
interpolation error dominates forces and HVP alike, so an absolute relJ
target below relF is unreachable for any BSPORDER. Measured ratio at the
production mesh: 1.6 (BaTiO₃). The d2-order loss only opens the J/F gap
where both are already small (mesh 0.5: ratio 3.7). Downstream, the
response observables are further damped through the k2-dominated Hessian:
linear_response!(PME, mesh 0.5) vs (Ewald), same α, NaCl-64 — ε
difference 2.3e-11, BEC difference 4.9e-7. The fitting path runs Ewald
and is exact regardless. Workloads needing more reciprocal accuracy
shrink `pme_meshsize` (already a parameter) before touching BSPORDER —
a 4→6 change rewrites the hardcoded spline kernels and costs ~3.4× in
spread/interpolation (Issue #117).

## 6. Preconditioner and the CG breakdown guard

`cg_hessian_free!` solves `H·x = b`, `H = -J`, by PCG with the Jacobi
preconditioner `diag(k2)` (unit entries for k2 = 0 species): k2 dominates
the Hessian diagonal (`H_ii = k2 + O(k4·d², Coulomb)`), and this is the
same object as the CG α-denominator `kvals` of the shell relaxation in
`calculate!` — the kernel 0th approximation of #117's XL-BOMD plan.

Breakdown guard convention: the pre-#147 guard `abs(pHp) < eps(T)`
compared a *dimensional* quantity against an absolute epsilon. With
Jacobi scaling (`p ~ r/k2`), `pHp ~ |r|²/k2` crosses eps long before
convergence — measured: the residual stalled at 6.3e-9 relative
*independent of tol*, flooring the acoustic sum rule at 7.6e-8. The
guard is now sign-based (`pHp ≤ 0`, scale-free, detects loss of
positive-definiteness only); with tol = 1e-12 the same solve reaches
|r|/|b| = 1.4e-16. A solve that exits unconverged (maxiter or breakdown)
emits a `@warn` — the downstream response quantities would otherwise look
plausible. The workspace itself guards its frozen-snapshot contract at
run time: `hvp!` errors if pos/shelldisp/lattice moved since
construction.

## 7. Independent numerical verification

Beyond FD-vs-analytic (which shares the force implementation), the tests
pin implementation-independent properties:

- **Complex-step derivatives** (`test_hvp.jl`): `Im f(x+ih)/h`, h = 1e-20
  — exact to machine precision, no subtractive cancellation, nothing
  shared with the δ-sweep — for `derfc_approx`, `d2ϕC`, `d2ϕB`
  (rtol 1e-12).
- **B-spline partition sums**: `Σ_l M_l = 1`, `Σ_l M′_l = 0`,
  `Σ_l M″_l = 0` at arbitrary fractional offsets (≤ 1e-13) — the property
  that makes a uniform shell translation exact on the PME grid.
- **Hessian symmetry** `wᵀ(Jv) = vᵀ(Jw)` at rtol 1e-10 (J is symmetric by
  construction; most sensitive to mismatched pair bookkeeping and the
  dipole-spread/gather adjointness of §5).
- **Translation identity** §1 → the acoustic sum rule is the summed CG
  residual: measured 7.6e-8 at tol 1e-8 (was ~1e-2·N with the FD path),
  1.8e-15 at tol 1e-12.
- **Sheared-cell fixtures**: the `inv(L)` chain of `vsc` and the gathers
  cancels transpose mistakes on cubic cells; the sheared NaCl-8 fixture
  breaks that degeneracy.

## 8. Cost

One `hvp!` evaluates both blocks in a single pass: short-range pair
sweep (force-kernel shaped) + per-atom closed forms + one reciprocal
solve (Ewald: one K loop; PME: one dipole spread + FFT round trip +
gathers). The FD path cost two full `shell_forces!` per J product and
two full `core_forces!` (with energy/virial waste) per K product.
Measured on NaCl-64, Ewald path, `linear_response!` end to end: 34 ms
(FD, main) → 12 ms (analytic) at identical ε and tighter ASR. The
FitShellModel CMA-ES loss loop (`calcloss`) inherits the factor
directly; the removal of the FD δ also removes the candidate-dependent
noise floor of #117 P1's motivation.

## Pinned by this note

Code sites:

- `src/Calculators/HVP.jl` — `hvp!`, `hvp_workspace`,
  `kernel_hvp_spring_intra!`, `kernel_gcb_hvp!`, `EwaldHvpCache`,
  `PMEHvpCache`, `kernel_spread_dipole!`, `kernel_gather_hvp!`,
  `cg_hessian_free!` (Jacobi preconditioner + sign-based guard),
  `linear_response!`
- `src/Calculators/CoulombBuckingham.jl` — `derfc_approx` (FPPREC
  coefficient products), `d2ϕC`, `d2ϕB`
- `src/Calculators/PME.jl` — `d2spline`, `kernel_calcweights2!` /
  `interpolation_weights2`, `_pme_convolve` (force path untouched)
- `src/Calculators/Ewald.jl` — reused `_ewald_workspace` /
  `_ewald_fill_kqr!` (bit-identical phases contract)
- `bench/check_hvp_bsporder.jl` — the §5 measurement protocol

Tests that fail if this note's results break:

- `test/calculators/test_hvp.jl` — scalar δ-sweep + complex-step oracles,
  B-spline sum rules, per-term and full-stack FD comparisons (Ewald and
  PME paths, sheared cell), Hessian symmetry, Ewald↔PME equivalence with
  convergence pin, workspace purity, umbrella and stale-snapshot guards
  (the DD guard has no serial-tier test; it is exercised when P4 wires
  the MPI path)
- `test/calculators/test_coreshell.jl` — ASR ≤ 1e-6 (analytic-HVP floor),
  Ewald↔PME response consistency (ε ≤ 1e-8, BEC ≤ 1e-5)
- `test/gpu/test_hvp_gpu.jl` — CPU↔GPU parity of the PME-path `hvp!`

Issues/PRs: #117 (SCF-free tracking; P4 consumes `Kv` and the
workspace-refresh design), #147 / this PR (implementation), PR #141
(derive-on-constrained-space precedent).
