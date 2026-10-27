# Shadow XL-BOMD for the core-shell model

**Status: normative.** This note derives the shadow extended-Lagrangian
scheme implemented in `src/Apps/XLBOMD.jl` (Issue #149, the P4 stage of
the SCF-free program #117): the shadow functional, its exact gradient and
its exact cell-strain derivative (Issue #151), the closed-form shadow
minimum, the conserved quantity reported in `log.txt`, and the measured
stability limit of the auxiliary coupling. It also records what the
scheme deliberately does **not** support (the polarization umbrella) and
why.

The scheme is opt-in (`params[:shell_dynamics] = :xlbomd`); the default
`:scf` path — per-step conjugate-gradient relaxation with a 3-point
predictor — is unchanged. Because the dynamics runs on a *shadow*
potential-energy surface rather than the fully relaxed one, results can
differ from the default path; the measured deviation is in
`docs/src/parameters.md`.

References: Niklasson, PRL 100, 123004 (2008) (XL-BOMD); Niklasson, JCP
147, 054103 (2017) (shadow potential formulation); Niklasson et al., JCP
130, 214109 (2009) (dissipative variants). The core-shell specialization,
its closed-form minimizer, and the κ limit below are this project's.

## 1. Conventions

Carried over from `coreshell-analytic-hvp.md` §1, which this note
depends on:

- Independent variables are the two position families `(x_c, x_s)`;
  `u = x_s - x_c` is `:shelldisp`, Cartesian.
- `forces = -∂E/∂x_c` at fixed `x_s`; `shellforces = -∂E/∂x_s` at fixed
  `x_c`.
- `J·v = (∂F_s/∂u)·v` and `K·v = (∂F_c/∂u)·v` are the analytic
  Hessian-vector products of `Calculators.hvp!`.

Energy split used throughout:

```
E(r, s) = E_pair(r, s) + E_spring(s)
E_spring(s) = Σ_i [ k2_i·|s_i|²/2 + k4_i·|s_i|⁴/24 ]
```

`E_pair` is everything else — real-space Coulomb + Buckingham, the
reciprocal sum, the intra-atom correction, the external field. The
auxiliary field is `ŝ`, propagated by its own equation of motion; the
physical shell configuration reported per step is the shadow minimizer
`s₁`.

## 2. The shadow functional

Linearize `E_pair` in the shell coordinate at `ŝ` and keep the spring
exact:

```
E_shadow(r; ŝ) = min_s [ E_pair(r, ŝ) + gᵀ(s - ŝ) + E_spring(s) ]
g = ∂E_pair/∂s |_ŝ = -F_pair,   F_pair = F_s(ŝ) + (k2 + k4|ŝ|²/6)·ŝ
```

Keeping the spring exact (rather than linearizing everything) is what
makes the minimization closed-form *and* keeps the quartic term — the
part that stiffens the shell against the polarization catastrophe —
inside the functional rather than in a discarded remainder.

`E_shadow` is a smooth function of the nuclear positions for frozen `ŝ`,
which is the property the dynamics needs: a conserved quantity exists
constructively, without any statement about how well the shells are
converged.

## 3. The shadow minimum in closed form

Stationarity of the bracket gives, per atom,

```
(k2 + k4|s₁|²/6)·s₁ = F_pair
```

so `s₁` is parallel to `F_pair` and its magnitude `m` solves the scalar
cubic

```
k4/6·m³ + k2·m = |F_pair|
```

For `k2 > 0`, `k4 ≥ 0` the left side is strictly increasing in `m`, so
the root is unique and Newton from the harmonic guess `m = |F_pair|/k2`
converges monotonically (`kernel_xl_shadow_min!`, ≤ 50 iterations, exits
on the machine-precision update). `k2 = 0` marks a shell-less species and
yields `s₁ = 0`.

## 4. The exact shadow force

The shells ride the cores at fixed *displacement* `ŝ`, so the derivative
with respect to a core position picks up both site families:

```
d/dr = ∂/∂x_c + ∂/∂x_s   (at fixed ŝ)
```

Differentiating `E_shadow` and using the envelope theorem (the explicit
`s₁`-dependence drops because `s₁` is the minimizer):

```
dE_shadow/dr = ∂E_pair/∂r|_ŝ + [∂g/∂r]ᵀ(s₁ - ŝ)
```

With `∂E_pair/∂r|_ŝ = -(F_c + F_s)` restricted to the pair part, and
`∂g/∂r = ∂²E_pair/∂r∂s → -(J + K)` restricted likewise, the implemented
force is

```
F_shadow = F_c(ŝ) + F_s(ŝ) + (J + K)·(s₁ - ŝ)
```

**Cancellation lemma (why every term is evaluated full-stack).** The
formula above is stated for the *pair* part, but `xl_shadow_forces!`
evaluates `F_c`, `F_s`, `J` and `K` over the complete stack, spring and
intra-atom terms included. This is exact, not an approximation: both
extra terms depend on the displacement `s` alone, so for them
`∂/∂x_c = -∂/∂x_s`, i.e. their contributions to `F_c + F_s` cancel
pairwise, and likewise their contributions to `J + K` (the intra and
spring blocks satisfy `K = -J` exactly — see
`coreshell-analytic-hvp.md` §3). Subtracting them term by term would be
equivalent and slower.

**The omitted terms are not optional.** Dropping `F_s(ŝ)` and `J·(s₁-ŝ)`
— keeping only the "core" force and the mixed block, which is the natural
first guess — leaves a force that is not the gradient of anything, and
the energy runs away within a few hundred steps. This was observed
directly during the Phase 1 evaluation (#117) and is the reason T1 below
is the load-bearing test.

## 4.1 The shadow virial

A barostat needs the cell-strain derivative of the same functional. Two
conventions have to be fixed before differentiating.

**What the reported virial already means.** `atoms.info[:virial]` is
`-∂E/∂ε` with *every* site riding the cell affinely: `kernel_gcb!`
accumulates each pair as `r⊗f` at the true site separation, and
`kernel_spring!` gives the intra-atomic core-shell "bond" its own
`-S(u)·u⊗u`. Along the relaxation path the distinction between "shells
affine" and "shells at fixed displacement" is invisible, because the two
differ by `Σ_i F_{s,i}⊗u_i` and `F_s = 0` at the minimum. The shadow path
runs off that minimum, so the distinction becomes real.

**ŝ does not scale.** The auxiliary field is an independent dynamical
variable propagated in Cartesian coordinates by §5's Verlet, with no
coupling to the cell — the barostat never rescales it. The strain
derivative is therefore taken at fixed Cartesian `ŝ`. (Declaring `ŝ`
affine instead is self-consistent too; it differs by
`Σ_i (H_pair·v)_i⊗ŝ_i` and would oblige the integrator to rescale `ŝ` and
`ŝ⁻` at every cell update. Nothing is bought by it.)

With that, differentiating `E_shadow` along any parameter `ξ` and using
the envelope theorem as in §4 gives
`dE_shadow/dξ = (∂E_pair/∂r)·(dr/dξ) + [dg/dξ]ᵀ(s₁ - ŝ)`, and for `ξ` the
strain this assembles into

```
W_shadow = W(ŝ) + D_v W(ŝ) - Σ_i sym[F_s(ŝ)_i ⊗ s₁_i]
                           - Σ_i sym[(J·v)_i ⊗ ŝ_i],     v = s₁ - ŝ
```

`W(ŝ)` is what `core_forces!` already left in `info[:virial]`, `F_s(ŝ)`
and `J·v` are already in hand from the force, and the two outer products
are per-atom closed forms. **The only new quantity is `D_v W`** — the
directional derivative of the virial with respect to the shell
displacements, the strain counterpart of `J`/`K`, computed by `hvp!`
alongside them.

**The u-only terms cancel.** For any contribution `E_x(u)` depending on
the displacement alone — the spring and the intra-atom correction — the
four pieces above are `-Σ∇E_x⊗ŝ`, `-Σ[(∇²E_x·v)⊗ŝ + ∇E_x⊗v]`,
`+Σ∇E_x⊗s₁` and `+Σ(∇²E_x·v)⊗ŝ`, which sum to zero identically. So the
assembly is evaluated full-stack, exactly as the force is (§4's
cancellation lemma), and the cancellation is *pinned by a test* rather
than exploited: a cell whose only interaction is the spring must report a
shadow virial of exactly zero (§10).

**The external field is excluded.** `ExternalField.calculate!` adds to
the shell force but contributes nothing to `info[:virial]` (nor to the
energy). The assembly therefore feeds `F_s(ŝ) - q_s·E` into the
conversion term: converting a shell force the reported virial does not
account for would report a strain derivative of an energy that is not
there. This inherits the field's missing virial from the `:scf` path
rather than fixing it — a run with `efield ≠ 0` has an incomplete stress
on both paths.

**Cost.** Neither term needs an extra force evaluation, an extra FFT or
an extra neighbour pass. In the short-range kernel the scalars the
accumulation needs (`c1 = (ϕ″ - ϕ_eff′/r)/r²` and `r·δ`) are the ones the
Hessian product already computes, so only the six-component accumulation
is new. In the reciprocal sum the shell displacements enter through the
charge grid alone, so `d|Q̂|²/dλ = 2·Re(conj(Q̂)·Q̂_v)` and both transforms
already exist — the charge grid's in the HVP cache, the dipole grid's
inside `hvp_coulomb!`. Measured on BaTiO₃ (§7's fixture), the pair kernel
costs +20.0% (CPU, 135 atoms) / +10.4% (GPU, 1080 atoms) and the grid
reduction 0.2% of a step, so the shadow virial is **+8.5% / +5.0% of a
shadow step**. A finite difference in `λ` — the obvious alternative —
would cost one extra full evaluation, +57% / +53%, and would put an FD
error into the barostat force.

`hvp!` computes the term unconditionally rather than on request: a
barostat driven by a stale virial is wrong without saying so, and an
opt-in output is one forgotten call site away from exactly that. The
price is paid by the callers that do not want it. The `Kv === nothing`
fast path can no longer skip the core-shell pair block — that block moves
with `u`, so half of its virial derivative is not optional — and now
costs 0.99× the full `Kv` call (measured, 135 atoms), against roughly
two-thirds of it before. `cg_hessian_free!` / `linear_response!` (the
fitting path, tens of products per solve) and the `:chebyshev` kernel's
`order - 1` extra products pay that, and discard the result.

**How wrong the previous output was.** Before this, `info[:virial]` and
the `peratomstress` columns carried `W(ŝ)` — the first term alone.
Measured on the shadow trajectory (2×2×2, 300 steps, the §7 protocol),
that is 2.5% of `max|W|` for the default kernel (2.0% with
`:chebyshev`), concentrated in the shear components: 8.9% (4.2%) of the
shear scale, against a hydrostatic offset of only 0.018 GPa. The trace
hides it; the deviatoric part does not.

**The dynamical check.** A barostat with no thermostat integrates the
MTTK invariant, which couples the cell to the virial exactly as the
nuclei are coupled to the force — so an inconsistent virial pumps the
cell and drifts the invariant even when every individual force is exact.
That is the strain counterpart of the runaway §4 describes, and the
sharpest end-to-end statement available. Measured (3×3×3, 600 steps of
NPH at `P = 0`, MTTK, drift from an OLS fit after a 200-step discard):

| mode | drift [Ha/atom/ps] | ⟨V⟩ [Å³] | steps/s (CPU) |
|---|---|---|---|
| `:scf`, `cgtol = 2e-5` (production default) | +4.8e-6 | 1743.82 | 18.4 |
| `:scf`, `cgtol = 1e-8` (tight reference) | −2.0e-6 | 1743.84 | 8.9 |
| `:xlbomd`, `:spring` | **−1.4e-6** | 1744.39 | 19.4 |
| `:xlbomd`, `:chebyshev` | **−1.2e-6** | 1744.13 | 10.3 |

The shadow path conserves the *barostatted* invariant as well as the
tight relaxation does — better than the production default — and the cell
settles within 0.03% of the volume the relaxed surface picks, well inside
the run-to-run spread (sd ≈ 9 Å³).

## 5. Auxiliary propagation and the conserved quantity

The auxiliary field follows a time-reversible Verlet in the shadow
minimizer,

```
ŝ⁺ = 2ŝ - ŝ⁻ + κ(s₁ - ŝ)
```

which discretizes `d²ŝ/dt² = ω²(s₁ - ŝ)` with `κ = ω²Δt²`. The propagator
is undamped on purpose: it is then exactly time-reversible, which is what
makes the quantity below an invariant. Should a long run ever need
damping, the correct form is Niklasson's *time-reversible* polynomial
filter over several past `ŝ` (JCP 130, 214109) — a plain velocity damping
`(2-γ)ŝ - (1-γ)ŝ⁻` destroys both the reversibility and the invariant, and
was measured to heat or diverge on this model (§6).

The quantity written to the `conserved` column of `log.txt` is the
ordinary one of the active thermostat/barostat combination
(`Apps.conserved_energy`), except that `atoms.info[:potential_energy]`
holds `E_shadow` rather than the relaxed energy. For a fixed-cell,
thermostat-free run this makes it exactly

```
H_shadow = K_nuclei + E_shadow(r; ŝ)
```

which is the invariant of the shadow dynamics. The auxiliary degrees of
freedom carry no mass term in this formulation and contribute nothing to
it. That is the honest reading of the column: it is conserved by the
shadow dynamics, and it approximates — but is not — the Born-Oppenheimer
energy of the relaxed surface.

## 6. Stability of the auxiliary coupling

Verlet applied to the auxiliary equation alone is stable for `κ < 4`, but
the auxiliary field is coupled to the physical system through `s₁`, and
the effective coupling is set by the shell Hessian: the practical bound
is `κ·λmax(spring⁻¹·H) < 4`. Shell models with a hard short-range wall
have stiff modes that push `λmax` well above unity, so the standard
literature value `κ = 1.8` is **not** transferable.

Measured on BaTiO₃ 4³ (320 atoms, 1 fs, NVE, Phase 1 evaluation of #117):

| κ | dissipation | outcome over 4–5 ps |
|---|---|---|
| 1.8 | 0 | diverges (auxiliary oscillation grows within ~10² steps) |
| 1.8 | 0.05 | diverges |
| 0.3 | 0.02 | survives but heats (T 400 → 460 K over 3 ps) |
| **0.5** | **0** | **stable; conserved-quantity drift at the measurement floor** |

The measurement floor referred to is the same one the tight-SCF reference
reaches: drift `+3.4e-8` Ha/atom/ps for `κ = 0.5` versus `+3.5e-8` for
`cgtol = 1e-8` relaxation, against `+5.2e-6` for the production default
(`cgtol = 2e-5` with the 3-point predictor). Hence the default
`xl_kappa = 0.5`.

The Phase-1 numbers above and the §7 table are **not directly
comparable**: different cell size (4³ vs 3³), run length, and fit window.
Each is internally consistent — the claim carried by both is the same
one, that the XL drift sits at the tight-relaxation level rather than the
production-default level.

**A surface term raises `λmax`.** The depolarization block adds a
positive rank-3 piece to `H_ss`, so `λmax(spring⁻¹H_ss)` — the quantity
both `κ` and a Chebyshev interval are declared against — can only grow.
Its contribution is closed-form (`surface-term.md` §4.1),

```
Δλ_surf = (4π/V)·(Σ_i q^s_i²/k2_i)·λmax(T_Ω),
```

and by Weyl's inequality it is a **lower bound on the full `λmax`**,
because the rest of `H_ss` is positive semi-definite at a stable shell
configuration. `xl_init!` therefore refuses two provably-wrong
declarations, **one per kernel**:

- `:chebyshev` — a declared `xl_kernel_spectrum` upper bound
  `b < Δλ_surf`. The kernel is anti-restoring above `b`, which §6.1
  identifies as the one unsafe direction of the interval.
- `:spring` — `κ·Δλ_surf ≥ 4`. The Verlet bound is on the coupling the
  auxiliary field actually sees, `κ·λp < 4`, and `λp = λ` holds for this
  kernel alone; §6.1's kernel flattens `λp` into `[1-ρ, 1+ρ] ⊂ (0, 2)`
  whatever `λmax` does, which is exactly why switching to `:chebyshev`
  (with an interval that covers the spectrum) is the escape from this
  refusal rather than another way to hit it.

Passing them is **not** a certificate: the rest of the spectrum is still
measured, not derived, so `bench/check_xl_spectrum.jl` remains the way to
fix `b` for a new material — it now measures both operators per frame,
with and without the block.

Measured there on BaTiO₃ 3³ (the §6.1 protocol, 8 frames along a 800-step
NVE run) with an `ε_out = 1` sphere, the extreme case `T = I/3`:

| quantity | median | max |
|---|---|---|
| `λmax` without the term | 2.7891 | 3.1283 |
| `λmax` with it | 2.7892 | 3.1290 |
| measured increase | 0.0002 | 0.0008 |
| closed-form `Δλ_surf` | 0.4491 | 0.4491 |

So the bound is honest but loose by three orders of magnitude here, and
the reason is structural rather than incidental: the block's single
direction is `w_i ∝ q^s_i/√k2_i`, a uniform (acoustic-like) shell mode,
whereas `λmax` belongs to a stiff local mode, and the Rayleigh quotient
of the top eigenvector picks up only the overlap between the two. **Both facts matter.** `Δλ_surf` is a
certificate of wrongness, so it belongs in a guard; the measured increase
is what should inform `b`, and on this material it means the shipped
`(0.4, 6.0)` needs no change when a surface term is switched on. A
conductor-limit medium (`ε_out → ∞`, so `T → 0`) contributes exactly
nothing by the formula — the consistency statement that the term reduces
to the tin-foil boundary the solvers already implement.

**Auxiliary frequency versus the phonon band.** `ω = √κ/Δt`; at
`κ = 0.5`, `Δt = 1 fs` this is 112 THz ≈ 3750 cm⁻¹, an order of magnitude
above the BaTiO₃ phonon band edge (~700 cm⁻¹), so the auxiliary mode does
not resonate with — or leave a signature in — the vibrational spectrum.
Users who shorten `Δt` should keep `κ` fixed (it is dimensionless), which
raises `ω` proportionally and keeps the separation.

## 6.1 The kernel: what the auxiliary Verlet actually solves

The driver `s₁ - ŝ` is not an arbitrary residual. From the definition of
the shadow minimum, `S(s₁) = F_pair = F_s(ŝ) + S(ŝ)`, so

```
S(s₁) - S(ŝ) = F_s(ŝ)          (exactly; = F_s(ŝ)/k2 when k4 = 0)
```

— the auxiliary field is driven by the **shell force at ŝ**, mapped
through the inverse of the spring. Writing `A` for the spring Hessian and
`H_ss = -J` for the full one, the linearized error `e = ŝ - s*` obeys

```
e⁺ = 2e - e⁻ - κ·K·H_ss·e,        K = A⁻¹ for the default kernel
```

so the effective coupling of a mode is `κ·λ` with `λ ∈ spec(A⁻¹H_ss)`, and
§6's bound `κ·λmax < 4` follows. Measured for the BaTiO₃ parameter set of
`example/ShellModel-MD` (3×3×3, 800 steps of NVE at ⟨T⟩ ≈ 465 K, Lanczos
with full reorthogonalization on `A^{-1/2}H_ss A^{-1/2}`, `k2 = 0` species
projected out):

| | min | median | max |
|---|---|---|---|
| `λmin` | 0.465 | **0.472** | 0.481 |
| `λmax` | 2.696 | **2.789** | 3.128 |

`λmin < 1` is the ferroelectric soft mode — the local field nearly cancels
the spring — and `λmax` is the Buckingham wall. The **factor ~6 spread is
what the kernel exists to remove**: at `κ = 0.5` the modes run at
effective couplings between 0.24 and 1.40, so the softest mode (the one
carrying ε(ω)) is tracked ~6× more loosely than the stiffest.

`xl_kernel = :chebyshev` replaces `A⁻¹` by `p(A⁻¹H_ss)·A⁻¹` with `p` the
minimax polynomial of `xl_kernel_order` on a declared interval `[a, b]`:

```
1 - λ·p(λ) = T_order((c - λ)/d) / T_order(c/d),   c = (a+b)/2, d = (b-a)/2
```

On `[a, b]` this puts the preconditioned spectrum inside `1 ± ρ` with
`ρ = 1/T_order(c/d) < 1`, at `order - 1` extra Hessian-vector products per
step. Two consequences:

- **κ becomes transferable.** `κ·λp < 2κ` for any declared interval (so
  the Verlet bound `κ·λp < 4` reduces to `κ < 2`, which the shipped
  default 1.8 satisfies; `xl_init!` does not guard it, because a κ above
  2 is only *possibly* unstable — the spectrum need not reach `1+ρ`), so
  the XL literature value `κ = 1.8` is stable by construction — hence the
  different default (`xl_init!`).
- **The interval is one-sided.** Declaring `a` too small only weakens the
  tracking of modes below it (`λ·p(λ)` decreases monotonically to 0 at
  `λ = 0` and never changes sign). "Weakens" is not "bounds", though: a
  mode whose effective coupling reaches 0 obeys `e⁺ = 2e - e⁻`, whose
  error grows linearly in the step count — slower than a divergence, and
  still caught by the deviation guard, but not a stationary error.
  Declaring `b` below the true `λmax`
  destabilizes: above `b` the polynomial leaves the band — sign reversal
  for even `order`, growth past `4/κ` for odd. **Widen `b` when in doubt**;
  `bench/check_xl_spectrum.jl` measures the interval for a given
  potential, and a violation surfaces as the `xl_max_deviation` fallback
  rather than as a wrong answer (the shadow force is the exact gradient of
  the shadow energy whatever the kernel does).

**Why a fixed polynomial and not a Krylov solve.** The natural-looking
alternative — run `m` conjugate-gradient iterations on `H_ss·x = F_s(ŝ)`
each step — was implemented and measured, and is *worse than no kernel at
all* at low rank (see §7). CG picks its polynomial from the current
right-hand side, so the effective propagator changes from step to step;
the auxiliary Verlet is then integrating a jittering operator and the
smooth tracking it depends on is lost. It only starts to pay at `m ≳ 6`,
by which point the step costs more than the `:scf` relaxation it replaces.
A fixed polynomial has no such freedom: it is a state-independent linear
map, so the propagation stays exactly time-reversible and the linearized
analysis above applies verbatim.

## 7. What the shadow surface costs and changes

Measured with `bench/check_xlbomd.jl` on the BaTiO₃ parameter set of
`example/ShellModel-MD` (r²SCAN-D4 Ewald), 1 fs, NVE, `κ = 0.5`.

**Conserved-quantity drift and per-step cost** (3×3×3 = 135 atoms, 600
steps, drift from an OLS fit after a 100-step discard):

| mode | drift [Ha/atom/ps] | ncg | steps/s (CPU) | steps/s (GPU) |
|---|---|---|---|---|
| `:scf`, `cgtol = 2e-5` (production default) | +5.9e-6 | 6.5 | 17.3 | 5.1 |
| `:scf`, `cgtol = 1e-8` (tight reference) | −1.3e-6 | 23.5 | 8.9 | 2.9 |
| `:xlbomd`, `xl_kernel = :spring` | **−2.2e-6** | **0** | 18.6 | **13.3** |
| `:xlbomd`, `xl_kernel = :chebyshev` | **−2.2e-6** | **0** | 10.3 | 8.0 |

(The `:xlbomd` throughputs are 7% / 19% below what they were before the
shadow virial landed on the CPU, 5% / 19% on the GPU — §4.1's mandatory
`hvp!` output, paid once per Hessian product and therefore `order` times
over for the kernel.)

The shadow dynamics conserves its invariant as well as the tight
relaxation conserves the physical one, at zero SCF iterations, and the
kernel does not change that — it changes only how closely `ŝ` tracks the
relaxed shells. On the GPU — the production backend — the shadow step is
**2.6× the default throughput** at this size (1.6× with the kernel) and
2.0× at 8×8×8 (2560 atoms: 2.25 → 4.59 steps/s, 2.61 with the kernel). On
the CPU the same step is only 1.08× faster, and the kernel turns that into
a 1.7× *slowdown*.

The reason is **not** that the shadow step rebuilds its PME intermediates,
which is what this note asserted until Issue #153 was measured. The
reciprocal path is 1.2% of a CPU shadow step at this parameter set (rc =
11 Å puts the cost in the real-space pair loop), and the HVP workspace
rebuild specifically is 0.3% — there is nothing there to win. The real
reason is that "~3 forces" and "~7.5 forces" are not the same unit of
work: the default's 7.5 are cheap *shell-only* evaluations, which on the
CPU run through a `@turbo` kernel (`cpukernel_gcb_onlyshell`, with the
`rc`-shift constants hoisted into tables) measured 3.3× faster than the
scalar KernelAbstractions version; the shadow step's two passes —
`kernel_gcb!` for the forces and `kernel_gcb_hvp!` for the Hessian — are
both on the scalar path. Per pair block the shadow step is therefore
paying roughly three times what the CG loop pays. A `@turbo` variant of
`kernel_gcb_hvp!` is the CPU lever (the Hessian pass is 423 ms of a 918 ms
step at 2560 atoms); it has not been written because the production
backend is the GPU.

The GPU benefits for a different reason: it trades a long chain of small
dependent CG kernels for a fixed, wider pipeline. Both halves of the work
scale with the number of evaluations, and the shadow step does two passes
where the relaxation does `ncg + 2`. Measured at 2560 atoms in Float32 —
the configuration production runs in — a `:scf` step is 23.6 ms at
`ncg = 12`, of which the reciprocal path is 3.9 ms (16%), against a shadow
step of 4.4 ms of which the reciprocal path is 1.0 ms (22%). The shadow
step's *share* is the larger of the two only because its real-space cost
fell so much further: 3.1 ms (one force pass plus one Hessian pass) against
13.8 ms. Issues #157 and #153 cut the per-call reciprocal cost on both
paths; what separates them is the call count.

**Force deviation from the relaxed surface.** The shadow force is the
exact gradient of `E_shadow`, not of the relaxed `E`, so the trajectory
follows a nearby but different surface. Measured *on* the shadow
trajectory (auxiliary field carried exactly as in `npt!`, tight
relaxation evaluated at the same positions), 2×2×2, 300 steps:

| kernel | force, max-norm | force, rms | polarization | extra HVP/step |
|---|---|---|---|---|
| `:spring`, `κ = 0.5` (default) | 5.8e-2 | 5.4e-2 | 5.7e-2 | 0 |
| `:chebyshev` order 2, `κ = 1.8` | 4.4e-2 | 4.3e-2 | 3.3e-2 | 1 |
| **`:chebyshev` order 3, `κ = 1.8`** (default order) | **4.1e-2** | **4.0e-2** | **2.1e-2** | 2 |
| `:chebyshev` order 4, `κ = 1.8` | 4.9e-2 | 4.2e-2 | 2.5e-2 | 3 |

(polarization = `|Σ q_s(s₁ − s*)| / |Σ q_s s*|`, the quantity ε(ω) is built
from; force deviations are relative to `max|F_relaxed|` and `‖F_relaxed‖`.)

This is the number to weigh before switching a production run: ~6% on the
worst atom with the default kernel, and no systematic energy consequence
(the drift column above). It is a property of the shadow *surface*, not an
error that shrinks with tighter settings. It does vanish with the tracking
lag, and to first order in it: `E_shadow` is stationary in `ŝ` at
`ŝ = s*`, so `F_shadow(ŝ) - F_relaxed = -(∂²E_shadow/∂r∂ŝ)·(ŝ - s*) + O(‖ŝ - s*‖²)`
— the constant term is absent (that is §10 T3), the linear one is not.
The only levers are therefore a stiffer auxiliary coupling and a better
kernel.

The three columns weight the modes differently, and the difference is the
whole story of §6.1: the max-norm force deviation is dominated by the
stiff modes, which the spring kernel already runs at `κ_eff = 1.40`, close
to optimal — so the kernel buys only ~1.4× there. The polarization is
carried by the soft mode, at `κ_eff = 0.24` with the spring kernel, and
that is where the kernel is worth its two Hessian-vector products (2.6×).
Observables that average over many configurations are far less sensitive
than the pointwise force bound suggests, but the honest advice is
unchanged: cross-check any new observable against a `:scf` segment before
quoting it.

At 8×8×8 (2560 atoms) the GPU throughput is 2.25 / 4.59 / 2.61 steps/s for
`:scf` / `:spring` / `:chebyshev`, so the kernel keeps a 1.16× advantage
over the default relaxation while halving the polarization deviation. On
the CPU it does not: the step falls to 10.3 steps/s against 17.3 for
`:scf`, because each kernel order adds a full scalar-path Hessian sweep
over the neighbour list while the `:scf` iterations it is competing with
are vectorized — the same accounting as above, multiplied by the order.

## 8. Guard, fallback, and the ncg column

`max|s₁ - ŝ|` is the deviation between the shadow minimizer and the
auxiliary field. It stays small on a well-behaved trajectory; growth
means the auxiliary field is leaving the physical surface, the failure
mode that ends in the polarization catastrophe. Exceeding
`xl_max_deviation` (or a NaN) triggers a full CG relaxation at the user's
`cgtol`, reseeds `ŝ = ŝ⁻ = s*`, restarts the warmup ladder, and warns.

The `ncg` column of `log.txt` records the CG evaluations of the step, so
`ncg > 0` in an XL run always means a warmup or fallback step. The
converse does **not** hold: a reseed whose shells are already inside
`cgtol` costs zero iterations. The authoritative fallback count is the
end-of-run `@info` summary.

## 9. Deliberate restrictions

- **The stress is the shadow stress** (§4.1), not the relaxed one. Since
  Issue #151 `atoms.info[:virial]` is the exact cell-strain derivative of
  `E_shadow`, so a barostat is driven by the surface the dynamics runs
  on and `barostat ≠ :none` is supported. What remains is the same
  surface offset the forces have: measured against a tight relaxation at
  the same configurations, 1.4% of `max|W|` with the default kernel
  (0.86% with `:chebyshev`), 0.012 GPa in the pressure. Quote it as a
  property of the shadow surface, and cross-check against a `:scf`
  segment before publishing a stress the way §7 advises for any other
  observable. The `peratomstress` columns of `step_<n>.xyz` carry the
  per-atom part of the same quantity; as on the `:scf` path they omit the
  reciprocal-space contribution, which has no per-atom attribution.
- **A species with `k2 = 0` and `k4 ≠ 0` is refused.** The closed-form
  minimizer of §3 is a Newton solve seeded at `|F_pair|/k2`; the purely
  quartic root exists but this branch cannot reach it, and returning
  `s₁ = 0` would disagree with what the `:scf` path does for the same
  parameters. `xl_init!` errors instead (`k2 = 0` with `k4 = 0` is the
  ordinary shell-less species and is fine).
- **No polarization umbrella.** Its bias has no analytic HVP yet
  (`hvp_workspace` refuses it), so the combination is refused. The
  **surface term is supported** since Issue #181 gave the depolarization
  field its analytic HVP (`surface-term.md` §4.1); what it brings with it
  is a spectral condition, below.
- **Domain decomposition** is supported through the PME path only; the
  Ewald HVP has no reduction wired and refuses under DD.
- **Restart** re-seeds the auxiliary field through the ordinary warmup;
  `ŝ` is not written to the trajectory file (same treatment as the
  predictor history `:shist*`).

## 10. Independent numerical verification

- **T1 (`test/apps/test_xlbomd.jl`)**: `xl_shadow_forces!` takes the
  auxiliary field as an argument, so the force can be compared against a
  central difference of the shadow energy **at frozen `ŝ`** — the exact
  statement of §4, checked on both Coulomb paths with and without the
  quartic spring.
- **T3**: with `ŝ` set to a converged SCF solution the minimizer
  coincides with it, the correction term vanishes, and the shadow force
  and energy must reproduce the ordinary relaxed ones. No finite
  differences are involved, so this is independent of the FD machinery.
- **T2**: the closed-form minimizer is checked against the residual of
  the defining cubic, recomputed outside the kernel.
- **T9** (the §4.1 counterparts): the reported virial against a central
  difference of the shadow energy under a *cell strain* at frozen `ŝ`;
  the T3 statement for the virial (`ŝ = s*` reproduces the relaxed
  virial, no FD); and the u-only cancellation as an exact zero — a cell
  whose only interaction is the spring has no strain derivative at all,
  so the spring virial `core_forces!` reports must be cancelled to
  roundoff by the three correction terms. `hvp!`'s virial output is
  checked separately in `test/calculators/test_hvp.jl` (T6) against a
  finite difference of `info[:virial]` on both Coulomb paths, plus two
  FD-free statements: linearity in the direction, and equality between
  the `Kv` and `Kv === nothing` branches.
- **T10** (Issue #181): the same statements with a surface term in the
  stack — T1's force FD and T9(a)'s strain FD with the term switched on,
  plus the mirror of T9(d): the surface shell force must **stay** in the
  conversion term (it does contribute to `info[:virial]`, unlike the
  external field), checked by hand-assembling the virial both ways. The
  spectral refusals above are checked on a deliberately soft cell where
  `Δλ_surf` dominates.
- MPI: `npt_dd/xlbomd`, `npt_dd/xlbomd_surface` and `hvp_dd/*` pin
  DD-vs-serial parity, which is what catches a missing dipole-grid or
  dipole-direction reduction (silently wrong, never an error).

## Pinned by this note

Code sites:

- `src/Apps/XLBOMD.jl` — `xl_shadow_forces!` (§2, §4, §4.1),
  `kernel_xl_shadow_min!` (§3), `kernel_xl_virial!` (§4.1),
  `xl_propagate!` (§5), `xl_step!` (§8), `xl_init!` defaults (§6)
- `src/Apps/NPT.jl` — the umbrella guard (§9), the `xl_init!` call site,
  the fallback summary (§8)
- `src/Apps/XLBOMD.jl:_xl_check_surface_spectrum` — the two one-sided
  refusals of §6
- `src/Apps/Driver.jl` — the `:shell_dynamics` dispatch (`evaluate_forces!`
  routes the shell path to `xl_step!` when `xl !== nothing`)
- `src/Apps/ParamSpec.jl` — `_check_control_params`, the `:xl_*`
  conditional-key rules (§9)
- `src/Calculators/HVP.jl` — the DD reductions the shadow step relies on,
  and `hvp!`'s virial output (§4.1: the pair-kernel accumulation, the
  spring/intra closed forms, the reciprocal cross term)

Tests:

- `test/apps/test_xlbomd.jl` — T1–T9 (§10), including the invariance of
  the default `:scf` path
- `test/calculators/test_hvp.jl` — T6, the virial output of `hvp!` (§4.1)
- `test/gpu/test_xlbomd_gpu.jl`, `test/gpu/test_hvp_gpu.jl` — backend
  parity, virial included
- `test/mpi/_check_hvp_dd.jl`, `npt_dd/xlbomd`, `npt_dd/xlbomd_baro` —
  DD parity (both reduction regimes of §4.1) and empty ranks

Issues/PRs: #117 (SCF-free tracking; Phase 1 measurements quoted in §6),
#147 / PR #148 (the analytic HVP this scheme consumes), #149 (this
implementation), #151 (the shadow virial of §4.1), #152 (the kernel of
§6.1), #181 (the surface term's HVP and the spectral condition of §6).
