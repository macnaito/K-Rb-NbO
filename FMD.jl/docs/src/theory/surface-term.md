# Anisotropic surface term: depolarization field of a finite external medium

**Status**: normative. Pins the tensor construction, the energy/force/virial
assembled by `Calculators/SurfaceTerm.jl:surface_term!`, and the lab-frame
convention for the embedding data.

## 1. Conventions

- `:lattice` `L` has **rows** equal to the cell vectors; fractional →
  Cartesian is `r = pos * L` row-wise. For a *column* vector of fractional
  moments the same map reads `M = Lᵀ · m_frac` (`calc_dipole_phys!`).
- Stress is `σ = −W / V` with `W` the stored Voigt-6 virial, i.e.
  **`W = −∂U/∂ε`**, tension positive. Voigt order `[xx, yy, zz, yz, xz, xy]`.
- The strain `ε` acts on Cartesian vectors as `v → (I + ε) v`, on the cell
  as `L → L (I + ε)ᵀ`, on the volume as `V → det(I + ε) V`. Shell
  displacements are Cartesian and co-transform.
- Gaussian (atomic) units: the Coulomb kernel is `q₁q₂/r`, so the continuum
  formulas below carry their textbook `2π/V` prefactors unchanged.
- **The external medium is lab-frame data.** `ε_out` and the embedding
  shape `A` describe the world outside the cell; they do **not** co-rotate
  or co-deform with the lattice, and therefore contribute no cell
  derivative. (The alternative — a medium glued to the cell axes — would
  add a `∂T/∂ε` term; it is not what "an external medium" means.)

## 2. The tensors

A uniformly polarised ellipsoid `Ω = {A·x : |x| < 1}` (symmetric
positive-definite `A`; a sphere is `A ∝ I`) embedded in an infinite
anisotropic dielectric `ε_out` feels the average reaction field
`Ē = −4π L_Ω · Q` with the **depolarization tensor**

```
L_Ω = (1/4π) ∫_{S²} (A⁻¹n̂)⊗(A⁻¹n̂) / (A⁻¹n̂ · ε_out · A⁻¹n̂) dω .
```

This is the angular part of the k-space solution of the anisotropic Poisson
equation; the radial integral is fixed by Parseval to `1/4π`, which is why a
plain average over uniform directions evaluates it (the overall scale of `A`
cancels). Derivation: research note *異方誘電率媒質中の表面項* (2026-06-21).

`Q` is the *effective* polarisation; eliminating it in favour of the real
excess polarisation `δP` gives the **real-polarisation tensor**

```
T_Ω = [I + L_Ω (ε_in − ε_out)]⁻¹ L_Ω ,        Ē = −4π T_Ω · δP .
```

For explicit MD charges `ε_in = I`: the cell's own dielectric response is
carried by the charges themselves (shell relaxation against the field below
IS the screening), so a background `ε_in ≠ I` would double-count it.

The surface energy of the `k = 0` polarisation mode is then

```
U = (2π/V) δM · T_Ω · δM ,      δM = M − M_ref(t) ,
M = Σ_i [q^c_i r_i + q^s_i (r_i + sd_i)] ,    M_ref(t) = Lᵀ(t) · m_frac .
```

`m_frac` is the total fractional dipole of the declared gauge frame (the
same gauge machinery as the polarisation umbrellas): `M_ref` co-transforms
with the cell, so pure cell deformation with atoms static in fractional
space keeps `δM = 0`.

Exact identities used as construction-time gates (violations are errors,
never warnings):

- `Tr(ε_out · L_Ω) = 1` for every shape. For the direction-average
  quadrature form this holds **identically** for any direction set (each
  direction contributes `(v·εv)/(v·εv)`), so the gate pins the
  normalisation contract (a dropped `1/n`) and cannot detect an
  unconverged quadrature — convergence is enforced by the doubling loop,
  and accuracy is verified independently against the closed forms of §5.
- `T_Ω⁻¹ = L_Ω⁻¹ + (ε_in − ε_out)` is symmetric, so `T_Ω` is symmetric.
- `T_Ω` is positive definite (`U ≥ 0`: a depolarization field costs
  energy). `L_Ω` is positive definite by inspection of the integrand.

Limits: isotropic sphere `L = I/(3ε_out)`, `T = I/(2ε_out + 1)` — the
textbook Ewald surface term `2π/((2ε_out+1)V)·|δM|²`; `ε_out → ∞` gives
`T → 0`, the conductor (tin-foil) boundary the Ewald/PME solvers already
implement — which is why *omitting* the term is that boundary condition;
`ε_out = 1`, sphere: the vacuum surface term `2π/(3V)·|δM|²`.

## 3. Forces

`∂M_γ/∂x_{i,α} = q_i δ_γα` for cores and `∂M_γ/∂sd_{i,α} = q^s_i δ_γα` for
shells, so with

```
g = ∂U/∂M = (4π/V) T_Ω · δM
```

the force is the **uniform per-charge field** `F_i = −q_i g` on cores and
`−q^s_i g` on shells — an `ExternalField` with `E = −g`, but carrying the
energy and virial that induced field actually has. On a charge-neutral
system `Σ_i F_i = 0` and the term is rigid-translation invariant; on a
charged system `M` itself is origin-dependent, which is why the constructor
refuses charged systems.

## 4. Virial

Differentiating `U(ε) = (2π/V(ε)) δM(ε)·T·δM(ε)` with `δM → (I+ε)δM`
(both `M` and `M_ref` transform identically — `r`, `sd` and `Lᵀ m_frac` all
pick up `(I+ε)`), `V → (1+Tr ε)V`, and `T` constant (§1):

```
W = −∂U/∂ε = U·I − sym(g ⊗ δM) ,      sym(a⊗b) = (a⊗b + b⊗a)/2 .
```

The `−sym(g⊗δM)` piece is exactly the atom-loop `Σ_i r_i ⊗ F_i` plus the
`M_ref` cell-covariance term: because the force is uniform per charge, the
atom loop telescopes to `−sym(g ⊗ M)` in closed form, and the reference
term adds `+sym(g ⊗ M_ref)`. `U·I` is the explicit `1/V` derivative. Trace
check: `Tr W = −2U + 3U = U`, consistent with `U ∝ λ⁻¹` under isotropic
dilation `λ` of all Cartesian data at fixed fractional coordinates.

Like every polarisation bias, `U` couples the vector `M` to lab-frame
tensors and is **not rotationally invariant**: `∂U/∂ε` has an antisymmetric
part that the symmetric Voigt-6 virial correctly does not carry (it is the
torque on the atoms, present in the forces; the barostats evolve a
symmetric 6-DOF cell — see `polarization-bias-virial.md` §4). FD checks
must therefore use `symmetrize = true`.

Under domain decomposition, `M` is Allreduced (inside `calcdipole!`), after
which `U`, `g` and `W` are replicated on every rank — they are added
directly, **not** through `add_reduced_ev!` (which would multiply them by
the rank count). The force scatter is rank-local.

## 4.1 Second derivatives: the analytic HVP (Issue #181)

Shadow XL-BOMD needs the shell-block and mixed Hessians of every term in
the stack, and the strain derivative of the virial along a shell
direction (`xlbomd-shell.md` §4.1). `U` depends on the positions through
`M` alone, so all three are rank-3 closed forms. With a direction `v` in
shell-displacement space and the **single reduction**

```
S = Σ_j q^s_j v_j        (a 3-vector; one Allreduce under DD)
```

the chain rule on `∂M_γ/∂x_{s,i,α} = q^s_i δ_γα`,
`∂M_γ/∂x_{c,i,α} = q^c_i δ_γα` gives

```
c    ≡ D_v g = (4π/V)·T_Ω·S
Jv_i = −(∂²U/∂x_s∂x_s·v)_i = −q^s_i·c
Kv_i = −(∂²U/∂x_c∂x_s·v)_i = −q^c_i·c
```

i.e. one 3-vector reduction and a uniform per-charge scatter — no FFT, no
neighbour pass, `O(N)`. The shell block is `H_surf = (4π/V)·(q^s q^sᵀ)⊗T`,
manifestly symmetric and positive semi-definite (`T ≻ 0`), so it cannot
break the SPD assumption `cg_hessian_free!` relies on; its null space is
every direction with `S = 0`, which is the statement that a shell motion
moving no dipole does not feel the depolarization field.

Differentiating the virial of §4 along the same direction (`T` constant,
`δM` and `g` linear in the displacements):

```
D_v W = (g·S)·I − sym(c⊗δM) − sym(g⊗S).
```

**Reduction regimes.** `S` is an owned partial sum and is Allreduced;
`δM` is already replicated (the cache freezes it at construction, through
`calcdipole`). Everything downstream is therefore replicated: the `Jv`/`Kv`
scatter is rank-local, and `D_v W` is a whole-system quantity added
directly — it rides `hvp!`'s **return value**, not its per-atom `Wv`, the
same regime as the reciprocal-space grid virial. Consequently the
`peratomstress` columns carry neither `W` nor `D_v W` of this term,
exactly as they carry no reciprocal one. (On the shadow path they are not
entirely free of it: the conversion terms `−sym(F_s⊗s₁)` and
`−sym(Jv⊗ŝ)` of `xlbomd-shell.md` §4.1 are per-atom and their `F_s`/`Jv`
do include the surface block. The whole-system total is right either way;
it is the per-atom split that has no unique answer here.)

**Spectral consequence.** `H_ss` gains a positive rank-3 block, so
`λmax(A⁻¹H_ss)` (`A` = spring Hessian) can only grow. The amount is
closed-form: with `w_i = q^s_i/√k2_i`, the symmetrised block
`A^{-1/2}H_surf A^{-1/2} = (4π/V)·(w wᵀ)⊗T` has non-zero eigenvalues
`(4π/V)|w|²·λ_k(T)`, so

```
Δλ_surf = (4π/V)·(Σ_i q^s_i²/k2_i)·λmax(T_Ω)
```

is a *lower bound* on the full `λmax` (Weyl, since the rest of `H_ss` is
positive semi-definite at a stable shell configuration). `xl_init!` uses
it for two one-sided refusals, one per kernel — a `:chebyshev`
`xl_kernel_spectrum` upper bound below `Δλ_surf`, and a `:spring`
`xl_kappa` with `κ·Δλ_surf ≥ 4` (only that kernel's Verlet coupling is
`λ` itself) — which are the conditions that are wrong with certainty; see
`xlbomd-shell.md` §6 and §6.1. Note the quantity is
intensive (`Σ q_s²/k2` grows with `N`, `4π/V` shrinks with it), so it does
not vanish in the thermodynamic limit and does not blow up either.

## 5. Independent verification

- The quadrature is verified against closed forms derived independently of
  it: the isotropic sphere, both branches of the uniaxial-sphere `L_∥`
  (arctan and artanh), the classical prolate-spheroid depolarization
  factors in vacuum, `Tr(ε_out·L) = 1` for a rotated ellipsoid in a rotated
  medium, and rotation covariance `L(RεRᵀ, RARᵀ) = R·L·Rᵀ`. Measured
  agreement at the default quadrature tolerance: 1e-9…1e-12 absolute.
- Energy: the `ε_out = 1` sphere reproduces `2π/(3V)|δM|²` on a hand-built
  polarised supercell, including the polarisation-quantum jump when one
  atom is moved by a full lattice vector (the branch is deliberately
  continuous — `:pos` is never wrapped).
- Forces, shell forces and the virial (with `symmetrize = true`) agree with
  central differences on a rattled fixture under a fully anisotropic
  `ε_out` and a rotated ellipsoid.
- Screening: for one core–shell pair the term renormalises the shell
  spring to `k₂·I + (4π/V)·q_s²·T_Ω`; the shell force vanishes at the
  analytically renormalised equilibrium under an external field, and the
  anisotropy of the response matches `T_Ω`.
- The HVP of §4.1 three ways: against the closed form above written out
  independently in the test (the *difference* between the products with
  and without the term, so nothing else can absorb an error); against the
  finite-difference oracles `fd_jvp!` / `fd_dFdu_dirderiv!` and a central
  difference of `info[:virial]`; and by the null-space statement — a
  direction with `Σ q^s_j v_j = 0` must leave every output bit-identical.
  DD parity (`hvp_dd/surface`, `npt_dd/xlbomd_surface`) covers the two
  reduction regimes, which fail silently rather than loudly.

## Pinned by this note

- `Calculators/SurfaceTerm.jl:depolarization_tensor` — the `1/4π`
  normalisation absorbed into the direction average, the trace gate.
- `Calculators/SurfaceTerm.jl:_surf_T_from_L` — the `ε_in = I` choice, the
  symmetry/SPD gates.
- `Calculators/SurfaceTerm.jl:surface_term!` — `g = (4π/V)T·δM`, the
  closed-form virial `U·I − sym(g⊗δM)`, the replicated (direct-add)
  reduction regime, and the lab-frame-`T` convention (no `∂T/∂ε` term).
- `Calculators/SurfaceTerm.jl:surface_hvp!` / `surface_hvp_cache` — §4.1:
  the single `S` reduction, the per-charge scatter, `D_v W` on the return
  value, and the frozen `δM`.
- `Calculators/SurfaceTerm.jl:surface_spectral_bump` — `Δλ_surf` and why
  it is a lower bound on `λmax`.
- `test/calculators/test_surface_term.jl` — every check in §5 except the
  HVP ones, which live in `test/calculators/test_hvp.jl` (T7).
- `test/mpi/_check_surface_term.jl` — DD parity of the replicated adds;
  `test/mpi/_check_hvp_dd.jl` (`surface`, `surface_tiny`) — the same for
  §4.1's two regimes, empty ranks included.

## Related

- `polarization-bias-virial.md` — the per-group analogue; §4 there is the
  symmetrisation argument this note inherits.
- `xlbomd-shell.md` — §4.1 (the assembly that consumes `D_v W`) and §6
  (the spectral bound `Δλ_surf` feeds).
- Issue #179, #181 (the analytic HVP of §4.1); research notes
  *異方誘電率媒質中の表面項* (2026-06-21) and
  *DWによる分極揺らぎと反電場の影響* (2026-04-21, the intended
  application).
