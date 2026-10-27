# Strain derivative of a group-polarisation bias

**Status**: normative. Pins the virial assembled by
`Calculators/Umbrella.jl:grad_group_p!` and the conventions its callers rely on.

## 1. Conventions

- `:lattice` `L` has **rows** equal to the cell vectors; fractional → Cartesian is
  `r = pos * L` row-wise (`src/Cores/Atoms.jl`).
- Stress is `σ = -W / V` with `W` the stored Voigt-6 virial
  (`Cores/Atoms.jl:stress`), i.e. **`W = -∂U/∂ε`**, tension positive.
- Voigt order is `[xx, yy, zz, yz, xz, xy]`.
- The strain `ε` acts on Cartesian vectors as `v → (I + ε) v`, on the cell as
  `L → L (I + ε)ᵀ`, and on the volume as `V → det(I + ε) V`.
- Group volume `V_g = V / n_g` with `n_g` the number of groups (a **constant**).

## 2. The object being differentiated

Every polarisation bias in this codebase is a function of the per-group
polarisation table alone,

```
U = U(P_g),      P_g = P_g^raw − P_g^ref(t),

P_g^raw = (1/V_g) Σ_{i∈g} [ q^c_i r_i + q^s_i (r_i + sd_i) ],
P_g^ref = (1/V_g) (M_g^frac · L),
```

where `M_g^frac` is the frozen fractional gauge moment (`group_mfrac`). This
holds for the least-squares amplitude + homogeneity bias
(`PolarizationUmbrella`), for a fixed target map, and for a path collective
variable: they differ only in the shape of `U`, never in how `P_g` depends on
the cell. Everything below therefore applies to all of them, and
`grad_group_p!` is the single place that implements it.

## 3. The three terms

Differentiating `P_g` at fixed fractional coordinates:

```
∂P_g,γ/∂ε_αβ = δ_γα (1/V_g) Σ_{i∈g}[q^c r_{i,β} + q^s (r+sd)_{i,β}]   (A)
             − δ_αβ P_g,γ                                             (B)
             − δ_γα P^ref_g,β                                          (C)
```

- **(A)** comes from `r → (I+ε) r` (and `sd → (I+ε) sd`, since the shell
  displacement is stored in Cartesian units).
- **(B)** comes from `1/V_g`, since `∂(1/V_g)/∂ε_αβ = −δ_αβ / V_g`. The constant
  `n_g` in `V_g = V/n_g` **cancels** — it must not appear in the result.
- **(C)** comes from the explicit `L` in `P_g^ref`; it enters `P_g` with a minus
  sign, hence the sign flip relative to (A).

Contracting with `∂U/∂P_g` and using `W = -∂U/∂ε`:

```
W_αβ = − Σ_g (∂U/∂P_g)_α (1/V_g) Σ_{i∈g}[…]_β        from (A)
     + δ_αβ Σ_g Σ_γ (∂U/∂P_g)_γ P_g,γ                from (B)
     + Σ_g (∂U/∂P_g)_α P^ref_g,β                     from (C)
```

Term (A) is exactly `Σ_i r_i ⊗ F_i` for the bias forces
`F_i,α = −(∂U/∂P_g)_α q_i / V_g` that `grad_group_p!` scatters, so it is
accumulated in the atom loop and needs no separate code.

**Term (B) is isotropic and involves the full contraction over γ.** A
per-component product `(∂U/∂P_g)_α P_g,α` would drop the cross terms and is not
the derivative of anything.

## 4. Symmetrisation

`U` depends on the *vector* `P_g` and on lab-frame data (the bias direction `ê`,
the target map, the mode pattern). It is therefore **not rotationally
invariant**, and `∂U/∂ε` has a non-zero antisymmetric part — unlike a pair
potential, where rotational invariance makes the stress symmetric identically.

The stored Voigt-6 virial keeps only the symmetric part, which is the correct
choice: the barostats evolve a symmetric 6-DOF cell
(`docs/src/theory/mttk-cell-dof.md`), so the antisymmetric part does no work on
any cell degree of freedom. It is not lost physics — it is the torque the bias
exerts on the atoms, and it is fully present in the atomic forces.

Consequence for testing: a finite-difference check of this virial must perturb
`(α,β)` and `(β,α)` **together** (`symmetrize = true` in
`assert_virial_matches_finite_diff`). A one-sided perturbation measures
`∂U/∂ε_αβ` including the antisymmetric part and will disagree with the stored
value in the shear components.

## 5. Independent verification

Central-difference `−∂U/∂ε` under the deformation of §1, at a rattled
3×4×5 two-atoms-per-group fixture with independent core and shell
displacements, agrees with the assembled virial for σ = 0, for σ > 0 with a
cosine mode, and for a flat-bottom threshold. The regression test asserts
`rtol = 1e-4` at `h = 1e-5`; a standalone check at `h = 1e-6` reproduces the
assembled virial to a relative 6e-8, which is the truncation floor of the
central difference rather than a discrepancy.

The same check run against the pre-fix implementation misses the diagonal by
O(1) *relative* — with the fixture above the xx component came out
`−0.42` against a true `+0.28` — while the shear components already agreed to
1e-8. A diagonal-only, sign-flipped error of that size is the signature of
term (B): it is the only term that contributes to the diagonal alone.

## Pinned by this note

- `Calculators/Umbrella.jl:grad_group_p!` — the atom-loop term (A), the
  isotropic term (B) (full contraction, no `1/n_g`), and the `Pg_ref` term (C).
- `test/calculators/test_umbrella.jl` — "PolarizationUmbrella virial = -∂U/∂ε
  (strain FD)" (breaks if any of the three terms is wrong) and "Fractional →
  Cartesian convention (non-orthorhombic cell)" (breaks if §1 is violated).
- `test/physical_consistency.jl:assert_virial_matches_finite_diff` — the
  `symmetrize` option and the reason it exists (§4).

## Related

- Issue #163 (term (B) was per-component and carried a spurious `1/n_g`),
  Issue #164 (the §1 convention was transposed in the polarisation path).
