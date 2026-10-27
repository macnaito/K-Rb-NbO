# Theory notes

Derivations and design decisions that fix **numerical constants,
conventions, or algorithm choices** in the code but are too long for a
source comment. Source comments state *what* a constant is and link
here; these notes carry the full argument, so that a future change to
the code can re-derive rather than cargo-cult.

Two kinds of notes live here:

- **Normative derivations** — project-specific theory (or a re-derivation
  of published theory under this codebase's conventions) whose result is
  baked into the code. Each note lists the exact code sites it pins and
  the tests that enforce it. If code and note disagree, one of them is
  wrong — fix that before merging.
- **Adoption studies** — comparisons of published methods recording which
  one was adopted and why (and what was rejected). These justify
  interface-level choices; they do not pin constants.

House rules for a new note:

1. State the codebase conventions up front (stress sign, lattice
   rows-vs-columns, DOF counting) — most disagreements with papers are
   convention mismatches, not errors.
2. End with a **Pinned by this note** section: code sites (file/function)
   and the tests that fail if the result is violated.
3. Every nontrivial constant needs a numerical verification that is
   independent of the implementation it pins (standalone MC, analytic
   limit, cross-code comparison) — "the code agrees with itself" is not
   evidence.
4. Cross-link the Issues/PRs where the question was settled.

## Notes

- [SCR barostat: 6-DOF derivation of the measure correction](scr-barostat.md)
  — normative; pins the `−2kT/V` drift term and the `Gamma(N)` test
  expectations (Issues #140, #132; PR #141).
- [MTTK cell thermostat: 6 vs 9 degrees of freedom](mttk-cell-dof.md)
  — normative; pins the `6kT` cell-NH target and `Q_cell` (Issue #132;
  PRs #131, #137).
- [Shadow XL-BOMD for the core-shell model](xlbomd-shell.md)
  — normative; pins the shadow functional, its exact gradient and its
  exact cell-strain derivative (including the spring/intra cancellation
  lemma in both), the closed-form shadow minimum, the conserved quantity,
  the auxiliary-field kernel, and the measured κ stability limit
  (Issues #149, #151, #152, #117).
- [Analytic Hessian-vector products for the core-shell model](coreshell-analytic-hvp.md)
  — normative; pins the (x_c, x_s) variable convention, every per-term
  Hessian block, the BSPORDER = 4 retention data, and the CG
  breakdown-guard convention (Issues #147, #117).
- [Strain derivative of a group-polarisation bias](polarization-bias-virial.md)
  — normative; pins the three-term decomposition assembled by
  `grad_group_p!` and the symmetrisation convention that follows from the
  bias not being rotationally invariant (Issues #163, #164; PRs #165, #166).
- [Anisotropic surface term](surface-term.md)
  — normative; pins the depolarization tensors `L_Ω`/`T_Ω` (quadrature
  normalisation, `ε_in = I`, trace/SPD gates), the closed-form virial
  `U·I − sym(g⊗δM)`, and the lab-frame convention for the external medium
  (Issue #179).
