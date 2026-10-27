using LinearAlgebra
using Random
using Distributions: Chisq

# ---------------------------------------------------------------------------
#  MD temperature / pressure control (Issue #116 / #136)
#
#  Thermostats and barostats are two orthogonal dispatch families selected
#  by params[:thermostat] / params[:barostat]:
#
#      Thermostat: NoThermostat (:none) | NoseHoover (:nh) | CSVR (:csvr)
#      Barostat:   NoBarostat  (:none)  | MTTK       (:mttk) | SCR (:scr)
#
#  npt! drives them through exactly three entry points:
#
#      update!(atoms, th, baro, c, arrtyp)     twice per MD step (symmetric)
#      scale_cell!(baro, atoms, c, arrtyp)     once per MD step (lattice)
#      conserved_energy(th, baro, atoms, c)    diagnostic H' for log.txt
#
#  `c` is the per-step control context from _control_context (ramped target
#  temperature, dt, global DOF count, target stress); `arrtyp` is the device
#  array type resolved once by _arrtyp in npt!.
#
#  Unlike the pre-#136 fused MTTK update, the particle thermostat no longer
#  sees the cell momenta: the MTTK barostat carries its OWN single-NH cell
#  thermostat (ζ_cell), per the MTTK 1996 (Mol. Phys. 87, 1117) §2.5
#  recommendation to thermostat particles and barostat independently (the
#  cell momenta are slow degrees of freedom whose weight in a shared
#  friction force is only ~6kT/3NkT — they were effectively unthermostatted).
#  This is the same split as LAMMPS fix npt (tchain / pchain).
#
#  All control state lives in host Float64 (rank-replicated under DD): every
#  input is an Allreduced scalar (kinetic_energy, stress, volume), so the
#  lattice update stays bit-identical across ranks (see Cores/GhostsDD.jl).
# ---------------------------------------------------------------------------

abstract type Thermostat end
abstract type Barostat end

"""
    NoThermostat()

`params[:thermostat] = :none` — no particle thermostat (NVE when combined
with [`NoBarostat`](@ref)). Every control hook is a no-op.
"""
struct NoThermostat <: Thermostat end

"""
    NoseHoover(zeta, work, tau)

Single Nosé–Hoover particle thermostat (`params[:thermostat] = :nh`).

- `zeta` — friction ζ; `dζ/dt = (2K − (3N−3)kT)/Q_T`, `Q_T = (3N−3)·kT·τ_t²`.
- `work` — accumulated thermostat energy `∫(3N−3)·kT·ζ dt`, the `g·kT·ξ`
  term of the conserved quantity. An energy accumulator (not a bare ξ
  integral) so that H′ stays well-defined under a temperature ramp.
- `tau`  — time constant τ_t (`params[:tau_t]`, a.u.).

Unlike the pre-#136 fused update, the force `G` contains **no cell terms**:
cell momenta are thermostatted inside [`MTTK`](@ref).
"""
mutable struct NoseHoover <: Thermostat
    zeta::Float64
    work::Float64
    tau::Float64
end

"""
    CSVR(tau, rng, work)

Canonical sampling through velocity rescaling (Bussi–Donadio–Parrinello,
J. Chem. Phys. 126, 014101 (2007)); `params[:thermostat] = :csvr`.

A stochastic scalar rescale of all velocities whose stationary kinetic-
energy distribution is EXACTLY the canonical one — no chains needed for
ergodicity (the noise supplies it). Like Nosé–Hoover it preserves the
zero COM momentum and any bond constraints (single global factor).

- `tau`  — relaxation time τ_t (`params[:tau_t]`, a.u.). The propagator is
  the exact solution of the paper's Eq. (7), so any application interval
  composes exactly — we apply it in the `_thermo_half!` slots (Δt = dt/4,
  four times per MD step), mirroring the NH cadence.
- `rng`  — dedicated host RNG, derived from `params[:seed]` with a
  CSVR-specific salt (see [`_control_rng`](@ref)): rank-replicated under
  DD and independent of the initial-velocity stream. Note the χ² draw
  goes through Distributions.jl, whose sampler internals may change
  between releases — seed reproducibility holds per Manifest, not across
  dependency upgrades.
- `work` — accumulated kinetic energy REMOVED by the rescalings,
  `Σ K·(1−α²)`; `K + U + work` is the paper's conserved effective energy
  H̃ (Eq. 15), logged via [`conserved_energy`](@ref).
"""
mutable struct CSVR <: Thermostat
    tau::Float64
    rng::Random.Xoshiro
    work::Float64
end

"""
    NoBarostat()

`params[:barostat] = :none` — fixed cell (NVT/NVE). The lattice is never
touched; `scale_cell!` and `_baro_half!` are no-ops.
"""
struct NoBarostat <: Barostat end

"""
    SCR(tau, kappa, rng, P0, Sigma)

Anisotropic stochastic cell rescaling (`params[:barostat] = :scr`;
Bernetti–Bussi, J. Chem. Phys. 153, 114107 (2020); fully anisotropic
formulation Del Tatto, arXiv:2111.06402 / Del Tatto–Raiteri–Bernetti–
Bussi, Appl. Sci. 12, 1139 (2022)).

A FIRST-ORDER (overdamped) stochastic barostat: no cell momentum, hence
no MTTK-style cell ringing — the spectral contamination is confined to
frequencies ≲ 1/(2π·τ_eff), complementary to MTTK's narrow high-lying
cell resonance (Issue #116). The stationary ensemble is the correct
anisotropic NST distribution for this codebase's frozen-COM convention —
its volume marginal V^(N−1)·e^(−β·P₀·V) = Gamma(N, kT/P₀), identical to
the MTTK one, is pinned by the ideal-gas tests (test_scr.jl; derivation
in docs/src/theory/scr-barostat.md) — regardless of `kappa`: the
compressibility estimate only sets the effective relaxation time
τ_eff = τ_p·κ_true/κ_input (exactness holds while μ stays perturbative;
`scale_cell!` guards |μ−I| < 0.25 against a grossly mis-scaled κ input).

Once per MD step (Euler integrator, thesis Eq. 2.27/2.28 restricted to
the symmetric 6-DOF strain space) the symmetric rescaling matrix

    μ = I − (κ_T·dt)/(3τ_p)·[(P₀I − P_int) − (2kT/V)·I + (1/V)·hΣhᵀ]
        + √(2κ_T·kT·dt/(3V·τ_p))·R_sym

is applied as `h ← μh` (columns = box vectors; in this codebase's
rows-are-lattice-vectors convention `lattice ← lattice·μ`, exact because
μ is symmetric), `v ← v·μ⁻¹` (momenta-scaled formulation, thesis
Eq. 2.29 — P_int then uses the INSTANTANEOUS kinetic stress, which
`stress(atoms, true)` provides), and fractional positions untouched.
Box rotations are eliminated to first order by symmetrizing μ (thesis
§2.6, the MTK-style strategy; products of symmetric matrices can still
accumulate a diffusive rotation at the commutator scale ~amp² per step —
negligible, and shared with the symmetric-ζ_b MTTK convention): `R_sym`
has N(0,1) diagonal and N(0,1/2) off-diagonal entries (the symmetrized
Wiener tensor (W+Wᵀ)/2).

- `tau`   — τ_p (`params[:tau_p]`, a.u.).
- `kappa` — isothermal-compressibility estimate κ_T (`params[:compressibility]`,
  inverse pressure a.u.; `UNITS[:GPa]/B_GPa` for a bulk modulus in GPa).
- `rng`   — replicated host RNG (salt `_SCR_SALT`), 6 draws per step,
  unconditional — rank-uniform under DD.
- `P0`, `Sigma` — hydrostatic target `P₀ = −tr(σ_ext)/3` and the
  Parrinello–Rahman strain coupling `Σ = V₀·h₀⁻¹(S_ext − P₀I)h₀⁻ᵀ` of the
  deviatoric target (thesis Eq. 2.19/2.20), frozen against the INITIAL
  cell h₀ by [`_init_barostat!`](@ref). Σ = 0 for hydrostatic targets.

There is NO conserved quantity for the Euler-integrated SCR (thesis
§2.4: the effective energy is not well-defined for this integrator);
the `conserved` log column reports the enthalpy-like K + U + P₀V
diagnostic instead — stationary in equilibrium, not an invariant.
"""
mutable struct SCR <: Barostat
    tau::Float64
    kappa::Float64
    rng::Random.Xoshiro
    P0::Float64
    Sigma::Matrix{Float64}
end

"""
    MTTK(zeta_b, zeta_cell, work_cell, work_ext, tau)

Martyna–Tobias–Tuckerman–Klein flexible-cell barostat
(`params[:barostat] = :mttk`) with an internal single-NH cell thermostat.

- `zeta_b`    — `3×3` cell strain rate ε̇ (host `Float64`; the MPI-replicated
  invariant). Symmetric by construction: symmetric init, symmetric force
  (stress tensors + isotropic MTK term), scalar damping. The symmetry
  carries TWO contracts: the 6-DOF count below, and the lattice update
  `lattice ← lattice·exp(ε̇dt)` (rows = lattice vectors), which matches
  the column-convention `h′ = exp(ε̇dt)·h` only through
  `exp(ε̇dt)ᵀ = exp(ε̇dt)` — an antisymmetric (cell-rotation) component
  must never be introduced without revisiting that line.
- `zeta_cell` — friction of the internal cell thermostat;
  `dζ_cell/dt = (2K_cell − 6kT)/Q_cell` with `K_cell = Q_b/2·Σζ_b²`,
  `Q_cell = 6·kT·τ_p²`. The count is **6** (not d² = 9): `zeta_b` is
  constrained to the symmetric subspace, which holds 6 independent DOF
  (off-diagonals appear twice in Σζ_b², matching ⟨2K_cell⟩ = 6kT), cf.
  arXiv:2603.24061 on MTK barostats with restricted cell DOF.
- `work_cell` — accumulated `∫6·kT·ζ_cell dt` (cell-thermostat H′ term).
- `work_ext`  — accumulated external-stress work `−∫V·σ_ext:dε`; reduces to
  `+P·ΔV` (enthalpy) for a hydrostatic target `σ_ext = −P·I`.
- `tau`       — time constant τ_p (`params[:tau_p]`, a.u.); sets both the
  barostat mass `Q_b = N·kT·τ_p²` and the cell-thermostat mass.

The conserved-quantity contribution is
`Q_b/2·Σζ_b² + Q_cell/2·ζ_cell² + work_cell + work_ext`.
"""
mutable struct MTTK <: Barostat
    zeta_b::Matrix{Float64}
    zeta_cell::Float64
    work_cell::Float64
    work_ext::Float64
    tau::Float64
end

# --------------------------------------------------------------------------
# Construction from params
# --------------------------------------------------------------------------

# Control-noise RNG salts. Compile-time constants (NOT hash(): its value
# is not stable across Julia versions, which would silently break seed
# reproducibility of archived trajectories). One salt per stochastic
# control so the streams are mutually independent and independent of the
# raw Xoshiro(seed) used for the initial velocities (NPT.jl).
const _CSVR_SALT = 0x43535652_c001b075  # "CSVR"
const _SCR_SALT  = 0x53435200_ce11ba70  # "SCR"

"""
    _control_rng(seed::Integer, salt::UInt64) -> Xoshiro

Dedicated RNG stream for a stochastic control, derived from
`params[:seed]` and a per-consumer salt. Rank-replicated by construction
under DD: every rank builds the identical state and every draw site is
unconditional and rank-uniform (never draw inside a rank-dependent
branch — the scale factors derived from these draws feed the lattice /
velocities, whose cross-rank bit-identity GhostsDD.jl verifies).
"""
_control_rng(seed::Integer, salt::UInt64) = Random.Xoshiro(rem(seed, UInt64) ⊻ salt)

"""
    _build_thermostat(params) -> Thermostat

Resolve `params[:thermostat]` (default `:nh`) into a thermostat struct.
Errors loudly on unknown values and on a missing time constant — direct
`npt!` callers bypass `validate_params`, so this is the last line of
defense (same pattern as [`_arrtyp`](@ref)).
"""
function _build_thermostat(params)
    kind = get(params, :thermostat, :nh)
    if kind === :nh
        haskey(params, :tau_t) ||
            error("npt!: thermostat = :nh requires params[:tau_t]")
        return NoseHoover(0.0, 0.0, Float64(params[:tau_t]))
    elseif kind === :csvr
        haskey(params, :tau_t) ||
            error("npt!: thermostat = :csvr requires params[:tau_t]")
        haskey(params, :seed) && params[:seed] isa Integer ||
            error("npt!: thermostat = :csvr requires an Integer params[:seed] " *
                  "(the rescaling noise is seeded for reproducibility)")
        get(params, :target_temperature, 0.0) >= 1.0 ||
            error("npt!: thermostat = :csvr requires target_temperature >= 1 K — " *
                  "the shared kT floor (max(T, 1 K)) would silently thermostat " *
                  "any lower target to 1 K")
        return CSVR(Float64(params[:tau_t]),
                    _control_rng(params[:seed], _CSVR_SALT), 0.0)
    end
    kind === :none && return NoThermostat()
    error("npt!: params[:thermostat] must be :nh, :csvr or :none (got $(repr(kind)))")
end

"""
    _build_barostat(params) -> Barostat

Resolve `params[:barostat]` (default `:mttk`) into a barostat struct.
See [`_build_thermostat`](@ref) for the error contract.
"""
function _build_barostat(params)
    kind = get(params, :barostat, :mttk)
    if kind === :mttk
        haskey(params, :tau_p) ||
            error("npt!: barostat = :mttk requires params[:tau_p]")
        haskey(params, :target_stress) ||
            error("npt!: barostat = :mttk requires params[:target_stress]")
        return MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(params[:tau_p]))
    elseif kind === :scr
        haskey(params, :tau_p) ||
            error("npt!: barostat = :scr requires params[:tau_p]")
        haskey(params, :target_stress) ||
            error("npt!: barostat = :scr requires params[:target_stress]")
        haskey(params, :compressibility) && params[:compressibility] > 0 ||
            error("npt!: barostat = :scr requires params[:compressibility] > 0 " *
                  "(isothermal compressibility estimate, inverse-pressure a.u.; " *
                  "it only sets the effective relaxation time, so a rough " *
                  "estimate is fine)")
        haskey(params, :seed) && params[:seed] isa Integer ||
            error("npt!: barostat = :scr requires an Integer params[:seed] " *
                  "(the cell noise is seeded for reproducibility)")
        get(params, :thermostat, :nh) === :none &&
            error("npt!: barostat = :scr requires a thermostat — an " *
                  "unthermostatted overdamped stochastic barostat samples no " *
                  "well-defined ensemble (its noise strength assumes a " *
                  "thermalized system)")
        # P0 / Sigma are frozen against the run's initial cell by
        # _init_barostat! (needs atoms), zeros until then.
        return SCR(Float64(params[:tau_p]), Float64(params[:compressibility]),
                   _control_rng(params[:seed], _SCR_SALT), 0.0, zeros(3, 3))
    end
    kind === :none && return NoBarostat()
    error("npt!: params[:barostat] must be :mttk, :scr or :none (got $(repr(kind)))")
end

"""
    _init_barostat!(baro, atoms, tstress)

One-time barostat initialization against the run's starting frame,
called by `npt!` right before the MD loop. [`SCR`](@ref): freeze the
hydrostatic split of the target stress and the Parrinello–Rahman strain
coupling `Σ = V₀·h₀⁻¹(S_ext − P₀I)h₀⁻ᵀ` against the initial cell h₀
(thesis Eq. 2.19). No-op for the other barostats.
"""
_init_barostat!(baro::Barostat, atoms, tstress) = nothing

function _init_barostat!(b::SCR, atoms, tstress)
    b.P0 = -tr(tstress) / 3
    # Pressure-tensor convention: S_ext = −σ_ext; the deviatoric part
    # S_ext − P₀I = −(σ_ext − tr(σ_ext)/3·I).
    Sdev = -(tstress - tr(tstress)/3 * I)
    h0 = Matrix(transpose(Array(atoms.info[:lattice])))  # columns = box vectors
    Sig = volume(atoms) .* (h0 \ (Sdev / transpose(h0)))
    # Symmetric analytically (h₀⁻¹·S·h₀⁻ᵀ, S symmetric); the two
    # triangular solves round differently, so symmetrize the stored Σ.
    b.Sigma = (Sig + transpose(Sig)) ./ 2
    return nothing
end

# --------------------------------------------------------------------------
# Per-step control context
# --------------------------------------------------------------------------

"""
    _target_temperature(params, step) -> Float64

Target temperature (Kelvin) at `step`: `params[:target_temperature]`,
ramped linearly to `params[:final_temperature]` over `params[:nsteps]`
when a positive final temperature is provided.
"""
function _target_temperature(params, step)
    if haskey(params, :final_temperature) && params[:final_temperature] > 0.0
        tdiff = (params[:final_temperature] - params[:target_temperature]) *
                (step / params[:nsteps])
    else
        tdiff = 0.0
    end
    return params[:target_temperature] + tdiff
end

"""
    _control_context(params, step, N, tstress) -> NamedTuple

Per-step control context shared by every thermostat/barostat operator:
`dt` (timestep), `kT` (ramped target temperature in Hartree, floored at
1 K — the historical guard), `N` (whole-system atom count) and `tstress`
(host `3×3` target stress). `N` and `tstress` are run constants hoisted
out of the MD loop by `npt!`. Note the 1 K floor enters not only the
extended-variable masses (which must stay positive) but also the
equipartition targets `(3N−3)kT` / `6kT` — a `target_temperature = 0`
run is effectively thermostatted at 1 K.
"""
_control_context(params, step, N, tstress) =
    (; dt = Float64(params[:timestep]),
       kT = max(_target_temperature(params, step), 1.0) / UNITS[:Kelvin],
       N, tstress)

# Extended-variable masses (recomputed per call from the ramped kT, matching
# the historical behavior; see docstrings of the structs for the formulas).
_mass_t(th::NoseHoover, c) = (3c.N - 3) * c.kT * th.tau^2
_mass_b(b::MTTK, c) = c.N * c.kT * b.tau^2
_mass_cell(b::MTTK, c) = 6 * c.kT * b.tau^2

# --------------------------------------------------------------------------
# Voigt / MTK helpers
# --------------------------------------------------------------------------

"""
    voigt2matrix(v::AbstractVector) -> Matrix

Expand a Voigt-6 vector `[σxx, σyy, σzz, σyz, σxz, σxy]` (in that order)
into the symmetric `3×3` tensor representation used internally for stress.
"""
function voigt2matrix(vector::Vector)
    v1, v2, v3, v4, v5, v6 = vector
    return [
        v1  v6  v5;
        v6  v2  v4;
        v5  v4  v3
    ]
end

"""
    _cell_projector(mode::Symbol, A::AbstractMatrix) -> Matrix
    _cell_ndof(mode::Symbol) -> Int

Constrain a symmetric cell-space tensor (a strain, a strain rate, or a
stress residual) to the subspace `mode` allows, and report that subspace's
dimension.

| `mode`  | projection      | `ndof` | equivalent |
|---------|-----------------|--------|------------|
| `:none` | `0`             | 0      | fixed cell |
| `:iso`  | `tr(A)/3 · I`   | 1      | LAMMPS `couple xyz` — volume only |
| `:diag` | `Diagonal(A)`   | 3      | axes free, no shear |
| `:full` | `A`             | 6      | fully flexible |

The projections are symmetric and idempotent, so applying one to both the
generalised force and the velocity keeps the trajectory in the subspace
exactly, for as long as it starts there.

Why a projection and not a boolean per-component mask: a mask cannot
express `:iso`. Masking to the three diagonal components lets `a`, `b`, `c`
relax **independently**, which for anything but a cubic crystal is a
different constraint than "hydrostatic only" — and one the user would have
no way to notice from the interface.

Why an enum and not a caller-supplied 6×6 projector: `ndof` has to be
exact. Nothing here needs it yet, but the barostats do — [`MTTK`](@ref)
hard-codes the cell thermostat's target as `6·kT` ([`_mttk_chain!`](@ref))
with the matching `6kT·ξ_cell` term in its conserved quantity, and under a
constraint those become `ndof·kT`. Deriving `ndof` from the numerical rank
of a user matrix would make the sampled ensemble depend on a rank
tolerance. With an enum it is 0/1/3/6 by construction.

Not wired into the barostats: relaxation uses this today (Issue #212).
Adopting it in MD additionally needs the `ndof·kT` substitutions above, and
for [`SCR`](@ref) a re-derivation of the fluctuation–dissipation amplitude
(`_sym_randn` draws all six components and `_scr_mu`'s normalisation
assumes them) — a theory-note-level change, deliberately out of scope here.
"""
function _cell_projector(mode::Symbol, A::AbstractMatrix)
    mode === :full && return A
    mode === :none && return zero(A)
    mode === :iso && return (tr(A) / 3) * one(A)
    mode === :diag && return Matrix(Diagonal(A))
    error("cell_dof must be :none, :iso, :diag or :full (got $(repr(mode)))")
end

function _cell_ndof(mode::Symbol)
    mode === :none && return 0
    mode === :iso && return 1
    mode === :diag && return 3
    mode === :full && return 6
    error("cell_dof must be :none, :iso, :diag or :full (got $(repr(mode)))")
end

"""
    _baro_velocity_exponent(zeta_b, nglob) -> Matrix

Exponent matrix `E` of the barostat velocity scaling `v ← v·exp(E·dt/2)`
in the MTTK particle equations (MTK 1994, eq. 2.12):
`E = -(ε̇ + Tr[ε̇]/N_f·I)` with `N_f = 3N-3` — the **same** `N_f` as the
MTK force term `2K/(3N-3)·I` in the barostat force; the MTTK conserved
quantity requires the two to match (Issue #130). In the isotropic limit
`ε̇ = v_ε I` this reduces to `-(1 + 3/N_f)·v_ε·I`, the textbook isotropic
MTK factor.
"""
_baro_velocity_exponent(zeta_b, nglob) =
    -(zeta_b + tr(zeta_b)/(3nglob - 3)*I)

# --------------------------------------------------------------------------
# Thermostat operators (dt/4 of thermostatting per call; called twice per
# update!, i.e. four times per MD step)
# --------------------------------------------------------------------------

_thermo_half!(::NoThermostat, atoms, c, arrtyp) = nothing

# Exact BDP propagator over Δt = dt/4 (paper appendix Eq. A7, recast as a
# completed square for manifest non-negativity):
#
#   b  = √c̃ + √((1−c̃)·K̄/(N_f·K))·R₁,     c̃ = e^{−Δt/τ},  K̄ = N_f·kT/2
#   α² = b² + (1−c̃)·K̄/(N_f·K)·S,          S ~ χ²_{N_f−1}
#   α  = sign(b)·√α²                        (faithful to the OU projection
#        of Eqs. A2/A5 — b is the component of the evolved velocity
#        vector along the current one. The canonical distribution is
#        invariant under a global v → −v, so the always-positive-α of
#        mainstream implementations samples the same ensemble; the
#        negative branch is kept for fidelity, not correctness.)
#
# The analytic solution composes exactly, so the dt/4 cadence (4× per
# step, mirroring NH) introduces no additional splitting error.
#
# DD invariant: the draws depend on nothing rank-local — NOT on K (which
# is Allreduced anyway) and not on any per-rank state — so the consumed
# stream length is identical on every rank and across serial/DD/backend
# replays. Do not change the sampling to anything K-conditional (e.g.
# drawing α from a K-dependent distribution with rejection) without
# revisiting this.
function _thermo_half!(th::CSVR, atoms, c, arrtyp)
    K = kinetic_energy(atoms)
    # K == 0 exactly: a cold start (initial_temperature = 0) — α diverges
    # (the rescale cannot heat a zero vector). Skip; the first force
    # half-kick populates K and the thermostat takes over from there.
    # (K is collective, so the skip is rank-uniform, and the draws below
    # are skipped on every rank alike.)
    K == 0 && return nothing
    Nf = 3c.N - 3
    ctil = exp(-(c.dt/4) / th.tau)
    fac = (Nf * c.kT / 2) / (Nf * K)          # K̄/(N_f·K)
    R1 = randn(th.rng)
    S = rand(th.rng, Chisq(Nf - 1))
    b = sqrt(ctil) + sqrt((1 - ctil) * fac) * R1
    alpha2 = b^2 + (1 - ctil) * fac * S
    # Book the work with the factor ACTUALLY applied (post-rounding), so
    # K + work is exact up to the reassociation of the kinetic-energy sum
    # — with the analytic alpha2 the α-rounding would leak into the
    # effective energy (invisible in Float64, ~1e-7/step in Float32).
    alpha = FPPREC(copysign(sqrt(alpha2), b))
    atoms.arrays[:velo] .*= alpha
    th.work += K * (1 - Float64(alpha)^2)
    return nothing
end

function _thermo_half!(th::NoseHoover, atoms, c, arrtyp)
    Q = _mass_t(th, c)
    gkT = (3c.N - 3) * c.kT
    th.zeta += c.dt/8 * (2kinetic_energy(atoms) - gkT) / Q
    atoms.arrays[:velo] .*= exp(-th.zeta * c.dt/4)
    # dξ/dt = ζ: the g·kT·ξ term advances with the scaling it compensates.
    th.work += gkT * th.zeta * c.dt/4
    th.zeta += c.dt/8 * (2kinetic_energy(atoms) - gkT) / Q
    return nothing
end

# --------------------------------------------------------------------------
# Barostat operators
# --------------------------------------------------------------------------

_baro_half!(::NoBarostat, atoms, c, arrtyp) = nothing

# One symmetric sub-chain of the MTTK cell update: ζ_cell(dt/8) → damp →
# force(dt/4) → damp → ζ_cell(dt/8). The damping friction is the barostat's
# OWN ζ_cell (pre-#136 it was the shared particle NH friction).
function _mttk_chain!(b::MTTK, atoms, c)
    Q_b = _mass_b(b, c)
    Q_cell = _mass_cell(b, c)
    ckT = 6 * c.kT
    b.zeta_cell += c.dt/8 * (Q_b * sum(abs2, b.zeta_b) - ckT) / Q_cell
    b.zeta_b .*= exp(-b.zeta_cell * c.dt/8)
    b.work_cell += ckT * b.zeta_cell * c.dt/8
    F = (c.tstress - voigt2matrix(stress(atoms, true))) * volume(atoms) +
        2kinetic_energy(atoms)/(3c.N - 3) * I
    b.zeta_b .+= c.dt/4 .* F ./ Q_b
    b.zeta_b .*= exp(-b.zeta_cell * c.dt/8)
    b.work_cell += ckT * b.zeta_cell * c.dt/8
    b.zeta_cell += c.dt/8 * (Q_b * sum(abs2, b.zeta_b) - ckT) / Q_cell
    return nothing
end

function _baro_half!(b::MTTK, atoms, c, arrtyp)
    _mttk_chain!(b, atoms, c)
    scale = conv(arrtyp, exp(_baro_velocity_exponent(b.zeta_b, c.N) * c.dt/2))
    atoms.arrays[:velo] .= atoms.arrays[:velo] * scale
    _mttk_chain!(b, atoms, c)
    return nothing
end

# SCR is a once-per-step first-order update: everything happens in
# scale_cell!; there is no cell momentum to propagate in the halves.
_baro_half!(::SCR, atoms, c, arrtyp) = nothing

# Symmetrized standard-normal 3×3 (the Wiener tensor (W + Wᵀ)/2, thesis
# Eq. 2.47): N(0,1) diagonal, N(0,1/2) off-diagonal. Exactly 6 draws in a
# FIXED order — rank-uniform and replay-stable.
function _sym_randn(rng)
    d1, d2, d3 = randn(rng), randn(rng), randn(rng)
    o23 = randn(rng) / sqrt(2.0)
    o13 = randn(rng) / sqrt(2.0)
    o12 = randn(rng) / sqrt(2.0)
    return [d1  o12 o13;
            o12 d2  o23;
            o13 o23 d3]
end

"""
    _scr_mu(b::SCR, S_int, V, L, kT, dt, rng) -> Matrix{Float64}

The symmetric SCR rescaling matrix μ (thesis Eq. 2.28, symmetrized per
§2.6) for one Euler step: `S_int` the internal stress tensor
(`voigt2matrix(stress(atoms, true))`, stress convention σ = −P), `V` the
volume, `L` the host lattice (ROWS = lattice vectors, so the thesis box
matrix is h = Lᵀ), `kT`/`dt` from the control context. Pure up to the 6
noise draws from `rng` — the unit-testable kernel of the barostat.
"""
function _scr_mu(b::SCR, S_int, V, L, kT, dt, rng)
    h = Matrix(transpose(L))
    # Pressure convention: P₀I − P_int = P₀I + S_int (P_int = −S_int).
    #
    # The measure correction is −2kT/V, NOT the thesis Eq. 2.17/2.20
    # value −kT/V. Fokker–Planck stationarity on the symmetric 6-dof
    # strain space (docs/src/theory/scr-barostat.md) shows the correction
    # is exactly the Itô drift of the multiplicative cell noise in lnV,
    # −½·E[tr ξ²]·amp², fixed by the noise's DOF structure alone:
    #   symmetric GOE-normalized ξ (this code): E[ξ²] = 2I → −2kT/V;
    #   full 9-dof iid Wiener tensor (thesis):  E[ξ²] = I  → −kT/V.
    # The thesis value is correct for its UNSYMMETRIZED 9-dof update
    # (its implementations remove rotations by QR after the update —
    # §2.6 strategy 2 — which leaves the det statistics of the 9-dof
    # noise untouched); the thesis does not re-derive the correction for
    # the symmetrized-noise variant used here — pairing strategy-1
    # symmetrization with the verbatim −kT/V is exactly the
    # miscalibration this constant replaces. The noise geometry is
    # pinned by the Monte Carlo test in test_scr.jl:
    # E[log det μ − tr(μ−I)] = −3·amp² (the 9-dof value would be half).
    #
    # No kinetic-statistics term enters: with the COM frozen in scaled
    # coordinates (_remove_com! in NPT.jl, N_f = 3N−3 in NoseHoover/CSVR)
    # the configurational measure is V^(N−1) and the kinetic-pressure
    # estimator averages (N−1)kT/V — the identity V·⟨S_int⟩ =
    # −kT·∂ln(V^(N−1)·q)/∂ε makes the same (N−1) appear on both sides of
    # the stationarity condition and cancel. The constant is therefore
    # independent of N — provided the kinetic-stress estimator and the
    # configurational count stay consistent (frozen COM ↔ N_f = 3N−3;
    # a thermostat targeting the full 3N would shift the sampled
    # V-power by one, not this constant).
    #
    # End-to-end check: the ideal-gas V-marginal is Gamma(N−3+c, kT/P₀)
    # for a correction −c·kT/V — verified at c ∈ {1,2,3} and N ∈ {4,8}
    # (standalone MC, 1–2%; c ∈ {1,3} also measured with this very code
    # in the PR #141 review cycle). c = 2 gives Gamma(N), identical to
    # the measured MTTK V-marginal — see the equivalence test in
    # test_scr.jl.
    #
    # h·Σ·hᵀ is symmetric analytically but not bitwise in floating point
    # (the chained matmuls round the (i,j)/(j,i) pairs differently for a
    # triclinic h); μ must be EXACTLY symmetric — the rotation-elimination
    # contract asserted per draw in test_scr.jl — so fold the ulp-level
    # asymmetry before it enters the bracket.
    pr = (h * b.Sigma * transpose(h)) ./ V
    bracket = (b.P0 - 2kT/V) * Matrix{Float64}(I, 3, 3) + S_int +
              (pr + transpose(pr)) ./ 2
    mu = Matrix{Float64}(I, 3, 3) .- (b.kappa * dt / (3b.tau)) .* bracket .+
         sqrt(2 * b.kappa * kT * dt / (3 * V * b.tau)) .* _sym_randn(rng)
    # Symmetric by construction (S_int, I, the symmetrized PR term,
    # R_sym) — the rotation-elimination contract AND the license for the
    # row-convention lattice update (μᵀ = μ).
    return mu
end

function scale_cell!(b::SCR, atoms, c, arrtyp)
    S_int = voigt2matrix(stress(atoms, true))
    V = volume(atoms)
    L = Matrix(Array(atoms.info[:lattice]))
    mu = _scr_mu(b, S_int, V, L, c.kT, c.dt, b.rng)
    # Euler stability guard: the ensemble is exact for any κ_T only while
    # μ stays perturbative. The classic unit mistake B·UNITS[:GPa]
    # (instead of UNITS[:GPa]/B_GPa) overshoots μ on the first step —
    # refuse to integrate garbage instead. The OPPOSITE mistake (1/B_GPa,
    # ~3e4× too small) is undetectable here: μ ≈ I is indistinguishable
    # from stiff physics, the sampled ensemble stays correct, and only
    # τ_eff stretches by the same factor (an effectively frozen cell) —
    # documented in docs/src/parameters.md; watch the cell columns of
    # log.txt. Threshold calibration: the N = 8
    # ideal-gas ensemble test (fluctuation-dominated by design) reaches
    # max|μ−I| ≈ 0.11 legitimately over 4×10⁵ steps, while the units
    # mistake overshoots on the first step — 0.25 separates the two by an
    # order of magnitude, and Gershgorin (3·0.25 < 1) still guarantees
    # det μ > 0 (no cell inversion).
    maximum(abs, mu - I) < 0.25 ||
        error("npt!: SCR rescaling left the perturbative regime " *
              "(max|μ−I| = $(maximum(abs, mu - I)) ≥ 0.25) — " *
              "params[:compressibility]·dt/tau_p is too large (check the " *
              "units: κ_T = UNITS[:GPa]/B_GPa) or the system is far from " *
              "the target stress; equilibrate with a larger tau_p first")
    # h ← μh (columns) ⇔ lattice ← lattice·μ (rows, μ symmetric);
    # fractional positions are untouched; velocities carry the inverse
    # factor (momenta-scaled formulation, thesis Eq. 2.29 — this is what
    # licenses the instantaneous kinetic term inside stress(atoms, true)).
    atoms.info[:lattice] .= atoms.info[:lattice] * conv(arrtyp, mu)
    atoms.arrays[:velo] .= atoms.arrays[:velo] * conv(arrtyp, inv(mu))
    return nothing
end

# --------------------------------------------------------------------------
# Per-step entry points
# --------------------------------------------------------------------------

"""
    update!(atoms, th::Thermostat, baro::Barostat, c, arrtyp)

Apply one half-timestep of temperature/pressure control: a symmetric
thermostat → barostat → thermostat sandwich. Called twice per MD step from
[`npt!`](@ref) (before the first and after the second velocity half-kick)
to give the symmetric splitting. With both controls `:none` this is a
no-op (NVE).
"""
function update!(atoms, th::Thermostat, baro::Barostat, c, arrtyp)
    _thermo_half!(th, atoms, c, arrtyp)
    _baro_half!(baro, atoms, c, arrtyp)
    _thermo_half!(th, atoms, c, arrtyp)
    return nothing
end

"""
    scale_cell!(baro::Barostat, atoms, c, arrtyp)

Apply the once-per-step cell update at the lattice-update site of the MD
loop. [`MTTK`](@ref): `lattice ← lattice·exp(ε̇·dt)` plus the
external-stress work accumulation; [`NoBarostat`](@ref): no-op (the
lattice is never touched). Fractional coordinates are unaffected by
construction (affine cell scaling).
"""
scale_cell!(::NoBarostat, atoms, c, arrtyp) = nothing

function scale_cell!(b::MTTK, atoms, c, arrtyp)
    deps = b.zeta_b .* c.dt
    # External-stress work −∫V·σ_ext:ζ_b dt over the step, exact for the
    # frozen-ζ_b exponential flow V(t) = V₀·e^{Tr[dε]·t/dt}: the time
    # integral of V is V₀·dt·(e^s − 1)/s with s = Tr[dε]. For a hydrostatic
    # target σ_ext = −P·I this telescopes to +P·ΔV (enthalpy) exactly.
    s = tr(deps)
    vfac = abs(s) > 1e-12 ? expm1(s)/s : 1 + s/2
    b.work_ext -= sum(c.tstress .* deps) * volume(atoms) * vfac
    atoms.info[:lattice] .= atoms.info[:lattice] * conv(arrtyp, exp(deps))
    return nothing
end

# --------------------------------------------------------------------------
# Conserved quantity (diagnostic)
# --------------------------------------------------------------------------

_control_energy(::NoThermostat, atoms, c) = 0.0
_control_energy(th::NoseHoover, atoms, c) =
    _mass_t(th, c)/2 * th.zeta^2 + th.work
_control_energy(th::CSVR, atoms, c) = th.work
_control_energy(::NoBarostat, atoms, c) = 0.0
_control_energy(b::MTTK, atoms, c) =
    _mass_b(b, c)/2 * sum(abs2, b.zeta_b) +
    _mass_cell(b, c)/2 * b.zeta_cell^2 + b.work_cell + b.work_ext
# SCR has no discrete invariant under the Euler integrator (thesis §2.4);
# the P₀V term turns the column into the enthalpy-like diagnostic
# documented in the SCR docstring. The deviatoric strain energy is not
# tracked (hydrostatic part only).
_control_energy(b::SCR, atoms, c) = b.P0 * volume(atoms)

"""
    conserved_energy(th, baro, atoms, c) -> Float64

Diagnostic conserved quantity H′ (Hartree) of the active
thermostat/barostat combination:

    H′ = K + U + [Q_T/2·ζ² + g·kT·ξ]            (NoseHoover)
             + [Σ K·(1−α²)]                      (CSVR effective energy H̃,
                                                  BDP 2007 Eq. 15)
             + [Q_b/2·Σζ_b² + Q_cell/2·ζ_cell²
                + 6kT·ξ_cell − ∫V·σ_ext:dε]      (MTTK)

with the ξ terms held as energy accumulators (`work*` fields). For NVE
this is exactly K + U. Logged per step in `log.txt` for drift monitoring;
exactly conserved (to integrator order) only while the target temperature
and stress are fixed — under a `:final_temperature` ramp it is a
diagnostic, not an invariant. For [`SCR`](@ref) the barostat term is
`P₀·V` and the column is an enthalpy-like diagnostic, NOT an invariant
(no discrete conserved quantity exists for the Euler-integrated SCR —
thesis §2.4); in equilibrium it must be stationary (no trend), which is
still a usable health signal.
"""
conserved_energy(th::Thermostat, baro::Barostat, atoms, c) =
    kinetic_energy(atoms) + Array(atoms.info[:potential_energy])[1] +
    _control_energy(th, atoms, c) + _control_energy(baro, atoms, c)
