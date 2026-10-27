# ============================================================================
# Shadow extended-Lagrangian BOMD for the core-shell model (Issue #149,
# derivation: docs/src/theory/xlbomd-shell.md).
#
# Opt-in alternative to the default per-step shell relaxation: instead of
# converging the shells at every step, an auxiliary displacement field ŝ is
# propagated by its own time-reversible Verlet and the dynamics runs on the
# EXACT gradient of a shadow energy built at ŝ. The conserved quantity is
# then constructive rather than a consequence of tight convergence, at a
# fixed ~3 force evaluations per step and zero SCF iterations.
#
# This is a different (shadow) potential-energy surface than the fully
# relaxed one, so results can differ from the default path — see
# docs/src/parameters.md for the measured deviation.
# ============================================================================

using KernelAbstractions
import AcceleratedKernels as AK
using ..FMD: Atoms, FPPREC

"""
    XLState

Scalar state of the shadow XL-BOMD integrator. The per-atom auxiliary
fields live in `atoms.arrays[:xl_shat]` / `[:xl_shatm]` so the `CellList`
re-sort permutes them along with every other per-atom array — the same
reason the extrapolation history `:shist*` lives there.

`count` is the number of steps since the last (re)seed: the first
`warmup` steps run the ordinary relaxation so the auxiliary field starts
from a converged trajectory. `nreset` counts fallback events for the
end-of-run report.

`kernel` selects the operator applied to the residual shell force before
it drives the auxiliary Verlet (Issue #152): `:spring` is the closed-form
spring inverse implicit in `s₁ - ŝ`, `:chebyshev` corrects it towards the
full shell Hessian with a fixed polynomial whose coefficients are `kcoef`
(`length(kcoef) - 1` extra Hessian-vector products per step).
"""
mutable struct XLState
    kappa::FPPREC
    kernel::Symbol
    kcoef::Vector{FPPREC}
    specmax::FPPREC
    warmup::Int
    threshold::FPPREC
    count::Int
    nreset::Int
end

"""
    cheb_kernel_coefs(order, a, b) -> Vector

Coefficients of the polynomial `p` (degree `order - 1`) that approximates
`λ⁻¹` on `[a, b]` in the minimax sense used for preconditioning:

    1 - λ·p(λ) = T_order((c - λ)/d) / T_order(c/d),   c = (a+b)/2, d = (b-a)/2

so `p` is applied as `x = Σ_j kcoef[j]·M^(j-1)·y` with `M = A⁻¹H_ss`. On
`[a, b]` the preconditioned spectrum satisfies `λ·p(λ) ∈ [1-ρ, 1+ρ]` with
`ρ = 1/|T_order(c/d)| < 1`, i.e. the auxiliary Verlet sees a nearly
uniform coupling instead of the factor ~6 spread the bare spring kernel
leaves (theory note §6.1).

Two one-sided properties make the interval safe to widen and dangerous to
shrink, and both are load-bearing for the guards in [`xl_init!`](@ref):

- below `a` (`λ < a`): `T_order` is monotone for arguments `≥ 1`, so
  `λ·p(λ)` decreases smoothly to `0` at `λ = 0` and never changes sign —
  under-resolved soft modes are merely tracked more weakly, never
  destabilized. Declaring `a` too small is therefore safe;
- above `b`: `|T_order|` grows without bound, so `λ·p(λ)` leaves the band.
  For even `order` it changes sign (the mode becomes anti-restoring); for
  odd `order` it grows past the Verlet bound `κ·λp < 4`. Either way the
  auxiliary field destabilizes, so **`b` must bound the true `λmax` from
  above** — the one direction of the interval that is not free.

The instability is loud (the `xl_max_deviation` guard fires and the run
falls back to a relaxation), not silent: the shadow force stays the exact
gradient of the shadow energy whatever the kernel does.
"""
function cheb_kernel_coefs(order::Integer, a::Real, b::Real)
    order >= 2 || error("cheb_kernel_coefs: order must be >= 2 (got $order)")
    0 < a < b || error("cheb_kernel_coefs: need 0 < a < b (got a = $a, b = $b)")
    c = (a + b) / 2
    d = (b - a) / 2
    # Polynomials in λ as coefficient vectors, built by the Chebyshev
    # three-term recurrence in u(λ) = (c - λ)/d.
    mul(u, v) = begin
        w = zeros(Float64, length(u) + length(v) - 1)
        for i in eachindex(u), j in eachindex(v)
            w[i+j-1] += u[i] * v[j]
        end
        w
    end
    add(u, v) = begin
        w = zeros(Float64, max(length(u), length(v)))
        for i in eachindex(u); w[i] += u[i]; end
        for i in eachindex(v); w[i] += v[i]; end
        w
    end
    u = [c/d, -1/d]
    Tm1 = [1.0]          # T_0
    T = u                # T_1
    for _ in 2:order
        Tm1, T = T, add(mul(2 .* u, T), -Tm1)
    end
    # T_order(c/d) is the value at λ = 0, i.e. the constant coefficient.
    σ = T[1]
    # c/d = (a+b)/(b-a) > 1 for a > 0 and T_order > 1 above 1, so this holds
    # analytically; it guards the floating-point limit a/b → 0, where σ
    # rounds to 1 and the band [1-ρ, 1+ρ] would touch zero (a mode the
    # auxiliary field would then stop tracking altogether).
    σ > 1 ||
        error("cheb_kernel_coefs: T_$order(c/d) = $σ ≤ 1 — the interval " *
              "[$a, $b] is degenerate at this precision")
    R = T ./ σ                                   # R(0) = 1 by construction
    coefs = FPPREC.(-R[2:end])                   # p(λ) = -(R - 1)/λ
    # The monomial basis loses conditioning as the degree grows, and it
    # loses it silently: the coefficients still evaluate, just not to the
    # polynomial that was designed. Check the design property on the
    # coefficients that will actually be used — the minimax residual on
    # [a, b] must be the theoretical ρ. This is what bounds the usable
    # `xl_kernel_order` instead of a hard-coded cap.
    ρ = 1 / σ
    worst = maximum(abs(1 - λ * evalpoly(λ, coefs))
                    for λ in range(a, b; length = 401))
    worst <= 1.05 * ρ + 1e-12 ||
        error("cheb_kernel_coefs: the order-$order polynomial on [$a, $b] " *
              "evaluates to a minimax residual of $worst against its design " *
              "value $ρ — the monomial coefficients have lost conditioning at " *
              "this order and interval; lower xl_kernel_order")
    return coefs
end

"""
    xl_init!(atoms, params) -> XLState | nothing

Register the auxiliary shell fields and read the XL knobs from `params`.
Returns `nothing` when `params[:shell_dynamics]` is `:scf` (the default),
which leaves [`npt!`](@ref) on the ordinary relaxation path.
"""
function xl_init!(atoms::Atoms, params)
    get(params, :shell_dynamics, :scf) === :xlbomd || return nothing
    # The closed-form shadow minimum solves (k2 + k4|s|²/6)·s = F_pair by
    # Newton from the harmonic guess |F_pair|/k2, which needs k2 > 0. A
    # species with k2 = 0 but a live quartic would still have a finite
    # root — the purely quartic one — that this branch cannot reach, so it
    # is refused rather than silently returning s₁ = 0 (the relaxation
    # path handles that case, so the two would disagree).
    k2 = Array(params[:k2])
    k4 = Array(params[:k4])
    bad = findall(i -> k2[i] <= 0 && k4[i] != 0, eachindex(k2))
    isempty(bad) ||
        error("shell_dynamics = :xlbomd requires k2 > 0 for every species with " *
              "a non-zero k4 (the closed-form shadow minimum is a Newton solve " *
              "seeded at |F_pair|/k2); species $(bad) have k2 = 0 with k4 ≠ 0")
    kern = get(params, :xl_kernel, :spring)
    # Mirrors the validator: direct npt! callers bypass validate_params.
    kern in (:spring, :chebyshev) ||
        error("xl_kernel must be :spring or :chebyshev (got $(repr(kern)))")
    # The kernel flattens the spectrum the auxiliary Verlet sees, so κ stops
    # being a per-material constant: :chebyshev gets the XL literature value
    # (safe for any interval, since ρ < 1 bounds κ·λp below 2κ), :spring
    # keeps the measured shell-model bound (theory note §6).
    κ = FPPREC(get(params, :xl_kappa, kern === :chebyshev ? 1.8 : 0.5))
    warm = Int(get(params, :xl_warmup, 3))
    thr = FPPREC(get(params, :xl_max_deviation, 0.05))
    κ > 0 || error("xl_kappa must be > 0 (got $κ)")
    warm >= 1 || error("xl_warmup must be >= 1 (got $warm)")
    thr > 0 || error("xl_max_deviation must be > 0 (got $thr)")
    kcoef = FPPREC[]
    specmax = zero(FPPREC)
    if kern === :chebyshev
        order = Int(get(params, :xl_kernel_order, 3))
        spec = get(params, :xl_kernel_spectrum, (0.4, 6.0))
        length(spec) == 2 ||
            error("xl_kernel_spectrum must be a 2-tuple (a, b) of the " *
                  "spring⁻¹·H_ss spectrum bounds (got $(repr(spec)))")
        kcoef = cheb_kernel_coefs(order, spec[1], spec[2])
        specmax = FPPREC(spec[2])
    end
    _xl_check_surface_spectrum(atoms, params, kern, κ, specmax)
    atoms.arrays[:xl_shat] = copy(atoms.arrays[:shelldisp])
    atoms.arrays[:xl_shatm] = copy(atoms.arrays[:shelldisp])
    atoms.arrays[:xl_dir] = zero(atoms.arrays[:shelldisp])
    return XLState(κ, kern, kcoef, specmax, warm, thr, 0, 0)
end

# Spectral necessary conditions when a surface term is present (Issue
# #181). The depolarization block adds a rank-3 positive piece to H_ss,
# so λmax(spring⁻¹H_ss) can only grow — and by exactly how much is a
# closed form (`surface_spectral_bump`). H_ss = H_rest + H_surf with
# H_rest positive semi-definite at a stable shell configuration, so
# Weyl's inequality makes that bump a LOWER bound on the true λmax: both
# checks below are then provably-wrong conditions, not heuristics.
#
# The two conditions apply to different kernels: the interval check is
# for :chebyshev (which declares one), the κ check for :spring (the only
# kernel whose Verlet coupling IS λmax — see §6.1).
#
# They are one-sided on purpose. Passing them does not certify stability
# — the rest of the spectrum is still measured, not derived
# (`bench/check_xl_spectrum.jl`, theory note §6/§6.1) — but failing them
# means the declared interval or κ is wrong with certainty, and the
# failure mode they catch is the silent one: a `b` below λmax makes the
# Chebyshev kernel anti-restoring on the modes it misses, and the run
# only reports it later as a deviation fallback (or not at all).
function _xl_check_surface_spectrum(atoms::Atoms, params, kern, κ, specmax)
    haskey(params, :surface_term) || return nothing
    Δλ = Calculators.surface_spectral_bump(params[:surface_term], atoms, params)
    if kern === :chebyshev && specmax < Δλ
        error("xl_kernel_spectrum's upper bound b = $specmax is below the " *
              "$(Δλ) the surface term alone contributes to " *
              "λmax(spring⁻¹·H_ss), so the declared interval provably " *
              "misses the top of the spectrum (the Chebyshev kernel turns " *
              "anti-restoring there).  Raise b past the measured λmax of " *
              "this system WITH the depolarization block — see " *
              "bench/check_xl_spectrum.jl")
    end
    # The κ bound is on the coupling the auxiliary Verlet actually sees,
    # κ·λp < 4.  Only the :spring kernel has λp = λ, so only there does
    # λmax enter it; :chebyshev flattens λp into [1-ρ, 1+ρ] ⊂ (0, 2)
    # whatever the spectrum is (§6.1), and its exposure to a raised λmax
    # is the interval check above, not this one.
    if kern === :spring && κ * Δλ >= 4
        error("xl_kappa = $κ with a surface term contributing $(Δλ) to " *
              "λmax(spring⁻¹·H_ss) violates the auxiliary Verlet stability " *
              "bound κ·λmax < 4 on the depolarization block alone.  Lower " *
              "xl_kappa (the bound is on the FULL λmax, which is larger " *
              "still), or switch to xl_kernel = :chebyshev, which flattens " *
              "the coupling into [1-ρ, 1+ρ] and is not bounded by λmax at " *
              "all — see docs/src/theory/xlbomd-shell.md §6 and §6.1")
    end
    return nothing
end

# Per-atom closed forms of the shadow minimum (backend-generic kernel).
#
# The shadow functional keeps the Spring term exact and linearizes
# everything else at ŝ, so its minimizer solves
#     (k2 + k4|s₁|²/6)·s₁ = F_pair,   F_pair = F_s(ŝ) + (k2 + k4|ŝ|²/6)·ŝ
# which is radial: s₁ = m·F̂_pair with k4/6·m³ + k2·m = |F_pair|. The cubic
# is strictly increasing in m for k2 > 0, k4 ≥ 0, so Newton from the
# harmonic guess m = |F_pair|/k2 converges monotonically.
#
# Writes s₁ into `s1`, and accumulates the two spring energies the shadow
# energy needs (at ŝ and at s₁) plus the linear term -F_pairᵀ(s₁ - ŝ) as
# per-atom contributions the caller reduces.
@kernel inbounds = true function kernel_xl_shadow_min!(s1, eterms,
    @Const(shellforces), @Const(shat), @Const(elemid), @Const(k2), @Const(k4))

    I = @index(Global)
    e = elemid[I]
    k2i = k2[e]
    k4i = k4[e]

    hx, hy, hz = shat[I,1], shat[I,2], shat[I,3]
    d2 = hx*hx + hy*hy + hz*hz
    c0 = k2i + k4i*d2/6
    gx = shellforces[I,1] + c0*hx
    gy = shellforces[I,2] + c0*hy
    gz = shellforces[I,3] + c0*hz

    gn = sqrt(gx*gx + gy*gy + gz*gz)
    m = zero(FPPREC)
    if k2i > 0 && gn > 0
        m = gn / k2i
        for _ in 1:50
            f = k4i/6*m^3 + k2i*m - gn
            df = k4i/2*m^2 + k2i
            Δ = f/df
            m -= Δ
            abs(Δ) <= eps(FPPREC)*max(m, one(FPPREC)) && break
        end
    end
    sx = gn > 0 ? m*gx/gn : zero(FPPREC)
    sy = gn > 0 ? m*gy/gn : zero(FPPREC)
    sz = gn > 0 ? m*gz/gn : zero(FPPREC)
    s1[I,1] = sx; s1[I,2] = sy; s1[I,3] = sz

    m2 = m*m
    # -E_spring(ŝ) + E_spring(s₁) - F_pairᵀ(s₁ - ŝ)
    eterms[I] = -(k2i*d2/2 + k4i*d2^2/24) + (k2i*m2/2 + k4i*m2^2/24) -
                (gx*(sx - hx) + gy*(sy - hy) + gz*(sz - hz))
end

# Shell-less species (k2 = 0) carry no shell degree of freedom: the shadow
# minimizer pins s₁ = 0 there, so the auxiliary field must be driven back
# to zero, which is what the spring direction (-ŝ on those rows) does.
#
# `PROJECT` zeroes those rows, `MASK` copies the spring direction into
# them. Both are needed and they are not interchangeable: `hvp_workspace`
# substitutes a UNIT spring for k2 = 0 in `kvals`, so without the
# projection a shell-less row would carry a residual of its own into the
# polynomial and — through `hvp!` — leak into the rows that do have
# shells. The operator being preconditioned would then be `A⁻¹H_ss` with a
# fictitious unit spring spliced in, whose eigenvalues have no reason to
# lie in the declared interval, while `bench/check_xl_spectrum.jl`
# measures the projected one. Same projection on both sides keeps the two
# the same operator.
@kernel inbounds = true function kernel_xl_shell_rows!(out, @Const(v), ::Val{MASK},
    @Const(elemid), @Const(k2)) where {MASK}

    I = @index(Global)
    if k2[elemid[I]] <= 0
        out[I,1] = MASK ? v[I,1] : zero(eltype(out))
        out[I,2] = MASK ? v[I,2] : zero(eltype(out))
        out[I,3] = MASK ? v[I,3] : zero(eltype(out))
    end
end

# Shadow virial (Issue #151, theory note §4.1): the cell-strain derivative
# of E_shadow at frozen ŝ, assembled per atom from quantities the shadow
# step already has,
#
#   W_shadow = W(ŝ) + D_v W(ŝ) - Σ_i sym[F_s(ŝ)_i ⊗ s₁_i]
#                              - Σ_i sym[(Jv)_i ⊗ ŝ_i]
#
# with v = s₁ - ŝ. `W(ŝ)` is already in info[:virial] from core_forces!,
# so this kernel builds only the correction: `Wv` is hvp!'s per-atom
# D_v W, and the two outer products convert the two conventions — the
# reported virial lets every site ride the cell affinely, whereas ŝ is a
# Cartesian auxiliary variable the barostat does not scale.
#
# The external field is subtracted from the shell force on purpose:
# `ExternalField.calculate!` adds to `shellforces` but by construction
# contributes nothing to `info[:virial]`, so its shell force must not be
# converted into a strain derivative here either — otherwise the assembly
# would report a term the energy it differentiates does not contain.
@kernel inbounds = true function kernel_xl_virial!(vir, @Const(Wv), @Const(shellforces),
    @Const(s1), @Const(Jv), @Const(shat), @Const(elemid), @Const(qs), @Const(efield))

    I = @index(Global)
    q = qs[elemid[I]]
    fx = shellforces[I,1] - q*efield[1]
    fy = shellforces[I,2] - q*efield[2]
    fz = shellforces[I,3] - q*efield[3]
    jx, jy, jz = Jv[I,1], Jv[I,2], Jv[I,3]
    sx, sy, sz = s1[I,1], s1[I,2], s1[I,3]
    hx, hy, hz = shat[I,1], shat[I,2], shat[I,3]

    vir[I,1] = Wv[I,1] - fx*sx - jx*hx
    vir[I,2] = Wv[I,2] - fy*sy - jy*hy
    vir[I,3] = Wv[I,3] - fz*sz - jz*hz
    vir[I,4] = Wv[I,4] - (fy*sz + fz*sy)/2 - (jy*hz + jz*hy)/2
    vir[I,5] = Wv[I,5] - (fz*sx + fx*sz)/2 - (jz*hx + jx*hz)/2
    vir[I,6] = Wv[I,6] - (fx*sy + fy*sx)/2 - (jx*hy + jy*hx)/2
end

# x = p(P·M·P)·P·y with M = A⁻¹H_ss = -A⁻¹J, y = A⁻¹F_s(ŝ) and P the
# projector onto the rows that have a shell. Evaluated by Horner in the
# monomial basis, one HVP per degree.
#
# The polynomial is FIXED — it depends on the declared spectral interval,
# not on the state — which is what keeps the propagation a time-reversible
# linear map. An adaptive Krylov solve here (recomputing the polynomial
# from the current residual every step) was measured to be worse than no
# kernel at all at low rank: the effective propagator jitters step to step
# and the smooth tracking the auxiliary Verlet relies on is lost (#152).
function _xl_cheb_dir!(dir, coefs, atoms::Atoms, ws, prms::NamedTuple, v)
    arr = atoms.arrays
    backend = Cores.get_backend(atoms)
    N = length(atoms)
    kv = ws.kvals
    w = similar(dir)
    Jw = similar(dir)
    project!(x) = N > 0 && kernel_xl_shell_rows!(backend, 512, N)(
        x, x, Val(false), arr[:elemid], prms[:k2])
    c1 = coefs[1]
    @. w = arr[:shellforces] / kv
    project!(w)
    @. dir = c1 * w
    for j in 2:length(coefs)
        # Collective under DD, and the iteration count is state-independent,
        # so every rank (empty ones included) makes the same calls. The
        # virial derivative these calls produce belongs to the kernel's
        # own directions, not to s₁ − ŝ, so it goes to the sink.
        Calculators.hvp!(Jw, nothing, ws.Wscratch, w, atoms, ws, prms)
        cj = coefs[j]
        @. w = -Jw / kv
        project!(w)
        @. dir += cj * w
    end
    N > 0 && kernel_xl_shell_rows!(backend, 512, N)(dir, v, Val(true),
                                                    arr[:elemid], prms[:k2])
    return dir
end

"""
    xl_shadow_forces!(atoms, nl, coulomb, prms, shat; kernel_coefs = nothing)
        -> (E_shadow, maxdev)

Evaluate the shadow energy and its exact gradient at the current atomic
positions for the frozen auxiliary field `shat`, writing the forces into
`atoms.arrays[:forces]`, the shadow energy into
`atoms.info[:potential_energy]`, the shadow virial into
`atoms.info[:virial]` (and its per-atom part into
`atoms.arrays[:peratom_virial]` when registered), and the shadow
minimizer into `atoms.arrays[:shelldisp]`. Returns the shadow energy and
`max|s₁ − ŝ|` (the deviation the caller monitors).

This is a pure function of the positions and `shat` — no history, no
convergence state — which is what makes the force/energy consistency
testable by finite differences at fixed `shat`.

The gradient is exact for the shadow functional:

    F = F_c(ŝ) + F_s(ŝ) + (J + K)·(s₁ − ŝ)

Both site families enter because the shells ride the cores at fixed
displacement, and the s₁-dependence drops by the envelope theorem. The
Spring and intra-atom contributions cancel pairwise between the force sum
and the Hessian sum, so every term is evaluated full-stack (theory note
§3).

The virial is exact for the same functional (theory note §4.1):

    W = W(ŝ) + D_v W(ŝ) − Σ sym[F_s(ŝ) ⊗ s₁] − Σ sym[(J·v) ⊗ ŝ]

so a barostat sees the cell-strain derivative of the surface the
dynamics actually runs on. The u-only terms (spring, intra) cancel
between the four pieces exactly as they do in the force — pinned by a
test rather than exploited, so the assembly stays uniform.

Besides the forces this fills `atoms.arrays[:xl_dir]` (when the array is
registered) with the direction the auxiliary Verlet is to be driven by —
either `s₁ − ŝ` (`kernel_coefs === nothing`, the spring kernel implicit in
the shadow minimum) or the Chebyshev-preconditioned residual
`p(A⁻¹H_ss)·A⁻¹F_s(ŝ)` (theory note §6.1), which costs
`length(kernel_coefs) - 1` extra Hessian-vector products. The kernel
changes neither the forces nor the energy: it only decides how `ŝ` moves
next, so the finite-difference statement above holds for either choice.
"""
function xl_shadow_forces!(atoms::Atoms, nl, coulomb, prms::NamedTuple, shat;
                           kernel_coefs = nothing)
    arr = atoms.arrays
    kernel_coefs === nothing || haskey(arr, :xl_dir) ||
        error("xl_shadow_forces!: a kernel needs the :xl_dir array " *
              "(registered by xl_init!)")
    backend = Cores.get_backend(atoms)
    N = length(atoms)
    dom = Cores.domain(atoms)

    # Refresh the Coulomb solver for the current cell before anything
    # reads it — `core_forces!` does not, and under a barostat (Issue
    # #151) the cell moves every step, which would otherwise leave the
    # reciprocal factors one step stale and silently wrong. Same contract
    # as `calculate!(::CoreShell, …)`, which does this at its entry.
    Calculators.cellupdate!(coulomb, atoms.info[:lattice], prms)

    # Full stack at ŝ: one pass gives E, F_c, F_s and the virial.
    arr[:shelldisp] .= shat
    Calculators.core_forces!(atoms, nl, coulomb, prms)

    s1 = similar(shat)
    eterms = KernelAbstractions.zeros(backend, FPPREC, N)
    if N > 0
        kernel_xl_shadow_min!(backend, 512, N)(s1, eterms,
            arr[:shellforces], shat, arr[:elemid], prms[:k2], prms[:k4])
    end

    # Mixed + shell Hessian blocks along (s₁ − ŝ). The workspace snapshot
    # is taken while :shelldisp still holds ŝ.
    ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
    v = similar(shat)
    @. v = s1 - shat
    Jv = similar(shat)
    Kv = similar(shat)
    Wv = KernelAbstractions.zeros(backend, FPPREC, N, 6)
    Wrec = Calculators.hvp!(Jv, Kv, Wv, v, atoms, ws, prms)
    @. arr[:forces] += arr[:shellforces] + Jv + Kv

    # Shadow virial: turn the virial core_forces! left at ŝ into the
    # strain derivative of E_shadow (kernel_xl_virial! above). Done here,
    # while :shelldisp still holds ŝ and F_s(ŝ) is still in the array.
    vir = KernelAbstractions.zeros(backend, FPPREC, N, 6)
    if N > 0
        kernel_xl_virial!(backend, 512, N)(vir, Wv, arr[:shellforces], s1, Jv,
            shat, arr[:elemid], prms[:qs], prms[:efield])
    end
    # Owned partial sums ride the Allreduce; the reciprocal remainder is
    # already whole-system (replicated grid) and is added directly — the
    # same split as the PME force path.
    Calculators.add_reduced_ev!(atoms, zero(FPPREC),
        N > 0 ? AK.sum(vir, dims = 1)[1, :] : zeros(FPPREC, 6))
    atoms.info[:virial] .+= Calculators._devcopy(backend, Wrec)
    # Per-atom decomposition, for the peratomstress columns: everything
    # except the reciprocal remainder, which has no per-atom attribution
    # in the force path either.
    if haskey(arr, :peratom_virial) && N > 0
        arr[:peratom_virial] .+= vir
    end

    # Direction for the next auxiliary Verlet step (Issue #152). Computed
    # here rather than in xl_propagate! because it belongs to the ŝ
    # snapshot: by the time the propagation runs, the nuclei have already
    # moved. `:shelldisp` still holds ŝ, which is what lets the kernel
    # re-enter hvp! on this workspace (its frozen-snapshot guard checks
    # exactly that). F_s(ŝ) is the exact residual of the shell equation —
    # Newton step toward the relaxed shells is H_ss⁻¹F_s(ŝ), and the
    # spring kernel s₁ − ŝ is its 0th-order approximation.
    if haskey(arr, :xl_dir)
        if kernel_coefs === nothing
            arr[:xl_dir] .= v
        else
            _xl_cheb_dir!(arr[:xl_dir], kernel_coefs, atoms, ws, prms, v)
        end
    end

    # Shadow energy: E(ŝ) with the exact spring at ŝ swapped for the one at
    # s₁ plus the linearized pair term (kernel_xl_shadow_min! accumulated
    # the three per-atom pieces).
    ecorr = N > 0 ? AK.sum(eterms) : zero(FPPREC)
    ecorr = Cores.allreduce_sum(dom, ecorr)
    atoms.info[:potential_energy] .+= ecorr
    E = Array(atoms.info[:potential_energy])[1]

    maxdev = N > 0 ? maximum(abs, v) : zero(FPPREC)
    maxdev = Cores.allreduce_max(dom, maxdev)

    arr[:shelldisp] .= s1
    return E, maxdev
end

"""
    xl_propagate!(atoms, st::XLState)

Time-reversible Verlet update of the auxiliary field,
`ŝ⁺ = 2ŝ − ŝ⁻ + κ·d`, where `d` is the kernel-preconditioned residual
[`xl_shadow_forces!`](@ref) left in `:xl_dir` for the configuration the
auxiliary field currently sits at (`s₁ − ŝ` for the spring kernel).

No dissipation term: the undamped propagator is exactly time-reversible,
which is what makes the shadow invariant of §5 an invariant at all. If a
long run ever needs damping, the form to add is Niklasson's
time-reversible polynomial filter over several past `ŝ`, not a velocity
damping — see the theory note. Time reversibility survives the kernel
because `d` is a linear map of the current state alone: the Chebyshev
coefficients are fixed at [`xl_init!`](@ref) from the declared spectral
interval, so nothing about the propagator depends on the history.
"""
function xl_propagate!(atoms::Atoms, st::XLState)
    shat = atoms.arrays[:xl_shat]
    shatm = atoms.arrays[:xl_shatm]
    dir = atoms.arrays[:xl_dir]
    κ = st.kappa
    tmp = similar(shat)
    @. tmp = 2*shat - shatm + κ*dir
    shatm .= shat
    shat .= tmp
    return nothing
end

# Relax the shells the ordinary way and (re)seed the auxiliary field from
# the converged state. Used for the warmup steps and after a fallback.
# Deliberately does not touch `st.count`: the callers own the warmup
# ladder (a seed inside the warmup advances it, a fallback restarts it).
function _xl_seed!(atoms::Atoms, nl, coulomb, prms::NamedTuple, st::XLState)
    ncg = Calculators.calculate!(Calculators.CoreShell(), atoms, nl, coulomb, prms)
    atoms.arrays[:xl_shat] .= atoms.arrays[:shelldisp]
    atoms.arrays[:xl_shatm] .= atoms.arrays[:shelldisp]
    # ŝ⁻ = ŝ and a zero driver: the auxiliary field starts at rest on the
    # relaxed state, whichever kernel is selected.
    fill!(atoms.arrays[:xl_dir], zero(FPPREC))
    return ncg
end

"""
    xl_step!(atoms, nl, coulomb, prms, st) -> Int

One shadow XL-BOMD step, returning the number of shell-force evaluations
spent inside a relaxation (0 for a pure XL step; the CG count during
warmup or after a fallback). That number is what `npt!` logs in the `ncg`
column, so a non-zero entry in an XL run marks a warmup or fallback step.
The kernel evaluations of `:xl_kernel = :chebyshev` are deliberately *not*
counted: they are a fixed per-step cost of the propagation, not the
variable relaxation cost the column exists to expose.

Falls back to a full relaxation — and reseeds the auxiliary field — when
`max|s₁ − ŝ|` exceeds `st.threshold`, which is the guard against the
auxiliary field drifting off the physical surface (a shell model can
otherwise walk into the polarization catastrophe unnoticed).
"""
function xl_step!(atoms::Atoms, nl, coulomb, prms::NamedTuple, st::XLState)
    if st.count < st.warmup
        st.count += 1
        return _xl_seed!(atoms, nl, coulomb, prms, st)
    end

    xl_propagate!(atoms, st)
    _, maxdev = xl_shadow_forces!(atoms, nl, coulomb, prms, atoms.arrays[:xl_shat];
                                  kernel_coefs = isempty(st.kcoef) ? nothing : st.kcoef)

    if !(maxdev <= st.threshold)      # also catches NaN
        st.nreset += 1
        # With a kernel the first thing to suspect is the declared spectral
        # upper bound: above it the polynomial leaves the stable band and
        # the auxiliary field is driven away from the surface (§6.1). That
        # is the failure this guard converts into a loud, recoverable event.
        hint = st.kernel === :chebyshev ?
            " — if this repeats, xl_kernel_spectrum's upper bound " *
            "($(st.specmax)) may be below the true λmax; measure it with " *
            "bench/check_xl_spectrum.jl and widen it" : ""
        @warn "shadow XL-BOMD: |s₁ − ŝ| exceeded the threshold — relaxing the " *
              "shells and reseeding the auxiliary field" * hint maxdev threshold = st.threshold maxlog = 5
        # Restart the warmup ladder: a reseeded auxiliary field needs the
        # same relaxed run-in as at t = 0 before it can carry the dynamics.
        st.count = 0
        return _xl_seed!(atoms, nl, coulomb, prms, st)
    end
    st.count += 1
    return 0
end
