# PMEError.jl — a-priori RMS force error estimation for the PME Coulomb
# path, and the α auto-determination built on it (Issue #216).
#
# The production apps fix the two cost knobs (rc, pme_meshsize) and let
# α — a splitting parameter with no physical meaning — be chosen to
# minimize the estimated total RMS force error. This file provides the
# estimate and the 1D minimization. Everything here is host-side Float64
# scalar math evaluated once at setup: the estimate must not depend on
# FPPREC (a Float32 build would lose the error floor before it is even
# compared), and nothing below touches device arrays.
#
# Estimate structure (validated against direct measurement in
# test/calculators/test_pme_error.jl):
#
#  * Real space: Kolafa–Perram [Mol. Simul. 9, 351 (1992)] — the classic
#    erfc-tail estimate for the cutoff error of the α-screened pair sum.
#  * Reciprocal space: the mean-square pair + self force error of THIS
#    implementation's discrete operator (spread → G̃ = expfactor·bsp2factor
#    → gather, analytic differentiation), evaluated exactly as a k-space
#    alias sum. No empirical calibration enters: the formula is derived on
#    the implementation's own constraint space (the same framework as
#    Deserno & Holm, J. Chem. Phys. 109, 7694 (1998) and Ballenegger,
#    Cerdà & Holm, JCTC 8, 936 (2012), specialized to SPME-ad with the
#    B-spline order and factor conventions of PME.jl).
#
# Reliability domain (documented, not enforced): both branches assume a
# HOMOGENEOUS, UNCORRELATED charge distribution and predict a site RMS.
# Measured behavior on this project's fixtures: within a few percent of
# the true RMS error for uniform-random configurations (est/meas
# 0.93–1.02 across α ∈ [0.25, 0.6], K ∈ [16, 32]); a factor 1.1–2.7
# OVERestimate for thermally-rattled crystals (charge correlation reduces
# the true error — the safe side for a warning). Two things it cannot see:
# max|ΔF| (≈ 4–5× the RMS for Gaussian error statistics), and the
# symmetric-lattice cancellation of Issue #211 (an ideal crystal on a
# commensurate grid has exactly zero error, far below any homogeneous
# estimate; the estimate describes the thermal ensemble, which is what
# MD error control needs).

export pme_error_estimate, pme_optimal_alpha

# |FT of the order-BSPORDER cardinal B-spline|, argument in units of the
# grid sampling frequency. Real and non-negative for the even BSPORDER.
_bspft(v) = v == 0 ? 1.0 : (sin(pi * v) / (pi * v))^BSPORDER

"""
    _pme_chi2_recip(cell, fftsize, alpha) -> (chi2_pair, chi2_self)

Mean-square reciprocal-space force error of the discrete PME operator,
per unit-charge pair (`chi2_pair`, positions uniform and independent) and
per unit charge against its own grid image (`chi2_self`, the "eggbox"
force). Host Float64.

Derivation sketch, in the conventions of PME.jl (fractional coordinates,
`rcell` rows = reciprocal vectors without 2π, `expfactor(m) =
exp(-(π/α)²m²)/(πVm²)`): the spread of a unit charge at fractional `f`
has DFT `Q̂(k) = Σ_n û(ν_n) e^{-2πi m_n·f}` over aliases `m_n = m₀(k) + nK`
with `û` the B-spline transform — the linear phases of the uncentered
spline cancel between the two factors of every |·|² and cross product, so
all coefficients below are real. The discrete pair energy is a DOUBLE
alias sum over the two particles' independent positions:

    U₁₂(f, g) = Σ_k G̃(k) Σ_{n,n'} û_n û_{n'} e^{2πi(m_n·f - m_{n'}·g)},

and it is NOT a function of f - g alone — that position dependence is the
eggbox. The exact interaction `Σ_m t(m) e^{2πim(f-g)}`, `t = expfactor`,
lives on the diagonal pairs `(m, m)` of the same basis. Analytic
differentiation makes the PME force the exact gradient of this energy
(pinned by the FD check in Issue #211's investigation), so with `f` and
`g` averaged INDEPENDENTLY the `(f, g)`-Fourier modes are orthogonal and
each `(n, n')` pair enters the mean square in quadrature:

    χ²_pair = Σ_k Σ_{n,n'} (2π|m_n|)² (G̃ û_n û_{n'} - t(m_n) δ_{nn'})²

with `G̃ = t(m₀)/W²`, `W = Σ_n û_n` (per axis, product) — the
`bsp2factor = 1/W²` alias-sum identity of the Euler-spline `b(m)`. The
n' sum collapses to `S₂ = Σ_{n'} û_{n'}²` (per axis, product) in the code
below. Collapsing the n' sum COHERENTLY to `W` instead would describe one
particle pinned at a grid point rather than the position average — a
systematic UNDERestimate, up to 2× at large α·h (caught by review;
measured est/meas on uniform-random configurations is 0.93–1.02 for this
quadrature form across α ∈ [0.25, 0.6], K ∈ [16, 32]).

The self term is the same computation for `|Q̂₁|²` of ONE particle: modes
at `Δ∘K` with coefficients `C_Δ = Σ_k G̃ Σ_n û_n û_{n+Δ}` — coherent in n
correctly, because these are deterministic functions of a single position.

Truncations (validated in the test file): aliases n ∈ {-1,0,1} per axis
(n = ±2 shifts the result by < 1%), self Δ ∈ {-1,0,1}³, and k-vectors
with `(π/α)²m₀² > 30` skipped — every term of an alias class scales with
`t₀`-squared-class products (`|m₀|` minimizes `|m|` over the class), so
the cut discards ≤ e⁻⁶⁰ ≈ 1e-26 of the peak contribution. The cut also
bounds the WORK: the surviving ball has radius `√30·αL/π` index units
independent of the grid size. Measured cost of the full α optimization
(~15 estimate evaluations, once per run): 3 s / 14 s / 26 s on 64³ /
128³ / 192³ production-scale cells — a setup cost against hours of MD
(the looser cut of 100 measured 25–80 s on the larger two).
"""
function _pme_chi2_recip(cell::AbstractMatrix{Float64}, fftsize::NTuple{3,Int},
                         alpha::Float64)
    K = fftsize
    rcell = (cell \ I)'
    V = abs(det(cell))
    # t(m) and |m_phys|² for an integer frequency triple
    function tfun(m1, m2, m3)
        rvx = m1*rcell[1,1] + m2*rcell[2,1] + m3*rcell[3,1]
        rvy = m1*rcell[1,2] + m2*rcell[2,2] + m3*rcell[3,2]
        rvz = m1*rcell[1,3] + m2*rcell[2,3] + m3*rcell[3,3]
        m2n = rvx*rvx + rvy*rvy + rvz*rvz
        m2n == 0.0 && return (0.0, 0.0)
        return (exp(-(pi/alpha)^2 * m2n) / (pi*V*m2n), m2n)
    end
    ctr(k, Kd) = k > Kd ÷ 2 ? k - Kd : k    # centered principal index (mi convention)

    chi2_pair = 0.0
    Cd = zeros(3, 3, 3)                     # self-mode accumulators C_Δ, Δ ∈ {-1,0,1}³
    ua = (zeros(3), zeros(3), zeros(3))     # per-axis û(ν_n), n ∈ -1:1
    for k3 in 0:K[3]-1, k2 in 0:K[2]-1, k1 in 0:K[1]-1
        (k1 == 0 && k2 == 0 && k3 == 0) && continue
        m0 = (ctr(k1, K[1]), ctr(k2, K[2]), ctr(k3, K[3]))
        t0, m20 = tfun(m0...)
        (pi/alpha)^2 * m20 > 30.0 && continue
        for d in 1:3, (i, n) in enumerate(-1:1)
            ua[d][i] = _bspft((m0[d] + n*K[d]) / K[d])
        end
        W = sum(ua[1]) * sum(ua[2]) * sum(ua[3])
        S2 = sum(abs2, ua[1]) * sum(abs2, ua[2]) * sum(abs2, ua[3])
        G = t0 / W^2
        # quadrature over the n' (second-particle) aliases collapses to S₂:
        # Σ_{n'} (G û_n û_{n'} - t_n δ_{nn'})² = G²û_n²S₂ - 2Gû_n²t_n + t_n²
        for i3 in 1:3, i2 in 1:3, i1 in 1:3
            u2 = (ua[1][i1] * ua[2][i2] * ua[3][i3])^2
            m = (m0[1] + (i1-2)*K[1], m0[2] + (i2-2)*K[2], m0[3] + (i3-2)*K[3])
            if m == (0, 0, 0)
                continue
            elseif m == m0
                # principal alias: t(m) = t0 exactly
                chi2_pair += (2pi)^2 * m20 * (G^2*u2*S2 - 2*G*u2*t0 + t0^2)
            else
                # far aliases with negligible weight on BOTH sides can be
                # skipped before the exp/div of tfun (u⁴ below 1e-24 puts
                # the term > 12 digits under the leading aliases)
                u2 < 1e-12 && continue
                tm, m2n = tfun(m...)
                chi2_pair += (2pi)^2 * m2n * (G^2*u2*S2 - 2*G*u2*tm + tm^2)
            end
        end
        for i3 in 1:3, i2 in 1:3, i1 in 1:3
            (i1 == 2 && i2 == 2 && i3 == 2) && continue
            s = 1.0
            for (d, Δ) in ((1, i1-2), (2, i2-2), (3, i3-2))
                sd = 0.0
                for i in 1:3
                    1 <= i + Δ <= 3 || continue
                    sd += ua[d][i] * ua[d][i+Δ]
                end
                s *= sd
            end
            Cd[i1, i2, i3] += G * s
        end
    end
    chi2_self = 0.0
    for i3 in 1:3, i2 in 1:3, i1 in 1:3
        (i1 == 2 && i2 == 2 && i3 == 2) && continue
        dv = ((i1-2)*K[1], (i2-2)*K[2], (i3-2)*K[3])
        rvx = dv[1]*rcell[1,1] + dv[2]*rcell[2,1] + dv[3]*rcell[3,1]
        rvy = dv[1]*rcell[1,2] + dv[2]*rcell[2,2] + dv[3]*rcell[3,2]
        rvz = dv[1]*rcell[1,3] + dv[2]*rcell[2,3] + dv[3]*rcell[3,3]
        chi2_self += (2pi)^2 * (rvx^2 + rvy^2 + rvz^2) * Cd[i1, i2, i3]^2 / 4
    end
    return chi2_pair, chi2_self
end

"""
    pme_error_estimate(cell, fftsize, alpha, rc, charges)
        -> (; total, real, recip)

A-priori site-RMS force error [Ha/Bohr] of the split Coulomb evaluation
(real-space erfc sum cut at `rc` + PME reciprocal on `fftsize`) at
splitting parameter `alpha`, for the point charges `charges` (every
charged site — cores AND shells — one entry each; zero-charge sites
contribute nothing and are harmless).

`real` is Kolafa–Perram; `recip` aggregates [`_pme_chi2_recip`](@ref)'s
pair and self terms with the standard quadrature weights
(`Σᵢqᵢ²(Q²-qᵢ²)/N` and `Σᵢqᵢ⁴/N`); `total` is the quadrature sum of the
two branches. Exactly-zero charges are dropped BEFORE the aggregation:
the RMS is over charged sites only, so a rigid-ion frame whose shell
column is all zeros reports the same number as one without shells (zero
entries would dilute the site average by √2 while changing no force). An
all-zero `charges` returns exact zeros. See the file header for the
reliability domain — in one line: a few percent on homogeneous random
systems, a safe-side overestimate (× 1.1–2.7 measured) on thermal
crystals, and RMS, not max. One more reading note for the core-shell
model: the RMS is over SITES; a core and its shell sit within a fraction
of a Bohr of each other, so their reciprocal errors are strongly
correlated and partially cancel in the net per-ion force — safe side
again, but "per site", not "per ion".
"""
function pme_error_estimate(cell::AbstractMatrix, fftsize::NTuple{3,Int},
                            alpha::Real, rc::Real, charges::AbstractVector)
    cell64 = Matrix{Float64}(cell)
    q = filter(!iszero, Float64.(charges))
    isempty(q) && return (; total = 0.0, real = 0.0, recip = 0.0)
    N = length(q)
    Q2 = sum(abs2, q)
    V = abs(det(cell64))
    wpair = sum(qi^2 * (Q2 - qi^2) for qi in q) / N
    wself = sum(qi^4 for qi in q) / N
    # Kolafa–Perram real-space RMS, in the same per-pair aggregation
    # (their 2Q²/√(N rc V) form is the equal-|q| special case of wpair)
    err_real = sqrt(wpair) * 2 * exp(-Float64(alpha)^2 * Float64(rc)^2) /
               sqrt(Float64(rc) * V)
    c2p, c2s = _pme_chi2_recip(cell64, fftsize, Float64(alpha))
    err_recip = sqrt(wpair * c2p + wself * c2s)
    return (; total = sqrt(err_real^2 + err_recip^2),
              real = err_real, recip = err_recip)
end

"""
    pme_optimal_alpha(cell, fftsize, rc, charges) -> Float64

The α minimizing [`pme_error_estimate`](@ref)`.total` at fixed cost knobs
(`rc`, `fftsize`) — golden-section on log α over [0.02, 3.0] a.u., which
brackets the crossover for any sane (rc, mesh): the real-space branch
grows like exp(+α²rc²) leftward and the reciprocal branch like a power of
α·h rightward, so the objective is unimodal in between and a 1% α
resolution is far below the estimate's own accuracy.

For a chargeless system (Q² = 0) every α gives an exactly zero estimate;
returns 0.35 (arbitrary, the Coulomb term is identically zero).

The result is deterministic in (cell, fftsize, rc, charges) alone — under
MPI every rank computes the same α from the replicated frame, so the
choice is rank-uniform by construction.
"""
function pme_optimal_alpha(cell::AbstractMatrix, fftsize::NTuple{3,Int},
                           rc::Real, charges::AbstractVector)
    iszero(sum(abs2, charges)) && return 0.35
    f(loga) = pme_error_estimate(cell, fftsize, exp(loga), rc, charges).total
    lo, hi = log(0.02), log(3.0)
    gr = (sqrt(5.0) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = f(x1), f(x2)
    while hi - lo > 0.01
        if f1 < f2
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = f(x1)
        else
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = f(x2)
        end
    end
    return exp((lo + hi) / 2)
end
