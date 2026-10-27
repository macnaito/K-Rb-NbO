# test/calculators/test_pme_error.jl
#
# Issue #216: (1) the 2·3·5·7-smooth FFT grid sizing, (2) the a-priori
# RMS force error estimate (Kolafa–Perram real space + the exact k-space
# alias sum for THIS implementation's reciprocal operator), and (3) the
# α auto-determination built on it.
#
# The estimator is validated against direct measurement, not against
# itself: the reciprocal branch against PME−Ewald force differences on a
# uniform-random charge configuration (the homogeneous system the formula
# describes — agreement there is a few percent, so the bounds below are
# derivation-bug detectors, not statistical slack), the real-space branch
# against explicit cutoff-difference forces, and the α optimum against a
# measured total-error scan.

using Test
using Random
using LinearAlgebra

# reciprocal-only evaluations (no real-space part)
_epe_pme_F(atoms, prms) = begin
    a = deepcopy_atoms(atoms)
    pme = Calculators.PMEInfo(a.info[:lattice], prms)
    reset!(a)
    Calculators.calculate!(pme, a, prms)
    copy(a.arrays[:forces])
end
_epe_ewald_F(atoms, prms) = begin
    a = deepcopy_atoms(atoms)
    ew = Calculators.EwaldInfo(a.info[:lattice], prms)
    reset!(a)
    Calculators.calculate!(ew, a, prms)
    copy(a.arrays[:forces])
end
# real-space-only evaluation at cutoff rc
_epe_cb_F(atoms, prms) = begin
    a = deepcopy_atoms(atoms)
    nl = neighborlist(a, prms[:rc], 0.0)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
    copy(a.arrays[:forces])
end
_epe_rms(F) = sqrt(sum(abs2, F) / size(F, 1))
# a pme_meshsize that makes ceil(L/mesh) == K exactly (K smooth ⇒ no rounding)
_epe_mesh_for(L, K) = FMD.FPPREC(L / (K - 0.4))
# 2·3·5·7-smoothness predicate (no Primes.jl dependency)
function _epe_smooth2357(n::Int)
    for p in (2, 3, 5, 7)
        while n % p == 0
            n ÷= p
        end
    end
    return n == 1
end

@testset "PME error estimate + α auto-determination (Issue #216)" begin

    @testset "pme_fftsize: 2357-smooth rounding" begin
        # prime ceil → next smooth: ceil(21.316/0.93) = 23 → 24
        @test Calculators.pme_fftsize(21.316, 21.316, 21.316, 0.93) == (24, 24, 24)
        # already smooth: untouched
        @test Calculators.pme_fftsize(21.316, 21.316, 21.316, 21.316 / 23.5) ==
              (24, 24, 24)
        @test Calculators.pme_fftsize(20.0, 20.0, 20.0, 1.0) == (20, 20, 20)
        # the production trap from the issue: ceil = 61 (prime) → 63 = 7·3²
        @test Calculators.pme_fftsize(121.0, 121.0, 121.0, 2.0) == (63, 63, 63)
        # BSPORDER floor survives the rounding (ceil ≤ 3 ⇒ clamp to 4)
        @test Calculators.pme_fftsize(2.0, 2.0, 2.0, 1.0) == (4, 4, 4)
        # per-axis independence + invariants on assorted sizes
        fs = Calculators.pme_fftsize(10.0, 33.1, 101.0, 0.7)
        raw = ceil.(Int, (10.0, 33.1, 101.0) ./ 0.7)
        for (n, r) in zip(fs, raw)
            @test n >= r                       # never coarser than requested
            @test _epe_smooth2357(n)
        end
        # PMEInfo actually uses it
        atoms = make_nacl_64()
        L = norm(Array(atoms.info[:lattice])[1, :])
        prms = (; nacl_params(atoms)..., pme_meshsize = FMD.FPPREC(0.93 * L / 21.316))
        pme = Calculators.PMEInfo(atoms.info[:lattice], prms)
        @test pme.fftsize ==
              Calculators.pme_fftsize(L, L, L, Float64(prms[:pme_meshsize]))
        @test all(_epe_smooth2357, pme.fftsize)
    end

    # ----------------------------------------------------------------
    # Fixtures for the measurement comparisons. Uniform-random positions
    # are the estimator's model system; the rattled crystal probes the
    # documented safe-side behavior on correlated configurations.
    # ----------------------------------------------------------------
    atoms0 = make_nacl_64()
    L = norm(Array(atoms0.info[:lattice])[1, :])
    V = L^3
    rng = MersenneTwister(216)
    at_rand = deepcopy_atoms(atoms0)
    at_rand.arrays[:pos] .= rand(rng, 64, 3)
    at_ratt = deepcopy_atoms(atoms0)
    at_ratt.arrays[:pos] .+= 0.02 .* randn(rng, 64, 3)
    base = nacl_params(atoms0; ewald_tolerance = 1e-12)
    q = Float64[atoms0.arrays[:elemid][i] == 1 ? 1.0 : -1.0 for i in 1:64]

    @testset "reciprocal estimate vs measured (uniform-random)" begin
        # α up to 0.60 on purpose: the coherent-alias form this replaced
        # (review of PR #217) agreed at small α·h and fell to a 2×
        # UNDERestimate at large α·h — the corner points are what pin the
        # quadrature form.
        for alpha in (0.25, 0.40, 0.60), K in (16, 32)
            prms = merge(NamedTuple(base),
                         (; alpha = FMD.FPPREC(alpha),
                          pme_meshsize = _epe_mesh_for(L, K)))
            F_pme = _epe_pme_F(at_rand, prms)
            F_ew = _epe_ewald_F(at_rand, prms)
            meas = _epe_rms(F_pme .- F_ew)
            est = Calculators.pme_error_estimate(
                Matrix{Float64}(at_rand.info[:lattice]), (K, K, K),
                alpha, Float64(base[:rc]), q).recip
            # measured agreement on this fixture is 0.93–1.02; the bound
            # is wide enough to be seed-robust and tight enough that a
            # missing/spurious factor (√2, an alias term, the self term,
            # coherent-vs-quadrature n' collapse) fails
            @test 0.75 < est / meas < 1.3
        end
        # zero-charge sites must not dilute the site RMS (rigid-ion frames
        # carry an all-zero shell column): appending zeros changes nothing
        est_q = Calculators.pme_error_estimate(
            Matrix{Float64}(at_rand.info[:lattice]), (24, 24, 24),
            0.30, Float64(base[:rc]), q)
        est_q0 = Calculators.pme_error_estimate(
            Matrix{Float64}(at_rand.info[:lattice]), (24, 24, 24),
            0.30, Float64(base[:rc]), vcat(q, zeros(64)))
        @test est_q0 == est_q
    end

    @testset "reciprocal estimate is safe-side on a thermal crystal" begin
        alpha, K = 0.30, 24
        prms = merge(NamedTuple(base),
                     (; alpha = FMD.FPPREC(alpha), pme_meshsize = _epe_mesh_for(L, K)))
        meas = _epe_rms(_epe_pme_F(at_ratt, prms) .- _epe_ewald_F(at_ratt, prms))
        est = Calculators.pme_error_estimate(
            Matrix{Float64}(at_ratt.info[:lattice]), (K, K, K),
            alpha, Float64(base[:rc]), q).recip
        # charge correlation reduces the true error (measured 1.1–2.7×
        # overestimate on this fixture family) — the estimate must not
        # come out BELOW the truth on the configuration class MD visits
        @test est > 0.9 * meas
    end

    @testset "real-space (Kolafa–Perram) estimate vs measured" begin
        rc_ref = FMD.FPPREC(0.478 * L)          # just inside minimum image
        for alpha in (0.35, 0.45), rc in (5.0, 8.0)
            pt = merge(NamedTuple(base), (; alpha = FMD.FPPREC(alpha),
                                          rc = FMD.FPPREC(rc)))
            pr = merge(NamedTuple(base), (; alpha = FMD.FPPREC(alpha), rc = rc_ref))
            meas = _epe_rms(_epe_cb_F(at_ratt, pt) .- _epe_cb_F(at_ratt, pr))
            est = Calculators.pme_error_estimate(
                Matrix{Float64}(at_ratt.info[:lattice]), (32, 32, 32),
                alpha, rc, q).real
            # KP is a statistical estimate: literature-grade agreement is
            # tens of percent (measured 0.57–1.23 here)
            @test 1 / 3 < est / meas < 3
        end
    end

    @testset "pme_optimal_alpha: measured error is minimal near α*" begin
        K = 12                                   # ceil(L/2.0) = 11 → smooth 12
        rc = Float64(base[:rc])
        cell = Matrix{Float64}(atoms0.info[:lattice])
        αopt = Calculators.pme_optimal_alpha(cell, (K, K, K), rc, q)
        @test 0.05 < αopt < 1.0                  # interior of the bracket
        rc_ref = FMD.FPPREC(0.478 * L)
        function meas_total(alpha)
            prms = merge(NamedTuple(base),
                         (; alpha = FMD.FPPREC(alpha), pme_meshsize = _epe_mesh_for(L, K)))
            F = _epe_cb_F(at_ratt, prms) .+ _epe_pme_F(at_ratt, prms)
            pref = merge(NamedTuple(base), (; alpha = FMD.FPPREC(alpha), rc = rc_ref))
            Fref = _epe_cb_F(at_ratt, pref) .+ _epe_ewald_F(at_ratt, pref)
            return _epe_rms(F .- Fref)
        end
        e0 = meas_total(αopt)
        # the measured optimum need not sit exactly at the estimated one
        # (the estimate is safe-side on this correlated fixture), but a
        # factor 1.7 away in either direction must be measurably worse
        @test e0 < 1.2 * meas_total(αopt / 1.7)
        @test e0 < 1.2 * meas_total(αopt * 1.7)
        # chargeless guard
        @test Calculators.pme_optimal_alpha(cell, (K, K, K), rc, zeros(4)) == 0.35
    end

    @testset "_resolve_alpha: auto, override, legacy warn" begin
        atoms = make_nacl_64()
        minimal = (; pme_meshsize = FMD.FPPREC(2.0), rc = base[:rc],
                   qc = base[:qc], qs = base[:qs])
        # auto: α determined, logged, injected as FPPREC
        out = @test_logs (:info, r"Ewald α = .*\(auto\)") match_mode = :any begin
            FMD.Apps._resolve_alpha(atoms, minimal, "test", false)
        end
        @test out.alpha isa FMD.FPPREC
        @test 0.05 < out.alpha < 1.0
        # deterministic (rank-uniformity relies on this)
        out2 = @test_logs (:info, r"Ewald α") match_mode = :any begin
            FMD.Apps._resolve_alpha(atoms, minimal, "test", false)
        end
        @test out2.alpha == out.alpha
        # explicit :alpha wins and is reported as caller-supplied
        outov = @test_logs (:info, r"caller-supplied") match_mode = :any begin
            FMD.Apps._resolve_alpha(atoms, (; minimal..., alpha = FMD.FPPREC(0.3)),
                                    "test", false)
        end
        @test outov.alpha == FMD.FPPREC(0.3)
        # legacy :ewald_tolerance without :alpha → warned and ignored
        outlg = @test_logs (:warn, r"ewald_tolerance is no longer used") (:info, r"Ewald α") match_mode = :any begin
            FMD.Apps._resolve_alpha(atoms, (; minimal..., ewald_tolerance = 1e-6),
                                    "test", false)
        end
        @test outlg.alpha == out.alpha           # tol did not move α
    end
end
