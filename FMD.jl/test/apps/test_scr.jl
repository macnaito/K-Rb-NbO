# test/apps/test_scr.jl
#
# Anisotropic stochastic cell rescaling barostat (Bernetti–Bussi 2020;
# Del Tatto arXiv:2111.06402) — Issue #140. The kernel `_scr_mu` is unit
# tested directly; the ensemble tests drive the propagator by hand on a
# force-free ideal gas (the only cheap system with an ANALYTIC volume
# distribution — the κ_T input never enters the stationary state, so a
# theory comparison needs a system whose true equation of state is
# known); the driver/backends are pinned by smoke + parity + MPI tests.
using Test
using Random
using LinearAlgebra
using FMD.Apps: npt!

@testset "_scr_mu kernel (symmetry / covariance / mean)" begin
    Random.seed!(140)
    rng = FMD.Apps._control_rng(0, FMD.Apps._SCR_SALT)
    kT = 1e-3
    dt = 1.0

    # TRICLINIC reference cell (rows = lattice vectors): a cubic h₀ is
    # symmetric and cannot distinguish the hΣhᵀ orientation from hᵀΣh —
    # this one can (the wrong orientation misses the round trip below by
    # 3.6e-4, measured, vs 2e-19 for the correct one).
    L0 = [10.0 0.0 0.0; 2.0 9.0 0.0; 1.0 -1.5 11.0]
    aT = make_atoms(L0, [0.1 0.2 0.3; 0.6 0.7 0.8], [1, 2])
    V = volume(aT)
    tst = FMD.Apps.voigt2matrix([-2e-3, -1e-3, -3e-3, 4e-4, -2e-4, 1e-4])
    # κ = 100 (not a physical value — E[μ] = I − pref·bracket is EXACT for
    # any κ, no small-parameter approximation) lifts the deterministic
    # bracket 100× above the noise so the orientation discrimination
    # below has real power; at κ = 1 the orientation fault sat at 0.83×
    # the 4σ mean bound, i.e. INSIDE it (first-review blind spot).
    b = FMD.Apps.SCR(100.0, 100.0, rng, 0.0, zeros(3, 3))
    FMD.Apps._init_barostat!(b, aT, tst)

    # _init_barostat! contract: P0 is the hydrostatic part and Σ inverts
    # back to the deviatoric target at the reference cell:
    # h₀·Σ·h₀ᵀ / V₀ = S_ext − P₀I = −dev(σ_ext).
    @test b.P0 ≈ -tr(tst) / 3
    h0 = Matrix(transpose(L0))
    Sdev = -(tst - tr(tst)/3 * I)
    @test h0 * b.Sigma * transpose(h0) ./ V ≈ Sdev atol = 1e-12

    # Exact symmetry (the rotation-elimination contract) + mean/covariance
    # of μ (thesis Eq. 2.28, symmetrized per §2.6; −2kT/V measure
    # correction — see the _scr_mu comment). The expected mean is built
    # from the TARGET STRESS directly (at h = h₀ the strain coupling is
    # exactly S_dev), not from b.Sigma:
    #   E[μ] = I − (κ·dt/3τ)·[(P₀ − 2kT/V)I + S_int + S_dev]
    #   Var[μ_ii] = amp², Var[μ_ij] = amp²/2,  amp² = 2κ·kT·dt/(3V·τ)
    # Orientation detection power: swapping h·Σ·hᵀ → hᵀ·Σ·h inside
    # _scr_mu shifts E[μ] by pref·3.6e-4 = 1.2e-4 = 8.25× the 4σ mean
    # bound at κ = 100, and the explicit Emu_wrong assert separates the
    # hypotheses on top. Measured healthy: |mean−Emu| = 0.51× the bound,
    # discrimination margin 5.3×. Fault-injected: both asserts fail —
    # |mean−Emu| = 8.4× the bound and the mean lands on the
    # wrong-orientation hypothesis (|mean−Emu_wrong| = 7.5e-6 ≪
    # 3·|mean−Emu| = 3.7e-4).
    S_int = FMD.Apps.voigt2matrix([-1.5e-3, -1.5e-3, -1.5e-3, 1e-4, 0.0, 0.0])
    pref = b.kappa * dt / (3b.tau)
    Emu = Matrix{Float64}(I, 3, 3) .-
          pref .* ((b.P0 - 2kT/V) * Matrix{Float64}(I, 3, 3) + S_int + Sdev)
    amp2 = 2 * b.kappa * kT * dt / (3V * b.tau)
    M = 50_000
    sums = zeros(3, 3)
    sq = zeros(3, 3)
    sym_ok = true
    for _ in 1:M
        mu = FMD.Apps._scr_mu(b, S_int, V, L0, kT, dt, rng)
        sym_ok &= mu == transpose(mu)      # exactly symmetric, every draw
        sums .+= mu
        sq .+= (mu .- Emu) .^ 2
    end
    @test sym_ok
    mean_ = sums ./ M
    var_ = sq ./ M
    @test maximum(abs, mean_ .- Emu) < 4 * sqrt(amp2 / M)
    # Wrong-orientation hypothesis: the strain coupling evaluated as
    # hᵀ·Σ·h (the coordinate-frame mixup a rows/columns confusion would
    # produce). The measured mean must sit ≥ 3× closer to the correct
    # bracket than to the wrong one.
    Sdev_wrong = transpose(h0) * b.Sigma * h0 ./ V
    Emu_wrong = Matrix{Float64}(I, 3, 3) .-
                pref .* ((b.P0 - 2kT/V) * Matrix{Float64}(I, 3, 3) + S_int +
                         Sdev_wrong)
    @test 3 * maximum(abs, mean_ .- Emu) < maximum(abs, mean_ .- Emu_wrong)
    for i in 1:3
        @test abs(var_[i, i] / amp2 - 1) < 0.05
    end
    @test abs(var_[1, 2] / (amp2 / 2) - 1) < 0.05
    @test abs(var_[1, 3] / (amp2 / 2) - 1) < 0.05
    @test abs(var_[2, 3] / (amp2 / 2) - 1) < 0.05

    # Seed determinism: same stream state ⇒ bitwise-identical μ.
    r1 = FMD.Apps._control_rng(9, FMD.Apps._SCR_SALT)
    r2 = FMD.Apps._control_rng(9, FMD.Apps._SCR_SALT)
    @test FMD.Apps._scr_mu(b, S_int, V, L0, kT, dt, r1) ==
          FMD.Apps._scr_mu(b, S_int, V, L0, kT, dt, r2)
end

@testset "_scr_mu Itô drift of ln det μ (symmetrized-noise value)" begin
    # The −2kT/V measure correction is defined as the cancellation of the
    # Itô drift of ln det μ (docs/src/theory/scr-barostat.md §4): for the
    # SYMMETRIZED Wiener tensor E[ln det μ − tr(μ−I)] = −3·amp², and the
    # trace of the −c·kT/V drift term is +c·(3/2)·amp², so c = 2. The
    # full 9-dof tensor of thesis Eq. 2.17 gives −1.5·amp² ⇒ the thesis
    # −kT/V. This control-variate Monte Carlo pins the factor directly.
    # Measured at M = 2e6: −1.99968e-8 vs theory −2.0e-8 (0.02%); the
    # per-sample residual has variance O(amp⁴), so M = 2e5 puts the two
    # hypotheses ≈ 30σ apart; the 15% band separates them by > 3×.
    rng = FMD.Apps._control_rng(1, FMD.Apps._SCR_SALT)
    kT = 1e-3
    dt = 1.0
    V = 1000.0
    L = Matrix{Float64}(10I, 3, 3)
    b = FMD.Apps.SCR(100.0, 1.0, rng, 5e-4, zeros(3, 3))
    S_int = FMD.Apps.voigt2matrix([-4e-4, -4e-4, -4e-4, 0.0, 0.0, 0.0])
    amp2 = 2 * b.kappa * kT * dt / (3V * b.tau)
    M = 200_000
    s = 0.0
    for _ in 1:M
        mu = FMD.Apps._scr_mu(b, S_int, V, L, kT, dt, rng)
        s += log(det(mu)) - tr(mu - Matrix{Float64}(I, 3, 3))
    end
    @test abs(s / M / (-3amp2) - 1) < 0.15
end

# Hand-driven npt! step skeleton on the force-free NaCl-8 ideal gas
# (U ≡ 0, virial ≡ 0 ⇒ stress(atoms, true) is the exact ideal-gas
# kinetic stress). Returns (mean, variance) of V sampled every 200 steps
# after a 10k-step warmup; for :mttk additionally the time average of the
# cell kinetic term Q_b·Σζ_b² (equipartition diagnostic, Issue #132).
function _gas_volume_stats(baro_kind, seed; nsteps = 100_000)
    Random.seed!(seed)
    a = make_nacl_8()
    N = length(a)
    T0 = 300.0
    kT = T0 / UNITS[:Kelvin]
    m = a.arrays[:masses]
    a.arrays[:velo] .= randn(FMD.FPPREC, N, 3) .* sqrt.(T0 / UNITS[:Kelvin] ./ m)
    a.arrays[:velo] .-= sum(a.arrays[:velo] .* m, dims = 1) ./ sum(m)
    P0 = N * kT / volume(a)            # Gamma(N) mean sits at the initial V
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    taup = 100dt
    tst = FMD.FPPREC.(-P0 .* [1, 1, 1, 0, 0, 0])
    tsm = FMD.Apps.voigt2matrix(Vector{Float64}(tst))
    prms = (; target_stress = tst, target_temperature = T0,
            timestep = dt, tau_t = 20dt, tau_p = taup, nsteps = 1)
    th = FMD.Apps.CSVR(Float64(20dt),
                       FMD.Apps._control_rng(seed, FMD.Apps._CSVR_SALT), 0.0)
    baro = if baro_kind === :scr
        s = FMD.Apps.SCR(Float64(taup), 1 / P0,
                         FMD.Apps._control_rng(seed, FMD.Apps._SCR_SALT),
                         0.0, zeros(3, 3))
        FMD.Apps._init_barostat!(s, a, tsm)
        s
    else
        FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(taup))
    end
    c = FMD.Apps._control_context(prms, 1, N, tsm)
    Vs = Float64[]
    cell2K = 0.0
    ncell = 0
    for step in 1:nsteps
        FMD.Apps.update!(a, th, baro, c, Array)
        FMD.Apps.scale_cell!(baro, a, c, Array)
        a.arrays[:pos] .+= (a.arrays[:velo] * c.dt) * (a.info[:lattice] \ I)
        FMD.Apps.update!(a, th, baro, c, Array)
        if step > 10_000 && step % 200 == 0
            push!(Vs, volume(a))
            if baro isa FMD.Apps.MTTK
                cell2K += FMD.Apps._mass_b(baro, c) * sum(abs2, baro.zeta_b)
                ncell += 1
            end
        end
    end
    mV = sum(Vs) / length(Vs)
    return (; mean = mV, var = sum(abs2, Vs .- mV) / (length(Vs) - 1),
            kT, P0, N, cell2K = ncell > 0 ? cell2K / ncell : NaN)
end

@testset "SCR ideal-gas volume distribution = Gamma(N) (Issues #140/#132)" begin
    # Stationary theory for N atoms at pressure P₀ with the COM frozen in
    # scaled coordinates (docs/src/theory/scr-barostat.md §3): the
    # configurational measure is V^(N−1), so P(V) ∝ V^(N−1)·e^(−βP₀V) =
    # Gamma(N, kT/P₀), i.e. ⟨V⟩ = N·kT/P₀ and Var(V) = N·(kT/P₀)². N = 8
    # makes the test sharp: one V-power moves the mean by 12.5%, so the
    # 5% bound (≈ 3σ of the ~450-sample mean, σ_V/⟨V⟩ ≈ 0.35)
    # discriminates every neighbouring hypothesis. Measured with −2kT/V
    # (seeds 7/11): mean/theory = 1.007/1.010, var/theory = 0.994/0.978.
    # The correction constant maps linearly onto the sampled power
    # (Gamma(N−3+c), theory note §4): the earlier −3kT/V calibration
    # measured Gamma(N+1) (mean/Gamma(N) ≈ 9/8 = 1.13) and the verbatim
    # thesis −kT/V measured Gamma(N−1) (mean/Gamma(N) ≈ 7/8 = 0.88) with
    # this very driver — both are the faults this test catches. The κ_T
    # input is deliberately exact (1/P₀): it only sets the relaxation
    # time; the sampled distribution must not depend on it.
    r = _gas_volume_stats(:scr, 7)
    theory_mean = r.N * r.kT / r.P0
    theory_var = r.N * (r.kT / r.P0)^2
    @test r.var > 0
    @test abs(r.mean / theory_mean - 1) < 0.05
    @test abs(r.var / theory_var - 1) < 0.20
end

@testset "MTTK ideal-gas Gamma(N) + cell equipartition (Issue #132)" begin
    # MTTK must sample the SAME ensemble as SCR — the 6-DOF cell
    # convention adopted in PR #137 (Q_cell = 6kT·τ_p², cell NH target
    # 6kT) makes both barostats target Gamma(N, kT/P₀) on the ideal gas.
    # Measured (seeds 11/7): mean/theory = 1.003/0.987, var/theory =
    # 1.035/0.992 — same bounds as the SCR testset (≥ 5× margin).
    #
    # Equipartition diagnostic (Issue #132): the symmetric ζ_b has 6
    # independent components, each carrying kT/2 — the off-diagonal
    # weight-2 entries of Σζ² do not change the count — so
    # ⟨Q_b·Σζ_b²⟩ = 6kT. Measured 1.008/1.029 of 6kT (seeds 11/7).
    # Role: the cell NH enforces its own target (⟨G_cell⟩ = 0 in
    # stationarity), so this assertion pins the IMPLEMENTED constant —
    # a revert detector, not an independent physics check; the physical
    # 6-DOF count is established analytically in
    # docs/src/theory/mttk-cell-dof.md. Revert-detection power, measured
    # by injecting the 9-DOF count (ckT = 9kT AND Q_cell = 9kT·τ_p²):
    # cell2K/6kT = 1.539/1.518 — 3.4× beyond the 15% bound. The same
    # fault moves the V stats only to var/theory = 1.15/1.17 (inside the
    # 20% band) and mean 1.003/0.990: the Gamma(N) assertions alone
    # would MISS it at this test's statistics, which is why this check
    # exists.
    r = _gas_volume_stats(:mttk, 11)
    theory_mean = r.N * r.kT / r.P0
    theory_var = r.N * (r.kT / r.P0)^2
    @test abs(r.mean / theory_mean - 1) < 0.05
    @test abs(r.var / theory_var - 1) < 0.20
    @test abs(r.cell2K / (6r.kT) - 1) < 0.15

    # Ensemble equivalence (the strengthened form of the old
    # (N+1)/N-ratio cross-check): identical Gamma(N) targets ⇒ the two
    # barostats' V means agree directly. Measured ratio 1.007 (seed 11).
    r_scr = _gas_volume_stats(:scr, 11)
    @test abs(r_scr.mean / r.mean - 1) < 0.05
end

@testset "SCR driver smoke + builder guardrails" begin
    Random.seed!(14)
    atoms = make_nacl_64()
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    lattice0 = copy(Matrix(atoms.info[:lattice]))
    common = (; base...,
              backend = :cpu,
              elements = Dict("Na" => 1, "Cl" => 2),
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              nsteps = 4,
              timestep = dt,
              tau_t = 20dt,
              tau_p = 500dt,
              seed = 5,
              initial_temperature = 50.0,
              target_temperature = 50.0,
              target_stress = zeros(FMD.FPPREC, 6),
              output_interval = 2)

    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., barostat = :scr,
                     compressibility = UNITS[:GPa] / 24.0,  # κ_T from B ≈ 24 GPa
                     output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test maximum(abs, out.arrays[:forces]) > 1e-6
        @test Matrix(out.info[:lattice]) != lattice0     # the cell moved
        lines = readlines(joinpath(dir, "log.txt"))
        @test length(lines) == 4
        cols = split(lines[end])
        @test length(cols) == 20
        @test isfinite(parse(Float64, cols[18]))
        @test parse(Int, cols[end]) >= 0
        # The rattled fixture is not pre-relaxed, so an all-zero ncg
        # column can only mean the counter (or the CG) stopped working.
        @test any(l -> parse(Int, last(split(l))) > 0, lines)
    end

    # Builder guardrails (direct npt! calls bypass validate_params).
    mktempdir() do dir
        @test_throws "requires params[:compressibility]" npt!(
            deepcopy_atoms(atoms),
            (; common..., barostat = :scr, output_dir = dir))
        @test_throws "requires a thermostat" npt!(
            deepcopy_atoms(atoms),
            (; common..., barostat = :scr, thermostat = :none,
              compressibility = 1.0, output_dir = dir))
        @test_throws "requires params[:target_stress]" npt!(
            deepcopy_atoms(atoms),
            (; Base.structdiff(common, NamedTuple{(:target_stress,)})...,
              barostat = :scr, compressibility = 1.0, output_dir = dir))
        # Euler stability guard: the too-LARGE unit mistake
        # (B·UNITS[:GPa] instead of UNITS[:GPa]/B) makes |μ−I| leave the
        # perturbative regime on the first step — refuse instead of
        # integrating garbage (CLAUDE.md: better broken than wrong).
        # The too-SMALL mistake (1/B) is NOT detectable — μ ≈ I mimics
        # stiff physics; the ensemble stays correct with τ_eff stretched
        # ~3e4× (documented in parameters.md), so no test asserts it.
        @test_throws "left the perturbative regime" npt!(
            deepcopy_atoms(atoms),
            (; common..., barostat = :scr,
              compressibility = 24.0 * UNITS[:GPa], output_dir = dir))
    end
end
