# test/apps/test_csvr.jl
#
# CSVR (stochastic velocity rescaling; Bussi–Donadio–Parrinello 2007)
# thermostat — Issue #138. The distribution tests drive the propagator by
# hand on a force-free ideal gas (U ≡ 0), the suite's pattern for
# isolating a control operator from force-integration error; the driver
# tests pin seed plumbing and the (csvr × mttk) / (csvr × none) paths
# end-to-end.
using Test
using Random
using LinearAlgebra
using FMD.Apps: npt!

# Hand-driven npt! step skeleton (F ≡ 0; COM stays zero under the scalar
# rescalings). One full MD step = two update! calls.
function _csvr_step!(a, th, c)
    nob = FMD.Apps.NoBarostat()
    FMD.Apps.update!(a, th, nob, c, Array)
    FMD.Apps.update!(a, th, nob, c, Array)
    return nothing
end

function _csvr_gas(seedv)
    Random.seed!(seedv)
    a = make_nacl_64()
    m = a.arrays[:masses]
    a.arrays[:velo] .= randn(FMD.FPPREC, length(a), 3) .*
                       sqrt.(300.0 / UNITS[:Kelvin] ./ m)
    a.arrays[:velo] .-= sum(a.arrays[:velo] .* m, dims = 1) ./ sum(m)
    return a
end

@testset "CSVR kinetic-energy distribution vs canonical χ² (Issue #138)" begin
    # Stationary theory (canonical, Nf = 3N−3 = 189 DOF):
    #   ⟨K⟩ = Nf·kT/2,  Var(K) = Nf·(kT)²/2.
    # 250 samples spaced 2τ apart (τ = 10·dt): rel. std of the sample
    # mean ≈ √(2/Nf)/√250 ≈ 0.65% → the 2% bound is ≈ 3σ; rel. std of the
    # sample variance ≈ √(2/250) ≈ 9% → the 30% bound is ≥ 3σ. Measured
    # on this seed: mean −0.19%, var −11.6%. The test has teeth: dropping
    # the noise term (the Berendsen limit of the propagator) collapses
    # the variance — measured var/theory − 1 = −1.0, far outside the
    # band, while the mean stays perfect.
    a = _csvr_gas(2007)
    N = length(a)
    T0 = 300.0
    kT = T0 / UNITS[:Kelvin]
    Nf = 3N - 3
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    prms = (; target_temperature = T0, timestep = dt, tau_t = 10dt, nsteps = 1)
    c = FMD.Apps._control_context(prms, 1, N, zeros(3, 3))
    th = FMD.Apps.CSVR(Float64(10dt),
                       FMD.Apps._control_rng(0, FMD.Apps._CSVR_SALT), 0.0)

    for _ in 1:500                      # warmup ≈ 50τ
        _csvr_step!(a, th, c)
    end
    Ks = Float64[]
    for step in 1:5000
        _csvr_step!(a, th, c)
        step % 20 == 0 && push!(Ks, kinetic_energy(a))
    end
    @test length(Ks) == 250
    mK = sum(Ks) / length(Ks)
    vK = sum(abs2, Ks .- mK) / (length(Ks) - 1)
    @test vK > 0                        # non-vacuity: the noise actually acts
    @test abs(mK / (Nf * kT / 2) - 1) < 0.02
    @test abs(vK / (Nf * kT^2 / 2) - 1) < 0.30
end

@testset "CSVR one-step conditional mean pins DOF and K̄" begin
    # The OU mean is exact for any Δt: E[K′|K] = c̃·K + (1−c̃)·K̄. Driven
    # at c̃ = e⁻¹ (τ = Δt, the strongest signal) from a FIXED off-target
    # K (K/K̄ ≈ 0.87), 10⁵ single applications pin the two quantities the
    # stationary-distribution test cannot resolve: an off-by-one in the
    # χ² DOF (S ~ χ²_Nf instead of χ²_{Nf−1}) shifts the mean by
    # (1−c̃)K̄/Nf ≈ +3.5e-3 relative — measured +3.53e-3 with the fault
    # injected — while the healthy propagator lands at +2.0e-5
    # (statistical σ ≈ 3e-4). The 1.5e-3 bound separates them by > 2×
    # either way. A wrong K̄ scale shifts the mean proportionally.
    Random.seed!(2007)
    a = make_nacl_64()
    N = length(a)
    T0 = 300.0
    kT = T0 / UNITS[:Kelvin]
    Nf = 3N - 3
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    m = a.arrays[:masses]
    a.arrays[:velo] .= randn(FMD.FPPREC, N, 3) .*
                       sqrt.(250.0 / UNITS[:Kelvin] ./ m)
    a.arrays[:velo] .-= sum(a.arrays[:velo] .* m, dims = 1) ./ sum(m)
    v0 = copy(a.arrays[:velo])
    K0 = kinetic_energy(a)

    prms = (; target_temperature = T0, timestep = dt, tau_t = dt, nsteps = 1)
    c = FMD.Apps._control_context(prms, 1, N, zeros(3, 3))
    # _thermo_half! propagates Δt = dt/4 with time constant th.tau:
    # th.tau = dt/4 gives c̃ = e⁻¹ exactly.
    th = FMD.Apps.CSVR(Float64(dt) / 4,
                       FMD.Apps._control_rng(2, FMD.Apps._CSVR_SALT), 0.0)
    M = 100_000
    s = 0.0
    for _ in 1:M
        a.arrays[:velo] .= v0
        FMD.Apps._thermo_half!(th, a, c, Array)
        s += kinetic_energy(a)
    end
    theory = exp(-1.0) * K0 + (1 - exp(-1.0)) * (Nf * kT / 2)
    @test abs(s / M / theory - 1) < 1.5e-3
end

@testset "CSVR effective energy is exactly conserved (force-free)" begin
    # With U ≡ 0 the BDP effective energy is K + work, and work books the
    # ACTUALLY-applied (post-rounding) α — the only residual is the FP
    # reassociation of the kinetic-energy sum across rescalings (measured
    # 6.9e-15 relative over 2000 steps before the exact-α booking, same
    # order after; 1e-12 gives ≥ 100× headroom while still catching any
    # real bookkeeping term, which would be O(kT/K) ≈ 1e-2 per step).
    a = _csvr_gas(2008)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    prms = (; target_temperature = 300.0, timestep = dt, tau_t = 10dt,
            nsteps = 1)
    c = FMD.Apps._control_context(prms, 1, length(a), zeros(3, 3))
    th = FMD.Apps.CSVR(Float64(10dt),
                       FMD.Apps._control_rng(1, FMD.Apps._CSVR_SALT), 0.0)
    E0 = kinetic_energy(a) + th.work
    worst = 0.0
    for _ in 1:2000
        _csvr_step!(a, th, c)
        worst = max(worst, abs(kinetic_energy(a) + th.work - E0))
    end
    @test worst / E0 < 1e-12
    @test th.work != 0.0                # non-vacuity: rescales happened
end

@testset "CSVR × MTTK conserved quantity, ideal gas" begin
    # The recommended production combination: H′ = K + CSVR work + MTTK
    # terms must stay flat through the composed splitting (a phase
    # mismatch between the CSVR work booking and the MTTK sandwich would
    # show up as secular drift). Measured on this fixture: 7.7e-6 @1500,
    # 1.5e-5 @3000, 2.5e-5 @5000 — the same slow linear growth (and
    # magnitude) as (nh, mttk); threshold 2e-5 @1500 sits ≥ 2.6× above.
    Random.seed!(130)
    a = _csvr_gas(130)
    N = length(a)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    prms = (; target_stress = zeros(FMD.FPPREC, 6), target_temperature = 300.0,
            timestep = dt, tau_t = 20dt, tau_p = 200dt, nsteps = 1)
    tst = FMD.Apps.voigt2matrix(Vector{Float64}(prms[:target_stress]))
    th = FMD.Apps.CSVR(Float64(20dt),
                       FMD.Apps._control_rng(0, FMD.Apps._CSVR_SALT), 0.0)
    baro = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    H(c) = FMD.Apps.conserved_energy(th, baro, a, c)
    H0 = H(FMD.Apps._control_context(prms, 1, N, tst))
    worst = 0.0
    for step in 1:1500
        c = FMD.Apps._control_context(prms, step, N, tst)
        FMD.Apps.update!(a, th, baro, c, Array)
        FMD.Apps.scale_cell!(baro, a, c, Array)
        a.arrays[:pos] .+= (a.arrays[:velo] * dt) * (a.info[:lattice] \ I)
        FMD.Apps.update!(a, th, baro, c, Array)
        worst = max(worst, abs(H(c) - H0))
    end
    @test worst / abs(H0) < 2e-5
    @test th.work != 0.0                       # both controls really acted
    @test abs(baro.zeta_cell) > 1e-5
end

@testset "CSVR seed plumbing (Issue #34 pattern)" begin
    Random.seed!(11)
    atoms = make_nacl_64()
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    common = (; base...,
              backend = :cpu,
              elements = Dict("Na" => 1, "Cl" => 2),
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              timestep = dt,
              tau_t = 20dt,
              initial_temperature = 50.0,
              target_temperature = 50.0,
              output_interval = 10)
    function run_csvr(seed, ambient; nsteps = 2, thermostat = :csvr)
        Random.seed!(ambient)          # must NOT influence the trajectory
        mktempdir() do dir
            npt!(deepcopy_atoms(atoms),
                 (; common..., thermostat, barostat = :none,
                   nsteps = nsteps, seed = seed, output_dir = dir))
        end
    end

    # Bitwise reproducibility under the same seed, independent of the
    # ambient global RNG (single-threaded CPU run).
    o1 = run_csvr(42, 1)
    o2 = run_csvr(42, 2)
    @test Array(o1.arrays[:velo]) == Array(o2.arrays[:velo])
    @test Array(o1.arrays[:pos]) == Array(o2.arrays[:pos])
    # A different seed changes the noise (hence the trajectory).
    o3 = run_csvr(7, 3)
    @test Array(o1.arrays[:velo]) != Array(o3.arrays[:velo])

    # Stream separation: the CSVR salt keeps the rescaling noise out of
    # the id-keyed initial-velocity stream — at nsteps = 0 (init only,
    # thermostat never applied) :csvr and :nh give bitwise-identical
    # draws for the same seed.
    z_csvr = run_csvr(42, 4; nsteps = 0)
    z_nh = run_csvr(42, 5; nsteps = 0, thermostat = :nh)
    @test Array(z_csvr.arrays[:velo]) == Array(z_nh.arrays[:velo])
end

@testset "CSVR driver smoke (csvr × mttk / csvr × none)" begin
    Random.seed!(12)
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
              thermostat = :csvr,
              seed = 5,
              initial_temperature = 50.0,
              target_temperature = 50.0,
              output_interval = 2)

    # csvr × mttk — the Issue #116 recommended production combination.
    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., tau_p = 500dt,
                     target_stress = zeros(FMD.FPPREC, 6), output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test maximum(abs, out.arrays[:forces]) > 1e-6
        @test Matrix(out.info[:lattice]) != lattice0
        lines = readlines(joinpath(dir, "log.txt"))
        @test length(lines) == 4
        cols = split(lines[end])
        @test length(cols) == 19
        @test isfinite(parse(Float64, cols[18]))
        @test parse(Int, cols[end]) >= 0
        # The rattled fixture is not pre-relaxed, so an all-zero ncg
        # column can only mean the counter (or the CG) stopped working.
        @test any(l -> parse(Int, last(split(l))) > 0, lines)
    end

    # csvr × none — the fixed-cell ε(ω)-production mode.
    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., barostat = :none, output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test Matrix(out.info[:lattice]) == lattice0
        @test countlines(joinpath(dir, "log.txt")) == 4
    end

    # Builder guardrails (direct npt! calls bypass validate_params).
    mktempdir() do dir
        @test_throws "requires an Integer params[:seed]" npt!(
            deepcopy_atoms(atoms),
            (; Base.structdiff(common, NamedTuple{(:seed,)})...,
              barostat = :none, initial_temperature = 0.0, output_dir = dir))
        @test_throws "requires target_temperature >= 1 K" npt!(
            deepcopy_atoms(atoms),
            (; common..., barostat = :none, target_temperature = 0.5,
              output_dir = dir))
        @test_throws "requires params[:tau_t]" npt!(
            deepcopy_atoms(atoms),
            (; Base.structdiff(common, NamedTuple{(:tau_t,)})...,
              barostat = :none, output_dir = dir))
    end
end
