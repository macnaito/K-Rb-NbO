# test/apps/test_nhc.jl
#
# Nosé–Hoover chain thermostat (Martyna–Klein–Tuckerman 1992). The
# propagator tests drive the control operators by hand on a force-free
# ideal gas (U ≡ 0), the suite's pattern for isolating a control operator
# from force-integration error (cf. test_csvr.jl / test_npt_cpu_smoke.jl);
# the driver test pins the params plumbing end-to-end.
using Test
using Random
using LinearAlgebra
using FMD.Apps: npt!

function _nhc_gas(seedv; T0 = 300.0)
    Random.seed!(seedv)
    a = make_nacl_64()
    m = a.arrays[:masses]
    a.arrays[:velo] .= randn(FMD.FPPREC, length(a), 3) .*
                       sqrt.(T0 / UNITS[:Kelvin] ./ m)
    a.arrays[:velo] .-= sum(a.arrays[:velo] .* m, dims = 1) ./ sum(m)
    return a
end

# Hand-driven npt! step (half-kicks are no-ops with F ≡ 0; COM stays zero
# under the uniform scalings).
function _nhc_step!(a, th, baro, c, dt)
    FMD.Apps.update!(a, th, baro, c, Array)
    FMD.Apps.scale_cell!(baro, a, c, Array)
    a.arrays[:pos] .+= (a.arrays[:velo] * dt) * (a.info[:lattice] \ I)
    FMD.Apps.update!(a, th, baro, c, Array)
    return nothing
end

_NHC_PRMS(dt) = (; target_stress = zeros(FMD.FPPREC, 6),
                   target_temperature = 300.0, timestep = dt,
                   tau_t = 20dt, tau_p = 200dt, nsteps = 1)

@testset "NHC with M = 1 is bitwise NoseHoover" begin
    # The chain propagator is written to perform exactly the NoseHoover
    # float operations when M = 1; any divergence (operator order, mass,
    # work booking) breaks bitwise identity of velocities, ζ and work.
    # Run under (·, mttk) so the barostat sandwich is exercised too.
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    prms = _NHC_PRMS(dt)
    tst = FMD.Apps.voigt2matrix(Vector{Float64}(prms[:target_stress]))
    a1 = _nhc_gas(21); a2 = _nhc_gas(21)
    N = length(a1)
    nh = FMD.Apps.NoseHoover(0.0, 0.0, Float64(20dt))
    nhc = FMD.Apps.NoseHooverChain(zeros(1), 0.0, Float64(20dt))
    b1 = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    b2 = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    for step in 1:300
        c = FMD.Apps._control_context(prms, step, N, tst)
        _nhc_step!(a1, nh, b1, c, dt)
        _nhc_step!(a2, nhc, b2, c, dt)
    end
    @test nh.zeta != 0.0                     # non-vacuity
    @test nhc.zeta[1] === nh.zeta
    @test nhc.work === nh.work
    @test a2.arrays[:velo] == a1.arrays[:velo]
    @test a2.info[:lattice] == a1.info[:lattice]
    c = FMD.Apps._control_context(prms, 300, N, tst)
    @test FMD.Apps.conserved_energy(nhc, b2, a2, c) ===
          FMD.Apps.conserved_energy(nh, b1, a1, c)
end

@testset "NHC conserved quantity, ideal gas" begin
    # Same fixture as the (nh, ·) cases in test_npt_cpu_smoke.jl. With
    # tau_t = 20·dt the light upper links (Q_j = kT·τ²) have canonical
    # rms ζ_j·dt = dt/τ = 0.05 (peaks ≈ 0.2), vs dt/(τ√(3N−3)) for the
    # single NH link, so the bounded O(dt²) wobble is larger than for NH.
    # Measured (M = 3): (·, mttk) below 2e-5 @1500; (·, none) 2.1e-5
    # @1500, bounded (≤ 2e-5 out to 12000 steps, no trend). A missing or
    # mis-signed chain term in the work booking / _control_energy is a
    # SECULAR O(kT·ζ_j·t) error, caught by the order test below rather than
    # by these magnitude bounds; with M = 3 every link must move.
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    prms = _NHC_PRMS(dt)
    tst = FMD.Apps.voigt2matrix(Vector{Float64}(prms[:target_stress]))
    for (baro_kind, nsteps, tol) in ((:mttk, 1500, 4e-5), (:none, 1500, 4e-5))
        a = _nhc_gas(130)
        N = length(a)
        th = FMD.Apps.NoseHooverChain(zeros(3), 0.0, Float64(20dt))
        baro = baro_kind === :mttk ?
            FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt)) :
            FMD.Apps.NoBarostat()
        lat0 = copy(a.info[:lattice])
        H(c) = FMD.Apps.conserved_energy(th, baro, a, c)
        H0 = H(FMD.Apps._control_context(prms, 1, N, tst))
        worst = 0.0
        for step in 1:nsteps
            c = FMD.Apps._control_context(prms, step, N, tst)
            _nhc_step!(a, th, baro, c, dt)
            worst = max(worst, abs(H(c) - H0))
        end
        @test worst / abs(H0) < tol
        @test all(!=(0.0), th.zeta)
        baro_kind === :none && @test a.info[:lattice] == lat0
    end
end

@testset "NHC conserved quantity converges at second order in dt" begin
    # Over a fixed physical time, halving dt must cut the worst |ΔH′| by
    # ≈ 4 (Strang splitting). Measured (M = 3, NVT, 1500 fs):
    # 3.02e-6 @0.5 fs → 6.72e-7 @0.25 fs, ratio 4.49. A bookkeeping error
    # in H′ (wrong link mass in _control_energy, a missing kT·ξ_j term)
    # does not vanish with dt, so the ratio collapses towards 1.
    fs = 1.0 / UNITS[:fs]
    function worst_drift(dt)
        a = _nhc_gas(130)
        N = length(a)
        prms = (; target_temperature = 300.0, timestep = dt, tau_t = 20fs,
                nsteps = 1)
        c = FMD.Apps._control_context(prms, 1, N, zeros(3, 3))
        th = FMD.Apps.NoseHooverChain(zeros(3), 0.0, 20fs)
        nob = FMD.Apps.NoBarostat()
        H() = FMD.Apps.conserved_energy(th, nob, a, c)
        H0 = H()
        worst = 0.0
        for _ in 1:round(Int, 1500fs / dt)
            FMD.Apps.update!(a, th, nob, c, Array)
            FMD.Apps.update!(a, th, nob, c, Array)
            worst = max(worst, abs(H() - H0))
        end
        return worst / abs(H0)
    end
    w1 = worst_drift(0.5fs)
    w2 = worst_drift(0.25fs)
    @test w2 > 0
    @test w1 / w2 > 3
end

@testset "NHC samples the canonical distribution; single NH does not" begin
    # Force-free gas + uniform velocity scaling: the velocity direction is
    # frozen, so the only dynamical variable is K (plus the chain). This is
    # the textbook non-ergodic case for a single NH thermostat — (ln K, ζ)
    # moves on a closed orbit set by the initial condition — and the case
    # the chain exists to fix. Canonical targets (g = 3N−3):
    #   K ~ Gamma(g/2, kT)  ⇒  ⟨2K⟩ = g·kT,  Var(K) = g·kT²/2
    #   ζ_j ~ N(0, kT/Q_j)  ⇒  ⟨Q_j ζ_j²⟩ = kT   (j = 1…M)
    # Only one of these is an identity of the dynamics: the top link is
    # undamped, dζ_M/dt = (Q_{M−1}ζ_{M−1}² − kT)/Q_M, so bounded ζ_M forces
    # ⟨Q_{M−1}ζ_{M−1}²⟩ = kT. Lower links obey ⟨G_j⟩ = Q_j⟨ζ_j ζ_{j+1}⟩,
    # which vanishes only under canonical sampling — so the remaining
    # checks measure canonicity, not bookkeeping. A wrong link mass or
    # link force shifts the corresponding average by O(1).
    # Starting at 200 K against a 300 K target, the single-NH orbit is
    # wide. Measured, 80k steps, seeds 77/1/2/3/4:
    #   M = 3: Var(K) ratio 1.004–1.012, ⟨2K⟩ ratio within 8e-4,
    #          ⟨Q_jζ_j²⟩/kT 0.99–1.034 (top-link identity 1.000)
    #   M = 1: Var(K) ratio 1.9–8.2 (2.11 for seed 77)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    T0 = 300.0
    kT = T0 / UNITS[:Kelvin]
    function sample(M)
        a = _nhc_gas(77; T0 = 200.0)
        N = length(a)
        prms = (; target_temperature = T0, timestep = dt, tau_t = 10dt, nsteps = 1)
        c = FMD.Apps._control_context(prms, 1, N, zeros(3, 3))
        th = FMD.Apps.NoseHooverChain(zeros(M), 0.0, Float64(10dt))
        nob = FMD.Apps.NoBarostat()
        for _ in 1:2000                         # warmup
            _nhc_step!(a, th, nob, c, dt)
        end
        nsamp = 80_000
        sK = 0.0; sK2 = 0.0
        sQz = zeros(M)
        for _ in 1:nsamp
            _nhc_step!(a, th, nob, c, dt)
            K = kinetic_energy(a)
            sK += K; sK2 += K^2
            for j in 1:M
                sQz[j] += FMD.Apps._mass_chain(th, c, j) * th.zeta[j]^2
            end
        end
        g = 3N - 3
        mK = sK / nsamp
        return (; twoK = 2mK / (g * kT),
                  varK = (sK2 / nsamp - mK^2) / (g * kT^2 / 2),
                  Qz = sQz ./ nsamp ./ kT)
    end
    M = 3
    r = sample(M)
    @test abs(r.Qz[M-1] - 1) < 0.005          # top-link identity
    @test abs(r.twoK - 1) < 0.005
    @test abs(r.varK - 1) < 0.05              # canonical KE fluctuation
    for j in 1:M
        @test abs(r.Qz[j] - 1) < 0.1          # every link, incl. the top one
    end
    r1 = sample(1)                            # control: single NH
    @test abs(r1.twoK - 1) < 0.005            # its one identity still holds …
    @test r1.varK > 1.5                       # … but K is not canonical
end

@testset "NHC builder / validation" begin
    @test FMD.Apps._build_thermostat((; thermostat = :nhc, tau_t = 1.0)) isa
          FMD.Apps.NoseHooverChain
    @test length(FMD.Apps._build_thermostat(
        (; thermostat = :nhc, tau_t = 1.0)).zeta) == 3          # default M
    @test length(FMD.Apps._build_thermostat(
        (; thermostat = :nhc, tau_t = 1.0, nhc_length = 5)).zeta) == 5
    @test_throws "requires params[:tau_t]" FMD.Apps._build_thermostat(
        (; thermostat = :nhc))
    @test_throws "nhc_length] must be an Integer >= 1" FMD.Apps._build_thermostat(
        (; thermostat = :nhc, tau_t = 1.0, nhc_length = 0))
    @test_throws "nhc_length] must be an Integer >= 1" FMD.Apps._build_thermostat(
        (; thermostat = :nhc, tau_t = 1.0, nhc_length = 2.0))
    @test_throws "nhc_length] must be an Integer >= 1" FMD.Apps._build_thermostat(
        (; thermostat = :nhc, tau_t = 1.0, nhc_length = true))
end

@testset "NHC driver smoke (nhc × mttk / nhc × none)" begin
    Random.seed!(13)
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
              thermostat = :nhc,
              nhc_length = 4,
              seed = 5,
              initial_temperature = 50.0,
              target_temperature = 50.0,
              output_interval = 2)

    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., tau_p = 500dt,
                     target_stress = zeros(FMD.FPPREC, 6), output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test Matrix(out.info[:lattice]) != lattice0
        lines = readlines(joinpath(dir, "log.txt"))
        @test length(lines) == 4
        @test all(l -> isfinite(parse(Float64, split(l)[18])), lines)
    end

    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., barostat = :none, output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test Matrix(out.info[:lattice]) == lattice0
        @test countlines(joinpath(dir, "log.txt")) == 4
    end

    mktempdir() do dir
        @test_throws "requires params[:tau_t]" npt!(
            deepcopy_atoms(atoms),
            (; Base.structdiff(common, NamedTuple{(:tau_t,)})...,
              barostat = :none, output_dir = dir))
    end
end
