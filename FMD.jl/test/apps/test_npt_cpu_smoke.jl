# test/apps/test_npt_cpu_smoke.jl
#
# CPU npt! smoke (params[:backend] = :cpu) — the first harness that calls
# npt! itself.  The GPU profile's npt_short mirrors the force/list pipeline
# but never enters the driver; this test pins the driver end-to-end on the
# default profile with a zero-coefficient SplineParam standing in for the
# MLIP (its Sub-derived lists and kernels still run).
using Test
using Random
using LinearAlgebra
using Logging: Warn
using FMD.Apps: npt!
import MPI

@testset "npt! CPU smoke (backend = :cpu)" begin
    Random.seed!(2)
    atoms = make_nacl_64()
    # Rattle: the ideal rocksalt has zero forces by symmetry, which would
    # make the run (and the shell CG) trivial.
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    splp = Calculators.SplineParam(FMD.FPPREC(0.6 * base[:rc]),
                                   FMD.FPPREC(0.4 * base[:rc]), 2, 4, 4)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    lattice0 = copy(Matrix(atoms.info[:lattice]))

    mktempdir() do dir
        # nsteps = 4 so the shell-history warmup completes and step 4 runs
        # the real 3-point extrapolation branch (shist_count reaches 3
        # after the third rotation).
        params = (; base...,
                  backend = :cpu,
                  splp = splp,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1),
                  cgtol = FMD.FPPREC(2e-5),
                  nsteps = 4,
                  timestep = dt,
                  tau_t = 20dt,
                  tau_p = 500dt,
                  initial_temperature = 0.0,   # keep the ambient RNG out (Issue #34)
                  target_temperature = 10.0,
                  target_stress = zeros(FMD.FPPREC, 6),
                  output_interval = 2,
                  output_dir = dir)
        logs, out = Test.collect_test_logs() do
            npt!(deepcopy_atoms(atoms), params)
        end

        # Stays on the host and finishes finite.
        @test out.arrays[:pos] isa Matrix{FMD.FPPREC}
        @test all(isfinite, out.arrays[:pos])
        @test all(isfinite, out.arrays[:forces])
        @test all(isfinite, out.arrays[:shelldisp])
        @test all(isfinite, Vector(out.info[:dipole_phys]))

        # The rattled crystal actually produced forces (non-vacuous run),
        # and the MTTK barostat actually moved the cell.
        @test maximum(abs, out.arrays[:forces]) > 1e-6
        @test Matrix(out.info[:lattice]) != lattice0

        # The rebuild path fired at least once during the run (one
        # "Building neighbor list..." for the initial build, plus one or
        # more mid-run rebuilds under skin = 0.1).
        nbuilds = count(l -> l.message == "Building neighbor list...", logs)
        @test nbuilds >= 2

        # Driver output: one log line per step, snapshots at the interval.
        @test countlines(joinpath(dir, "log.txt")) == 4
        # 20 columns: temp | lattice(9) | dipole(3) | dipole_phys(3) |
        # volume | conserved (Issue #136) | shell_rms | ncg (Issue #144);
        # conserved finite, ncg a non-negative count.
        cols = split(first(eachline(joinpath(dir, "log.txt"))))
        @test length(cols) == 20
        @test isfinite(parse(Float64, cols[18]))
        @test parse(Int, cols[end]) >= 0
        # shell_rms: every step ran the CG, so finite on every line; and
        # rms ≤ max < cgtol on every step that exited by convergence
        # (the stopping test bounds the maximum component, and the rms of
        # the same components cannot exceed it).
        for l in eachline(joinpath(dir, "log.txt"))
            c = split(l)
            rms = parse(Float64, c[19])
            @test isfinite(rms) && rms >= 0
            parse(Int, c[20]) < params[:cgmaxiter] && @test rms < params[:cgtol]
        end
        @test isfile(joinpath(dir, "step_2.xyz"))
        @test isfile(joinpath(dir, "step_4.xyz"))
        @test !isfile(joinpath(dir, "step_3.xyz"))
    end

    # Unknown backend values error loudly (message-matched so this pins the
    # backend check itself, not some later missing-key error).
    @test_throws "must be :cuda or :cpu" npt!(deepcopy_atoms(atoms),
                                              (; base..., backend = :gpu))

    # DD opt-in failure path (no mpiexec involved). The MPI-uninitialized
    # message is only asserted while MPI really is uninitialized — the
    # fitting smokes later in the suite call MPI.Init() in-process, which
    # is exactly why the opt-in is explicit rather than inferred.
    if !MPI.Initialized()
        @test_throws "MPI.Init() has not been called" npt!(deepcopy_atoms(atoms),
            (; base..., backend = :cpu, mpi = true))
    end
end

@testset "npt! pure shell model — no MLIP (Issue #133)" begin
    Random.seed!(3)
    atoms = make_nacl_64()
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    lattice0 = copy(Matrix(atoms.info[:lattice]))

    mktempdir() do dir
        # Same driver run as the smoke above but with NO MLIP at all — the
        # entry path Issue #133 opened. nacl_params has Buckingham a ≡ 0
        # (Coulomb + springs only), which is also exactly the
        # configuration the no-repulsion warning targets: one run pins
        # both the no-MLIP driver and the warning.
        params = (; base...,
                  backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1),
                  cgtol = FMD.FPPREC(2e-5),
                  nsteps = 4,
                  timestep = dt,
                  tau_t = 20dt,
                  tau_p = 500dt,
                  initial_temperature = 0.0,
                  target_temperature = 10.0,
                  target_stress = zeros(FMD.FPPREC, 6),
                  output_interval = 2,
                  output_dir = dir)
        logs, out = Test.collect_test_logs() do
            npt!(deepcopy_atoms(atoms), params)
        end

        # Finishes finite on the host, with non-vacuous forces and a
        # moving cell — the driver really ran with no MLIP term.
        @test out.arrays[:pos] isa Matrix{FMD.FPPREC}
        @test all(isfinite, out.arrays[:pos])
        @test all(isfinite, out.arrays[:forces])
        @test all(isfinite, out.arrays[:shelldisp])
        @test maximum(abs, out.arrays[:forces]) > 1e-6
        @test Matrix(out.info[:lattice]) != lattice0
        @test countlines(joinpath(dir, "log.txt")) == 4
        @test isfile(joinpath(dir, "step_2.xyz"))
        @test isfile(joinpath(dir, "step_4.xyz"))

        # No MLIP and a ≡ 0: the bare Coulomb + springs warning fired,
        # exactly once (entry-time, not per-step).
        warns = filter(l -> l.level == Warn, logs)
        @test length(warns) == 1
        @test occursin("no short-range repulsion", warns[1].message)

        # Equivalence pin: a zero-coefficient SplineParam must reproduce
        # the no-MLIP trajectory bitwise — apply_mlip!(::Nothing, ...) is
        # a true no-op and no shell-side term was dropped (single-threaded
        # CPU: adding an all-zero MLIP contribution is exact).
        splp0 = Calculators.SplineParam(FMD.FPPREC(0.6 * base[:rc]),
                                        FMD.FPPREC(0.4 * base[:rc]), 2, 4, 4)
        out0 = mktempdir() do dir0
            Test.collect_test_logs() do
                npt!(deepcopy_atoms(atoms),
                     (; params..., splp = splp0, output_dir = dir0))
            end[2]
        end
        @test Array(out0.arrays[:pos]) == Array(out.arrays[:pos])
        @test Array(out0.arrays[:forces]) == Array(out.arrays[:forces])
        @test Array(out0.arrays[:shelldisp]) == Array(out.arrays[:shelldisp])
        @test Matrix(out0.info[:lattice]) == Matrix(out.info[:lattice])

        # With a Buckingham repulsion present the warning must not fire.
        # nsteps = 0 stops right after initialization — entry checks are
        # all this case needs.
        p_rep = (; params..., a = fill(FMD.FPPREC(0.1), 2, 2), nsteps = 0)
        logs0, _ = Test.collect_test_logs() do
            npt!(deepcopy_atoms(atoms), p_rep)
        end
        @test count(l -> l.level == Warn, logs0) == 0
    end
end

@testset "fixed-cell driver smoke (barostat = :none, Issue #136)" begin
    Random.seed!(4)
    atoms = make_nacl_64()
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    lattice0 = copy(Matrix(atoms.info[:lattice]))
    # Deliberately NO :target_stress: fixed-cell runs must not require it
    # (conditionally required, barostat = :mttk only — Issue #136).
    common = (; base...,
              backend = :cpu,
              elements = Dict("Na" => 1, "Cl" => 2),
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              nsteps = 4,
              timestep = dt,
              initial_temperature = 0.0,
              target_temperature = 10.0,
              output_interval = 2)

    # NVT (nh × none): forces/positions evolve but the lattice must come
    # back BITWISE identical — nothing may touch it on the :none path.
    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., tau_t = 20dt, barostat = :none,
                     output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test maximum(abs, out.arrays[:forces]) > 1e-6
        @test Matrix(out.info[:lattice]) == lattice0
        lines = readlines(joinpath(dir, "log.txt"))
        @test length(lines) == 4
        cols = split(lines[end])
        @test length(cols) == 20
        @test isfinite(parse(Float64, cols[18]))
        # shell_rms of the final step is that of the returned frame's
        # last CG exit, up to the %15.8e print precision.
        @test parse(Float64, cols[19]) ≈ out.info[:shell_residual_rms][1] rtol = 1e-7
        # The log is sampled at the end of the step (after the closing
        # half-kick), so the last temperature is that of the returned
        # velocities, up to the %15.8e print precision.
        @test parse(Float64, cols[1]) ≈
              FMD.Apps._report_temperature(out, FMD.Cores.SerialDomain()) rtol = 1e-7
        @test parse(Int, cols[end]) >= 0
        # The rattled fixture is not pre-relaxed, so an all-zero ncg
        # column can only mean the counter (or the CG) stopped working.
        @test any(l -> parse(Int, last(split(l))) > 0, lines)
    end

    # NVE (none × none) with seeded initial velocities: the control stack
    # is fully off; integrator + shell CG still run and the cell is fixed.
    mktempdir() do dir
        out = npt!(deepcopy_atoms(atoms),
                   (; common..., thermostat = :none, barostat = :none,
                     initial_temperature = 50.0, seed = 7,
                     output_dir = dir))
        @test all(isfinite, out.arrays[:pos])
        @test Matrix(out.info[:lattice]) == lattice0
        @test maximum(abs, out.arrays[:velo]) > 0
        @test countlines(joinpath(dir, "log.txt")) == 4
    end

    # Builder guardrails for direct npt! calls (validate_params bypassed):
    # unknown selections and a missing τ for the active method fail
    # loudly before any MD state is touched.
    mktempdir() do dir
        @test_throws "must be :nh, :csvr or :none" npt!(deepcopy_atoms(atoms),
            (; common..., tau_t = 20dt, tau_p = 500dt, thermostat = :berendsen,
              output_dir = dir))
        @test_throws "must be :mttk, :scr or :none" npt!(deepcopy_atoms(atoms),
            (; common..., tau_t = 20dt, tau_p = 500dt, barostat = :berendsen,
              output_dir = dir))
        # common carries no :tau_t — the default (:nh) thermostat demands it.
        @test_throws "requires params[:tau_t]" npt!(deepcopy_atoms(atoms),
            (; common..., output_dir = dir))
        @test_throws "requires params[:tau_p]" npt!(deepcopy_atoms(atoms),
            (; common..., tau_t = 20dt, output_dir = dir))
        # …and the default (:mttk) barostat demands a target stress.
        @test_throws "requires params[:target_stress]" npt!(deepcopy_atoms(atoms),
            (; common..., tau_t = 20dt, tau_p = 500dt, output_dir = dir))
    end
end

# SurfaceTerm through the whole driver (Issue #179): the term rides the
# shell CG stack, so this pins the wiring (conv passthrough of the host
# struct, per-step evaluation, dipole_phys gauge pickup) rather than the
# physics — the calculator tests own the values.
@testset "npt! with a surface term (fixed cell)" begin
    Random.seed!(179)
    atoms = make_nacl_64()
    prototype = deepcopy_atoms(atoms)          # declared P = 0 frame
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    st = Calculators.SurfaceTerm(atoms, base[:qc], base[:qs];
                                 eps_out = [2.0, 3.0, 5.0],
                                 gauge = prototype)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])

    run_it(extra) = mktempdir() do dir
        params = (; base..., extra...,
                  backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(2e-5),
                  nsteps = 3, timestep = dt, tau_t = 20dt,
                  thermostat = :nh, barostat = :none,
                  initial_temperature = 0.0, target_temperature = 10.0,
                  output_interval = 10, output_dir = dir)
        npt!(deepcopy_atoms(atoms), params)
    end

    out = run_it((; surface_term = st))
    @test all(isfinite, out.arrays[:pos])
    @test all(isfinite, out.arrays[:forces])
    @test all(isfinite, out.arrays[:shelldisp])
    @test isfinite(st.results["bias"])

    # Non-vacuity: the depolarization field really steered the trajectory.
    out0 = run_it((;))
    @test Array(out.arrays[:pos]) != Array(out0.arrays[:pos])

    # dipole_phys is reported against the surface term's gauge (the
    # mfr_total chain), i.e. against the declared prototype rather than
    # this run's rattled start frame.
    prms = (; qc = base[:qc], qs = base[:qs])
    M_abs = Calculators.calcdipole!(out, prms)
    expected = M_abs .- transpose(Matrix(out.info[:lattice])) * st.M_frac_ref
    @test isapprox(Vector(out.info[:dipole_phys]), expected;
                   rtol = 1e-10, atol = 1e-12)

    # Shadow XL-BOMD used to be refused here; since Issue #181 the
    # depolarization field has an analytic HVP and the combination runs.
    # (The physics of that path — force, virial, DD parity — is pinned in
    # test/apps/test_xlbomd.jl T10; this is the driver-door smoke.)
    outx = run_it((; surface_term = st, shell_dynamics = :xlbomd,
                   xl_warmup = 2))
    @test all(isfinite, outx.arrays[:pos])
    @test all(isfinite, outx.arrays[:shelldisp])
end

@testset "reported temperature uses 3N-3 DOF (Issue #93)" begin
    # Hand-built frame: 8 atoms, unit masses, one atom with |v|² = 2
    # ⇒ kinetic = 1 Hartree. The log temperature must count 3·8-3 = 21
    # DOF (COM momentum is removed every step; the MTTK sigma uses the
    # same count) — dividing by 3N = 24 was the Issue #93 bug.
    a = make_nacl_8()
    a.arrays[:masses] .= 1.0
    a.arrays[:velo][1, :] .= FMD.FPPREC[1.0, 1.0, 0.0]
    dom = Cores.domain(a)
    @test kinetic_energy(a) ≈ 1.0
    @test FMD.Apps._report_temperature(a, dom) ≈ 2 * UNITS[:Kelvin] / 21
end

@testset "barostat velocity exponent uses N_f = 3N-3 (Issue #130)" begin
    N = 64
    # Isotropic limit: must reduce to -(1 + 3/N_f)·v_ε·I with the SAME
    # N_f = 3N-3 as the MTK force term 2K/(3N-3) — the MTTK conserved
    # quantity requires the two to match (3N+3 was the Issue #130 bug).
    vε = 0.01
    E = FMD.Apps._baro_velocity_exponent(vε * Matrix{Float64}(I, 3, 3), N)
    @test E ≈ -(1 + 3 / (3N - 3)) * vε * Matrix{Float64}(I, 3, 3)
    # A traceless ε̇ picks up no isotropic contraction at any N.
    A = 0.01 .* [0.0 1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
    @test FMD.Apps._baro_velocity_exponent(A, N) ≈ -A
end

@testset "control conserved quantity, ideal gas (Issues #130/#136)" begin
    # U = 0 (no forces) isolates the thermostat/barostat propagator — the
    # MTK velocity-scaling divisor (Issue #130), the split particle/cell
    # thermostats and the 6-DOF cell count (Issue #136) — from any
    # force-integration error. The loop mirrors npt!'s step sequence with
    # the real update!/scale_cell!; H' is conserved_energy (see
    # MDControl.jl for the per-combination formula).
    #
    # Thresholds are set ≥ 2× above the drift measured on this fixture:
    #   (nh, mttk) σ_ext = 0:  9.7e-6 @1500, 1.7e-5 @3000, 2.7e-5 @5000,
    #     5.3e-5 @10000, 1.05e-4 @20000 (clean linear growth ~5.3e-9/step
    #     — the work accumulators sample ζ at the scaling events, an
    #     O(dt³)-per-step bookkeeping bias; documented in parameters.md)
    #   (nh, mttk) hydrostatic: 1.7e-5 @3000; with the work_ext SIGN
    #     flipped the same run drifts by ~2·|work_ext|/H0 ≈ 6.7e-3 —
    #     a 170× separation, so this case pins the +P·ΔV convention
    #   (none, mttk): 1.34e-6, saturated (identical @1500/3000/5000)
    #   (nh, none) NVT: 1.74e-6, saturated (identical @1500/3000/5000)
    #   (none, none) NVE: exactly 0 (no operator touches velocities)
    Random.seed!(130)
    a = make_nacl_64()
    N = length(a)
    T0 = 300.0
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    m = a.arrays[:masses]
    a.arrays[:velo] .= randn(FMD.FPPREC, N, 3) .* sqrt.(T0 / UNITS[:Kelvin] ./ m)
    a.arrays[:velo] .-= sum(a.arrays[:velo] .* m, dims = 1) ./ sum(m)
    v0 = copy(a.arrays[:velo])
    lat0 = copy(a.info[:lattice])
    pos0 = copy(a.arrays[:pos])
    reset!() = (a.arrays[:velo] .= v0; a.info[:lattice] .= lat0;
                a.arrays[:pos] .= pos0)

    # Hand-driven npt! step skeleton (half-kicks are no-ops with F ≡ 0;
    # COM stays zero under the uniform scalings, so _remove_com! is
    # omitted). Returns the worst relative |H' − H'₀|.
    function drive!(th, baro, prms, nsteps)
        tst = FMD.Apps.voigt2matrix(Vector{Float64}(prms[:target_stress]))
        H(c) = FMD.Apps.conserved_energy(th, baro, a, c)
        H0 = H(FMD.Apps._control_context(prms, 1, N, tst))
        worst = 0.0
        for step in 1:nsteps
            c = FMD.Apps._control_context(prms, step, N, tst)
            FMD.Apps.update!(a, th, baro, c, Array)
            FMD.Apps.scale_cell!(baro, a, c, Array)
            a.arrays[:pos] .+= (a.arrays[:velo] * prms[:timestep]) *
                               (a.info[:lattice] \ I)
            FMD.Apps.update!(a, th, baro, c, Array)
            worst = max(worst, abs(H(c) - H0))
        end
        return worst / abs(H0)
    end

    prms = (; target_stress = zeros(FMD.FPPREC, 6),
            target_temperature = T0, timestep = dt,
            tau_t = 20dt, tau_p = 200dt, nsteps = 1500)

    # (nh, mttk), σ_ext = 0.
    th = FMD.Apps.NoseHoover(0.0, 0.0, Float64(20dt))
    baro = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    @test drive!(th, baro, prms, 1500) < 2e-5
    # Non-vacuity: every control DOF actually moved — in particular the
    # internal cell thermostat (ζ_cell ≈ 6.8e-4 on this fixture), the
    # piece the pre-#136 shared-friction integrator did not have.
    @test th.zeta != 0.0
    @test any(!=(0.0), baro.zeta_b)
    @test abs(baro.zeta_cell) > 1e-5
    @test a.info[:lattice] != lat0

    # (nh, mttk), hydrostatic σ_ext = −P·I at the ideal-gas equilibrium
    # pressure P = N·kT/V₀ — pins the work_ext sign (+P·ΔV enthalpy
    # convention): a sign error turns the bounded wobble into a secular
    # ~2·|work_ext| drift (see threshold comment above).
    reset!()
    kT = T0 / UNITS[:Kelvin]
    P = N * kT / volume(a)
    prms_p = (; prms..., target_stress = FMD.FPPREC.(-P .* [1, 1, 1, 0, 0, 0]))
    th = FMD.Apps.NoseHoover(0.0, 0.0, Float64(20dt))
    baro = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    @test drive!(th, baro, prms_p, 3000) < 4e-5
    @test baro.work_ext != 0.0

    # (none, mttk) — the fourth quadrant of the 2×2. NOT a textbook NPH:
    # the cell momenta still carry the barostat's INTERNAL NH thermostat
    # (a design constant of the MTTK struct; noted in parameters.md).
    reset!()
    baro = FMD.Apps.MTTK(zeros(3, 3), 0.0, 0.0, 0.0, Float64(200dt))
    @test drive!(FMD.Apps.NoThermostat(), baro, prms, 1500) < 3e-6
    @test abs(baro.zeta_cell) > 0.0

    # (nh, none) — NVT: conserved, and the lattice bitwise untouched
    # (nothing may write to it on the :none path).
    reset!()
    th = FMD.Apps.NoseHoover(0.0, 0.0, Float64(20dt))
    @test drive!(th, FMD.Apps.NoBarostat(), prms, 1500) < 4e-6
    @test a.info[:lattice] == lat0

    # (none, none) — NVE: the control stack is a strict no-op, so H' = K
    # is conserved EXACTLY and the velocities come back bitwise.
    reset!()
    @test drive!(FMD.Apps.NoThermostat(), FMD.Apps.NoBarostat(), prms, 10) == 0.0
    @test a.arrays[:velo] == v0
    @test a.info[:lattice] == lat0
end

@testset "npt! seeded initial velocities (Issue #34)" begin
    Random.seed!(11)
    atoms = make_nacl_64()
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    splp = Calculators.SplineParam(FMD.FPPREC(0.6 * base[:rc]),
                                   FMD.FPPREC(0.4 * base[:rc]), 2, 4, 4)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    T0 = 50.0
    function run_seeded(seed, ambient; frame = atoms, nsteps = 1)
        Random.seed!(ambient)          # must NOT influence the trajectory
        mktempdir() do dir
            npt!(frame,
                 (; base..., backend = :cpu, splp = splp,
                   elements = Dict("Na" => 1, "Cl" => 2),
                   pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                   skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(2e-5),
                   nsteps = nsteps, timestep = dt,
                   tau_t = 20dt, tau_p = 500dt,
                   initial_temperature = T0, target_temperature = T0,
                   target_stress = zeros(FMD.FPPREC, 6),
                   output_interval = 10, output_dir = dir, seed = seed))
        end
    end
    # npt! never mutates the caller's frame (change_arrtyp copies), so the
    # shared fixture can be passed directly — pinned below.
    o1 = run_seeded(42, 1)
    o2 = run_seeded(42, 2)   # same seed, different ambient RNG state

    # Bitwise reproducibility under the same seed, independent of the
    # ambient global RNG (single-threaded CPU run: no atomics noise).
    @test Array(o1.arrays[:velo]) == Array(o2.arrays[:velo])
    @test Array(o1.arrays[:pos]) == Array(o2.arrays[:pos])
    # The caller's frame is untouched by the runs above.
    @test all(==(0), atoms.arrays[:velo])

    # nsteps = 0 stops right after initialization — the returned velocities
    # are exactly the drawn ones.
    z1 = run_seeded(42, 3; nsteps = 0)
    z3 = run_seeded(7, 4; nsteps = 0)
    @test Array(z1.arrays[:velo]) != Array(z3.arrays[:velo])   # seed matters

    # Physical invariants of the draw: COM momentum removed exactly,
    # temperature in a sane band around T0 (single realization of ~190
    # dof fluctuates by ~10%, so this is a sanity band, not a mean test).
    # Same 3N-3 estimator as the driver log (Issue #93) — the draw is
    # unbiased for it: E[2K] = (3N-3)·k·T0 after COM removal.
    v = Array(z1.arrays[:velo]); m = Array(z1.arrays[:masses])
    @test maximum(abs, vec(sum(v .* m, dims = 1))) < 1e-9
    Tinit = FMD.Apps._report_temperature(z1, Cores.domain(z1))
    @test 0.5 * T0 < Tinit < 1.5 * T0

    # The draw is keyed by :id, not by row order: a row-shuffled frame
    # (the restart scenario — writexyz stores cell-sorted rows with their
    # :id column) receives the same velocity per atom id. The raw draw is
    # bitwise identical; the COM-removal sums reassociate across row
    # orders, so the comparison allows FP noise (≲ N·eps relative).
    perm = randperm(Random.Xoshiro(5), length(atoms))
    shuffled = atoms[perm]
    z2 = run_seeded(42, 5; frame = shuffled, nsteps = 0)
    P1 = sortperm(Array(z1.arrays[:id]))
    P2 = sortperm(Array(z2.arrays[:id]))
    v1 = Array(z1.arrays[:velo])[P1, :]
    v2 = Array(z2.arrays[:velo])[P2, :]
    @test maximum(abs, v1 .- v2) < 1e-12 * maximum(abs, v1)

    # Direct npt! calls without :seed fail loudly when velocities are drawn.
    @test_throws "params[:seed] is required" npt!(atoms,
        (; base..., backend = :cpu, splp = splp, initial_temperature = T0))
    @test_throws "must be an Integer" npt!(atoms,
        (; base..., backend = :cpu, splp = splp, initial_temperature = T0,
          seed = 1.5))
end

# The reported dipole_phys is measured against a zero-point.  Before Issue
# #122 that zero-point was implicit — the umbrella's construction frame when
# an umbrella was present, this run's starting frame otherwise — so the same
# trajectory reported different polarisations depending on whether a bias was
# attached, and a restart silently re-zeroed the column.  With an explicit
# :polarization_gauge the value is a property of the structure alone.
@testset "dipole_phys uses the declared gauge, with or without a bias" begin
    Random.seed!(3)
    atoms = make_nacl_64()
    # Pair each Na with a Cl so every group is charge neutral, then give the
    # groups a spatial layout (ASE flat order) matching a (2,2,2) grid.
    let w = Cores.wrapfrac.(atoms.arrays[:pos])
        gx = min.(Int.(floor.(w[:, 1] .* 2)), 1)
        gy = min.(Int.(floor.(w[:, 2] .* 2)), 1)
        gz = min.(Int.(floor.(w[:, 3] .* 2)), 1)
        atoms.arrays[:groupid] = gx .* 4 .+ gy .* 2 .+ gz .+ 1
    end
    prototype = deepcopy_atoms(atoms)          # the declared P = 0 structure
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))

    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    gauge = Calculators.PolarizationGauge(prototype, (2, 2, 2),
                                          base[:qc], base[:qs])
    # k = 0: the umbrella must not perturb the trajectory, otherwise the two
    # runs below would differ through the dynamics rather than through the
    # reporting path this test is about.
    pu = Calculators.PolarizationUmbrella(
        atoms, 0.0, 0.0, 0.0, FMD.FPPREC[1, 0, 0], (2, 2, 2),
        base[:qc], base[:qs]; gauge = gauge, m = (1, 0, 0))
    dt = FMD.FPPREC(1.0 / UNITS[:fs])

    run_it(extra) = mktempdir() do dir
        params = (; base..., extra...,
                  backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(2e-5),
                  # 1 step: dipole_phys is written inside the step loop.
                  nsteps = 1, timestep = dt, tau_t = 20dt, tau_p = 500dt,
                  initial_temperature = 0.0, target_temperature = 10.0,
                  target_stress = zeros(FMD.FPPREC, 6),
                  output_interval = 100, output_dir = dir)
        return Vector(npt!(deepcopy_atoms(atoms), params).info[:dipole_phys])
    end

    d_gauge_only = run_it((; polarization_gauge = gauge))
    d_gauge_bias = run_it((; polarization_gauge = gauge,
                             polarization_umbrella = pu))
    d_implicit = run_it((;))

    # Same gauge → same answer whether or not a bias is attached.
    @test isapprox(d_gauge_only, d_gauge_bias; rtol = 1e-12)
    # It is the polarisation relative to the prototype, so it carries the
    # rattle.  The implicit fallback re-zeroes on this run's start frame and
    # therefore reports only the drift accumulated during the run — orders
    # of magnitude smaller, and meaningless across a restart.
    @test norm(d_gauge_only) > 1e-6
    @test norm(d_implicit) < 0.01 * norm(d_gauge_only)

    # A gauge from a different system is rejected before the run starts.
    other = make_nacl_8()
    other.arrays[:groupid] = vcat(1:4, 1:4)
    @test_throws ErrorException run_it((;
        polarization_gauge = Calculators.PolarizationGauge(
            other, (1, 2, 2), base[:qc], base[:qs])))
end

# End-to-end: does the map bias actually move the system to the map?  The
# calculator tests pin the gradient and the energy; this one pins the thing a
# user cares about — that driving with it converges, and that the CV written
# to Bias.log is the quantity that reports it (Issue #120).
@testset "npt! with a target-map umbrella converges to the map" begin
    Random.seed!(4)
    atoms = make_nacl_64()
    # Charge-neutral octant groups, assigned from the ideal positions (the
    # rattle below would otherwise scatter boundary atoms and charge them).
    let w = Cores.wrapfrac.(atoms.arrays[:pos])
        gx = min.(Int.(floor.(w[:, 1] .* 2)), 1)
        gy = min.(Int.(floor.(w[:, 2] .* 2)), 1)
        gz = min.(Int.(floor.(w[:, 3] .* 2)), 1)
        atoms.arrays[:groupid] = gx .* 4 .+ gy .* 2 .+ gz .+ 1
    end
    prototype = deepcopy_atoms(atoms)
    atoms.arrays[:pos] .+= 0.004 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))

    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    gauge = Calculators.PolarizationGauge(prototype, (2, 2, 2),
                                          base[:qc], base[:qs])
    # A target the material can actually reach: 30% of the polarisation the
    # rattle produced.  (NaCl has no soft polar mode, so an imposed pattern
    # would be a fight with the lattice rather than a test of the bias.)
    Pg0 = Calculators.group_p_gauged(atoms, gauge, base[:qc], base[:qs])
    tgt = FMD.FPPREC(0.3) .* Pg0
    pu = Calculators.PolarizationMapUmbrella(
        atoms, 2e5, tgt, (2, 2, 2), base[:qc], base[:qs]; gauge = gauge)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])

    mktempdir() do dir
        params = (; base...,
                  backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(2e-5),
                  nsteps = 60, timestep = dt, tau_t = 20dt,
                  initial_temperature = 0.0, target_temperature = 1.0,
                  thermostat = :nh, barostat = :none,
                  output_interval = 60, output_dir = dir,
                  polarization_gauge = gauge,
                  polarization_umbrella = pu)
        out = npt!(deepcopy_atoms(atoms), params)

        cvs = Float64[]
        for line in eachline(joinpath(dir, "Bias.log"))
            startswith(line, "CV") && continue
            push!(cvs, parse(Float64, split(line, ',')[1]))
        end
        @test length(cvs) == 60
        # Started meaningfully off-target and got substantially closer.
        @test cvs[1] > 1e-5
        @test cvs[end] < 0.5 * cvs[1]
        # The CV really is the distance to the map: recompute it from the
        # final frame (which the driver returns cell-sorted — the gauge pairs
        # by :id, which is what makes this measurable at all).
        Pg1 = Calculators.group_p_gauged(Cores.change_arrtyp(out, Array),
                                         gauge, base[:qc], base[:qs])
        rms = sqrt(sum(abs2, Pg1 .- tgt) / size(tgt, 1))
        @test isapprox(rms, cvs[end]; rtol = 0.5)   # same order, same trend
        @test maximum(abs, Pg1 .- tgt) < 0.5 * maximum(abs, Pg0 .- tgt)
    end
end
