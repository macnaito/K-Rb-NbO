# test/apps/test_relax.jl
#
# FIRE 2.0 structure relaxation (Issue #209), fixed cell.
#
# The central test is a RESTORATION test: the ideal rocksalt has zero force
# by symmetry, so it is a stationary point whose location is known without
# reference to this implementation. Rattle it, relax, and the atoms must
# come back — an oracle the driver cannot fake by converging to whatever it
# happens to sit on.
#
# Every fixture here uses `buckingham = true`. With the default a ≡ 0 there
# is no short-range wall at all, rocksalt is not a minimum, and a
# relaxation would (correctly) collapse the structure.
#
# `pme_meshsize = 0.25 Å` is not arbitrary. The PME mesh error is what
# limits the reachable `ftol`, and on a high-symmetry structure it does not
# shrink monotonically with the spacing — it tracks the grid's
# commensurability with the crystal. Measured on this fixture, whose exact
# (analytic-Ewald) forces vanish to 1e-17:
#
#     meshsize 0.50 Å -> fftsize 23 -> max|F| = 1.8e-05   <- spurious!
#     meshsize 0.30 Å -> fftsize 38 -> max|F| = 1.7e-17
#     meshsize 0.25 Å -> fftsize 46 -> max|F| = 1.9e-17
#     meshsize 0.20 Å -> fftsize 57 -> max|F| = 1.2e-06   <- spurious!
#
# (Measured pre-#216. The pattern is parity: the NaCl-64 coordinates are
# multiples of 1/4, so every EVEN grid is commensurate and the symmetric
# error cancels exactly. Since the 2357-smooth rounding of Issue #216,
# 0.25 Å lands on fftsize 48 — still even, still cancelling — and the odd
# grids the escalation testsets rely on must be targeted explicitly, see
# there.)
#
# On a rattled frame the PME-vs-Ewald force difference is 3.8e-06 at
# 0.25 Å (5.0e-06 at fftsize 48; 1.9e-05 at 0.50 Å). So 0.25 Å keeps the
# solver error a factor ~2-3 below the ftol used here, which is what makes
# the assertions below statements about the structure rather than about
# the grid.
#
# The probes below therefore evaluate at the RUN'S α — since Issue #216 the
# drivers honor the explicit `alpha` in these params (0.30) rather than
# re-deriving one from `ewald_tolerance`, so "the run's α" is simply
# `params[:alpha]`. Measured on a rattled frame (at the pre-#216 α = 0.326;
# the split barely moves at 0.30):
#
#     PME vs Ewald, same α              3.75e-06   <- the mesh error
#     PME vs Ewald, α = 0.502           3.87e-06
#     Ewald α mismatch (0.326 vs 0.502)  1.84e-07  <- the α/truncation part
#
# i.e. the budget is ~95% mesh error; matching α keeps it that way instead
# of quietly folding in a real-space truncation difference.
const RELAX_PME_FORCE_TOL = FMD.FPPREC(5e-6)   # measured mesh error, rounded up
using Test
using Random
using LinearAlgebra
using Logging: Warn
using FMD.Apps: relax!

# Shared fixture: ideal NaCl-64 + the calibrated Born–Mayer parameters, at
# a cutoff that reaches the nearest neighbours (see nacl_params).
function _relax_fixture(; rattle = 0.0, seed = 5)
    Random.seed!(seed)
    atoms = make_nacl_64()
    rattle > 0 && (atoms.arrays[:pos] .+= (rattle / UNITS[:Å]) .*
                   randn(FMD.FPPREC, size(atoms.arrays[:pos])) ./
                   norm(Array(atoms.info[:lattice])[1, :]))
    return atoms
end

function _relax_params(atoms, dir; kwargs...)
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true, buckingham = true)
    p = (; base...,
         backend = :cpu,
         elements = Dict("Na" => 1, "Cl" => 2),
         pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]),
         skin = FMD.FPPREC(0.1),
         cgtol = FMD.FPPREC(1e-7),
         cgmaxiter = 30,
         ftol = FMD.FPPREC(1e-5),
         maxsteps = 400,
         dt0 = FMD.FPPREC(1.0 / UNITS[:fs]),
         dt_max = FMD.FPPREC(10.0 / UNITS[:fs]),
         dmax = FMD.FPPREC(0.1 / UNITS[:Å]),
         output_interval = 25,
         output_dir = dir)
    return (; p..., kwargs...)
end

@testset "relax! restores a rattled rocksalt (FIRE 2.0)" begin
    ideal = _relax_fixture()
    pos_ideal = copy(Array(ideal.arrays[:pos]))
    rattled = _relax_fixture(; rattle = 0.08)
    # Non-vacuous: the rattle actually moved things and made real forces.
    @test maximum(abs, Array(rattled.arrays[:pos]) .- pos_ideal) > 1e-4

    mktempdir() do dir
        params = _relax_params(rattled, dir)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(rattled), params)
        end
        out, converged = res
        @test converged
        # This fixture asks for ftol = 1e-5 against a ~4e-6 solver error, so
        # it legitimately escalates — the margin-free regime. The "a healthy
        # run is not told to tighten anything" pin lives in its own testset
        # below, where the margin is real.

        # --- the oracle: back to the ideal positions --------------------
        # gather by :id — the cell-list re-sort permutes rows.
        ids = Array(out.arrays[:id])
        got = Array(out.arrays[:pos])[sortperm(ids), :]
        # Fixed cell, periodic: the minimum is degenerate under a rigid
        # translation (and under the periodic image), so compare the
        # translation-free difference wrapped to the nearest image.
        d = got .- pos_ideal
        d .-= round.(d)                       # nearest periodic image
        d .-= sum(d, dims = 1) ./ size(d, 1)  # remove the rigid translation
        lat = Array(out.info[:lattice])
        dcart = d * lat
        @test maximum(abs, dcart) < 5e-3      # Bohr — well inside the rattle

        # --- independent verification of the criterion -------------------
        # Two levels, because they fail differently.
        #
        # (1) Same Coulomb solver, but rebuilt from scratch and with the
        #     inner shell CG driven far tighter than the run used: this
        #     checks the driver's claim without inheriting its neighbour
        #     list, its PME accelerator, or its inner tolerance.
        probe = deepcopy_atoms(out)
        reset!(probe)
        nl_probe = neighborlist(probe, params[:rc], FMD.FPPREC(0.0))
        pme_probe = Calculators.PMEInfo(probe.info[:lattice], params)
        Calculators.calculate!(Calculators.CoreShell(), probe, nl_probe, pme_probe,
                               (; params...,
                                  cgtol = FMD.FPPREC(1e-9), cgmaxiter = 60))
        @test maximum(abs, probe.arrays[:forces]) < params[:ftol]

        # (2) A DIFFERENT Coulomb solver: analytic Ewald instead of PME.
        #     This is the solver-independent statement — it cannot be
        #     satisfied by a structure that only looks relaxed on the run's
        #     particular FFT grid. The budget is ftol plus the measured
        #     PME-vs-Ewald discrepancy at this mesh.
        ewald_probe = deepcopy_atoms(out)
        reset!(ewald_probe)
        Calculators.calculate!(Calculators.CoreShell(), ewald_probe,
                               (; params..., cgtol = FMD.FPPREC(1e-9),
                                  cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
        @test maximum(abs, ewald_probe.arrays[:forces]) <
              params[:ftol] + RELAX_PME_FORCE_TOL

        # --- it is a MINIMUM, not just a stationary point ----------------
        # Every random displacement must raise the energy.
        e0 = Array(ewald_probe.info[:potential_energy])[1]
        rng = Random.Xoshiro(17)
        for trial in 1:5
            bumped = deepcopy_atoms(out)
            δ = FMD.FPPREC(2e-3) .* randn(rng, FMD.FPPREC, size(bumped.arrays[:pos]))
            bumped.arrays[:pos] .+= δ * inv(lat)
            reset!(bumped)
            Calculators.calculate!(Calculators.CoreShell(), bumped,
                                   (; params..., cgtol = FMD.FPPREC(1e-9),
                                      cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
            @test Array(bumped.info[:potential_energy])[1] > e0
        end

        # --- outputs -----------------------------------------------------
        @test isfile(joinpath(dir, "relaxed.xyz"))
        lines = readlines(joinpath(dir, "log.txt"))
        # step 0 (the input structure) plus one line per iteration.
        @test length(lines) >= 2
        @test parse(Int, first(split(lines[1]))) == 0
        # 22 columns: step | max|F| | energy | sigma(6) | volume | dt |
        # alpha | ncg | lattice(9)
        @test length(split(lines[1])) == 22
        # max|F| decreased overall, and the last logged value met ftol.
        f_first = parse(Float64, split(lines[1])[2])
        f_last = parse(Float64, split(lines[end])[2])
        @test f_last < f_first
        @test f_last < params[:ftol]
        # Fixed cell: the volume column never moved.
        vols = [parse(Float64, split(l)[10]) for l in lines]
        @test all(v -> v == vols[1], vols)
    end
end

@testset "relax! leaves an already-relaxed structure alone (and says so)" begin
    ideal = _relax_fixture()
    mktempdir() do dir
        params = _relax_params(ideal, dir)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(ideal), params)
        end
        out, converged = res
        @test converged
        # The ideal rocksalt is a symmetry-forced stationary point, so the
        # driver must stop at step 0 without moving anything.
        @test Array(out.arrays[:pos])[sortperm(Array(out.arrays[:id])), :] ≈
              Array(ideal.arrays[:pos])
        # ...and warn that a stationary point may be a saddle.
        warns = filter(l -> l.level == Warn, logs)
        @test any(l -> occursin("may be a saddle", string(l.message)), warns)
    end
end

@testset "relax! reports non-convergence rather than erroring" begin
    rattled = _relax_fixture(; rattle = 0.08)
    mktempdir() do dir
        # One step cannot relax a 0.08-rattle.
        params = _relax_params(rattled, dir; maxsteps = 1)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(rattled), params)
        end
        out, converged = res
        @test !converged
        # The partially relaxed structure is still written — it is a usable
        # result, as long as nobody mistakes it for a converged one.
        @test isfile(joinpath(dir, "relaxed.xyz"))
        @test any(l -> l.level == Warn && occursin("NOT converged", string(l.message)),
                  logs)
        @test all(isfinite, out.arrays[:pos])
    end
end

@testset "relax! guards" begin
    atoms = _relax_fixture(; rattle = 0.02)

    mktempdir() do dir
        # ftol at/below cgtol is unreachable in principle: the inner shell CG
        # leaves an O(cgtol) error in the core forces, so the outer criterion
        # would be chasing its own noise.
        @test_throws "not reachable with cgtol" relax!(deepcopy_atoms(atoms),
            _relax_params(atoms, dir; ftol = FMD.FPPREC(1e-7), cgtol = FMD.FPPREC(1e-7)))
        # The margin is a rule of thumb, not a law — overridable, but the
        # override has to be explicit (the message names the key).
        p_override = _relax_params(atoms, dir; ftol = FMD.FPPREC(1e-7),
                                   cgtol = FMD.FPPREC(1e-7),
                                   ftol_cgtol_margin = 1.0, maxsteps = 1)
        @test FMD.Apps._check_relax_tolerances(p_override) === nothing

        # A non-positive ftol would make the criterion unsatisfiable.
        @test_throws "ftol] must be > 0" relax!(deepcopy_atoms(atoms),
            _relax_params(atoms, dir; ftol = FMD.FPPREC(0.0)))

        # XL-BOMD is a dynamics device — no meaning for relaxation.
        @test_throws "no meaning for relaxation" relax!(deepcopy_atoms(atoms),
            _relax_params(atoms, dir; shell_dynamics = :xlbomd))

        # The umbrella is a bias, not a physical potential: its minimum is a
        # biased minimum, not the equilibrium structure the app reports.
        base = nacl_params(atoms; with_shell = true, buckingham = true)
        withgroups = deepcopy_atoms(atoms)
        withgroups.arrays[:groupid] = ones(Int, length(withgroups))
        pu = Calculators.PolarizationUmbrella(
            withgroups, 1.0, 1.0, 0.05, FMD.FPPREC[1, 0, 0], (1, 1, 1),
            base[:qc], base[:qs];
            gauge = withgroups, m = (1, 0, 0), sigma = 0.5)
        @test_throws "bias, not a physical potential" relax!(deepcopy_atoms(withgroups),
            _relax_params(atoms, dir; polarization_umbrella = pu))
    end
end

@testset "a converged verdict survives an independent evaluation (starved CG)" begin
    # The acceptance test of the whole app. `cgmaxiter = 1` starves the inner
    # shell CG so it returns long before reaching cgtol, which is exactly the
    # regime where the outer criterion can be met by inner-solver noise.
    #
    # The invariant asserted here is the one that matters and holds
    # unconditionally: whatever the driver REPORTS must hold up against an
    # evaluation it did not perform — a different Coulomb solver (analytic
    # Ewald) and a shell CG driven to convergence. A `converged = true`
    # produced by inner-solver noise fails this; there is no branch in which
    # the test asserts nothing.
    rattled = _relax_fixture(; rattle = 0.05)
    mktempdir() do dir
        params = _relax_params(rattled, dir;
                               cgmaxiter = 1,
                               cgtol = FMD.FPPREC(1e-6),
                               ftol = FMD.FPPREC(1e-4),
                               maxsteps = 300)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(rattled), params)
        end
        out, converged = res

        # Non-vacuous: the starved CG really did hit its cap, so the forces
        # the driver worked with carry the O(cgtol) error this is about.
        ncgs = [parse(Int, split(l)[13]) for l in readlines(joinpath(dir, "log.txt"))]
        @test maximum(ncgs) >= params[:cgmaxiter]

        probe = deepcopy_atoms(out)
        reset!(probe)
        Calculators.calculate!(Calculators.CoreShell(), probe,
                               (; params..., cgtol = FMD.FPPREC(1e-9),
                                  cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
        maxf_true = maximum(abs, probe.arrays[:forces])
        if converged
            @test maxf_true < params[:ftol] + RELAX_PME_FORCE_TOL
        else
            # Not converged is always an allowed answer — but it has to be
            # said out loud, and the structure still written.
            @test any(l -> l.level == Warn && occursin("NOT converged",
                                                       string(l.message)), logs)
            @test isfile(joinpath(dir, "relaxed.xyz"))
        end
    end
end

@testset "relax! escalates rather than reporting a solver artifact" begin
    # A deliberately coarse ODD PME grid puts the solver error at ~4e-5,
    # above the ftol asked for. Odd on purpose: the NaCl-64 coordinates are
    # multiples of 1/4, so on any EVEN grid the B-spline lattice is
    # commensurate and the mesh error cancels by symmetry near the relaxed
    # structure (Issue #211) — no escalation to test. Since the 2357-smooth
    # rounding (Issue #216) the mesh must target an odd smooth size
    # directly: L/20.5 ⇒ ceil = 21 = 3·7, which the rounding keeps.
    # The run will meet ftol against that error; the acceptance test must
    # catch it, and the driver must then keep relaxing on the tightened
    # solvers rather than either lying or giving up.
    rattled = _relax_fixture(; rattle = 0.05)
    L_fix = norm(Array(rattled.info[:lattice])[1, :])
    mktempdir() do dir
        params = _relax_params(rattled, dir;
                               pme_meshsize = FMD.FPPREC(L_fix / 20.5),
                               ftol = FMD.FPPREC(1e-5),
                               maxsteps = 400)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(rattled), params)
        end
        out, converged = res
        @test any(l -> occursin("within the solver error", string(l.message)), logs)
        # Whatever the outcome, it must be honest: a `converged = true` has
        # to hold against the tightened solvers, which is what the escalated
        # loop measured.
        if converged
            ewald_probe = deepcopy_atoms(out)
            reset!(ewald_probe)
            Calculators.calculate!(Calculators.CoreShell(), ewald_probe,
                                   (; params..., cgtol = FMD.FPPREC(1e-9),
                                      cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
            @test maximum(abs, ewald_probe.arrays[:forces]) <
                  params[:ftol] + RELAX_PME_FORCE_TOL
        else
            @test any(l -> l.level == Warn && occursin("NOT converged",
                                                       string(l.message)), logs)
        end
    end
end

@testset "app[:Relax] job entry + default_params" begin
    atoms = _relax_fixture(; rattle = 0.03)
    mktempdir() do dir
        input = joinpath(dir, "input.xyz")
        elements = Dict("Na" => 1, "Cl" => 2)
        open(input, "w") do io
            writexyz(io, atoms, (; elements = elements))
        end
        fit = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                          rc_factor = 0.45, with_shell = true, buckingham = true)
        user = (; elements = elements,
                input = input,
                output_dir = joinpath(dir, "out"),
                backend = :cpu,
                pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]),
                rc = fit[:rc], skin = FMD.FPPREC(0.1),
                qc = fit[:qc], qs = fit[:qs], k2 = fit[:k2], k4 = fit[:k4],
                eta_c = fit[:eta_c], eta_s = fit[:eta_s],
                a = fit[:a], rho = fit[:rho], b = fit[:b], c = fit[:c], e = fit[:e],
                maxsteps = 300)
        params = (; FMD.default_params(Val(:Relax), user)..., user...)
        # The defaults must themselves satisfy the schema (a default that
        # fails validation would make the documented idiom unusable).
        @test FMD.Apps.validate_params(:Relax, params) === nothing
        @test FMD.app[:Relax](params) == 0
        @test isfile(joinpath(dir, "out", "relaxed.xyz"))
        @test isfile(joinpath(dir, "out", "log.txt"))

        # :nsteps belongs to the MD/fit jobs; :Relax caps with :maxsteps and
        # must say so rather than hint at a typo.
        @test_throws "not a :Relax key" FMD.Apps.validate_params(:Relax, (; params..., nsteps = 10))
        # A non-converged run reports 1, not 0 — a batch driver has to be
        # able to branch on it.
        @test FMD.app[:Relax]((; params..., maxsteps = 1,
                               output_dir = joinpath(dir, "out2"))) == 1
    end
end

@testset "a run with solver margin is not advised to tighten anything" begin
    # Regression pin: `escalated` must mean "the acceptance test FAILED and
    # the run continued", not "the acceptance test ran". Here ftol = 1e-4
    # sits a factor ~25 above the 3.8e-6 mesh error, so the check passes on
    # the first try and the advice must stay silent.
    rattled = _relax_fixture(; rattle = 0.05)
    mktempdir() do dir
        params = _relax_params(rattled, dir; ftol = FMD.FPPREC(1e-4))
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(rattled), params)
        end
        _, converged = res
        @test converged
        @test !any(l -> occursin("tighten them mid-flight", string(l.message)) ||
                        occursin("within the solver error", string(l.message)), logs)
        # The verdict is still verified — it just passed.
        @test any(l -> occursin("verified on solvers tightened by", string(l.message)),
                  logs)
    end
end

# ---------------------------------------------------------------------------
# Cell degrees of freedom (Issue #212)
# ---------------------------------------------------------------------------

# The cutoff must be FIXED across the comparison. `nacl_params` derives
# `rc = rc_factor · L`, so calling it per lattice constant would give the
# oracle an a-dependent Hamiltonian while the relaxation keeps the one it
# started with — and the two would minimise different functions.
const RELAX_RC = nacl_params(make_nacl_64(); rc_factor = 0.45)[:rc]

function _cell_params(atoms, dir; kwargs...)
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                       rc_factor = 0.45, with_shell = true, buckingham = true)
    p = (; base...,
         rc = RELAX_RC,
         backend = :cpu,
         elements = Dict("Na" => 1, "Cl" => 2),
         pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]),
         skin = FMD.FPPREC(0.1),
         cgtol = FMD.FPPREC(1e-8), cgmaxiter = 40,
         ftol = FMD.FPPREC(1e-5), stol = FMD.FPPREC(1e-7),
         target_stress = zeros(FMD.FPPREC, 6),
         cell_dof = :iso,
         maxsteps = 600,
         dt0 = FMD.FPPREC(1.0 / UNITS[:fs]),
         dt_max = FMD.FPPREC(10.0 / UNITS[:fs]),
         dmax = FMD.FPPREC(0.1 / UNITS[:Å]),
         emax = FMD.FPPREC(1e-3),
         output_interval = 10000,
         output_dir = dir)
    return (; p..., kwargs...)
end

# Static evaluation of the ideal rocksalt at lattice constant `aA`, with
# analytic Ewald (no PME grid anywhere) and the FIXED cutoff.
function _ideal_static(aA)
    at = make_nacl_64(; a_bohr = aA / UNITS[:Å])
    p = nacl_params(at; alpha = 0.30, ewald_tolerance = 1e-8, rc_factor = 0.45,
                    with_shell = true, buckingham = true)
    p = (; p..., rc = RELAX_RC, cgtol = FMD.FPPREC(1e-10), cgmaxiter = 60,
         skin = FMD.FPPREC(0.0))
    reset!(at)
    Calculators.calculate!(Calculators.CoreShell(), at, p)
    return at
end

# `a` of a relaxed frame: make_nacl_64 is a 2x2x2 conventional cell, L = 2a.
_lattice_a(out) = norm(Array(out.info[:lattice])[1, :]) / 2 * UNITS[:Å]

@testset "cell relaxation reproduces a 1-D minimisation of E(a)" begin
    # The oracle. Golden section on the energy, and independently the
    # bisection of σ_xx = 0 — two different 1-D methods on the analytic-Ewald
    # Hamiltonian, neither of which involves the FIRE driver or a PME grid.
    E_of_a(aA) = Array(_ideal_static(aA).info[:potential_energy])[1]
    s_of_a(aA) = stress(_ideal_static(aA), false)[1]
    function golden(f, lo, hi)
        φ = (sqrt(5) - 1) / 2
        c, d = hi - φ*(hi-lo), lo + φ*(hi-lo)
        fc, fd = f(c), f(d)
        for _ in 1:45
            if fc < fd
                hi, d, fd = d, c, fc; c = hi - φ*(hi-lo); fc = f(c)
            else
                lo, c, fc = c, d, fd; d = lo + φ*(hi-lo); fd = f(d)
            end
        end
        return (lo + hi) / 2
    end
    function bisect(f, lo, hi)
        for _ in 1:55
            m = (lo + hi) / 2
            f(m) < 0 ? (lo = m) : (hi = m)
        end
        return (lo + hi) / 2
    end
    a_energy = golden(E_of_a, 5.4, 5.9)
    a_stress = bisect(s_of_a, 5.4, 5.9)
    # The energy minimum and the stress zero must be the same point — this
    # is the virial convention (σ = +(1/V)∂E/∂ε) checked end-to-end, on the
    # very quantity the cell relaxation is driven by.
    @test isapprox(a_energy, a_stress; atol = 1e-5)

    # The relaxation must find it from either side. Approaching a minimum
    # from above and below is the cheap test for a hysteretic optimiser.
    @testset "start = $a_start Å" for a_start in (5.40, 5.90)
        atoms = make_nacl_64(; a_bohr = a_start / UNITS[:Å])
        mktempdir() do dir
            out, converged = relax!(deepcopy_atoms(atoms),
                                    _cell_params(atoms, dir; cell_dof = :iso))
            @test converged
            @test isapprox(_lattice_a(out), a_energy; atol = 1e-3)
            # Independent verification: the converged stress, re-evaluated
            # with a different Coulomb solver, really is at the target.
            probe = deepcopy_atoms(out)
            reset!(probe)
            Calculators.calculate!(Calculators.CoreShell(), probe,
                                   (; _cell_params(atoms, dir)...,
                                      cgtol = FMD.FPPREC(1e-10), cgmaxiter = 60,
                                      skin = FMD.FPPREC(0.0)))
            @test maximum(abs, stress(probe, false)) < 1e-7
        end
    end
end

@testset "cell relaxation reaches a non-zero target stress" begin
    # Hydrostatic compression: target_stress is a STRESS (tension positive),
    # so a compressive pressure P enters as -P on the diagonal.
    P = FMD.FPPREC(2.0 / UNITS[:GPa])
    atoms = make_nacl_64()
    v0 = volume(atoms)
    mktempdir() do dir
        params = _cell_params(atoms, dir; cell_dof = :iso,
                              target_stress = FMD.FPPREC[-P, -P, -P, 0, 0, 0])
        out, converged = relax!(deepcopy_atoms(atoms), params)
        @test converged
        # Under compression the cell must shrink.
        @test volume(out) < v0
        # And the converged stress must be the target, verified independently.
        probe = deepcopy_atoms(out)
        reset!(probe)
        Calculators.calculate!(Calculators.CoreShell(), probe,
                               (; params..., cgtol = FMD.FPPREC(1e-10),
                                  cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
        sig = stress(probe, false)
        @test maximum(abs, sig[1:3] .- (-P)) < 1e-6
        @test maximum(abs, sig[4:6]) < 1e-7
    end
end

@testset "cell_dof selects the right subspace" begin
    @testset ":none leaves the lattice bitwise untouched" begin
        atoms = _relax_fixture(; rattle = 0.03)
        lat0 = copy(Array(atoms.info[:lattice]))
        mktempdir() do dir
            out, _ = relax!(deepcopy_atoms(atoms),
                            _cell_params(atoms, dir; cell_dof = :none,
                                         maxsteps = 40))
            @test Array(out.info[:lattice]) == lat0
        end
    end

    @testset ":iso keeps a cubic cell cubic" begin
        atoms = make_nacl_64(; a_bohr = 5.40 / UNITS[:Å])
        mktempdir() do dir
            out, converged = relax!(deepcopy_atoms(atoms),
                                    _cell_params(atoms, dir; cell_dof = :iso))
            @test converged
            L = Array(out.info[:lattice])
            # Hydrostatic strain is a multiple of I, so off-diagonals stay
            # exactly zero and the three axes stay exactly equal.
            @test maximum(abs, L .- Diagonal(diag(L))) == 0
            @test L[1, 1] == L[2, 2] == L[3, 3]
        end
    end

    @testset ":diag frees the axes independently but never shears" begin
        # Uniaxial target: σ_zz differs, so c must separate from a = b —
        # something :iso structurally cannot do.
        atoms = make_nacl_64()
        Pz = FMD.FPPREC(2.0 / UNITS[:GPa])
        mktempdir() do dir
            out, converged = relax!(deepcopy_atoms(atoms),
                _cell_params(atoms, dir; cell_dof = :diag,
                             target_stress = FMD.FPPREC[0, 0, -Pz, 0, 0, 0]))
            @test converged
            L = Array(out.info[:lattice])
            @test maximum(abs, L .- Diagonal(diag(L))) == 0     # no shear
            @test L[3, 3] < L[1, 1]                             # compressed along z
            # x and y are equivalent by symmetry, but under :diag they are
            # INDEPENDENT variables converging separately, so they agree to
            # the convergence tolerance, not exactly. The bound follows from
            # stol: the axes can differ by the strain that stol permits,
            # Δε ≲ stol/C₁₁. With stol = 1e-7 Ha/Bohr³ and C₁₁ ≈ 3B ≈ 75 GPa
            # = 2.5e-3 Ha/Bohr³ that is Δε ≲ 4e-5, so 1e-4 is a bound with
            # margin rather than a number chosen to make the test pass
            # (measured: 6.4e-6).
            @test isapprox(L[1, 1], L[2, 2]; rtol = 1e-4)
        end
    end

    @testset ":full relaxes shear away" begin
        # A SYMMETRIC shear: L = L₀(I + ε) with ε = γ(E₁₂ + E₂₁).
        #
        # The distinction matters. The naive `lat[2,1] += γ·lat[2,2]` is
        # L₀(I + γE₂₁), and I + γE₂₁ decomposes into a symmetric strain
        # γ/2·(E₁₂+E₂₁) PLUS a rigid rotation γ/2·(E₂₁−E₁₂). The strain
        # parameterisation here is symmetric by construction — which is
        # exactly how the three rigid-rotation degrees of freedom of a
        # general 3×3 are kept out of the cell force — so a relaxation can
        # remove the strain half and must leave the rotation half alone.
        # Feeding it the naive shear and demanding the whole thing vanish
        # would be asserting that the driver does something wrong.
        γ = FMD.FPPREC(0.01)
        atoms = make_nacl_64()
        lat0 = Array(atoms.info[:lattice])
        eps = FMD.FPPREC[0 γ 0; γ 0 0; 0 0 0]
        atoms.info[:lattice] = lat0 * (I + eps)
        shear0 = abs(Array(atoms.info[:lattice])[1, 2])
        @test shear0 > 0.1                      # the fixture really is sheared
        mktempdir() do dir
            out, converged = relax!(deepcopy_atoms(atoms),
                                    _cell_params(atoms, dir; cell_dof = :full))
            @test converged
            L = Array(out.info[:lattice])
            @test abs(L[1, 2]) < 0.05 * shear0
            @test abs(L[2, 1]) < 0.05 * shear0
            # Rotation-invariant statement of the same thing: the metric
            # G = L·Lᵀ of a cube (rotated or not) is a multiple of I.
            G = L * L'
            @test maximum(abs, G .- Diagonal(diag(G))) / G[1, 1] < 1e-3
            probe = deepcopy_atoms(out)
            reset!(probe)
            Calculators.calculate!(Calculators.CoreShell(), probe,
                                   (; _cell_params(atoms, dir)...,
                                      cgtol = FMD.FPPREC(1e-10), cgmaxiter = 60,
                                      skin = FMD.FPPREC(0.0)))
            @test maximum(abs, stress(probe, false)[4:6]) < 1e-7
        end
    end

    @testset "a rigid rotation of the cell is NOT relaxed away" begin
        # The counterpart of the note above, asserted directly: the naive
        # shear L₀(I + γE₂₁) is half strain, half rotation. The driver must
        # remove the strain and leave the rotation — the metric becomes
        # cubic again while the lattice matrix stays visibly non-diagonal.
        γ = FMD.FPPREC(0.01)
        atoms = make_nacl_64()
        lat = Array(atoms.info[:lattice])
        lat[2, 1] += γ * lat[2, 2]
        atoms.info[:lattice] = lat
        mktempdir() do dir
            out, converged = relax!(deepcopy_atoms(atoms),
                                    _cell_params(atoms, dir; cell_dof = :full))
            @test converged
            L = Array(out.info[:lattice])
            G = L * L'
            @test maximum(abs, G .- Diagonal(diag(G))) / G[1, 1] < 1e-3   # cubic again
            @test abs(L[2, 1]) > 0.3 * abs(lat[2, 1])                     # still rotated
        end
    end
end

@testset "cell inner product counts shear twice (Voigt trap)" begin
    # Regression pin for the reason the cell dof is a symmetric 3x3 rather
    # than a Voigt-6 vector. `voigt2matrix` stores TENSOR components, so
    # σ:ε = Σᵢ σᵢᵢεᵢᵢ + 2Σᵢ<ⱼ σᵢⱼεᵢⱼ. A `dot` of two Voigt-6 vectors would
    # weight every shear component by 1/2 and under-drive the shear modes.
    dom = Cores.SerialDomain()
    v = zeros(FMD.FPPREC, 0, 3)          # no atoms: isolate the cell term
    f = zeros(FMD.FPPREC, 0, 3)
    # Pure shear in xy, both in the force and the velocity.
    fc = FMD.FPPREC[0 2 0; 2 0 0; 0 0 0]
    vc = FMD.FPPREC[0 3 0; 3 0 0; 0 0 0]
    P, vnorm, fnorm = FMD.Apps._relax_reductions(dom, v, f, (vc, fc))
    # Frobenius: 2*(2*3) = 12. A Voigt dot would give 6.
    @test P == 12
    @test vnorm ≈ sqrt(18)
    @test fnorm ≈ sqrt(8)
end

@testset "cell_dof guards" begin
    atoms = _relax_fixture(; rattle = 0.02)
    mktempdir() do dir
        # ExternalField contributes no energy and no virial, so the field's
        # strain derivative is missing from σ and the cell would relax to
        # the wrong shape. Atoms-only is still fine, hence the pairing.
        efield = FMD.FPPREC[0.0, 0.0, 1e-4]
        @test_throws "would relax to the wrong shape" relax!(deepcopy_atoms(atoms),
            _cell_params(atoms, dir; cell_dof = :full, efield = efield))
        @test_throws "would relax to the wrong shape" relax!(deepcopy_atoms(atoms),
            _cell_params(atoms, dir; cell_dof = :iso, efield = efield))
        # ...and with a fixed cell the same field is accepted.
        out, _ = relax!(deepcopy_atoms(atoms),
                        _cell_params(atoms, dir; cell_dof = :none,
                                     efield = efield, maxsteps = 3))
        @test all(isfinite, out.arrays[:forces])

        # An unknown cell_dof must not be silently treated as :none.
        @test_throws "cell_dof must be" relax!(deepcopy_atoms(atoms),
            _cell_params(atoms, dir; cell_dof = :hydrostatic))

        # stol / target_stress are required once the cell moves.
        p = _cell_params(atoms, dir; cell_dof = :iso)
        @test_throws "stol] is required" relax!(deepcopy_atoms(atoms),
            Base.structdiff(p, NamedTuple{(:stol,)}))
        @test_throws "target_stress] is required" relax!(deepcopy_atoms(atoms),
            Base.structdiff(p, NamedTuple{(:target_stress,)}))
    end
end

@testset "PME is rebuilt on edge-length change, not volume" begin
    # An isochoric deformation: stretch x, compress y by the same factor.
    # The volume is unchanged to first order, so a volume-based trigger
    # would sleep through it while the mesh spacing along x degrades.
    atoms = make_nacl_64()
    lat = Array(atoms.info[:lattice])
    lat[1, :] .*= 1.10
    lat[2, :] ./= 1.10
    atoms.info[:lattice] = lat
    p0 = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8, rc_factor = 0.45,
                     with_shell = true, buckingham = true)
    ms = FMD.FPPREC(0.25 / UNITS[:Å])
    pme_a = Calculators.PMEInfo(atoms.info[:lattice],
                                (; p0..., pme_meshsize = ms, alpha = ewaldcoeff(p0)))
    ref = make_nacl_64()
    pme_ref = Calculators.PMEInfo(ref.info[:lattice],
                                  (; p0..., pme_meshsize = ms, alpha = ewaldcoeff(p0)))
    # Same volume to <1%, different grids: this is the case the trigger is for.
    @test isapprox(volume(atoms), volume(ref); rtol = 0.01)
    @test pme_a.fftsize != pme_ref.fftsize

    # And the helper fires on the edge-length ratio.
    fs_like = (; pmeinfo = pme_ref)
    lat_ref = Array(ref.info[:lattice])
    len0 = (norm(lat_ref[1, :]), norm(lat_ref[2, :]), norm(lat_ref[3, :]))
    mktempdir() do dir
        params = _cell_params(atoms, dir; cell_dof = :full, maxsteps = 1)
        a2, p2, arrtyp2, dom2 = FMD.Apps._setup_frame(deepcopy_atoms(ref), params, "test")
        fs, a2 = FMD.Apps.build_force_stack(a2, p2, arrtyp2, dom2)
        @test FMD.Apps._maybe_rebuild_pme!(fs, a2, p2, dom2, len0) == false
        a2.info[:lattice] = FMD.FPPREC.(lat)      # deform past the budget
        @test FMD.Apps._maybe_rebuild_pme!(fs, a2, p2, dom2, len0) == true
        @test fs.pmeinfo.fftsize == pme_a.fftsize

        # The grid must come from the params HANDED to the helper, not from
        # anything it closes over. This is what makes the call site's choice
        # matter: after an acceptance-test escalation the run is on tightened
        # params, and rebuilding from the caller's original `params` would
        # silently restore the coarse mesh while still reporting "verified on
        # tightened solvers". relax! therefore passes `eff`, never `params`.
        tight = FMD.Apps._tighten_solvers(p2)
        len_now = (norm(lat[1, :]), norm(lat[2, :]), norm(lat[3, :]))
        a2.info[:lattice] = FMD.FPPREC.(lat .* 1.2)     # trigger again
        @test FMD.Apps._maybe_rebuild_pme!(fs, a2, tight, dom2, len_now) == true
        fine = fs.pmeinfo.fftsize
        @test all(fine .> pme_a.fftsize)
        # ...and handing it the coarse params instead would undo that.
        a2.info[:lattice] = FMD.FPPREC.(lat .* 1.5)
        @test FMD.Apps._maybe_rebuild_pme!(fs, a2, p2, dom2, len_now) == true
        @test all(fs.pmeinfo.fftsize .< fine)
    end
end

@testset ":cell_dof passed to :NPTMD is rejected with a real reason" begin
    # The default unknown-key report would say "did you mean :cell_factor?",
    # which is worse than useless here — the key exists, it is just not a
    # thing the barostats can honour yet.
    atoms = _relax_fixture()
    mktempdir() do dir
        input = joinpath(dir, "input.xyz")
        elements = Dict("Na" => 1, "Cl" => 2)
        open(input, "w") do io
            writexyz(io, atoms, (; elements = elements))
        end
        fit = nacl_params(atoms; with_shell = true, buckingham = true)
        dt = FMD.FPPREC(1.0 / UNITS[:fs])
        user = (; elements = elements, input = input,
                output_dir = joinpath(dir, "md"),
                pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]),
                rc = fit[:rc], qc = fit[:qc], qs = fit[:qs], k2 = fit[:k2],
                k4 = fit[:k4], eta_c = fit[:eta_c], eta_s = fit[:eta_s],
                a = fit[:a], rho = fit[:rho], b = fit[:b], c = fit[:c], e = fit[:e],
                timestep = dt, tau_t = 20dt, tau_p = 500dt,
                initial_temperature = 0.0, target_temperature = 10.0,
                target_stress = zeros(FMD.FPPREC, 6), output_interval = 1)
        params = (; FMD.default_params(Val(:NPTMD), user)..., user...)
        @test FMD.Apps.validate_params(:NPTMD, params) === nothing
        @test_throws "not supported by :NPTMD" FMD.Apps.validate_params(
            :NPTMD, (; params..., cell_dof = :iso))
    end
end

@testset "escalation and cell-driven PME rebuild coexist" begin
    # Both machineries firing in one run, and the verdict still surviving an
    # independent check.
    #
    # NOTE on what this does NOT cover: the call site passes `eff` (the
    # solver settings in force) rather than `params` to
    # `_maybe_rebuild_pme!`, so an escalation cannot be undone by a later
    # rebuild. That swap is not observable end-to-end — a run escalates only
    # once it has essentially converged, after which the cell no longer moves
    # far enough to trigger another rebuild. The discriminating assertion is
    # the helper-contract one in the testset above; this one pins that the two
    # paths coexist without breaking the result.
    #
    # Forced with `pme_rebuild_ratio` just above 1, so any cell motion at all
    # triggers a rebuild, and a mesh that puts the FINAL cell (a → 5.64 Å,
    # L → 21.32 Bohr) on the coarse ODD grid 21 = 3·7, so the acceptance
    # test escalates there. Odd for the same Issue #211/#216 reason as the
    # escalation testset above: an even grid is commensurate with the
    # relaxed rocksalt and its mesh error cancels — nothing would escalate.
    atoms = make_nacl_64(; a_bohr = 5.45 / UNITS[:Å])
    L_final = 2 * 5.64 / UNITS[:Å]        # relaxed cell edge (calibrated fit)
    mktempdir() do dir
        params = _cell_params(atoms, dir;
                              cell_dof = :iso,
                              pme_meshsize = FMD.FPPREC(L_final / 20.5),
                              pme_rebuild_ratio = 1.0005,
                              ftol = FMD.FPPREC(1e-5),
                              stol = FMD.FPPREC(1e-7),
                              maxsteps = 800)
        logs, res = Test.collect_test_logs() do
            relax!(deepcopy_atoms(atoms), params)
        end
        out, converged = res
        escalated = any(l -> occursin("within the solver error", string(l.message)),
                        logs)
        rebuilt = any(l -> occursin("past the PME mesh budget", string(l.message)),
                      logs)
        # Non-vacuous: both paths really did fire in this run.
        @test escalated
        @test rebuilt
        # The grid the run ended on must be at least as fine as the one the
        # escalation moved to — never back to the coarse starting grid.
        # Whatever it reports, the verdict must survive an independent check.
        if converged
            probe = deepcopy_atoms(out)
            reset!(probe)
            Calculators.calculate!(Calculators.CoreShell(), probe,
                                   (; params..., cgtol = FMD.FPPREC(1e-10),
                                      cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
            @test maximum(abs, probe.arrays[:forces]) <
                  params[:ftol] + RELAX_PME_FORCE_TOL
            @test maximum(abs, stress(probe, false)) < 10 * params[:stol]
        end
    end
end
