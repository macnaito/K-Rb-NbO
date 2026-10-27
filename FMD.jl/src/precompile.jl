# src/precompile.jl
#
# PrecompileTools workload — bakes the type inference of the serial-CPU
# npt! stack (CellList / neighborlist, CoreShell CG, PME, Ewald, Spline
# MLIP, xyz/log output) into the FMD pkgimage. Without it every Julia
# process pays the full first-call JIT of that stack (~35 s, measured on
# the NaCl-64 test fixture; the workload below reproduces the same code
# paths on a NaCl-8 cell), which dominates the MPI test tiers where each
# check spawns fresh child processes (Issue #111).
#
# Inference caching is keyed on TYPES, not values — in particular on the
# concrete NamedTuple type of `params` (key set AND order): npt! rebuilds
# params through a runtime splat, so the whole downstream stack is
# inferred against that concrete shape. The params below deliberately
# mirror nacl_params + _npt_dd_params from the test suite (same keys,
# same order). If those fixtures gain/lose/reorder keys the cache stops
# matching SILENTLY — tests stay green, only the per-process JIT comes
# back. Re-measure a child (e.g. `time mpiexec -n 2 julia --project
# test/mpi/_run_single.jl npt_dd/normal out dir cpu`) when touching
# either side.
#
# The cutoffs are chosen so the loops do real work on NaCl-8 (nearest
# neighbour a/2 = 5.33 Bohr): rc = 0.6a = 6.40 keeps Coulomb real-space
# pairs, rmax2 = rmax3 = 0.85 rc = 5.44 keeps Spline 2-body pairs and
# 3-body triplets, and the position jitter makes the shell CG iterate
# (a perfect lattice converges in 0 steps, leaving the CG body cold).
# GPU kernels (runtime codegen) and MPI-only branches cannot execute at
# precompile time and are deliberately absent; the ghost unification
# (Phase 0) keeps the force kernels shared with the DD path, so DD
# children still hit this cache.
#
# `relax!` is deliberately NOT precompiled, despite having a different
# concrete params NamedTuple type. The concern was that the type mismatch
# would miss this cache and cost a fresh ~35 s JIT of the shared stack in
# every MPI child. Measured (2-rank children, Issue #214): npt_dd/normal
# 27.0 s, relax_dd/normal 22.7 s — the relax child is not paying an extra
# JIT, so a second workload would roughly double the precompile time below
# to buy nothing. Note this is not a controlled comparison (the two checks
# run different workloads): it bounds the relax child's JIT from above,
# which is all the decision needs. Re-measure before adding one.
#
# Operational note: executing npt! here means precompiling FMD needs a
# writable temp dir (mktempdir) and ~40 s extra, once per src change. To
# build FMD without the workload (e.g. on a restricted node), set
#     [FMD]
#     precompile_workload = false
# in LocalPreferences.toml (PrecompileTools per-package switch).

using PrecompileTools: @setup_workload, @compile_workload
import Logging
import Random

@setup_workload begin
    a_bohr = FPPREC(5.64 / Cores.UNITS[:Å])        # NaCl lattice constant
    lat = FPPREC[a_bohr 0 0; 0 a_bohr 0; 0 0 a_bohr]
    pos = FPPREC[                                   # 4 Na (fcc) + 4 Cl
        0 0 0; 0 0.5 0.5; 0.5 0 0.5; 0.5 0.5 0;
        0.5 0.5 0.5; 0.5 0 0; 0 0.5 0; 0 0 0.5]
    # Deterministic jitter (dedicated RNG — the ambient global RNG state
    # is left untouched, Issue #34).
    pos .+= FPPREC(0.005) .* Random.randn(Random.Xoshiro(0), FPPREC, 8, 3)
    dal = Cores.UNITS[:dalton]
    N = 8
    info = Dict{Symbol,Any}(
        :lattice => lat,
        :potential_energy => FPPREC[0],
        :virial => zeros(FPPREC, 6),
        :dielectric_tensor => zeros(FPPREC, 6),
        :method => Symbol[:precompile],
        :species => ["Na", "Cl"],
    )
    arrays = Dict{Symbol,Array}(
        :pos => pos,
        :elemid => [1, 1, 1, 1, 2, 2, 2, 2],
        :forces => zeros(FPPREC, N, 3),
        :shellforces => zeros(FPPREC, N, 3),
        :shelldisp => zeros(FPPREC, N, 3),
        :velo => zeros(FPPREC, N, 3),
        :id => collect(1:N),
        :masses => FPPREC[fill(22.99dal, 4); fill(35.45dal, 4)],
    )
    atoms = Atoms{Array}(info, arrays)

    nelem = 2
    rc = FPPREC(0.6) * a_bohr                       # > a/2: real-space pairs exist
    dt = FPPREC(1.0 / Cores.UNITS[:fs])
    # No :alpha and no :ewald_tolerance on purpose: npt! then exercises
    # the auto-α resolution (pme_optimal_alpha + pme_error_estimate,
    # Issue #216) — the production first-call path. The Ewald block below
    # supplies both keys locally for the fitting-path calling convention.
    params = (;
        rc = rc,
        qc = FPPREC[+1, -1],
        qs = FPPREC[-0.2, +0.2],
        eta_c = fill(FPPREC(1e-3), nelem),
        eta_s = fill(FPPREC(1e-3), nelem),
        a = zeros(FPPREC, nelem, nelem),
        b = zeros(FPPREC, nelem, nelem),
        rho = ones(FPPREC, nelem, nelem),
        c = zeros(FPPREC, nelem, nelem),
        e = FPPREC(1.0),
        k2 = FPPREC[10, 20],
        k4 = zeros(FPPREC, nelem),
        efield = zeros(FPPREC, 3),
        cgtol = FPPREC(2e-5),
        cgmaxiter = 30,
        splp = Calculators.SplineParam(FPPREC(0.85) * rc, FPPREC(0.85) * rc, 2, 4, 4),
        elements = Dict("Na" => 1, "Cl" => 2),
        pme_meshsize = FPPREC(0.5 / Cores.UNITS[:Å]),
        skin = FPPREC(0.0),            # rebuild (CellList + nlists) every step
        nsteps = 2,
        timestep = dt,
        tau_t = 20dt,
        tau_p = 500dt,
        initial_temperature = 300.0,
        target_temperature = 300.0,
        target_stress = zeros(FPPREC, 6),
        seed = 0,
        output_interval = 1,           # writexyz + log.txt every step
        output_dir = mktempdir(),
        backend = :cpu,
        mpi = false,
    )

    @compile_workload begin
        # npt! prints progress (@info); silence the info-level
        # chatter but keep warnings and errors visible so a broken
        # workload still explains itself in the precompile log.
        redirect_stdout(devnull) do
            Logging.with_logger(Logging.ConsoleLogger(stderr, Logging.Warn)) do
                a = Apps.npt!(atoms, params)
                # The npt! loop above always evaluates Coulomb through PME;
                # cover the Ewald reciprocal 2-pass (used by the fitting
                # apps and the coreshell/reduce test children) as well.
                # Same calling convention as the fitting entries: supply
                # :ewald_tolerance and resolve :alpha via ewaldcoeff before
                # building the info (see the EwaldInfo docstring).
                prms = (; params..., ewald_tolerance = FPPREC(1e-6))
                prms = (; prms..., alpha = Calculators.ewaldcoeff(prms))
                cl, a, _ = Cores.CellList(a, prms.rc)
                nl = Cores.neighborlist(a, cl, prms.rc, prms.skin)
                Cores.clear!(a)
                Calculators.calculate!(Calculators.CoreShell(), a, nl,
                                       Calculators.EwaldInfo(a.info[:lattice], prms),
                                       prms)
            end
        end
    end
end
