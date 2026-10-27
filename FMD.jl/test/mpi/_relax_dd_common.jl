# test/mpi/_relax_dd_common.jl
#
# Shared fixture + params for the relax! DD check — included by BOTH the
# parent (serial reference, no MPI.Init) and the child (DD run under
# mpiexec) so the two runs are guaranteed identical inputs.

function _relax_dd_fixture(tiny::Bool)
    Random.seed!(0)
    full = tiny ? make_nacl_8() : make_nacl_64()
    full.arrays[:pos] .+= (tiny ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    return full
end

function _relax_dd_params(full, outdir; tiny::Bool, mpi::Bool = false,
                          cell_dof::Symbol = :none, backend::Symbol = :cpu)
    # buckingham = true: with a ≡ 0 there is no short-range wall, rocksalt is
    # not a minimum, and the run would collapse rather than relax.
    #
    # make_nacl_8 (L = a) cannot reach its own nearest neighbours at
    # rc_factor = 0.45, so the tiny fixture is bare Coulomb + springs. That
    # is fine here: this check is about the collectives being rank-uniform,
    # not about converging to a physical structure — it runs a fixed, small
    # number of steps and compares serial against DD.
    base = nacl_params(full; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true,
                       buckingham = !tiny)
    return (; base...,
            backend = backend,
            mpi = mpi,
            elements = Dict("Na" => 1, "Cl" => 2),
            pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]),
            # Small skin so the rebuild (and hence migrate) fires during the
            # run: the DD rebuild path must actually execute.
            skin = FMD.FPPREC(0.02),
            cgtol = FMD.FPPREC(1e-7),
            cgmaxiter = 30,
            # ftol below what these few steps can reach, on purpose: the run
            # exercises the full step budget instead of stopping early, so
            # every collective is hit many times.
            ftol = FMD.FPPREC(1e-8),
            ftol_cgtol_margin = 0.1,
            maxsteps = tiny ? 6 : 10,
            dt0 = FMD.FPPREC(1.0 / UNITS[:fs]),
            dt_max = FMD.FPPREC(10.0 / UNITS[:fs]),
            # Deliberately tight: forces the cap to bind, so
            # _cap_displacement!'s Allreduce is exercised rather than
            # short-circuited.
            dmax = FMD.FPPREC(0.002 / UNITS[:Å]),
            # The cell degrees of freedom add three more rank-uniform
            # decisions: the projected cell force (built from a replicated
            # stress, so it must be added AFTER the atom reduction), the
            # strain cap, and the edge-length PME rebuild trigger. emax is
            # tight for the same reason dmax is: the cap must bind.
            cell_dof = cell_dof,
            target_stress = zeros(FMD.FPPREC, 6),
            stol = FMD.FPPREC(1e-12),
            emax = FMD.FPPREC(2e-4),
            cell_factor = FMD.FPPREC(length(full)),
            output_interval = 1000,
            output_dir = outdir)
end
