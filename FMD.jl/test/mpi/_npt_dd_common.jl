# test/mpi/_npt_dd_common.jl
#
# Shared fixture + params for the npt! DD tier-2 test — included by BOTH
# the parent (serial reference, no MPI.Init) and the child (DD run under
# mpiexec) so the two runs are guaranteed identical inputs.

function _npt_dd_fixture(tiny::Bool; umbrella::Bool = false)
    Random.seed!(0)
    full = tiny ? make_nacl_8() : make_nacl_64()
    if umbrella
        # Spatial octant groups (iz fastest) matching a (2,2,2) grid —
        # under a brick decomposition a rank owns only its bricks' groups.
        # Assigned BEFORE the rattle: half the rocksalt atoms sit exactly on
        # an octant boundary, and rattling first would scatter them across
        # the faces and leave the octants charged, which PolarizationGauge
        # rejects.
        octant_groupid!(full, (2, 2, 2))
    end
    full.arrays[:pos] .+= (tiny ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    return full
end

function _npt_dd_params(full, outdir; tiny::Bool, mpi::Bool = false,
                        umbrella::Bool = false, csvr::Bool = false,
                        scr::Bool = false, xl::Bool = false,
                        cheb::Bool = false, xlbaro::Bool = false,
                        nlh::Bool = false, surface::Bool = false,
                        backend::Symbol = :cpu)
    base = nacl_params(full; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    splp = Calculators.SplineParam(FMD.FPPREC(0.6 * base[:rc]),
                                   FMD.FPPREC(0.4 * base[:rc]), 2, 4, 4)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    # The umbrella is constructed from the FULL frame (documented DD
    # convention) — identical on every process since the fixture is
    # deterministic. Its compute backend is baked in at construction;
    # CUDA.CUDABackend() is only referenced on the cuda path (the cpu
    # parent never loads it).
    pu_kw = backend === :cuda ? (; backend = CUDA.CUDABackend()) : (;)
    extra = umbrella ?
            (; polarization_umbrella = Calculators.PolarizationUmbrella(
                 full, 1.0, 1.0, 0.05, FMD.FPPREC[1, 0, 0], (2, 2, 2),
                 base[:qc], base[:qs]; gauge = full, m = (1, 0, 0), sigma = 0.5,
                 pu_kw...),
               # Local polarization map (Issue #220) rides this flavour: the
               # same gauge frame as the umbrella, one root-only file per
               # output step — the DD map must equal the serial one
               # (verify_npt_umbrella).
               polarization_gauge = Calculators.PolarizationGauge(
                   full, (2, 2, 2), base[:qc], base[:qs]),
               polmap_interval = 3) :
            (;)
    # skin = 0.005 makes the rebuild (and hence migrate) fire within a few
    # steps at 300 K — the migration path must actually run.
    p = (; base..., extra...,
         backend = backend,
         mpi = mpi,
         splp = splp,
         elements = Dict("Na" => 1, "Cl" => 2),
         pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
         skin = FMD.FPPREC(0.005),
         cgtol = FMD.FPPREC(2e-5),
         nsteps = 6,
         timestep = dt,
         tau_t = 20dt,
         tau_p = 500dt,
         initial_temperature = 300.0,
         target_temperature = 300.0,
         target_stress = zeros(FMD.FPPREC, 6),
         seed = 3,
         output_interval = 3,
         output_dir = outdir)
    # Appended only for the csvr/scr flavours so the default flavour keeps
    # the exact NamedTuple shape mirrored by src/precompile.jl (keys AND
    # order — see its header comment).
    csvr && (p = (; p..., thermostat = :csvr))
    scr && (p = (; p..., barostat = :scr,
                 compressibility = UNITS[:GPa] / 24.0))
    # Shadow XL-BOMD with a warmup short enough that the 6-step run
    # reaches the shadow steps.
    xl && (p = (; p..., barostat = :none, shell_dynamics = :xlbomd,
                xl_warmup = 2))
    # Chebyshev kernel flavour (Issue #152): the extra Hessian-vector
    # products of the preconditioned direction are collective, and their
    # count is state-independent, so DD parity is the property to pin.
    cheb && (p = (; p..., xl_kernel = :chebyshev))
    # NLH baseline (Issue #124): a core-core gather kernel whose per-atom
    # energy/virial ride the same owned-partial-sum Allreduce as the other
    # short-range terms, on a list derived from the rc parent. The cutoff
    # must stay inside rc (the derived list would otherwise be illegal) AND
    # outside the NaCl nearest-neighbour distance a/2 = 2.82 Å, or the term
    # contributes exactly zero and the DD parity check tests nothing:
    # 0.9·rc = 4.57 Å on the NaCl-64 fixture covers the 2.82 Å and 3.99 Å
    # shells. (On the NaCl-8 `tiny` fixture rc itself is only 2.54 Å, so the
    # term is necessarily zero there — that flavour exists to drive the
    # empty-rank launch/collective path, not the physics.)
    nlh && (p = (; p..., nlh = Calculators.NLHParam(Dict("Na" => 1, "Cl" => 2),
                                                    FMD.FPPREC(0.9 * base[:rc]))))
    # Shadow virial under a barostat (Issue #151): the correction splits
    # into an owned per-atom part (Allreduced) and a replicated
    # reciprocal remainder (not), so a wrong reduction regime moves the
    # cell differently on P ranks than in serial — which is exactly what
    # comparing the final lattice catches.
    xlbaro && (p = (; p..., barostat = :mttk, shell_dynamics = :xlbomd,
                    xl_warmup = 2))
    # Shadow XL-BOMD with the depolarization term (Issue #181): the
    # surface HVP block adds a reduction of its own (Σ q_s v) and a
    # replicated virial derivative, both inside the shadow step. Under a
    # barostat the cell integrates that virial, so a wrong reduction
    # regime moves the lattice differently on P ranks than in serial.
    # The gauge is the UNRATTLED prototype: with the run frame as its own
    # gauge δM = 0 and the whole term is inert. |q_s| = 1 (qc + qs = ±0.8,
    # neutral) keeps the block well above the comparison tolerance.
    if surface
        p = (; p..., qc = FMD.FPPREC[1.8, -1.8], qs = FMD.FPPREC[-1.0, 1.0])
        proto = _npt_dd_fixture_proto(full)
        p = (; p..., barostat = :mttk, shell_dynamics = :xlbomd,
             xl_warmup = 2,
             surface_term = Calculators.SurfaceTerm(full, p[:qc], p[:qs];
                 eps_out = [2.0, 3.0, 5.0], gauge = proto))
    end
    return p
end

# The gauge frame the `surface` flavour measures δM from: the same cell
# without the rattle. Rebuilt rather than threaded through, so the parent
# (serial reference) and the child (DD run) construct it identically.
function _npt_dd_fixture_proto(full)
    proto = length(full) <= 8 ? make_nacl_8() : make_nacl_64()
    length(proto) == length(full) ||
        error("_npt_dd_fixture_proto: fixture size mismatch")
    return proto
end
