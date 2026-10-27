# test/mpi/_check_coreshell.jl
#
# Shell-CG parity under DD for both long-range backends (PME and Ewald; on
# backend = :cuda the Ewald pair is skipped — its reciprocal kernel is
# host-only by design, PME is the GPU long-range). The lockstep property
# (identical fcount on every rank) is the structural guarantee. The DD and
# serial runs follow the same iterate sequence up to reduction
# reassociation (~1e-16 per scalar), so the converged states agree far
# tighter than the |F| < cgtol ball radius; the tolerances below are still
# deliberately looser than the observed reassociation scale (worst-case
# bounds, flake-resistant), tight enough to catch any lost ghost pair or
# mis-folded contribution. Note E is 2nd order in the shell displacement
# error but virial/forces are 1st order — only the reassociation-scale
# agreement, not the stationary-point argument, justifies the virial
# tolerance.

function check_coreshell(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    Random.seed!(0)
    full = make_nacl_64()
    full.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(full.arrays[:pos]))
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    prms = (; prms..., cgtol = FMD.FPPREC(2e-5), cgmaxiter = 30,
            pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
    rc = prms[:rc]
    sk = FMD.FPPREC(0.1)
    rc_eff = FMD.FPPREC(rc * (1 + sk))
    nfull = length(orig)

    prms_b = BACKEND == "cuda" ? arrtyp_prms(CuArray, prms) : prms

    function serial_ref(make_coulomb)
        a = deepcopy_atoms(orig)
        BACKEND == "cuda" && (a = Cores.change_arrtyp(a, CuArray))   # GPU serial ref
        cl, a, _ = CellList(a, rc_eff)
        nlq = neighborlist(a, cl, rc, sk)
        coulomb = make_coulomb(a)
        clear!(a)
        ncg = Calculators.calculate!(Calculators.CoreShell(), a, nlq, coulomb, prms_b)
        ids = Array(a.arrays[:id])
        byid(x) = (o = zeros(FMD.FPPREC, nfull, size(x, 2)); o[ids, :] .= Array(x); o)
        return (ncg = ncg, E = Array(a.info[:potential_energy])[1],
                V = collect(FMD.FPPREC, Array(a.info[:virial])),
                F = byid(a.arrays[:forces]), S = byid(a.arrays[:shelldisp]))
    end

    function dd_run(make_coulomb, ref)
        dom = Cores.MPIDomain(nfull; comm = comm)
        loc = Cores.distribute(deepcopy_atoms(full), dom)
        loc = Cores.migrate(loc)
        BACKEND == "cuda" && Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        cl, loc, _ = CellList(loc, rc_eff)   # host DD build (+ ghost permute)
        if BACKEND == "cuda"
            # Phase 2 target order: host build, convert atoms + cell list,
            # rebuild the list on the device (entry sync = device DD exchange).
            loc = Cores.change_arrtyp(loc, CuArray)
            cl = Cores.change_arrtyp(cl, CuArray)
        end
        nlq = neighborlist(loc, cl, rc, sk)
        coulomb = make_coulomb(loc)
        clear!(loc)
        ncg = Calculators.calculate!(Calculators.CoreShell(), loc, nlq, coulomb, prms_b)

        # Lockstep: every rank must have executed the identical iteration
        # sequence — a diverging count means a rank-local branch survived.
        counts = MPI.Allgather(Int32(ncg), comm)
        ok_lockstep = all(==(counts[1]), counts)

        ids = Array(loc.arrays[:id])
        E = Array(loc.info[:potential_energy])[1]
        V = collect(FMD.FPPREC, Array(loc.info[:virial]))
        ok = abs(E - ref.E) < 1e-8 * max(1.0, abs(ref.E)) &&
             maximum(abs, V .- ref.V) < 1e-8 * max(1.0, maximum(abs, ref.V))
        if length(loc) > 0
            fscale = max(one(FMD.FPPREC), maximum(abs, ref.F))
            ok &= maximum(abs, Array(loc.arrays[:forces]) .- ref.F[ids, :]) < 1e-6 * fscale
            ok &= maximum(abs, Array(loc.arrays[:shelldisp]) .- ref.S[ids, :]) < 1e-6
        end
        ok = MPI.Allreduce(ok, &, comm)
        ok_lockstep = MPI.Allreduce(ok_lockstep, &, comm)
        return (ncg = ncg, ok = ok, ok_lockstep = ok_lockstep)
    end

    ref_pme = serial_ref(a -> Calculators.PMEInfo(a.info[:lattice], prms_b))
    res_pme = dd_run(a -> Calculators.PMEInfo(a.info[:lattice], prms_b), ref_pme)

    if BACKEND == "cpu"
        ref_ew = serial_ref(a -> Calculators.EwaldInfo(a.info[:lattice], prms_b))
        res_ew = dd_run(a -> Calculators.EwaldInfo(a.info[:lattice], prms_b), ref_ew)
    else
        # Ewald reciprocal is host-only by design. Keep the saved key set
        # constant across backends, but store NaN sentinels — a future
        # `@test d["ok_ew"]` on the cuda output must fail loudly (non-boolean),
        # never pass on a value that means "not checked".
        ref_ew = (ncg = 0,)
        res_ew = (ncg = 0, ok = NaN, ok_lockstep = NaN)
    end

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, backend = BACKEND,
                     ncg_ref_pme = ref_pme.ncg, ncg_pme = res_pme.ncg,
                     ok_pme = res_pme.ok, ok_lockstep_pme = res_pme.ok_lockstep,
                     ncg_ref_ew = ref_ew.ncg, ncg_ew = res_ew.ncg,
                     ok_ew = res_ew.ok, ok_lockstep_ew = res_ew.ok_lockstep)
    end
    return nothing
end
