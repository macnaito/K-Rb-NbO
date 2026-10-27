# test/mpi/_check_hvp_dd.jl
#
# Analytic HVP parity under domain decomposition (Issue #149). The
# reciprocal path is the reason this check exists: the dipole grid of the
# direction vector is spread from owned atoms only, so without the same
# Allreduce the charge grid gets in the force path, every rank would
# convolve its own partial grid — wrong forces with no error raised. The
# real-space and closed-form terms ride the ghost machinery already
# exercised by `coreshell`, but they are included so the comparison is on
# the product a caller actually gets.
#
# mode ∈ normal | tiny | surface | surface_tiny. `tiny` uses NaCl-8 which
# leaves ranks with zero owned atoms at P = 6: the check is then that
# hvp! still completes (the grid reduction and the staged ghost
# replication are collective, so an empty rank that returns early
# deadlocks).
#
# The `surface` flavours add a SurfaceTerm (Issue #181). Its block has a
# reduction of its own — the 3-vector `Σ_j q^s_j v_j` — and a virial
# derivative in the OTHER regime (replicated, returned rather than summed
# per atom). Both mistakes are silent: a missing Allreduce gives every
# rank its own partial dipole direction, and summing the returned
# remainder again multiplies it by the rank count. `surface_tiny`
# additionally drives empty ranks through that Allreduce.
#
# Ewald is deliberately not covered: its HVP has no DD path and must
# refuse loudly instead (asserted here).

function check_hvp_dd(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)
    tiny = mode in ("tiny", "surface_tiny")
    surf = mode in ("surface", "surface_tiny")

    Random.seed!(0)
    full = tiny ? make_nacl_8() : make_nacl_64()
    proto = deepcopy_atoms(full)              # gauge frame (unrattled)
    full.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(full.arrays[:pos]))
    # Non-zero shells: the whole point is the shell-displacement Hessian.
    full.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(full.arrays[:shelldisp]))
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    prms = (; prms..., pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
    if surf
        # |q_s| = 1 (qc + qs = ±0.8, still neutral): the block scales as
        # q_s², and the fixture's ±0.2 would leave it near the comparison
        # tolerance. The gauge is the unrattled prototype — with the run
        # frame as its own gauge δM = 0 and the virial derivative vanishes
        # identically, which would make its reduction regime untested.
        prms = (; prms..., qc = FMD.FPPREC[1.8, -1.8],
                qs = FMD.FPPREC[-1.0, 1.0])
        # One instance for both runs: surface_hvp! only reads it (unlike
        # surface_term!, it writes no `results`).
        prms = (; prms...,
                surface_term = Calculators.SurfaceTerm(orig, prms[:qc],
                    prms[:qs]; eps_out = [2.0, 3.0, 5.0], gauge = proto))
    end
    rc = prms[:rc]
    sk = FMD.FPPREC(0.1)
    rc_eff = FMD.FPPREC(rc * (1 + sk))
    nfull = length(orig)
    prms_b = BACKEND == "cuda" ? arrtyp_prms(CuArray, prms) : prms

    # One global direction, distributed by :id so both runs see the same
    # per-atom vector regardless of the local row order.
    vfull = randn(FMD.FPPREC, nfull, 3)
    vfull ./= sqrt(sum(abs2, vfull))

    function serial_ref()
        a = deepcopy_atoms(orig)
        BACKEND == "cuda" && (a = Cores.change_arrtyp(a, CuArray))
        cl, a, _ = CellList(a, rc_eff)
        nlq = neighborlist(a, cl, rc, sk)
        pme = Calculators.PMEInfo(a.info[:lattice], prms_b)
        ws = Calculators.hvp_workspace(a, nlq, pme, prms_b)
        ids = Array(a.arrays[:id])
        v = BACKEND == "cuda" ? CuArray(vfull[ids, :]) : vfull[ids, :]
        Jv = similar(a.arrays[:shelldisp])
        Kv = similar(a.arrays[:shelldisp])
        Wv = KernelAbstractions.zeros(Cores.get_backend(a), FMD.FPPREC, length(a), 6)
        Wrec = Calculators.hvp!(Jv, Kv, Wv, v, a, ws, prms_b)
        byid(x) = (o = zeros(FMD.FPPREC, nfull, 3); o[ids, :] .= Array(x); o)
        return (J = byid(Jv), K = byid(Kv),
                W = vec(sum(Array(Wv), dims = 1)) .+ Wrec)
    end

    function dd_run(ref)
        dom = Cores.MPIDomain(nfull; comm = comm)
        loc = Cores.distribute(deepcopy_atoms(full), dom)
        loc = Cores.migrate(loc)
        BACKEND == "cuda" && Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        cl, loc, _ = CellList(loc, rc_eff)
        if BACKEND == "cuda"
            loc = Cores.change_arrtyp(loc, CuArray)
            cl = Cores.change_arrtyp(cl, CuArray)
        end
        nlq = neighborlist(loc, cl, rc, sk)
        pme = Calculators.PMEInfo(loc.info[:lattice], prms_b)
        ws = Calculators.hvp_workspace(loc, nlq, pme, prms_b)

        ids = Array(loc.arrays[:id])
        vloc_h = length(loc) > 0 ? vfull[ids, :] : zeros(FMD.FPPREC, 0, 3)
        v = BACKEND == "cuda" ? CuArray(vloc_h) : vloc_h
        Jv = similar(loc.arrays[:shelldisp])
        Kv = similar(loc.arrays[:shelldisp])
        Wv = KernelAbstractions.zeros(Cores.get_backend(loc), FMD.FPPREC,
                                      length(loc), 6)
        Wrec = Calculators.hvp!(Jv, Kv, Wv, v, loc, ws, prms_b)

        # Reaching here at all is half the check under `tiny` (an empty
        # rank that skipped the collectives would have hung above).
        ok = true
        if length(loc) > 0
            jscale = max(one(FMD.FPPREC), maximum(abs, ref.J))
            kscale = max(one(FMD.FPPREC), maximum(abs, ref.K))
            ok &= maximum(abs, Array(Jv) .- ref.J[ids, :]) < 1e-9 * jscale
            ok &= maximum(abs, Array(Kv) .- ref.K[ids, :]) < 1e-9 * kscale
        end
        # Virial derivative (Issue #151): the per-atom part is an owned
        # partial sum, the returned reciprocal remainder is replicated —
        # get either regime wrong (reduce the grid term, or forget to
        # reduce the per-atom one) and this diverges from serial.
        Wloc = length(loc) > 0 ? vec(sum(Array(Wv), dims = 1)) : zeros(FMD.FPPREC, 6)
        MPI.Allreduce!(Wloc, +, comm)
        Wloc .+= Wrec
        ok &= maximum(abs, Wloc .- ref.W) < 1e-9 * max(one(FMD.FPPREC),
                                                       maximum(abs, ref.W))
        ok = MPI.Allreduce(ok, &, comm)
        nempty = MPI.Allreduce(Int(length(loc) == 0), +, comm)
        return (ok = ok, nempty = nempty)
    end

    ref = serial_ref()
    res = dd_run(ref)

    # The Ewald HVP has no DD wiring; it must refuse rather than return a
    # rank-local structure factor.
    ok_ewald_refused = false
    if BACKEND == "cpu"
        dom = Cores.MPIDomain(nfull; comm = comm)
        loc = Cores.distribute(deepcopy_atoms(full), dom)
        loc = Cores.migrate(loc)
        cl, loc, _ = CellList(loc, rc_eff)
        nlq = neighborlist(loc, cl, rc, sk)
        ew = Calculators.EwaldInfo(orig.info[:lattice], prms_b)
        try
            Calculators.hvp_workspace(loc, nlq, ew, prms_b)
        catch
            ok_ewald_refused = true
        end
        ok_ewald_refused = MPI.Allreduce(ok_ewald_refused, &, comm)
    else
        ok_ewald_refused = NaN   # sentinel: a boolean @test would fail loudly
    end

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, backend = BACKEND, mode = mode,
                     ok = res.ok, nempty = res.nempty,
                     jmax = maximum(abs, ref.J), kmax = maximum(abs, ref.K),
                     ok_ewald_refused = ok_ewald_refused)
    end
    return nothing
end
