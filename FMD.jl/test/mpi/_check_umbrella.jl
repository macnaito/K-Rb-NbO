# test/mpi/_check_umbrella.jl
#
# Polarization umbrella parity under DD. The umbrella is constructed from
# the full replicated frame (the documented convention); umbrella_p! then
# runs on the distributed frame — Pg Allreduce, replicated group-lattice
# terms, owned-partial atom virial.
#
# backend = :cuda: the umbrella is built with the CUDA backend from the
# (host) full frame, the reference is the GPU serial run, and the DD frame
# is converted after distribute/migrate — isolating DD deviations from
# backend deviations.

function check_umbrella(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    Random.seed!(0)
    full = make_nacl_64()
    # Spatial group layout (octants of the box, ASE convention: iz fastest,
    # ix slowest) matching the (2,2,2) group grid. Crucially, under a brick
    # decomposition a rank owns only the groups inside its brick — the
    # rank-local groupid maximum is then NOT the global one, which is exactly
    # what _global_ngroups exists for (an id-interleaved layout would let a
    # rank-local maximum slip through).
    #
    # Assigned from the IDEAL positions, before the rattle: half the rocksalt
    # atoms sit exactly on an octant boundary, so rattling first scatters them
    # across the faces at random and leaves the octants charged (measured: net
    # up to ±4 e, 5-10 atoms instead of 8), which PolarizationGauge rejects.
    let w = Cores.wrapfrac.(full.arrays[:pos])
        gx = min.(Int.(floor.(w[:, 1] .* 2)), 1)
        gy = min.(Int.(floor.(w[:, 2] .* 2)), 1)
        gz = min.(Int.(floor.(w[:, 3] .* 2)), 1)
        full.arrays[:groupid] = gx .* 4 .+ gy .* 2 .+ gz .+ 1
    end
    full.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(full.arrays[:pos]))
    h(id, c) = FMD.FPPREC(0.001 * id + 0.01 * c)
    for r in 1:length(full), c in 1:3
        full.arrays[:shelldisp][r, c] = 0.01 * h(full.arrays[:id][r], c)
    end
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    nfull = length(orig)

    # Umbrella from the FULL frame (documented DD convention; the constructor
    # takes the host frame and bakes the requested compute backend in). Two
    # identical instances so the serial and DD evaluations do not share
    # mutable state (results Dict / workspace buffers).
    pu_backend = BACKEND == "cuda" ? CUDA.CUDABackend() : KernelAbstractions.CPU()
    # mode "map" swaps the bias for PolarizationMapUmbrella on the same frame
    # and the same group grid: it shares the DD path (group_p Allreduce,
    # replicated group-lattice terms, owned-partial atom virial) with the mode
    # umbrella, so the parity contract below is identical and no extra child
    # process is needed.
    ng = 2 * 2 * 2
    tgt = FMD.FPPREC[0.002 * sin(2.0i + 3.0d) for i in 1:ng, d in 1:3]
    wmask = FMD.FPPREC[i == 3 ? 0.0 : 0.5 + 0.5 * (i % 2) for i in 1:ng]
    mk_pu(a) = mode == "map" ?
        Calculators.PolarizationMapUmbrella(
            a, 50.0, tgt, (2, 2, 2), prms[:qc], prms[:qs];
            gauge = a, weight = wmask, sigma = 0.5, r_th = 1e-4,
            backend = pu_backend) :
        Calculators.PolarizationUmbrella(
            a, 1.0, 1.0, 0.05, FMD.FPPREC[1, 0, 0], (2, 2, 2),
            prms[:qc], prms[:qs]; gauge = a, m = (1, 0, 0), sigma = 0.5,
            backend = pu_backend)
    pu_ref = mk_pu(orig)
    pu_dd = mk_pu(orig)

    prms_b = BACKEND == "cuda" ? arrtyp_prms(CuArray, prms) : prms

    # --- serial reference (GPU serial when backend = cuda) -----------------
    a0 = deepcopy_atoms(orig)
    BACKEND == "cuda" && (a0 = Cores.change_arrtyp(a0, CuArray))
    clear!(a0)
    fill!(a0.arrays[:forces], 0)
    fill!(a0.arrays[:shellforces], 0)
    Calculators.umbrella_p!(a0, pu_ref, prms_b)
    ids0 = Array(a0.arrays[:id])
    byid(x) = (o = zeros(FMD.FPPREC, nfull, size(x, 2)); o[ids0, :] .= Array(x); o)
    E_ref = Array(a0.info[:potential_energy])[1]
    V_ref = collect(FMD.FPPREC, Array(a0.info[:virial]))
    F_ref = byid(a0.arrays[:forces])
    S_ref = byid(a0.arrays[:shellforces])
    cv_ref = pu_ref.results["cv"]
    bias_ref = pu_ref.results["bias"]

    # --- DD ----------------------------------------------------------------
    dom = Cores.MPIDomain(nfull; comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)
    if BACKEND == "cuda"
        # Phase 2 target order (distribute/migrate on host, then convert);
        # npt! adopts this in the final Phase 2 PR.
        Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        loc = Cores.change_arrtyp(loc, CuArray)
    end
    clear!(loc)
    fill!(loc.arrays[:forces], 0)
    fill!(loc.arrays[:shellforces], 0)
    Calculators.umbrella_p!(loc, pu_dd, prms_b)
    idsl = Array(loc.arrays[:id])

    E = Array(loc.info[:potential_energy])[1]
    V = collect(FMD.FPPREC, Array(loc.info[:virial]))
    ok = abs(E - E_ref) < 1e-12 * max(1.0, abs(E_ref)) &&
         maximum(abs, V .- V_ref) < 1e-12 * max(1.0, maximum(abs, V_ref)) &&
         abs(pu_dd.results["cv"] - cv_ref) < 1e-12 * max(1.0, abs(cv_ref)) &&
         abs(pu_dd.results["bias"] - bias_ref) < 1e-12 * max(1.0, abs(bias_ref))
    if length(loc) > 0
        fscale = max(one(FMD.FPPREC), maximum(abs, F_ref))
        ok &= maximum(abs, Array(loc.arrays[:forces]) .- F_ref[idsl, :]) < 1e-12 * fscale
        sscale = max(one(FMD.FPPREC), maximum(abs, S_ref))
        ok &= maximum(abs, Array(loc.arrays[:shellforces]) .- S_ref[idsl, :]) < 1e-12 * sscale
    end
    ok = MPI.Allreduce(ok, &, comm)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, ok = ok,
                     e_ref = E_ref, cv_ref = cv_ref, bias_ref = bias_ref,
                     fmax_ref = maximum(abs, F_ref))
    end
    return nothing
end
