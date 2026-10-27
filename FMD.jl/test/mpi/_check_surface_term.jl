# test/mpi/_check_surface_term.jl
#
# SurfaceTerm parity under DD. The term is constructed from the full
# replicated frame (the gauge convention shared with the umbrellas);
# surface_term! then runs on the distributed frame — the whole-system
# dipole is an owned partial sum Allreduced inside calcdipole!, and every
# derived quantity (energy, virial, the uniform per-charge force) is
# replicated.  A missing reduction shows up as a rank-count-scaled energy;
# a force applied from a rank-local dipole shows up in the per-atom
# comparison by :id.
#
# mode "tiny": NaCl-8 at P = 6 — the x ∈ [2/3, 1) bricks own no atoms, so
# empty ranks must still ride the dipole Allreduce (the term's only
# collective) without hanging or contributing garbage.
#
# backend = :cuda: the DD frame is converted after distribute/migrate, the
# reference is the GPU serial run — isolating DD deviations from backend
# deviations (same protocol as check_umbrella).

function check_surface_term(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    Random.seed!(0)
    full = mode == "tiny" ? make_nacl_8() : make_nacl_64()
    proto = deepcopy_atoms(full)              # gauge frame (unrattled)
    full.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(full.arrays[:pos]))
    # Deterministic shell displacements keyed by :id, so the fixture is
    # identical no matter how ranks later reorder rows.
    for r in 1:length(full), c in 1:3
        full.arrays[:shelldisp][r, c] =
            FMD.FPPREC(0.001 * full.arrays[:id][r] + 0.01 * c) * 0.01
    end
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; with_shell = true)
    nfull = length(orig)

    # Anisotropic ε_out so all three dipole components couple.  Two
    # identical instances so serial and DD do not share the results Dict.
    mk_st(a) = Calculators.SurfaceTerm(a, prms[:qc], prms[:qs];
                                       eps_out = [2.0, 3.0, 5.0],
                                       gauge = proto)
    st_ref = mk_st(orig)
    st_dd = mk_st(orig)

    prms_b = BACKEND == "cuda" ? arrtyp_prms(CuArray, prms) : prms

    # --- serial reference (GPU serial when backend = cuda) -----------------
    a0 = deepcopy_atoms(orig)
    BACKEND == "cuda" && (a0 = Cores.change_arrtyp(a0, CuArray))
    clear!(a0)
    Calculators.surface_term!(a0, st_ref, prms_b)
    ids0 = Array(a0.arrays[:id])
    byid(x) = (o = zeros(FMD.FPPREC, nfull, size(x, 2)); o[ids0, :] .= Array(x); o)
    E_ref = Array(a0.info[:potential_energy])[1]
    V_ref = collect(FMD.FPPREC, Array(a0.info[:virial]))
    F_ref = byid(a0.arrays[:forces])
    S_ref = byid(a0.arrays[:shellforces])
    bias_ref = st_ref.results["bias"]

    # --- DD ----------------------------------------------------------------
    dom = Cores.MPIDomain(nfull; comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)
    if BACKEND == "cuda"
        Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        loc = Cores.change_arrtyp(loc, CuArray)
    end
    clear!(loc)
    Calculators.surface_term!(loc, st_dd, prms_b)
    idsl = Array(loc.arrays[:id])

    E = Array(loc.info[:potential_energy])[1]
    V = collect(FMD.FPPREC, Array(loc.info[:virial]))
    ok = abs(E - E_ref) < 1e-12 * max(1.0, abs(E_ref)) &&
         maximum(abs, V .- V_ref) < 1e-12 * max(1.0, maximum(abs, V_ref)) &&
         abs(st_dd.results["bias"] - bias_ref) < 1e-12 * max(1.0, abs(bias_ref))
    if length(loc) > 0
        fscale = max(one(FMD.FPPREC), maximum(abs, F_ref))
        ok &= maximum(abs, Array(loc.arrays[:forces]) .- F_ref[idsl, :]) < 1e-12 * fscale
        sscale = max(one(FMD.FPPREC), maximum(abs, S_ref))
        ok &= maximum(abs, Array(loc.arrays[:shellforces]) .- S_ref[idsl, :]) < 1e-12 * sscale
    end
    ok = MPI.Allreduce(ok, &, comm)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, ok = ok, e_ref = E_ref, bias_ref = bias_ref,
                     fmax_ref = maximum(abs, F_ref))
    end
    return nothing
end
