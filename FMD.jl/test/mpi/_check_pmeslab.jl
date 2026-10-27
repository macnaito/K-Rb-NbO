# test/mpi/_check_pmeslab.jl
#
# z-slab distributed PME (Issue #195) — DD parity of the slab solve
# against the in-process serial (replicated) reference, in the
# _check_reduce.jl style: E / virial on every rank, forces and
# shellforces per id, neutral AND charged cell, plus two slab-specific
# pins the replicated check cannot give:
#   * same-run bit-identity of OnlyShellForce vs full shellforces (the
#     structural sharing of _pme_convolve_dd!),
#   * yoff ≠ 0 correctness of the ky-slab factor slices (P = 1 tests
#     degenerate to yoff = 0; here every rank asserts its own slice).
#
#   mode ∈ normal | tiny —
#     normal: NaCl-64, production-like mesh; P ∈ {2, 4} exercises uneven
#       z/ky partitions and the transpose across all rank pairs.
#     tiny: NaCl-8 at P = 6 with a mesh coarse enough that fftsize floors
#       at BSPORDER (4,4,4) → zcounts = ycounts = [1,1,1,1,0,0]: the
#       zero-plane / zero-ky ranks (nothing plans, dummy MPI buffers,
#       Mloc == 0 virial contraction) coincide with the empty-atom ranks
#       of the (3,2,1) brick grid — the triple degeneracy in one run.
#   backend ∈ :cpu | :cuda — cuda: serial reference is the GPU serial run
#     (isolates DD deviation from backend deviation), device grids ride
#     the CUDA-aware segmented collectives.

function check_pmeslab(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    Random.seed!(0)
    full = mode == "tiny" ? make_nacl_8() : make_nacl_64()
    full.arrays[:pos] .+= (mode == "tiny" ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    h(id, c) = FMD.FPPREC(0.001 * id + 0.01 * c)
    for r in 1:length(full), c in 1:3
        full.arrays[:shelldisp][r, c] = 0.01 * h(full.arrays[:id][r], c)
    end
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    nfull = length(orig)

    # tiny: mesh coarser than side/BSPORDER floors the grid at (4,4,4).
    side = norm(Array(orig.info[:lattice])[1, :])
    meshsize = mode == "tiny" ? FMD.FPPREC(side / 2.5) :
                                FMD.FPPREC(0.30 / UNITS[:Å])
    prms_pme = (; prms..., pme_meshsize = meshsize)
    prms_pme_q = (; prms_pme..., qc = prms_pme[:qc] .+ FMD.FPPREC(0.05))
    bprms(p) = BACKEND == "cuda" ? arrtyp_prms(CuArray, p) : p
    prms_pme_b = bprms(prms_pme)
    prms_pme_q_b = bprms(prms_pme_q)

    # --- serial (replicated) reference --------------------------------------
    a0 = deepcopy_atoms(orig)
    BACKEND == "cuda" && (a0 = Cores.change_arrtyp(a0, CuArray))
    ids0 = Array(a0.arrays[:id])

    function run_ref(f!)
        clear!(a0)
        fill!(a0.arrays[:forces], 0)
        fill!(a0.arrays[:shellforces], 0)
        f!(a0)
        E = Array(a0.info[:potential_energy])[1]
        V = collect(FMD.FPPREC, Array(a0.info[:virial]))
        Fbyid = zeros(FMD.FPPREC, nfull, 3)
        Fbyid[ids0, :] .= Array(a0.arrays[:forces])
        Sbyid = zeros(FMD.FPPREC, nfull, 3)
        Sbyid[ids0, :] .= Array(a0.arrays[:shellforces])
        return (E = E, V = V, F = Fbyid, S = Sbyid)
    end

    pme0 = Calculators.PMEInfo(a0.info[:lattice], prms_pme_b)
    ref = run_ref(a -> Calculators.calculate!(pme0, a, prms_pme_b))
    ref_q = run_ref(a -> Calculators.calculate!(pme0, a, prms_pme_q_b))

    # --- DD side ------------------------------------------------------------
    dom = Cores.MPIDomain(nfull; comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)
    if BACKEND == "cuda"
        Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        loc = Cores.change_arrtyp(loc, CuArray)
    end
    idsl = Array(loc.arrays[:id])

    # Domain-aware construction must actually arm the slab solver here
    # (P > 1, pme_dd defaulting to true) — a silently replicated run would
    # make every parity below vacuous.
    pme_dd = Calculators.PMEInfo(loc.info[:lattice], prms_pme_b, dom)
    ok_armed = MPI.Allreduce(pme_dd.dd !== nothing, &, comm)
    # ... and the opt-out must disarm it.
    pme_off = Calculators.PMEInfo(loc.info[:lattice],
                                  (; prms_pme_b..., pme_dd = false), dom)
    ok_optout = MPI.Allreduce(pme_off.dd === nothing, &, comm)

    # ky-slab factor slices at this rank's own yoff (yoff ≠ 0 for every
    # rank but the first — the P = 1 serial test cannot see this).
    dd = pme_dd.dd
    yr = dd.yoffs[dd.myrank+1]+1 : dd.yoffs[dd.myrank+1]+dd.nyloc
    ok_slice = Array(dd.bsp2factor_dd) == Array(pme_dd.bsp2factor)[:, yr, :] &&
               Array(dd.expfactor_dd) == Array(pme_dd.expfactor)[:, yr, :] &&
               Array(dd.presfactor_dd) == Array(pme_dd.presfactor)[:, yr, :, :]
    ok_slice = MPI.Allreduce(ok_slice, &, comm)

    function check_dd(f!, r; check_shell = false)
        clear!(loc)
        fill!(loc.arrays[:forces], 0)
        fill!(loc.arrays[:shellforces], 0)
        f!(loc)
        E = Array(loc.info[:potential_energy])[1]
        V = collect(FMD.FPPREC, Array(loc.info[:virial]))
        ok = abs(E - r.E) < 1e-12 * max(one(FMD.FPPREC), abs(r.E)) &&
             maximum(abs, V .- r.V) < 1e-12 * max(one(FMD.FPPREC), maximum(abs, r.V))
        if length(loc) > 0
            fscale = max(one(FMD.FPPREC), maximum(abs, r.F))
            ok &= maximum(abs, Array(loc.arrays[:forces]) .- r.F[idsl, :]) < 1e-12 * fscale
            if check_shell
                sscale = max(one(FMD.FPPREC), maximum(abs, r.S))
                ok &= maximum(abs, Array(loc.arrays[:shellforces]) .- r.S[idsl, :]) <
                      1e-12 * sscale
            end
        end
        return MPI.Allreduce(ok, &, comm)
    end

    ok_pme = check_dd(a -> Calculators.calculate!(pme_dd, a, prms_pme_b), ref;
                      check_shell = true)
    ok_pme_q = check_dd(a -> Calculators.calculate!(pme_dd, a, prms_pme_q_b), ref_q;
                        check_shell = true)

    # Same-run bit-identity: full then OnlyShellForce on the same frame —
    # identical convolution by structural sharing, identical gather, so
    # the shellforces must be `==` (P-fixed, one process: bitwise is
    # legitimate, unlike cross-P comparisons). Neutral and charged cell
    # (the correction never touches shellforces — pinned rather than
    # assumed).
    function bit_full_vs_shellonly(p)
        clear!(loc)
        fill!(loc.arrays[:forces], 0)
        fill!(loc.arrays[:shellforces], 0)
        Calculators.calculate!(pme_dd, loc, p)
        S_full = copy(Array(loc.arrays[:shellforces]))
        fill!(loc.arrays[:forces], 0)
        fill!(loc.arrays[:shellforces], 0)
        Calculators.calculate!(pme_dd, Calculators.OnlyShellForce(), loc, p)
        return length(loc) == 0 || Array(loc.arrays[:shellforces]) == S_full
    end
    ok_bit = MPI.Allreduce(bit_full_vs_shellonly(prms_pme_b) &&
                           bit_full_vs_shellonly(prms_pme_q_b), &, comm)

    ntot = MPI.Allreduce(length(loc), +, comm)
    nmax = MPI.Allreduce(length(loc), max, comm)
    nmin = MPI.Allreduce(length(loc), min, comm)
    nz_zero = count(==(0), dd.zcounts)   # partition degeneracy actually hit

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, backend = BACKEND, fftsize = collect(pme_dd.fftsize),
                     e_pme = ref.E, dE_q = ref_q.E - ref.E,
                     vnorm = norm(ref.V),
                     ntot = ntot, nmax = nmax, nmin = nmin, nz_zero = nz_zero,
                     ok_armed = ok_armed, ok_optout = ok_optout, ok_slice = ok_slice,
                     ok_pme = ok_pme, ok_pme_q = ok_pme_q, ok_bit = ok_bit)
    end
    return nothing
end
