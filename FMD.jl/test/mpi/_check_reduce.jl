# test/mpi/_check_reduce.jl
#
# Per-calculator DD parity under the reduction regime — after every
# calculate!, info[:potential_energy] / info[:virial] must hold the
# whole-system value on every rank, equal to the serial reference; forces
# (and shellforces) must match per id. Also kinetic_energy / stress /
# dipole parity.
#
#   mode ∈ normal | tiny — tiny: NaCl-8 at P = 6 → empty ranks; only the
#     empty-rank-safe checks (Ewald/PME + global scalars) run — short-range
#     kernels are hardened for empty ranks with the DD driver integration.
#   backend ∈ :cpu | :cuda — cuda: the serial reference is the GPU *serial*
#     run (isolating DD deviations from backend deviations). The DD frame
#     is built on the host (distribute/migrate/CellList), then atoms and
#     cell list are converted with change_arrtyp and the neighbor list is
#     rebuilt on the device — its entry sync runs the device DD ghost
#     exchange, so the CB/ACE/Spline parities exercise the full device
#     short-range stack. Ewald stays cpu-only (reciprocal kernel is
#     host-only by design; the Q paths run through PME on cuda).

function check_reduce(out, dir, mode, backend, comm)
    BACKEND = String(backend)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    # Which calculator families run in this configuration.
    RUN_NL = mode != "tiny"        # neighbor-list calculators (both backends)
    RUN_SPRING = mode != "tiny"    # on-site, backend-generic
    RUN_EWALD = BACKEND == "cpu"   # Ewald reciprocal is CPU-only by design
                                   # (GPU long-range goes through PME)

    # Identical replicated frame with nonzero shells and velocities (keyed by
    # id so any row order gives the same physical state).
    Random.seed!(0)
    full = mode == "tiny" ? make_nacl_8() : make_nacl_64()
    full.arrays[:pos] .+= (mode == "tiny" ? 0.005 : 0.01) .*
                          randn(FMD.FPPREC, size(full.arrays[:pos]))
    mode == "tiny" || (full.arrays[:pos][1, :] .+= 1.0)
    h(id, c) = FMD.FPPREC(0.001 * id + 0.01 * c)
    for r in 1:length(full), c in 1:3
        full.arrays[:shelldisp][r, c] = 0.01 * h(full.arrays[:id][r], c)
        full.arrays[:velo][r, c] = 1e-4 * h(full.arrays[:id][r], c)
    end
    orig = deepcopy_atoms(full)
    prms = nacl_params(orig; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    rc = prms[:rc]
    sk = FMD.FPPREC(0.1)
    rc_eff = FMD.FPPREC(rc * (1 + sk))
    nfull = length(orig)

    # Host param variants first, then the backend conversion (identity on cpu).
    prms_q = (; prms..., qc = prms[:qc] .+ FMD.FPPREC(0.05))
    prms_pme = (; prms..., pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
    prms_pme_q = (; prms_pme..., qc = prms_pme[:qc] .+ FMD.FPPREC(0.05))
    bprms(p) = BACKEND == "cuda" ? arrtyp_prms(CuArray, p) : p
    prms_b = bprms(prms)
    prms_q_b = bprms(prms_q)
    prms_pme_b = bprms(prms_pme)
    prms_pme_q_b = bprms(prms_pme_q)

    # Deterministic MLIP fixtures (identical on every rank; built on the host,
    # adapted to the device for cuda — Adapt preserves FPPREC, unlike cu).
    rng = Random.Xoshiro(7)
    acep = ACEParam(FMD.FPPREC(1.0 / UNITS[:Å]), FMD.FPPREC(0.8 * rc), 2, 4, 4, 3, 3)
    for fn in fieldnames(typeof(acep))
        x = getfield(acep, fn)
        x isa AbstractArray && (x .= 0.01 .* randn(rng, FMD.FPPREC, size(x)...))
    end
    splp = Calculators.SplineParam(FMD.FPPREC(0.8 * rc), FMD.FPPREC(0.6 * rc), 2, 4, 4)
    for fn in (:c1, :c2, :c3)
        x = getfield(splp, fn)
        x .= 0.01 .* randn(rng, FMD.FPPREC, size(x)...)
    end
    Calculators._spl_symmetrize_c3!(splp.c3)   # neighbour-order invariance contract
    acep_b = BACKEND == "cuda" ? Adapt.adapt(CuArray, acep) : acep
    splp_b = BACKEND == "cuda" ? Adapt.adapt(CuArray, splp) : splp

    # --- serial references -------------------------------------------------
    a0 = deepcopy_atoms(orig)
    BACKEND == "cuda" && (a0 = Cores.change_arrtyp(a0, CuArray))   # GPU serial frame
    local nl0, nl0_ace, nl0_2, nl0_3
    if RUN_NL
        # Serial builders are backend-generic — on cuda this is the existing
        # GPU serial CellList/list path.
        cl0, a0, _ = CellList(a0, rc_eff)
        nl0 = neighborlist(a0, cl0, rc, sk)
        nl0_ace = neighborlist(a0, nl0, acep.rmax, sk)
        nl0_2 = neighborlist(a0, nl0, splp.rmax2, sk)
        nl0_3 = neighborlist(a0, nl0, splp.rmax3, sk)
    elseif BACKEND == "cpu"
        # tiny cpu keeps its historical CellList sort of the reference frame
        # (the row order feeds the reduction order of the surviving refs).
        cl0, a0, _ = CellList(a0, rc_eff)
    end
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

    refs = Dict{String,Any}()
    if RUN_NL
        refs["cb"] = run_ref(a -> Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl0, prms_b))
        refs["ace"] = run_ref(a -> Calculators.calculate!(acep_b, a, nl0_ace))
        refs["spline"] = run_ref(a -> Calculators.calculate!(splp_b, a, nl0_2, nl0_3))
    end
    if RUN_SPRING
        refs["spring"] = run_ref(a -> Calculators.calculate!(Calculators.Spring(), a, prms_b))
    end
    # Ewald (two-pass, DD-exact; CPU-only): neutral cell + a charged cell
    # (Q ≠ 0 exercises the Q Allreduce and the replicated charged-cell
    # correction). On cuda the same Q paths are exercised through PME below.
    if RUN_EWALD
        ewald0 = Calculators.EwaldInfo(a0.info[:lattice], prms_b)
        refs["ewald"] = run_ref(a -> Calculators.calculate!(ewald0, a, prms_b))
        refs["ewald_q"] = run_ref(a -> Calculators.calculate!(ewald0, a, prms_q_b))
    end
    # PME (replicated grid): the DD value must match the serial PME per rank;
    # together with the Ewald parities and the serial Ewald == PME equivalence
    # suite this pins DD-Ewald ↔ DD-PME transitively.
    pme0 = Calculators.PMEInfo(a0.info[:lattice], prms_pme_b)
    refs["pme"] = run_ref(a -> Calculators.calculate!(pme0, a, prms_pme_b))
    refs["pme_q"] = run_ref(a -> Calculators.calculate!(pme0, a, prms_pme_q_b))

    ke_ref = kinetic_energy(a0)
    copyto!(a0.info[:virial], FMD.FPPREC[1, 2, 3, 4, 5, 6])
    st_ref = stress(a0, true)
    dip_ref = Calculators.calcdipole!(a0, prms_b)
    mfr_ref = Calculators.total_mfrac_ref(a0, prms_b)

    # --- DD side -------------------------------------------------------------
    dom = Cores.MPIDomain(nfull; comm = comm)
    loc = Cores.distribute(full, dom)
    loc = Cores.migrate(loc)
    local nl, nl_ace, nl_2, nl_3
    if BACKEND == "cpu"
        cl, loc, _ = CellList(loc, rc_eff)
        nl = neighborlist(loc, cl, rc, sk)
    else
        # Phase 2 target order (distribute/migrate/CellList on the host frame,
        # then convert and rebuild the list on the device): npt! adopts this
        # in the final Phase 2 PR — today it still rejects GPU DD at entry.
        Cores.probe_device_mpi!(dom, CUDA.CUDABackend())
        if RUN_NL
            cl_h, loc, _ = CellList(loc, rc_eff)   # host DD build (+ ghost permute)
            loc = Cores.change_arrtyp(loc, CuArray)
            cl = Cores.change_arrtyp(cl_h, CuArray)
            nl = neighborlist(loc, cl, rc, sk)     # device kernels + device DD sync
        else
            loc = Cores.change_arrtyp(loc, CuArray)
        end
    end
    if RUN_NL
        nl_ace = neighborlist(loc, nl, acep.rmax, sk)
        nl_2 = neighborlist(loc, nl, splp.rmax2, sk)
        nl_3 = neighborlist(loc, nl, splp.rmax3, sk)
    end
    idsl = Array(loc.arrays[:id])

    function check_dd(f!, ref; ftol, check_shell = false)
        clear!(loc)
        fill!(loc.arrays[:forces], 0)
        fill!(loc.arrays[:shellforces], 0)
        f!(loc)
        E = Array(loc.info[:potential_energy])[1]
        V = collect(FMD.FPPREC, Array(loc.info[:virial]))
        escale = max(one(FMD.FPPREC), abs(ref.E))
        vscale = max(one(FMD.FPPREC), maximum(abs, ref.V))
        ok = abs(E - ref.E) < 1e-12 * escale &&
             maximum(abs, V .- ref.V) < 1e-12 * vscale
        if length(loc) > 0
            fscale = max(one(FMD.FPPREC), maximum(abs, ref.F))
            ok &= maximum(abs, Array(loc.arrays[:forces]) .- ref.F[idsl, :]) < ftol * fscale
            if check_shell
                sscale = max(one(FMD.FPPREC), maximum(abs, ref.S))
                ok &= maximum(abs, Array(loc.arrays[:shellforces]) .- ref.S[idsl, :]) <
                      ftol * sscale
            end
        end
        return MPI.Allreduce(ok, &, comm)
    end

    if RUN_EWALD
        ewald_dd = Calculators.EwaldInfo(loc.info[:lattice], prms_b)
        ok_ewald = check_dd(a -> Calculators.calculate!(ewald_dd, a, prms_b),
                            refs["ewald"]; ftol = 1e-12, check_shell = true)
        ok_ewald_q = check_dd(a -> Calculators.calculate!(ewald_dd, a, prms_q_b),
                              refs["ewald_q"]; ftol = 1e-12, check_shell = true)
    else
        ok_ewald = ok_ewald_q = true
    end
    pme_dd = Calculators.PMEInfo(loc.info[:lattice], prms_pme_b)
    ok_pme = check_dd(a -> Calculators.calculate!(pme_dd, a, prms_pme_b),
                      refs["pme"]; ftol = 1e-12, check_shell = true)
    ok_pme_q = check_dd(a -> Calculators.calculate!(pme_dd, a, prms_pme_q_b),
                        refs["pme_q"]; ftol = 1e-12, check_shell = true)
    if RUN_NL
        ok_cb = check_dd(a -> Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms_b),
                         refs["cb"]; ftol = 1e-12, check_shell = true)
        ok_ace = check_dd(a -> Calculators.calculate!(acep_b, a, nl_ace),
                          refs["ace"]; ftol = 1e-10)
        ok_spline = check_dd(a -> Calculators.calculate!(splp_b, a, nl_2, nl_3),
                             refs["spline"]; ftol = 1e-10)
    else
        ok_cb = ok_ace = ok_spline = true
    end
    ok_spring = RUN_SPRING ?
        check_dd(a -> Calculators.calculate!(Calculators.Spring(), a, prms_b),
                 refs["spring"]; ftol = 1e-12, check_shell = true) : true

    ke = kinetic_energy(loc)
    ok_ke = MPI.Allreduce(abs(ke - ke_ref) < 1e-12 * max(1.0, abs(ke_ref)), &, comm)

    copyto!(loc.info[:virial], FMD.FPPREC[1, 2, 3, 4, 5, 6])   # same value on every rank
    st = stress(loc, true)
    ok_st = MPI.Allreduce(maximum(abs, st .- st_ref) <
                          1e-12 * max(1.0, maximum(abs, st_ref)), &, comm)

    dip = Calculators.calcdipole!(loc, prms_b)
    ok_dip = MPI.Allreduce(maximum(abs, Array(dip) .- Array(dip_ref)) <
                           1e-12 * max(1.0, maximum(abs, Array(dip_ref))), &, comm)

    mfr = Calculators.total_mfrac_ref(loc, prms_b)
    ok_mfr = MPI.Allreduce(maximum(abs, mfr .- mfr_ref) <
                           1e-12 * max(1.0, maximum(abs, mfr_ref)), &, comm)

    vkeys = String[]
    RUN_NL && append!(vkeys, ["cb", "ace", "spline"])
    RUN_SPRING && push!(vkeys, "spring")
    push!(vkeys, RUN_EWALD ? "ewald" : "pme")
    vmin = minimum(norm(refs[k].V) for k in vkeys)
    ntot = MPI.Allreduce(length(loc), +, comm)
    nmax = MPI.Allreduce(length(loc), max, comm)
    nmin = MPI.Allreduce(length(loc), min, comm)

    getE(k) = haskey(refs, k) ? refs[k].E : NaN

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, backend = BACKEND,
                     e_cb = getE("cb"), e_ace = getE("ace"), e_spline = getE("spline"),
                     e_spring = getE("spring"), e_ewald = getE("ewald"),
                     e_pme = refs["pme"].E,
                     dE_q = RUN_EWALD ? refs["ewald_q"].E - refs["ewald"].E :
                                        refs["pme_q"].E - refs["pme"].E,
                     vmin = vmin, ntot = ntot, nmax = nmax, nmin = nmin,
                     natoms = nfull,
                     ok_ewald = ok_ewald, ok_ewald_q = ok_ewald_q,
                     ok_pme = ok_pme, ok_pme_q = ok_pme_q,
                     ok_cb = ok_cb, ok_spring = ok_spring, ok_ace = ok_ace,
                     ok_spline = ok_spline, ok_ke = ok_ke, ok_st = ok_st,
                     ok_dip = ok_dip, ok_mfr = ok_mfr)
    end
    return nothing
end
