# test/mpi/_verify_checks.jl
#
# Parent-side verification for the bundled MPI tier (Issue #113): one
# verify_* function per feature, holding the assertion bodies that lived
# in the per-feature parent files (test_domain_mpi.jl etc.) before the
# bundling. Each takes the check's JLD2 dict `d` — or `nothing` when the
# artifact is missing or the check errored — plus whatever parameters the
# assertions need. P-dependent expressions always take P explicitly.

# Locate and load one check's artifact from a bundle result (a NamedTuple
# with fields ok/log/outdir). On failure print the diagnostics and return
# nothing — the verify functions then fail their leading @test.
function load_check(bundle, name)
    s = bundle_slug(name)
    errfiles = filter(f -> startswith(f, s * ".error."), readdir(bundle.outdir))
    if !isempty(errfiles)
        for f in errfiles
            @info "bundle check $name errored ($f):\n" *
                  read(joinpath(bundle.outdir, f), String)
        end
        return nothing
    end
    path = joinpath(bundle.outdir, s * ".jld2")
    if !isfile(path)
        @info "bundle check $name left no artifact (crashed or never ran; " *
              "see the bundle output above)"
        return nothing
    end
    return JLD2.load(path)
end

# Shared leading assertion: the artifact must exist. Returns whether the
# caller can proceed.
function _artifact_ok(d)
    @test d !== nothing
    return d !== nothing
end

# --- allreduce -------------------------------------------------------------

function verify_allreduce(d)
    _artifact_ok(d) || return
    @test d["mpisize"] == 2
    @test d["ok"] === true
    @test all(d["v"] .== d["expected"])
end

function verify_allreduce_gpu(d, log, P)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["ok"] === true
    # The probe announced itself and passed its exact value check on the
    # root rank (its output travels through the bundle log — a process-
    # level property, so matching the whole log is correct).
    @test occursin("Probing CUDA-aware MPI", log)
    @test occursin("CUDA-aware MPI probe passed", log)
    # Which transport the MPI build *reports* is informational only —
    # MPICH-ABI CUDA-aware builds report false; the value checks above
    # are the actual guarantee.
    @info "GPU+MPI transport (P=$P)" mpi_has_cuda = d["has_cuda"]
end

# --- segmented collectives (Issue #191) ------------------------------------

function verify_collectives(d, P)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["ok"] === true
    # Non-vacuity: the count layout really contained data and a zero-count
    # rank (P ≥ 2 always has the trailing empty rank by construction).
    @test d["ntot"] > 0
    @test d["nzero"] == 1
end

# --- z-slab distributed PME (Issue #195) -----------------------------------

function verify_pmeslab(d, P, mode)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    # The slab solver was actually armed (and the opt-out disarms it) —
    # without this the parities below would silently test the replicated
    # path.
    @test d["ok_armed"] === true
    @test d["ok_optout"] === true
    @test d["ok_slice"] === true          # yoff ≠ 0 ky-slab factor slices
    @test d["ok_pme"] === true            # neutral: E/V/F/S vs serial reference
    @test d["ok_pme_q"] === true          # charged cell
    @test d["ok_bit"] === true            # same-run OnlyShellForce == full
    # Non-vacuity.
    @test abs(d["e_pme"]) > 1e-6
    @test abs(d["dE_q"]) > 1e-6
    @test d["vnorm"] > 1e-12
    @test d["ntot"] == (mode == "tiny" ? 8 : 64)
    if mode == "tiny"
        # The triple degeneracy really happened: empty-atom ranks AND
        # zero-plane ranks (fftsize floored at BSPORDER, P = 6).
        @test d["nmin"] == 0
        @test d["fftsize"] == [4, 4, 4]
        @test d["nz_zero"] == 2
    else
        # The frame is genuinely distributed (no rank holds everything —
        # else the DD parity degenerates to serial-vs-serial).
        @test 0 < d["nmax"] < d["ntot"]
        @test d["nz_zero"] == 0
    end
end

# --- domain ----------------------------------------------------------------

function verify_domain(d, P)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test prod(d["dims"]) == P
    @test d["ok_dims"] && d["ok_bounds"] && d["ok_formula"] && d["ok_nbr"]
    # The bricks tile [0,1)³: every probe point (incl. FP edges and every
    # brick boundary) is owned by exactly one rank.
    @test d["ok_partition"]
    # Reduction helpers.
    @test d["s"] == P * (P - 1) ÷ 2
    @test d["sv"] == FMD.FPPREC[P + P * (P - 1) ÷ 2, P * (P - 1)]
    @test d["mx"] == P - 1
    @test d["land_all"] === true
    @test d["land_one"] === false     # rank 0 votes false at every P
    @test d["roots"] == 1
    @test d["ng"] == 1000
end

# --- migrate ---------------------------------------------------------------

function verify_migrate(d, P)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    # distribute: bricks partition the ids; every row bitwise equal to
    # the reference frame (unwrapped :pos included — gauge intact);
    # a sliced frame is rejected loudly.
    @test d["ok_partition0"] && d["ok_bits0"]
    @test d["ok_argcheck"]
    # migrate without movement is a bitwise no-op.
    @test d["ok_noop"]
    # migrate after a drift: ids still a partition, every row owned,
    # transferred rows bitwise equal to reference + drift.
    @test d["ok_partition1"] && d["ok_owned"] && d["ok_bits1"]
    # Non-vacuous: atoms actually changed ranks when x is decomposed.
    if d["dimsx"] > 1
        @test d["moved_total"] > 0
    end
end

# --- ghosts (host) ---------------------------------------------------------

function _verify_ghosts_common(d)
    @test d["nghost"] > 0                    # non-vacuous
    # (id, shift) multiset equality against the brute-force oracle.
    @test d["ok_oracle"]
    # Reverse (fold) chains deliver exactly the analytic ghost sums.
    @test d["ok_fold"]
    # Shell displacements ride the sync bitwise (no seam shift).
    @test d["ok_sdisp"]
    # extend_ghosts! replicates owned values onto every ghost row.
    @test d["ok_extend"]
    # The naive builder refuses to run under DD.
    @test d["ok_naive"]
end

function verify_ghosts(d, P, mode)
    _artifact_ok(d) || return
    if mode == "normal"
        @test d["mpisize"] == P
        _verify_ghosts_common(d)
        # P = 1: the all-self-stage DD build reproduces the serial pair
        # content exactly.
        P == 1 && @test d["ok_p1"]
        # Forces through the DD list + MPI sync match the serial reference
        # per id (gather kernel: reassociation only, tight tolerance).
        @test d["fmax"] > 1e-3               # non-vacuous (rattled)
        @test d["ok_forces"]
    elseif mode == "bigrc"
        _verify_ghosts_common(d)
        @test d["fmax"] > 1e-3
        @test d["ok_forces"]
    elseif mode == "tiny"
        @test d["min_nlocal"] == 0           # an empty rank really occurred
        _verify_ghosts_common(d)
    else
        error("verify_ghosts: unknown mode $mode")
    end
end

function verify_ghosts_deadlock(d)
    _artifact_ok(d) || return
    # Every rank must see the same verdict (false) although only rank 0
    # moved — a missing Allreduce splits the votes.
    @test d["votes"] == Int32[0, 0]
end

# --- ghosts (device exchange) ----------------------------------------------

function verify_ghosts_gpu(d, P, mode)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    # Non-vacuous: ghosts actually exist (every mode), and the tiny mode
    # really produced an empty rank.
    @test d["nghost_tot"] > 0
    mode == "tiny" && @test d["min_nlocal"] == 0
    @test d["ok"]
    # Individual flags for failure localization.
    @test d["ok_upos"] && d["ok_sdisp"] && d["ok_xcart"]
    @test d["ok_fold"] && d["ok_fext"] && d["ok_fold2"]
    @test d["ok_extend"]
    @test d["ok_ncol"]
    @test d["ok_resync"]
end

# --- reduce ----------------------------------------------------------------

function verify_reduce(d, P, mode, backend)
    _artifact_ok(d) || return
    @test d["backend"] == String(backend)
    if mode == "normal" && backend === :cpu
        @test d["mpisize"] == P
        # Non-vacuous: every calculator produced a real energy AND virial,
        # and the frame was actually split at P > 1 (no rank holds it all).
        @test abs(d["e_cb"]) > 1e-6
        @test abs(d["e_spring"]) > 1e-10
        @test abs(d["e_ace"]) > 1e-10
        @test abs(d["e_spline"]) > 1e-10
        @test d["vmin"] > 1e-12
        @test d["ntot"] == d["natoms"] == 64
        P > 1 && @test d["nmax"] < 64
        # info holds the whole-system E/virial on every rank, equal to the
        # serial reference; forces/shellforces match per id.
        @test d["ok_cb"]
        # Ewald two-pass (S(k) Allreduce): neutral and charged cells; the
        # charged-cell correction must actually contribute (Q Allreduce).
        @test abs(d["e_ewald"]) > 1e-6
        @test abs(d["dE_q"]) > 1e-6
        @test d["ok_ewald"]
        @test d["ok_ewald_q"]
        # PME replicated grid: DD == serial PME per rank; the two serial
        # long-range methods agree (rough sanity — the tight serial
        # equivalence lives in test_ewald_pme_equivalence.jl), so DD-Ewald
        # ↔ DD-PME holds transitively.
        @test d["ok_pme"]
        @test d["ok_pme_q"]                  # Q Allreduce path (PME side)
        @test abs(d["e_pme"] - d["e_ewald"]) < 1e-4 * abs(d["e_ewald"])
        @test d["ok_spring"]
        @test d["ok_ace"]
        @test d["ok_spline"]
        # Global scalar/vector helpers.
        @test d["ok_ke"]
        @test d["ok_st"]
        @test d["ok_dip"]
        @test d["ok_mfr"]
    elseif mode == "tiny" && backend === :cpu
        # NaCl-8 with empty ranks: only the empty-rank-safe checks ran.
        @test d["nmin"] == 0                 # an empty rank really occurred
        @test d["ntot"] == d["natoms"] == 8
        @test abs(d["e_ewald"]) > 1e-8
        @test abs(d["dE_q"]) > 1e-8
        @test d["ok_ewald"]
        @test d["ok_ewald_q"]
        @test d["ok_pme"]                    # empty ranks ride the grid Allreduce
        @test d["ok_ke"] && d["ok_st"] && d["ok_dip"] && d["ok_mfr"]
    elseif mode == "normal" && backend === :cuda
        @test d["mpisize"] == P
        # Non-vacuous, and the frame was actually split at P > 1. (No Ewald
        # here: the Ewald reciprocal kernel is CPU-only by design — GPU
        # long-range goes through PME, whose serial GPU ↔ CPU-Ewald
        # equivalence lives in test_ewald_pme_equivalence.jl; dE_q is the
        # PME charged-cell energy shift, pinning the Q Allreduce.)
        @test abs(d["e_pme"]) > 1e-6
        @test abs(d["e_cb"]) > 1e-6
        @test abs(d["e_spring"]) > 1e-10
        @test abs(d["e_ace"]) > 1e-10
        @test abs(d["e_spline"]) > 1e-10
        @test abs(d["dE_q"]) > 1e-6
        @test d["vmin"] > 1e-12
        @test d["ntot"] == d["natoms"] == 64
        P > 1 && @test d["nmax"] < 64
        # PME replicated grid: the charge-grid Allreduce runs on the device
        # buffer through the CUDA-aware transport.
        @test d["ok_pme"]
        @test d["ok_pme_q"]
        # On-site spring (backend-generic, add_reduced_ev! path on device).
        @test d["ok_spring"]
        # Neighbor-list stack through the device DD ghost exchange.
        @test d["ok_cb"]
        @test d["ok_ace"]
        @test d["ok_spline"]
        # Global scalar/vector helpers on device frames.
        @test d["ok_ke"]
        @test d["ok_st"]
        @test d["ok_dip"]
        @test d["ok_mfr"]
    elseif mode == "tiny" && backend === :cuda
        # Empty ranks with device frames ride the device grid Allreduce
        # and the global scalars.
        @test d["nmin"] == 0                 # an empty rank really occurred
        @test d["ntot"] == d["natoms"] == 8
        @test abs(d["e_pme"]) > 1e-8
        @test abs(d["dE_q"]) > 1e-8
        @test d["ok_pme"]
        @test d["ok_pme_q"]
        @test d["ok_ke"] && d["ok_st"] && d["ok_dip"] && d["ok_mfr"]
    else
        error("verify_reduce: unknown (mode, backend) = ($mode, $backend)")
    end
end

# --- coreshell -------------------------------------------------------------

function verify_hvp_dd(d, P, backend, mode)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["backend"] == String(backend)
    @test d["mode"] == mode
    # Non-vacuous: a zero product would make the parity comparison empty.
    @test d["jmax"] > 1e-6
    @test d["kmax"] > 1e-6
    # DD == serial for both Hessian blocks (the dipole-grid reduction is
    # what this pins; without it each rank convolves a partial grid).
    @test d["ok"]
    if occursin("tiny", mode)
        # The point of the tiny modes: ranks with no owned atoms exist and
        # the run still completed (they ride the collectives — including
        # the surface term's dipole-direction Allreduce).
        @test d["nempty"] > 0
    else
        @test d["nempty"] == 0
    end
    if backend === :cpu
        # Ewald has no DD path and must refuse rather than answer with a
        # rank-local structure factor.
        @test d["ok_ewald_refused"]
    else
        @test isnan(d["ok_ewald_refused"])
    end
end

function verify_coreshell(d, P, backend)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["backend"] == String(backend)
    if backend === :cpu
        for tag in ("pme", "ew")
            # Non-vacuous AND non-saturated: the CG actually iterated and
            # actually converged (a run pinned at cgmaxiter would make the
            # parity a comparison of two unconverged states).
            @test 0 < d["ncg_ref_$tag"] < 30
            @test 0 < d["ncg_$tag"] < 30
            # Lockstep: identical iteration count on every rank (the
            # structural deadlock-freedom property).
            @test d["ok_lockstep_$tag"]
            # DD vs serial: the iteration count may flip by one when a
            # convergence test lands within reassociation noise of cgtol.
            @test abs(d["ncg_$tag"] - d["ncg_ref_$tag"]) <= 1
            # Converged state agrees at the cgtol scale; E/virial tighter
            # (stationary point).
            @test d["ok_$tag"]
        end
    else
        # cuda: PME only (Ewald reciprocal is host-only; its keys hold NaN
        # sentinels in the artifact and are deliberately never asserted).
        @test 1 < d["ncg_ref_pme"] < 30
        @test 0 < d["ncg_pme"] < 30
        @test d["ok_lockstep_pme"]
        # Same iterate count up to the convergence-boundary wiggle: when
        # the residual lands within reassociation noise of cgtol (or the
        # uphill-correction branch flips), the count moves by one.
        @test abs(d["ncg_pme"] - d["ncg_ref_pme"]) <= 1
        @test d["ok_pme"]
    end
end

# --- umbrella --------------------------------------------------------------

function verify_umbrella(d, P)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    # Non-vacuous: the bias engaged and produced real forces. (The CV
    # itself is legitimately ≈ 0 here — the gauge reference cancels the
    # rattle polarization; the bias energy comes from the target offset
    # and its gradient drives the forces.)
    @test abs(d["e_ref"]) > 1e-10
    @test d["fmax_ref"] > 1e-10
    # E / virial / forces / shellforces / results all match the serial
    # reference on every rank.
    @test d["ok"]
end

# --- surface term ----------------------------------------------------------

function verify_surface_term(d, P; tiny::Bool = false)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    # Non-vacuous: the rattled fixture produced a real δM, so the term
    # carries energy and forces (U_surf ≥ 0 by the SPD construction of T).
    @test d["e_ref"] > 1e-10
    @test d["bias_ref"] == d["e_ref"]
    @test d["fmax_ref"] > (tiny ? 1e-12 : 1e-10)
    # E / virial / forces / shellforces match the serial reference on
    # every rank (the tiny flavour additionally proves empty ranks ride
    # the dipole Allreduce without hanging).
    @test d["ok"]
end

# --- npt_dd ----------------------------------------------------------------

# Serial reference computed in THIS process (no MPI.Init → SerialDomain);
# on backend = :cuda this is the GPU serial run. One call per fixture
# flavour, shared across every P.
function npt_dd_serial_ref(backend; csvr::Bool = false, scr::Bool = false,
                          xl::Bool = false, cheb::Bool = false,
                          xlbaro::Bool = false, nlh::Bool = false,
                          surface::Bool = false)
    full = _npt_dd_fixture(false)
    dir = mktempdir()
    ref = npt!(deepcopy_atoms(full),
               _npt_dd_params(full, dir; tiny = false, csvr = csvr,
                              scr = scr, xl = xl, cheb = cheb,
                              xlbaro = xlbaro, nlh = nlh, surface = surface,
                              backend = backend))
    ids_ref = Array(ref.arrays[:id])
    byid(x) = (o = zeros(FMD.FPPREC, length(ids_ref), size(x, 2));
               o[ids_ref, :] .= Array(x); o)
    return (; pos = byid(ref.arrays[:pos]), velo = byid(ref.arrays[:velo]),
            lattice = Matrix(Array(ref.info[:lattice])), dir = dir)
end

function relax_dd_serial_ref(backend; tiny::Bool = false, cell_dof::Symbol = :none)
    full = _relax_dd_fixture(tiny)
    dir = mktempdir()
    ref, converged = FMD.Apps.relax!(deepcopy_atoms(full),
                                     _relax_dd_params(full, dir; tiny = tiny,
                                                      cell_dof = cell_dof,
                                                      backend = backend))
    ids_ref = Array(ref.arrays[:id])
    n = length(ids_ref)
    byid(x) = (o = zeros(FMD.FPPREC, n, size(x, 2)); o[ids_ref, :] .= Array(x); o)
    return (; pos = byid(ref.arrays[:pos]),
            shelldisp = byid(ref.arrays[:shelldisp]),
            fire_velo = byid(ref.arrays[:fire_velo]),
            lattice = Matrix(Array(ref.info[:lattice])),
            converged = converged, n = n, dir = dir)
end

# relax! under DD must reproduce the serial trajectory. Every FIRE branch is
# taken from an Allreduce-derived scalar (the power's sign, the convergence
# test, the step-cap factor), so a rank-local value would not merely perturb
# the result — it would put ranks on different branches and the structures
# would separate by O(1), far above any tolerance here. The bounds are the
# accumulated reduction-reassociation scale, as for npt_dd.
function verify_relax_normal(d, dd_dir, ref, P, backend; cell::Bool = false)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["backend"] == String(backend)
    @test d["ids"] == collect(1:ref.n)
    # The verdict itself is a collective decision: ranks that disagreed
    # would have exited the loop at different steps.
    @test d["converged"] == ref.converged
    dpos = maximum(abs, d["pos"] .- ref.pos)
    dsd = maximum(abs, d["shelldisp"] .- ref.shelldisp)
    dvel = maximum(abs, d["fire_velo"] .- ref.fire_velo)
    dlat = maximum(abs, d["lattice"] .- ref.lattice)
    @info "relax_dd deviations vs serial (backend=$backend, P=$P)" dpos dsd dvel dlat
    tol = backend === :cpu ? (P == 1 ? 1e-9 : 1e-5) : (P == 1 ? 1e-9 : 1e-7)
    # A fixed cell must not move AT ALL; a relaxing one must track the serial
    # run to the same reassociation scale as the positions.
    lattol = cell ? tol : 1e-12
    @test dpos < tol
    @test dsd < tol
    @test dvel < tol
    @test dlat < lattol
    # The root-written log must match the serial one. Column 10 is the
    # volume: a fixed-cell relaxation may never touch it.
    logref = [parse.(FMD.FPPREC, split(l)) for l in eachline(joinpath(ref.dir, "log.txt"))]
    logdd = [parse.(FMD.FPPREC, split(l)) for l in eachline(joinpath(dd_dir, "log.txt"))]
    @test length(logdd) == length(logref)
    # 22 columns: step | max|F| | energy | sigma(6) | volume | dt | alpha |
    # ncg | lattice(9).
    @test all(length(r) == 22 for r in logdd)
    reldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-8))
    # Columns 1..12 (through alpha) and 14..22 (the lattice); column 13 is
    # ncg, whose count can differ by an iteration right at the cgtol
    # threshold, as in verify_npt_normal.
    cols = vcat(1:12, 14:22)
    @test reldiff(logdd[1][cols], logref[1][cols]) < 1e-6
    @test maximum(reldiff(logdd[i][cols], logref[i][cols])
                  for i in eachindex(logdd)) < 1e-3
    # Column 10 is the volume. With a fixed cell it must be bitwise constant;
    # with a relaxing one it must not be (or the cell dof did nothing and the
    # check would be vacuous).
    if cell
        @test any(r[10] != logdd[1][10] for r in logdd)
    else
        @test all(r[10] == logdd[1][10] for r in logdd)
    end
end

function npt_dd_serial_ref_umbrella(backend)
    full_u = _npt_dd_fixture(false; umbrella = true)
    dir = mktempdir()
    npt!(deepcopy_atoms(full_u),
         _npt_dd_params(full_u, dir; tiny = false, umbrella = true,
                        backend = backend))
    return dir
end

function verify_npt_normal(d, dd_dir, ref, P, backend)
    _artifact_ok(d) || return
    @test d["mpisize"] == P
    @test d["backend"] == String(backend)   # the child really took this path
    # Conservation: the gathered frame holds every id exactly once.
    @test d["ids"] == collect(1:64)
    # 6 steps with a rebuild in between: trajectories agree to the
    # accumulated-reassociation scale (a lost atom or a broken migration
    # would be O(1)). CPU P = 1 runs the DD machinery on the whole box —
    # its ghost/list build order differs from serial, so it is not bitwise,
    # but it must be far tighter than P > 1. On cuda the reference is the
    # GPU serial run (GPU atomics make even two serial GPU runs
    # non-bitwise; measured noise floor ~1e-16, see PR #108) — the looser
    # P > 1 thresholds still sit far below any single-ghost-desync /
    # stale-shell signature (~1e-7 would slip through a 1e-5 bound).
    dpos = maximum(abs, d["pos"] .- ref.pos)
    dvel = maximum(abs, d["velo"] .- ref.velo)
    dlat = maximum(abs, d["lattice"] .- ref.lattice)
    @info "npt_dd deviations vs serial (backend=$backend, P=$P)" dpos dvel dlat
    tol = backend === :cpu ? (P == 1 ? 1e-9 : 1e-5) : (P == 1 ? 1e-9 : 1e-7)
    lattol = backend === :cpu ? (P == 1 ? 1e-10 : 1e-6) : (P == 1 ? 1e-10 : 1e-8)
    @test dpos < tol
    @test dvel < tol
    @test dlat < lattol
    # The collective log quantities (temp | lattice | dipole | dipole_phys
    # | volume | conserved) written by the root must match the serial log —
    # step 1 tightly (minimal divergence), all steps loosely. Column 19,
    # shell_rms, is left out of that comparison: it is the residual at the
    # CG exit, so the one-iteration flip described next changes it by a
    # whole CG contraction factor; it is only required finite on every CG
    # step and NaN exactly where the serial run logged one (a shadow
    # XL-BOMD step has no CG, so no residual — see evaluate_forces!). The ncg
    # column (20, Issue #144) is compared separately as a count: the CG
    # branching is Allreduce-derived (identical across ranks), but serial
    # vs DD summation-order roundoff can flip one iteration right at the
    # cgtol threshold — and the uphill-correction branch counts two
    # evaluations per iteration, so a single flipped iteration can move
    # the count by 2. The bound still catches systematic divergence
    # (e.g. DD persistently maxing out cgmaxiter).
    logref = [parse.(FMD.FPPREC, split(l)) for l in eachline(joinpath(ref.dir, "log.txt"))]
    logdd = [parse.(FMD.FPPREC, split(l)) for l in eachline(joinpath(dd_dir, "log.txt"))]
    @test length(logdd) == length(logref) == 6
    @test all(length(logdd[i]) == length(logref[i]) == 20 for i in 1:6)
    reldiff(a, b) = maximum(abs.(a .- b) ./ max.(abs.(b), 1e-8))
    @test reldiff(logdd[1][1:18], logref[1][1:18]) < 1e-6
    @test maximum(reldiff(logdd[i][1:18], logref[i][1:18]) for i in 1:6) < 1e-3
    @test all(isfinite(logdd[i][19]) == isfinite(logref[i][19]) for i in 1:6)
    # NaN only on a step that ran no CG (ncg = 0), never as a stale or
    # broken value on a CG step.
    @test all(isfinite(logdd[i][19]) || (isnan(logdd[i][19]) && logdd[i][20] == 0) for i in 1:6)
    @test isfinite(logdd[1][19])                 # step 1 is a CG step on every flavour
    @test maximum(abs(logdd[i][20] - logref[i][20]) for i in 1:6) <= 2
    # The rebuild (→ migrate; on cuda a host round trip) path actually
    # fired: the initial build plus at least one mid-run rebuild, counted
    # by the child's logger (pre-#113: a stdout grep on the per-check
    # process log).
    @test d["nl_rebuilds"] >= 2
    # Root-only output: snapshots readable and complete.
    for s in (3, 6)
        snap = joinpath(dd_dir, "step_$(s).xyz")
        @test isfile(snap)
        if isfile(snap)
            back = Cores.readxyz(open(snap, "r"))[1]
            @test length(back) == 64
            @test sort(Array(back.arrays[:id])) == collect(1:64)
        end
    end
end

function verify_npt_umbrella(d, dd_dir, ref_dir_u, backend)
    _artifact_ok(d) || return
    @test d["backend"] == String(backend)
    @test d["ids"] == collect(1:64)
    # Bias.log must match the serial run (root-only, per-step CV/bias).
    bias_ref = [split(l, ",") for l in eachline(joinpath(ref_dir_u, "Bias.log"))]
    bias_dd = [split(l, ",") for l in eachline(joinpath(dd_dir, "Bias.log"))]
    @test length(bias_dd) == length(bias_ref) == 7   # header + 6 steps
    v(x) = parse.(FMD.FPPREC, x)
    @test maximum(abs, v(bias_dd[2]) .- v(bias_ref[2])) < 1e-8
    for i in 3:7
        @test maximum(abs, v(bias_dd[i]) .- v(bias_ref[i])) < 1e-4
    end
    # Local polarization maps (Issue #220): the group table is Allreduced,
    # so the root's file must equal the serial file to the trajectory-
    # divergence scale of the log comparison (1e-3 relative at P = 2).
    for s in (3, 6)
        f_dd = joinpath(dd_dir, "polmap_$(s).txt")
        f_ref = joinpath(ref_dir_u, "polmap_$(s).txt")
        @test isfile(f_dd)
        @test isfile(f_ref)
        (isfile(f_dd) && isfile(f_ref)) || continue
        a = Calculators.read_polmap(f_dd)
        b = Calculators.read_polmap(f_ref)
        @test a.step == s
        @test a.gridsize == (2, 2, 2)
        @test maximum(abs, b.P) > 1e-6        # non-vacuous (300 K + rattle)
        @test isapprox(a.P, b.P; rtol = 1e-3, atol = 1e-8)
    end
    @test !isfile(joinpath(dd_dir, "polmap_1.txt"))
end

function verify_npt_tiny(d, dd_dir, backend)
    _artifact_ok(d) || return
    @test d["backend"] == String(backend)
    @test d["nmin"] == 0                     # an empty rank really occurred
    @test d["ids"] == collect(1:8)
    @test countlines(joinpath(dd_dir, "log.txt")) == 6
end

# --- fit -------------------------------------------------------------------

function verify_fitspline(d1, d2)
    ok1 = _artifact_ok(d1)
    ok2 = _artifact_ok(d2)
    (ok1 && ok2) || return
    @test d1["mpisize"] == 1
    @test d2["mpisize"] == 2
    # Tier 1: partial-sum parity (machine precision)
    @test isapprox(d1["y"],  d2["y"];  rtol = 1e-12)
    @test isapprox(d1["Hy"], d2["Hy"]; rtol = 1e-12)
    # Tier 2: converged-fit parity. The residual guard makes "CG
    # converged" a check, not an assumption — without it a regression
    # into the maxiter-limited regime would go unnoticed.
    @test d1["resid"] <= 1e-11
    @test d2["resid"] <= 1e-11
    @test isapprox(d1["c1"], d2["c1"]; rtol = 1e-6, atol = 1e-10)
    @test isapprox(d1["c2"], d2["c2"]; rtol = 1e-6, atol = 1e-10)
    @test isapprox(d1["c3"], d2["c3"]; rtol = 1e-6, atol = 1e-10)
end

# Machine-precision parity of the blockchol-si Gram assembly itself
# (Issue #184: panel accumulation + hierarchical node-window reduce). The
# caller's `rtol` reflects what differs between the two artifacts: same P
# with different node splits reorders only the two-level Reduce (1e-13);
# different P also reorders the frame sharding (1e-12). The strict lower
# triangles are exactly zero on both sides, so full-matrix comparison is
# equivalent to comparing the valid upper triangles.
function verify_ace_gram(d1, d2; P1, P2, rtol)
    ok1 = _artifact_ok(d1)
    ok2 = _artifact_ok(d2)
    (ok1 && ok2) || return
    @test d1["mpisize"] == P1
    @test d2["mpisize"] == P2
    g1 = d1["grams"]
    g2 = d2["grams"]
    @test length(g1) == length(g2)
    for i in eachindex(g1)
        @test isapprox(g1[i], g2[i]; rtol = rtol)
        # the assembly skips zero-filling the shared window because every
        # element is written by exactly one panel Reduce; the strict lower
        # triangle (never touched by gemm/syrk) must then be exactly zero
        @test all(iszero, tril(g1[i], -1))
        @test all(iszero, tril(g2[i], -1))
    end
end

# `resid_tol`: the plain-CG fixture converges to ≲1e-12; the blockchol-si
# variant's TRUE residual bottoms out at the preconditioner-solve rounding
# floor ~ε·κ(block) (κ ≈ λmax/Λ_RIGID ≈ 1e6 on this fixture, so ~1e-10 —
# observed 6e-11 at P=2). The guard only has to sit far below the tier-2
# comparison tolerance (rtol 1e-6), so the precond caller passes 1e-9.
function verify_fitace(d1, d2; resid_tol = 1e-11)
    ok1 = _artifact_ok(d1)
    ok2 = _artifact_ok(d2)
    (ok1 && ok2) || return
    @test d1["mpisize"] == 1
    @test d2["mpisize"] == 2
    # Tier 1: partial-sum parity (machine precision)
    @test isapprox(d1["y"],  d2["y"];  rtol = 1e-12)
    @test isapprox(d1["Hy"], d2["Hy"]; rtol = 1e-12)
    # Tier 2: converged-fit parity (+ convergence guard, see
    # verify_fitspline)
    @test d1["resid"] <= resid_tol
    @test d2["resid"] <= resid_tol
    @test isapprox(d1["vec_acep"], d2["vec_acep"]; rtol = 1e-6, atol = 1e-10)
end
