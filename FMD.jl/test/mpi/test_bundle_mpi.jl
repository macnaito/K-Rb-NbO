# test/mpi/test_bundle_mpi.jl
#
# CPU-tier parent driver for the bundled MPI tests (Issue #113). Stage 1
# launches ONE child process per P with every CPU check of that P bundled
# (_run_bundle.jl); stage 2 asserts per feature from the artifacts, with
# the same testset names and granularity as the pre-#113 per-feature
# parents. The two deliberate-abort checks (migrate keyset / farjump) run
# through _run_single.jl — they must kill their process.
using Test
using JLD2
using FMD.Apps: npt!

include(joinpath(@__DIR__, "driver.jl"))
include(joinpath(@__DIR__, "_npt_dd_common.jl"))
include(joinpath(@__DIR__, "_relax_dd_common.jl"))
include(joinpath(@__DIR__, "_verify_checks.jl"))

# Light checks first, npt last: the artifacts are written incrementally,
# so an early crash/hang still leaves the cheap checks verifiable.
const CPU_BUNDLES = [
    (1, ["domain", "migrate/normal", "ghosts/normal", "reduce/normal",
         "coreshell", "umbrella", "surface_term", "fitspline",
         "fitspline/nlh", "fitace",
         "fitace/precond", "acegram/flat", "npt_dd/normal",
         "relax_dd/normal", "relax_dd/cell"]),
    (2, ["allreduce", "collectives", "domain", "migrate/normal", "ghosts/normal",
         "ghosts/bigrc", "ghosts/deadlock", "reduce/normal", "pmeslab/normal",
         "coreshell",
         "hvp_dd/normal", "umbrella", "umbrella/map", "surface_term",
         "fitspline", "fitspline/nlh", "fitace", "fitace/precond",
         "npt_dd/normal", "npt_dd/nlh",
         "npt_dd/umbrella", "npt_dd/csvr", "npt_dd/scr",
         "hvp_dd/surface",
         "npt_dd/xlbomd", "npt_dd/xlbomd_cheb", "npt_dd/xlbomd_baro",
         "npt_dd/xlbomd_surface",
         "relax_dd/normal", "relax_dd/cell", "relax_dd/cell_iso"]),
    (4, ["domain", "migrate/normal", "ghosts/normal", "ghosts/bigrc",
         "reduce/normal", "pmeslab/normal", "coreshell", "acegram/flat",
         "acegram/split", "npt_dd/normal"]),
    (6, ["collectives", "migrate/normal", "ghosts/normal", "ghosts/tiny", "reduce/tiny",
         "pmeslab/tiny", "hvp_dd/tiny", "hvp_dd/surface_tiny",
         "surface_term/tiny", "npt_dd/tiny",
         "npt_dd/nlh_tiny", "relax_dd/tiny"]),
]

# ---- stage 1: run the bundles (results kept for stage 2; the outdirs are
# mktempdir-owned and vanish with this process) ------------------------------
cpu_bundles = Dict{Int,Any}()
for (P, names) in CPU_BUNDLES
    @testset "bundle execution (cpu, P = $P)" begin
        outdir = mktempdir()
        # Measured bundle times are 1-3 min; the budget stays well below
        # the HPC job elapse (45 min) so a hang still yields the [TIMEOUT]
        # diagnostics (last "begin" line) instead of losing them to pjsub.
        ok, log = run_mpi_script(joinpath(@__DIR__, "_run_bundle.jl"), P;
                                 args = vcat([outdir, "cpu"], names),
                                 timeout_sec = 1200)
        @test ok
        ok || @info "bundle subprocess output (cpu, P=$P):\n$log"
        cpu_bundles[P] = (; ok, log, outdir)
    end
end

# ---- stage 2: per-feature verification ------------------------------------

@testset "MPI Allreduce identity" begin
    verify_allreduce(load_check(cpu_bundles[2], "allreduce"))
end

@testset "z-slab distributed PME (Issue #195)" begin
    # P = 2 / 4: uneven z/ky partitions + full transpose routing; P = 6
    # tiny: zero-plane / zero-ky / empty-atom triple degeneracy.
    for P in (2, 4)
        verify_pmeslab(load_check(cpu_bundles[P], "pmeslab/normal"), P, "normal")
    end
    verify_pmeslab(load_check(cpu_bundles[6], "pmeslab/tiny"), 6, "tiny")
end

@testset "segmented collectives (P ∈ {2, 6})" begin
    # P = 2: uneven counts + trailing zero-count rank; P = 6: the same
    # layout at a size where alltoallv routes are genuinely all-to-all.
    for P in (2, 6)
        verify_collectives(load_check(cpu_bundles[P], "collectives"), P)
    end
end

@testset "Domain layer under MPI (P ∈ {1, 2, 4})" begin
    for P in (1, 2, 4)
        verify_domain(load_check(cpu_bundles[P], "domain"), P)
    end
end

@testset "migrate / distribute under MPI" begin
    # P = 6 gives dims = (3,2,1): the x drift then exercises the periodic
    # seam (last brick → first), where a naive w-vs-bounds direction test
    # inverts — only the minimal-image routing survives it.
    @testset "conservation + gauge (P = $P)" for P in (1, 2, 4, 6)
        verify_migrate(load_check(cpu_bundles[P], "migrate/normal"), P)
    end

    @testset "keyset mismatch fails loudly (P = 2)" begin
        ok, log = run_mpi_script(joinpath(@__DIR__, "_run_single.jl"), 2;
                                 args = ["migrate/keyset", tempname() * ".jld2",
                                         mktempdir(), "cpu"],
                                 timeout_sec = 180)
        @test !ok
        @test occursin("key sets differ across ranks", log)
    end

    @testset "|Δ| ≥ 2 brick jump fails loudly (P = 5, dims_x = 5)" begin
        # dims_x = 5 so a +2-brick jump stays Δ = +2 in minimal image
        # (at dims_x = 3 it would be a legitimate wrap-around -1 move).
        ok, log = run_mpi_script(joinpath(@__DIR__, "_run_single.jl"), 5;
                                 args = ["migrate/farjump", tempname() * ".jld2",
                                         mktempdir(), "cpu"],
                                 timeout_sec = 180)
        @test !ok
        @test occursin("more than one brick", log)
    end
end

@testset "DD ghosts under MPI" begin
    # P = 6 → dims (3,2,1): brick_x < rc, so H_x = 2 exchange rounds and a
    # 2-layer halo grid — the multi-round forwarding and the periodic seam
    # are both on the line. P = 1 additionally pins DD ≡ serial.
    @testset "oracle + parity (P = $P)" for P in (1, 2, 4, 6)
        verify_ghosts(load_check(cpu_bundles[P], "ghosts/normal"), P, "normal")
    end

    # rc = L → δ ≈ 1.1: multi-round on dims == 2 (both directions share
    # one peer, tag-separated) AND on dims == 1 (self-stages, |shift| ≥ 2
    # images) — the two degeneracies the standard runs never combine with
    # H ≥ 2.
    @testset "multi-round degeneracies, rc = L (P = $P)" for P in (2, 4)
        verify_ghosts(load_check(cpu_bundles[P], "ghosts/bigrc"), P, "bigrc")
    end

    # NaCl-8 at P = 6 (dims (3,2,1)): the x ∈ [2/3, 1) bricks hold no
    # atoms — empty ranks must ride every collective in build/sync/fold.
    @testset "empty ranks (NaCl-8, P = 6)" begin
        verify_ghosts(load_check(cpu_bundles[6], "ghosts/tiny"), 6, "tiny")
    end

    @testset "collective isreusable (asymmetric displacement, P = 2)" begin
        verify_ghosts_deadlock(load_check(cpu_bundles[2], "ghosts/deadlock"))
    end
end

@testset "reduction regime: per-calculator DD parity (P ∈ {1, 2, 4})" begin
    for P in (1, 2, 4)
        verify_reduce(load_check(cpu_bundles[P], "reduce/normal"), P, "normal", :cpu)
    end

    # NaCl-8 at P = 6 (dims (3,2,1)): empty ranks must ride the S(k)
    # Allreduce, the replicated E/virial adds, and the global scalars.
    @testset "empty ranks (Ewald + globals, NaCl-8, P = 6)" begin
        verify_reduce(load_check(cpu_bundles[6], "reduce/tiny"), 6, "tiny", :cpu)
    end
end

@testset "shell-CG collective parity (P ∈ {1, 2, 4})" begin
    for P in (1, 2, 4)
        verify_coreshell(load_check(cpu_bundles[P], "coreshell"), P, :cpu)
    end
end

@testset "analytic HVP DD parity (P = 2)" begin
    verify_hvp_dd(load_check(cpu_bundles[2], "hvp_dd/normal"), 2, :cpu, "normal")
    # With a SurfaceTerm (Issue #181): the block's own 3-vector Allreduce
    # and the replicated regime of its virial derivative.
    verify_hvp_dd(load_check(cpu_bundles[2], "hvp_dd/surface"), 2, :cpu,
                  "surface")
end

@testset "analytic HVP, empty ranks (NaCl-8, P = 6)" begin
    verify_hvp_dd(load_check(cpu_bundles[6], "hvp_dd/tiny"), 6, :cpu, "tiny")
    verify_hvp_dd(load_check(cpu_bundles[6], "hvp_dd/surface_tiny"), 6, :cpu,
                  "surface_tiny")
end

@testset "relax! DD (FIRE collectives)" begin
    # The four collectives relax! adds (_relax_reductions, _max_force,
    # _cap_displacement!, the PMEInfo rebuild in _verify_convergence!) only
    # execute at P > 1, and every FIRE branch is decided from one of them.
    ref = relax_dd_serial_ref(:cpu)
    @testset "relaxation parity vs serial (P = $P)" for P in (1, 2)
        verify_relax_normal(load_check(cpu_bundles[P], "relax_dd/normal"),
                            joinpath(cpu_bundles[P].outdir, bundle_slug("relax_dd/normal")),
                            ref, P, :cpu)
    end
    # The cell degrees of freedom: the cell force is built from a REPLICATED
    # stress and must be folded in after the atom reduction, so a P-dependent
    # result here is the signature of that bug.
    @testset "cell dof through the DD driver (cell_dof = $(dof), P = $P)" for
            (dof, P) in ((:full, 1), (:full, 2), (:iso, 2))
        ref_c = relax_dd_serial_ref(:cpu; cell_dof = dof)
        name = dof === :full ? "relax_dd/cell" : "relax_dd/cell_iso"
        verify_relax_normal(load_check(cpu_bundles[P], name),
                            joinpath(cpu_bundles[P].outdir, bundle_slug(name)),
                            ref_c, P, :cpu; cell = true)
    end

    # P = 6 on the 8-atom fixture: several ranks own one atom or none, so
    # the empty-rank path through every reduction is exercised.
    @testset "empty-rank path (P = 6, tiny)" begin
        ref_t = relax_dd_serial_ref(:cpu; tiny = true)
        verify_relax_normal(load_check(cpu_bundles[6], "relax_dd/tiny"),
                            joinpath(cpu_bundles[6].outdir, bundle_slug("relax_dd/tiny")),
                            ref_t, 6, :cpu)
    end
end

@testset "npt! DD tier-2 (driver integration)" begin
    # Serial reference in THIS process (no MPI.Init here → SerialDomain),
    # once per fixture flavour, shared across every P.
    ref = npt_dd_serial_ref(:cpu)

    @testset "trajectory parity vs serial (P = $P)" for P in (1, 2, 4)
        verify_npt_normal(load_check(cpu_bundles[P], "npt_dd/normal"),
                          joinpath(cpu_bundles[P].outdir, bundle_slug("npt_dd/normal")),
                          ref, P, :cpu)
    end

    # Umbrella through the whole DD driver (P = 2): the interim guard is
    # gone — the Pg Allreduce runs inside every CG force evaluation, and
    # Bias.log must match the serial run (root-only, per-step CV/bias).
    @testset "umbrella through the DD driver (P = 2)" begin
        ref_dir_u = npt_dd_serial_ref_umbrella(:cpu)
        verify_npt_umbrella(load_check(cpu_bundles[2], "npt_dd/umbrella"),
                            joinpath(cpu_bundles[2].outdir, bundle_slug("npt_dd/umbrella")),
                            ref_dir_u, :cpu)
    end

    # CSVR through the DD driver (P = 2, Issue #138): the rescaling noise
    # comes from the seed-derived replicated stream, so the DD trajectory
    # must match the serial one at the same tolerances as the
    # deterministic NH flavour (any rank divergence would also trip the
    # GhostsDD lattice-hash abort long before these bounds).
    @testset "CSVR through the DD driver (P = 2)" begin
        ref_c = npt_dd_serial_ref(:cpu; csvr = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/csvr"),
                          joinpath(cpu_bundles[2].outdir, bundle_slug("npt_dd/csvr")),
                          ref_c, 2, :cpu)
    end

    # SCR through the DD driver (P = 2, Issue #140): the stochastic cell
    # update is the highest-risk MPI property of the barostat — every μ
    # input is an Allreduced scalar/tensor plus the replicated noise
    # stream, so DD must match serial at the NH tolerances (rank
    # divergence would also trip the GhostsDD lattice-hash abort).
    @testset "SCR through the DD driver (P = 2)" begin
        ref_s = npt_dd_serial_ref(:cpu; scr = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/scr"),
                          joinpath(cpu_bundles[2].outdir, bundle_slug("npt_dd/scr")),
                          ref_s, 2, :cpu)
    end

    # Shadow XL-BOMD through the DD driver (P = 2, Issue #149): the shadow
    # step evaluates the analytic HVP every step, whose dipole-grid
    # Allreduce is the property that makes DD parity possible at all — a
    # missing reduction shows up here as a trajectory mismatch, not an
    # error.
    @testset "shadow XL-BOMD through the DD driver (P = 2)" begin
        ref_x = npt_dd_serial_ref(:cpu; xl = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/xlbomd"),
                          joinpath(cpu_bundles[2].outdir, bundle_slug("npt_dd/xlbomd")),
                          ref_x, 2, :cpu)
        # Chebyshev kernel (Issue #152): the preconditioned direction adds
        # order-1 further HVPs per step. Their count is state-independent,
        # so every rank — empty ones included — makes the same collective
        # calls; a divergence here would be a hang, a missing reduction a
        # silent trajectory mismatch.
        ref_xc = npt_dd_serial_ref(:cpu; xl = true, cheb = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/xlbomd_cheb"),
                          joinpath(cpu_bundles[2].outdir,
                                   bundle_slug("npt_dd/xlbomd_cheb")),
                          ref_xc, 2, :cpu)
        # Shadow virial under a barostat (Issue #151): the correction is
        # an owned per-atom sum plus a REPLICATED reciprocal remainder, so
        # a wrong reduction regime scales the cell-driving stress by the
        # rank count. Comparing the final lattice against the serial run
        # is what catches it.
        ref_xb = npt_dd_serial_ref(:cpu; xlbaro = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/xlbomd_baro"),
                          joinpath(cpu_bundles[2].outdir,
                                   bundle_slug("npt_dd/xlbomd_baro")),
                          ref_xb, 2, :cpu)
        # NLH baseline (Issue #124): a gather kernel on a list derived from
        # the rc parent, whose per-atom energy/virial are owned partial sums.
        # A ghost-row or reduction-regime slip shows up as a trajectory
        # divergence from the serial run.
        ref_nlh = npt_dd_serial_ref(:cpu; nlh = true)
        # Non-vacuity: the baseline must actually change the trajectory,
        # otherwise "DD matches serial" would hold for a term that is
        # identically zero (it is, if the cutoff lands inside the nearest-
        # neighbour distance — see the note in _npt_dd_params).
        @test !isapprox(ref_nlh.pos, ref.pos; rtol = 1e-10)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/nlh"),
                          joinpath(cpu_bundles[2].outdir,
                                   bundle_slug("npt_dd/nlh")),
                          ref_nlh, 2, :cpu)
    end

    # Shadow XL-BOMD + surface term under a barostat (Issue #181): the
    # depolarization block's own Allreduce lives inside the shadow force,
    # and its virial derivative is replicated (returned, not summed per
    # atom). Both mistakes are silent and both move the cell, so the
    # lattice comparison against serial is what catches them.
    @testset "XL-BOMD + surface term through the DD driver (P = 2)" begin
        ref_xs = npt_dd_serial_ref(:cpu; surface = true)
        verify_npt_normal(load_check(cpu_bundles[2], "npt_dd/xlbomd_surface"),
                          joinpath(cpu_bundles[2].outdir,
                                   bundle_slug("npt_dd/xlbomd_surface")),
                          ref_xs, 2, :cpu)
    end

    # NaCl-8 at P = 6: empty ranks ride the whole driver (distribute,
    # collectives, rebuild + migrate, gather).
    @testset "empty ranks drive the full loop (NaCl-8, P = 6)" begin
        verify_npt_tiny(load_check(cpu_bundles[6], "npt_dd/tiny"),
                        joinpath(cpu_bundles[6].outdir, bundle_slug("npt_dd/tiny")),
                        :cpu)
        # Same loop with the NLH baseline configured: an empty rank must
        # skip the kernel launch but still ride add_reduced_ev!'s collective.
        # The term itself is zero at this cell size (rc < nearest neighbour),
        # so this flavour pins the empty-rank code path, not the physics —
        # npt_dd/nlh at P = 2 covers the values.
        verify_npt_tiny(load_check(cpu_bundles[6], "npt_dd/nlh_tiny"),
                        joinpath(cpu_bundles[6].outdir, bundle_slug("npt_dd/nlh_tiny")),
                        :cpu)
    end
end

@testset "polarization umbrella DD parity (P ∈ {1, 2})" begin
    for P in (1, 2)
        verify_umbrella(load_check(cpu_bundles[P], "umbrella"), P)
    end
    # Same parity contract for the target-map bias (Issue #120): it rides the
    # same DD path, so a divergence here would mean the new bias broke the
    # replicated/owned split rather than the machinery beneath it.
    verify_umbrella(load_check(cpu_bundles[2], "umbrella/map"), 2)
end

@testset "surface term DD parity (P ∈ {1, 2}; empty ranks at P = 6)" begin
    # The term's only collective is the dipole Allreduce inside
    # calcdipole!; a missing reduction scales the energy by the rank
    # count, a rank-local dipole shows in the per-atom force comparison.
    for P in (1, 2)
        verify_surface_term(load_check(cpu_bundles[P], "surface_term"), P)
    end
    # NaCl-8 at P = 6 (dims (3,2,1)): empty ranks must ride the Allreduce.
    verify_surface_term(load_check(cpu_bundles[6], "surface_term/tiny"), 6;
                        tiny = true)
end

@testset "FitSpline 1-rank vs 2-rank" begin
    verify_fitspline(load_check(cpu_bundles[1], "fitspline"),
                     load_check(cpu_bundles[2], "fitspline"))
end

# Issue #124: the NLH baseline is subtracted per rank on that rank's shard
# of the dataset, so a rank-dependent slip there lands in the RHS partial
# sums that tier 1 compares at machine precision.
@testset "FitSpline + NLH baseline, 1-rank vs 2-rank" begin
    verify_fitspline(load_check(cpu_bundles[1], "fitspline/nlh"),
                     load_check(cpu_bundles[2], "fitspline/nlh"))
end

@testset "FitACE 1-rank vs 2-rank" begin
    verify_fitace(load_check(cpu_bundles[1], "fitace"),
                  load_check(cpu_bundles[2], "fitace"))
end

@testset "FitACE + blockchol_si preconditioner, 1-rank vs 2-rank" begin
    d1 = load_check(cpu_bundles[1], "fitace/precond")
    d2 = load_check(cpu_bundles[2], "fitace/precond")
    # resid_tol 1e-9: PCG's true-residual floor is the preconditioner-solve
    # rounding level ε·κ(block) — see the note on verify_fitace.
    verify_fitace(d1, d2; resid_tol = 1e-9)
    # Cross-mode: the preconditioner must not change WHAT is solved — the
    # converged PCG solution equals the converged plain-CG solution of the
    # same fixture (both guarded by resid <= 1e-11 inside verify_fitace).
    dplain = load_check(cpu_bundles[1], "fitace")
    if _artifact_ok(d1) && _artifact_ok(dplain)
        @test isapprox(d1["vec_acep"], dplain["vec_acep"];
                       rtol = 1e-6, atol = 1e-10)
    end
end

# Issue #184: machine-precision parity of the Gram ASSEMBLY itself — the
# column-panel accumulation and the hierarchical (node window + leader)
# reduce, which the fitace tiers above cannot isolate (tier 1 is
# preconditioner-independent, tier 2 compares solutions at 1e-6).
@testset "ACE Gram assembly, artificial node split ≡ flat (P = 4)" begin
    dflat = load_check(cpu_bundles[4], "acegram/flat")
    dsplit = load_check(cpu_bundles[4], "acegram/split")
    # same P and frame shard — only the Reduce topology differs
    verify_ace_gram(dflat, dsplit; P1 = 4, P2 = 4, rtol = 1e-13)
    # the split must exercise the owner-is-not-a-node-leader window read
    # (leaders of the rank%2 split at P = 4 are ranks 0 and 1)
    if dsplit !== nothing
        @test any(>=(2), dsplit["owner"])
    end
end

@testset "ACE Gram assembly, 1-rank vs 4-rank" begin
    verify_ace_gram(load_check(cpu_bundles[1], "acegram/flat"),
                    load_check(cpu_bundles[4], "acegram/flat");
                    P1 = 1, P2 = 4, rtol = 1e-12)
end
