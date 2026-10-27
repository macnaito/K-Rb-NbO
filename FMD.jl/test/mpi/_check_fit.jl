# test/mpi/_check_fit.jl
#
# 1-rank vs 2-rank fitting parity children for Spline and ACE, sharing one
# dataset builder (the two pre-#113 scripts were identical here). Each
# check writes, from rank 0:
#   * the fitted parameter vector, and
#   * a partial-sum parity probe (RHS `y` and one operator application
#     `H*y` from Calculators.build_normal_system).
#
# The probe is the machine-precision tier of the parity test (Issue #23):
# it is deterministic regardless of conditioning, so a 1-rank vs 2-rank
# mismatch there isolates a genuine partial-sum / sharding bug. The fit
# itself is kept well-posed (both frames jittered + Tikhonov `rigid` +
# CG run to convergence) so the solution tier is meaningful too: an
# unregularised fixture is numerically singular and CG at the breakdown
# iteration amplifies reduction-order roundoff (~1e-16) into ~1e-3
# solution differences.

# 2 frames of NaCl 64; CoreShell+@turbo path has real neighbors here.
function _fit_dataset()
    ds = Atoms[]
    for k in 1:2
        a = make_nacl_64()
        # Jitter EVERY frame (1e-3·k) to avoid exact lattice symmetries.
        # NOTE: measured effect on conditioning is minor — the unregularised
        # effective rank stays ~5/530 — so this alone does NOT make the fit
        # well-posed; the Tikhonov `rigid` is what does (Issue #23).
        a.arrays[:pos] .+= 1e-3 * k .* randn(FMD.FPPREC, size(a.arrays[:pos]))
        a.info[:potential_energy] = FMD.FPPREC[1e-2 * k]
        a.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(a.arrays[:forces]))
        a.info[:virial] = FMD.FPPREC[1e-4 * k, 1e-4 * k, 1e-4 * k, 0, 0, 0]
        a.info[:method] = [:smoke]
        push!(ds, a)
    end
    return ds
end

function check_fitspline(out, dir, mode, backend, comm)
    Random.seed!(0)
    dataset = _fit_dataset()

    make_splp() = Calculators.SplineParam(FMD.FPPREC(8.0), FMD.FPPREC(6.0),
                                          2, 4, 4)

    # Tikhonov weight ≈ 1e-6·λmax of this fixture's normal operator
    # (λmax ≈ 3e10, measured in Issue #23). Keeps CG convergent so the
    # solution-parity tier tests a converged fit, not Krylov-breakdown noise.
    # Conditioning-only choice for this parity fixture — NOT a physically
    # recommended regularisation for production fits (cf. app defaults in
    # src/Apps/FitSpline.jl).
    Λ_RIGID = FMD.FPPREC(3.0e4)

    base = nacl_params(dataset[1]; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; c1 = Λ_RIGID, c2 = Λ_RIGID, c3 = Λ_RIGID),
        nsteps        = 50,
    )
    # "nlh" flavour (Issue #124): the Δ-learning baseline subtraction runs
    # per rank on that rank's dataset shard, so a rank-dependent slip there
    # would show up as a partial-sum (RHS) mismatch in the tier-1 probe.
    if mode == "nlh"
        params = (; params...,
                  nlh = Calculators.NLHParam(Dict("Na" => 1, "Cl" => 2),
                                             FMD.FPPREC(0.9 * base[:rc])))
    end

    # ---- machine-precision tier: partial-sum parity probe -----------------
    probe = Calculators.build_normal_system(make_splp(), dataset, params)
    Hy = zeros(length(probe.y))
    probe.Hmul!(Hy, copy(probe.y))

    # ---- solution tier: the actual fit ------------------------------------
    splp = make_splp()
    Calculators.fitting!(splp, dataset, params)

    # Converged-fit guard: recompute the true residual of the solved system.
    # Tier-2 parity is only meaningful on a converged fit; asserting this in
    # the parent turns "CG converged" from an assumption into a check, and
    # catches regressions back into the maxiter-limited regime (Issue #23).
    xfit = vec(splp)
    Hx = zeros(length(xfit))
    probe.Hmul!(Hx, xfit)
    resid = norm(Hx .- probe.y) / norm(probe.y)

    if MPI.Comm_rank(comm) == 0
        JLD2.jldsave(out;
                     c1 = Array(splp.c1),
                     c2 = Array(splp.c2),
                     c3 = Array(splp.c3),
                     y = probe.y,
                     Hy = Hy,
                     resid = resid,
                     mpisize = MPI.Comm_size(comm))
    end
    return nothing
end

function check_fitace(out, dir, mode, backend, comm)
    Random.seed!(0)
    dataset = _fit_dataset()

    make_acep() = Calculators.ACEParam(FMD.FPPREC(1.0), FMD.FPPREC(6.0),
                                       2, 4, 4, 3, 3)

    # Tikhonov weight ≈ 1e-6·λmax of this fixture's normal operator
    # (λmax ≈ 2.3e14, measured in Issue #23) — same rationale as the spline
    # check above. Conditioning-only choice for this parity fixture — NOT a
    # physically recommended regularisation for production fits.
    Λ_RIGID = FMD.FPPREC(2.0e8)

    base = nacl_params(dataset[1]; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; cp = Λ_RIGID, c0 = Λ_RIGID, c00 = Λ_RIGID,
                           c11 = Λ_RIGID, c22 = Λ_RIGID, c33 = Λ_RIGID,
                           c011 = Λ_RIGID, c022 = Λ_RIGID, c033 = Λ_RIGID),
        nsteps        = 50,
    )
    # mode "precond": identical fixture solved with the blockchol-si PCG
    # (Issue #177). Tier-1 (y/Hy parity) is preconditioner-independent; the
    # tier-2 solution must agree with the plain-CG artifact within the
    # verify_fitace tolerances (factors differ at roundoff between rank
    # counts because the block Reduce order differs).
    if mode == "precond"
        params = (; params..., cg_preconditioner = :blockchol_si)
    end

    # ---- machine-precision tier: partial-sum parity probe -----------------
    probe = Calculators.build_normal_system(make_acep(), dataset, params)
    Hy = zeros(length(probe.y))
    probe.Hmul!(Hy, copy(probe.y))

    # ---- solution tier: the actual fit ------------------------------------
    acep = make_acep()
    Calculators.fitting!(acep, dataset, params)

    # Converged-fit guard — see check_fitspline.
    xfit = vec(acep)
    Hx = zeros(length(xfit))
    probe.Hmul!(Hx, xfit)
    resid = norm(Hx .- probe.y) / norm(probe.y)

    if MPI.Comm_rank(comm) == 0
        JLD2.jldsave(out;
                     vec_acep = vec(acep),
                     y = probe.y,
                     Hy = Hy,
                     resid = resid,
                     mpisize = MPI.Comm_size(comm))
    end
    return nothing
end
