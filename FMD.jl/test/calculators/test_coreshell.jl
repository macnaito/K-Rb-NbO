# test/calculators/test_coreshell.jl
#
# CoreShell SCF on the CPU is the path that production FitSpline / FitACE
# (script/flow.sh, mpiexec -n N julia ...) takes inside `fitting!`, so this
# file exercises the SCF entry point with non-zero shell charges so the
# `cpukernel_gcb_onlyshell` kernel actually runs.

using Test
using LinearAlgebra
using Random

@testset "CoreShell SCF (CPU)" begin
    atoms = make_nacl_64()
    prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    prms = (; prms..., efield = FMD.FPPREC[1e-3, 0.0, 0.0])

    @testset "shell forces converge below cgtol" begin
        a = deepcopy_atoms(atoms)
        ncg = Calculators.calculate!(Calculators.CoreShell(), a, prms)
        @test ncg >= 0
        # cgmaxiter + 1: the loop guard is fcount < cgmaxiter and the
        # uphill-correction branch adds a second evaluation inside an
        # iteration, so the count can exceed cgmaxiter by one.
        @test ncg <= prms[:cgmaxiter] + 1
        # Post-condition: ‖shell_forces‖∞ ≤ cgtol (allow a small slack).
        @test maximum(abs.(a.arrays[:shellforces])) < prms[:cgtol] * 100
        @test all(isfinite, a.arrays[:forces])
        @test all(isfinite, a.info[:potential_energy])
        # shell_residual_rms is the CG loop's shell-only evaluation at the
        # exit positions; the full-stack core evaluation re-derives
        # :shellforces at the same positions through different kernels,
        # so the two rms must agree to roundoff — and an rms can never
        # exceed the maximum of the same components.
        sf = a.arrays[:shellforces]
        rms = a.info[:shell_residual_rms][1]
        @test rms ≈ sqrt(sum(abs2, sf) / length(sf)) rtol = 1e-8
        @test 0 < rms <= maximum(abs, sf)
    end

    @testset "tighter cgtol → at least as many CG iterations" begin
        a_loose = deepcopy_atoms(atoms)
        a_tight = deepcopy_atoms(atoms)
        prms_loose = (; prms..., cgtol = FMD.FPPREC(1e-3))
        prms_tight = (; prms..., cgtol = FMD.FPPREC(1e-7))
        ncg_loose = Calculators.calculate!(Calculators.CoreShell(), a_loose, prms_loose)
        ncg_tight = Calculators.calculate!(Calculators.CoreShell(), a_tight, prms_tight)
        @test ncg_tight >= ncg_loose
        # Tight result must yield a smaller or equal shell residual.
        @test maximum(abs.(a_tight.arrays[:shellforces])) <=
              maximum(abs.(a_loose.arrays[:shellforces])) + 1e-12
    end

    @testset "Zero external field: shell displacement stays small" begin
        prms_z = (; prms..., efield = zeros(FMD.FPPREC, 3))
        a = deepcopy_atoms(atoms)
        Calculators.calculate!(Calculators.CoreShell(), a, prms_z)
        # Symmetry: with no driving field, equilibrium shell displacement
        # should be near machine precision in this idealized cubic NaCl.
        @test maximum(abs.(a.arrays[:shelldisp])) < 1e-3
    end

    @testset "Empty-neighbor atoms are handled (perovskite_5)" begin
        # rc=0.45·a in a 4-Å perovskite leaves every atom with zero
        # neighbors; the @turbo guard in cpukernel_gcb_onlyshell
        # must allow this without segfault.
        a = make_perovskite_5(; a_ang = 4.0)
        p = perovskite_params(a; alpha = 0.20, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)
        ncg = Calculators.calculate!(Calculators.CoreShell(), a, p)
        @test ncg >= 0
        @test all(isfinite, a.arrays[:shellforces])
    end
end

@testset "CoreShell relaxed core force = -∇E* (Hellmann-Feynman FD)" begin
    # After shell relaxation, the core force returned by CoreShell must equal
    # the gradient of the *relaxed* energy surface E*(r) = min_shell E(r, ·)
    # (Hellmann–Feynman).  Previously only the SCF residual was checked, not
    # that the resulting core force is consistent with the energy.  We FD E*
    # by re-relaxing the shells at each displaced core configuration.  An
    # asymmetric fixture gives non-zero forces; efield = 0 (nacl_params
    # default) so ExternalField adds no energy-free force.
    Random.seed!(21)
    atoms = make_nacl_8()
    atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
    prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-8,
                       rc_factor = 0.45, with_shell = true)
    prms = (; prms..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 300)

    calc! = a -> Calculators.calculate!(Calculators.CoreShell(), a, prms)

    let a = deepcopy_atoms(atoms)
        _, F = evaluate(calc!, a)
        @test maximum(abs, F) > 1e-3        # relaxed forces are genuinely non-zero
    end

    assert_force_matches_finite_diff(calc!, atoms; h = 1e-4,
                                     atom_indices = 1:length(atoms), axes = (1, 2, 3),
                                     rtol = 1e-3, atol = 1e-5,
                                     label = "CoreShell relaxed core force (FD)")
end

@testset "CoreShell core force = -∂E/∂r at fixed shelldisp (CG residual left)" begin
    # MD moves the cores at fixed :shelldisp, so the shells ride along and
    # the force the integrator needs is the gradient of E at those frozen
    # shells — F_core + F_shell — not the relaxed-surface force above. The
    # two agree only when F_shell = 0, so a loose cgtol leaves a residual
    # large enough to tell them apart.
    Random.seed!(22)
    atoms = make_nacl_8()
    atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
    prms = nacl_params(atoms; alpha = 0.40, ewald_tolerance = 1e-8,
                       rc_factor = 0.45, with_shell = true)
    prms = (; prms..., cgtol = FMD.FPPREC(1e-2), cgmaxiter = 300)

    relaxed = deepcopy_atoms(atoms)
    Calculators.calculate!(Calculators.CoreShell(), relaxed, prms)
    Fs = copy(relaxed.arrays[:shellforces])
    @test maximum(abs, Fs) > 1e-4          # the residual is really there

    # Energy and force at the frozen shells: one full-stack pass, no CG,
    # with the Ewald α the convenience calculate! substitutes.
    prmsE = (; prms..., alpha = Calculators.ewaldcoeff(prms))
    frozen! = b -> begin
        nl = neighborlist(b, prmsE[:rc], prmsE[:skin])
        Calculators.core_forces!(b, nl, Calculators.EwaldInfo(b.info[:lattice], prmsE), prmsE)
        b.arrays[:forces] .+= b.arrays[:shellforces]
    end
    let b = deepcopy_atoms(relaxed)
        _, F = evaluate(frozen!, b)
        # calculate! returns exactly this force (same positions, same stack).
        @test relaxed.arrays[:forces] ≈ F rtol = 1e-10 atol = 1e-14
    end

    assert_force_matches_finite_diff(frozen!, relaxed; h = 1e-4,
                                     atom_indices = 1:length(relaxed), axes = (1, 2, 3),
                                     rtol = 1e-3, atol = 1e-5,
                                     label = "F_core + F_shell = -∂E/∂r|shelldisp (FD)")

    # Sensitivity: the core force alone misses the residual by more than
    # the FD tolerance, so the check above would catch dropping the term.
    ilattice = inv(Array(relaxed.info[:lattice]))
    i, ax = Tuple(argmax(abs.(Fs)))
    δ = zeros(FMD.FPPREC, 3); δ[ax] = 1e-4
    ap = deepcopy_atoms(relaxed); ap.arrays[:pos][i, :] .+= vec(δ' * ilattice)
    am = deepcopy_atoms(relaxed); am.arrays[:pos][i, :] .-= vec(δ' * ilattice)
    num = -(evaluate(frozen!, ap)[1] - evaluate(frozen!, am)[1]) / 2e-4
    Fcore = relaxed.arrays[:forces][i, ax] - Fs[i, ax]
    @test abs(Fcore - num) > 100 * (1e-5 + 1e-3 * abs(num))
end

@testset "linear_response! (CPU only)" begin
    atoms = make_nacl_64()
    prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    a = deepcopy_atoms(atoms)
    # `linear_response!` writes Born-effective-charge slabs into
    # :becex/y/z; pre-allocate them.  The function does call clear! on
    # those keys but does not initialize them when missing.
    N = length(a)
    a.arrays[:becex] = zeros(FMD.FPPREC, N, 3)
    a.arrays[:becey] = zeros(FMD.FPPREC, N, 3)
    a.arrays[:becez] = zeros(FMD.FPPREC, N, 3)

    scfcnt = Calculators.linear_response!(a, prms)
    @test scfcnt >= 0
    @test all(isfinite, a.info[:dielectric_tensor])
    # Permittivity diagonal must be ≥ 1 (vacuum baseline).
    @test a.info[:dielectric_tensor][1] >= 1 - 1e-8
    @test a.info[:dielectric_tensor][2] >= 1 - 1e-8
    @test a.info[:dielectric_tensor][3] >= 1 - 1e-8

    # Acoustic sum rule: Σ_i Z*_{αβ}(i) ≈ 0 for each Cartesian (α, β).
    # With the analytic HVP (Issue #147) the identity Σ_i(J+K)_ij = 0
    # holds to machine precision, so the ASR defect is exactly the summed
    # CG residual — bounded by the solve tolerance, not by an FD floor.
    # Measured 7.6e-8 at the default tol = 1e-8 (was ~1e-2·N pre-#147);
    # ×~10 margin here.
    # Column-wise (per Cartesian α), not the whole-matrix sum — the
    # identity is per column and a matrix total lets components cancel.
    @test maximum(abs, sum(a.arrays[:becex], dims = 1)) < 1e-6
    @test maximum(abs, sum(a.arrays[:becey], dims = 1)) < 1e-6
    @test maximum(abs, sum(a.arrays[:becez], dims = 1)) < 1e-6

    # Ewald ↔ PME consistency of the response quantities (same α): the
    # reciprocal discretization error is damped through the k2-dominated
    # Hessian, so the observables agree far tighter than the raw HVP
    # blocks (measured: ε 2.3e-11, BEC 4.9e-7 at mesh 0.5).
    prms2 = (; prms..., alpha = Calculators.ewaldcoeff(prms),
             efield = zeros(FMD.FPPREC, 3), pme_meshsize = FMD.FPPREC(0.5))
    bE = deepcopy_atoms(atoms)
    bP = deepcopy_atoms(atoms)
    for at in (bE, bP), k in (:becex, :becey, :becez)
        at.arrays[k] = zeros(FMD.FPPREC, N, 3)
    end
    nlE = neighborlist(bE, prms2[:rc], prms2[:skin])
    Calculators.linear_response!(bE, nlE, Calculators.EwaldInfo(bE.info[:lattice], prms2), prms2)
    nlP = neighborlist(bP, prms2[:rc], prms2[:skin])
    Calculators.linear_response!(bP, nlP, Calculators.PMEInfo(bP.info[:lattice], prms2), prms2)
    @test maximum(abs, bP.info[:dielectric_tensor] .- bE.info[:dielectric_tensor]) < 1e-8
    @test maximum(abs, bP.arrays[:becex] .- bE.arrays[:becex]) < 1e-5
    @test maximum(abs, bP.arrays[:becez] .- bE.arrays[:becez]) < 1e-5
end
