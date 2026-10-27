# test/apps/test_fit_spline_smoke.jl
#
# Single-rank smoke for Calculators.fitting!(::SplineParam, ...) on CPU.
# Exercises the data-parallel CG loop with mpisize == 1.  This is the
# same code path script/flow.sh drives in production via mpiexec -n N.

using Test
using Random
using MPI

@testset "FitSpline smoke (1 rank, CPU + CoreShell)" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(0)

    # Use the NaCl 64 supercell so the inner CoreShell SCF has real
    # neighbors and the cpukernel_gcb_onlyshell @turbo path runs.
    atoms = make_nacl_64()
    atoms.info[:potential_energy] = FMD.FPPREC[0.01]
    atoms.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(atoms.arrays[:forces]))
    # Non-zero virial so σ_stress > 0 inside fitting!'s normalization.
    atoms.info[:virial] = FMD.FPPREC[1e-4, 1e-4, 1e-4, 0, 0, 0]
    atoms.info[:method] = [:smoke]
    dataset = Atoms[atoms]

    # SplineParam with a small basis (3 Bohr / 6 Bohr cutoffs already in Bohr).
    splp = Calculators.SplineParam(FMD.FPPREC(8.0), FMD.FPPREC(6.0),
                                   2, 4, 4)

    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; ),
        nsteps        = 3,
    )

    # Issue #28: fitting! must be RNG-transparent — it neither consumes nor
    # reseeds the global RNG, so the stream after the call equals the stream
    # untouched (exact equality: same seed, same draw).  Note this also
    # requires fitting! to spawn no tasks: task creation splits (advances)
    # the parent's task-local RNG even without drawing from it.
    Random.seed!(123)
    rngprobe = rand()
    Random.seed!(123)
    @test_nowarn Calculators.fitting!(splp, dataset, params)
    @test rand() == rngprobe

    @test all(isfinite, splp.c1)
    @test all(isfinite, splp.c2)
    @test all(isfinite, splp.c3)

    # Issue #1: fitting! must return a c3 in the involution-symmetric
    # subspace `c3[a,b,θ,sa,sb,si] == c3[b,a,θ,sb,sa,si]`.
    let c3 = splp.c3
        for si in axes(c3, 6), sb in axes(c3, 5), sa in axes(c3, 4)
            slc_ab = c3[:, :, :, sa, sb, si]
            slc_ba = c3[:, :, :, sb, sa, si]
            @test isapprox(slc_ab, permutedims(slc_ba, (2, 1, 3));
                           atol = 1e-12)
        end
    end
    # Issue #63: bit-exactly, not just approximately — the load-time
    # contract check (`_check_splp_c3sym`) uses iszero, so fit output
    # must measure exactly zero or saved potentials would fail to load.
    @test Calculators._spl_c3_asymmetry(splp.c3) == 0.0
end

@testset "FitSpline smoke (rigid_curv P-spline penalty)" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(0)

    atoms = make_nacl_64()
    atoms.info[:potential_energy] = FMD.FPPREC[0.01]
    atoms.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(atoms.arrays[:forces]))
    atoms.info[:virial] = FMD.FPPREC[1e-4, 1e-4, 1e-4, 0, 0, 0]
    atoms.info[:method] = [:smoke]
    dataset = Atoms[atoms]

    splp = Calculators.SplineParam(FMD.FPPREC(8.0), FMD.FPPREC(6.0),
                                   2, 6, 5)

    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; ),
        rigid_curv    = (c2 = 1e-2, c3_r = 1e-2, c3_θ = 1e-3,
                         c2_pin = 1e-2, c3_r_pin = 1e-2),
        nsteps        = 3,
    )

    @test_nowarn Calculators.fitting!(splp, dataset, params)
    @test all(isfinite, splp.c2)
    @test all(isfinite, splp.c3)

    # Issue #1: same end-to-end symmetry assertion under the curvature /
    # boundary-pin regularizers.  The projection inside fitting! must
    # produce a symmetric c3 even when curv/pin terms are active.
    let c3 = splp.c3
        for si in axes(c3, 6), sb in axes(c3, 5), sa in axes(c3, 4)
            slc_ab = c3[:, :, :, sa, sb, si]
            slc_ba = c3[:, :, :, sb, sa, si]
            @test isapprox(slc_ab, permutedims(slc_ba, (2, 1, 3));
                           atol = 1e-12)
        end
    end
    # Issue #63: bit-exact symmetry also under curv/pin regularizers.
    @test Calculators._spl_c3_asymmetry(splp.c3) == 0.0
end

@testset "FitSpline Δ-learning with an NLH baseline" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(124)

    atoms = make_nacl_64()
    atoms.info[:method] = [:smoke]
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    # Na–Cl is a legal NLH pair; keep the baseline cutoff inside rc so the
    # MD path could derive its list from the Coulomb parent.
    nlh = Calculators.NLHParam(Dict("Na" => 1, "Cl" => 2), 0.9 * base[:rc])
    # Tikhonov weight ≈ 1e-6·λmax of this fixture's normal operator (λmax ≈
    # 3e10, measured in Issue #23) — the same conditioning choice the MPI
    # parity fixture uses, so CG actually converges here.
    Λ_RIGID = FMD.FPPREC(3.0e4)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; c1 = Λ_RIGID, c2 = Λ_RIGID, c3 = Λ_RIGID),
        nsteps        = 200,
        nlh           = nlh,
    )

    # Reference E/F/V := the baseline itself (core-shell + NLH), so the
    # residual the MLIP is asked to model is identically zero.
    ref = deepcopy_atoms(atoms)
    reset!(ref)
    Calculators.add_baseline!(ref, params)
    atoms.info[:potential_energy] = FMD.FPPREC[ref.info[:potential_energy][1]]
    atoms.arrays[:forces] = copy(ref.arrays[:forces])
    atoms.info[:virial]   = copy(ref.info[:virial])
    dataset = Atoms[atoms]

    mk() = Calculators.SplineParam(FMD.FPPREC(8.0), FMD.FPPREC(6.0), 2, 4, 4)

    # The normal-equation RHS is A^T(residual): subtracting the same baseline
    # the reference was built from must cancel it to round-off. Forgetting the
    # NLH term (params without :nlh) leaves the whole NLH contribution in the
    # target — that is the failure this pins, and it is many orders larger.
    y_with    = Calculators.build_normal_system(mk(), dataset, params).y
    y_without = Calculators.build_normal_system(mk(), dataset,
                    Base.structdiff(params, NamedTuple{(:nlh,)})).y
    @test maximum(abs, y_without) > 0
    @test maximum(abs, y_with) < 1e-10 * maximum(abs, y_without)

    # The solve on top is only smoke: with y ≈ 0 the exact minimiser is
    # x = 0, but how close CG gets in a fixed iteration budget is a property
    # of this fixture's conditioning (κ ~ 1e6 here), not of the baseline
    # bookkeeping — that is what the RHS assertion above pins.
    splp = mk()
    @test_nowarn Calculators.fitting!(splp, dataset, params)
    @test all(isfinite, vec(splp))
end
