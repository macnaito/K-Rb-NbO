# test/apps/test_fit_ace_smoke.jl
#
# Single-rank smoke for Calculators.fitting!(::ACEParam, ...) on CPU.

using Test
using Random
using MPI

@testset "FitACE smoke (1 rank, CPU + CoreShell)" begin
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

    rmin = 1.0; rmax = 6.0  # Bohr
    acep = Calculators.ACEParam(FMD.FPPREC(rmin), FMD.FPPREC(rmax),
                                2, 4, 4, 3, 3)

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
    @test_nowarn Calculators.fitting!(acep, dataset, params)
    @test rand() == rngprobe

    @test all(isfinite, vec(acep))
end

@testset "FitACE Δ-learning with an NLH baseline" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(124)

    atoms = make_nacl_64()
    atoms.info[:method] = [:smoke]
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    nlh = Calculators.NLHParam(Dict("Na" => 1, "Cl" => 2), 0.9 * base[:rc])
    # Tikhonov weight chosen (as in the Spline counterpart) so CG converges
    # on this fixture rather than stalling on an ill-conditioned operator.
    Λ = FMD.FPPREC(3.0e4)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; cp = Λ, c0 = Λ, c00 = Λ, c11 = Λ, c22 = Λ, c33 = Λ,
                         c011 = Λ, c022 = Λ, c033 = Λ),
        nsteps        = 200,
        nlh           = nlh,
    )

    # Reference := the baseline (core-shell + NLH) ⇒ zero residual target.
    ref = deepcopy_atoms(atoms)
    reset!(ref)
    Calculators.add_baseline!(ref, params)
    atoms.info[:potential_energy] = FMD.FPPREC[ref.info[:potential_energy][1]]
    atoms.arrays[:forces] = copy(ref.arrays[:forces])
    atoms.info[:virial]   = copy(ref.info[:virial])
    dataset = Atoms[atoms]

    mk() = Calculators.ACEParam(FMD.FPPREC(1.0), FMD.FPPREC(6.0), 2, 4, 4, 3, 3)

    y_with    = Calculators.build_normal_system(mk(), dataset, params).y
    y_without = Calculators.build_normal_system(mk(), dataset,
                    Base.structdiff(params, NamedTuple{(:nlh,)})).y
    @test maximum(abs, y_without) > 0
    @test maximum(abs, y_with) < 1e-10 * maximum(abs, y_without)

    # Solve on top is smoke only — see the note in the Spline counterpart.
    acep = mk()
    @test_nowarn Calculators.fitting!(acep, dataset, params)
    @test all(isfinite, vec(acep))
end

@testset "FitACE smoke with blockchol_si preconditioner" begin
    if !MPI.Initialized()
        MPI.Init()
    end
    Random.seed!(125)

    atoms = make_nacl_64()
    atoms.info[:potential_energy] = FMD.FPPREC[0.01]
    atoms.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(atoms.arrays[:forces]))
    atoms.info[:virial] = FMD.FPPREC[1e-4, 1e-4, 1e-4, 0, 0, 0]
    atoms.info[:method] = [:smoke]
    dataset = Atoms[atoms]

    acep = Calculators.ACEParam(FMD.FPPREC(1.0), FMD.FPPREC(6.0), 2, 4, 4, 3, 3)
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    Λ = FMD.FPPREC(3.0e4)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; cp = Λ, c0 = Λ, c00 = Λ, c11 = Λ, c22 = Λ, c33 = Λ,
                         c011 = Λ, c022 = Λ, c033 = Λ),
        nsteps        = 3,
        cg_preconditioner = :blockchol_si,
    )

    # Issue #28 contract holds with the preconditioner too: the blockchol-si
    # assembly is deterministic loops + BLAS + MPI — no RNG, no task spawn.
    Random.seed!(123)
    rngprobe = rand()
    Random.seed!(123)
    @test_nowarn Calculators.fitting!(acep, dataset, params)
    @test rand() == rngprobe

    @test all(isfinite, vec(acep))
end
