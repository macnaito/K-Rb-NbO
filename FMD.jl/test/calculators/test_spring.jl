# test/calculators/test_spring.jl
using Test

@testset "Spring (core-shell harmonic + quartic)" begin
    atoms = make_perovskite_5(; a_ang = 4.0)
    # Manually populate shell displacements with known values.
    atoms.arrays[:shelldisp] .= [
        0.05  0.0   0.0;
       -0.03  0.04  0.0;
        0.0   0.0  -0.02;
        0.0   0.01  0.0;
        0.0   0.0   0.03;
    ]

    nelem = 3
    prms = (
        k2 = FMD.FPPREC[10.0, 30.0, 20.0],
        k4 = FMD.FPPREC[0.0, 0.0, 0.0],   # disable quartic for closed-form check
    )

    @testset "Energy = ½ Σ k₂ Δ²" begin
        reset!(atoms)
        Calculators.calculate!(Calculators.Spring(), atoms, prms)

        u = atoms.arrays[:shelldisp]
        e = atoms.arrays[:elemid]
        E_expected = sum(prms[:k2][e[i]] * sum(u[i, :].^2) / 2 for i in 1:length(atoms))
        @test isapprox(atoms.info[:potential_energy][1], E_expected; rtol = 1e-12)
    end

    @testset "Force = -k₂ · Δ on shells; equal-and-opposite on cores" begin
        reset!(atoms)
        Calculators.calculate!(Calculators.Spring(), atoms, prms)

        u = atoms.arrays[:shelldisp]
        e = atoms.arrays[:elemid]
        for i in 1:length(atoms)
            k2i = prms[:k2][e[i]]
            for k in 1:3
                # rx = -shelldisp; F_shell += k2 * rx; F_core -= k2 * rx
                # so F_shell = -k2 * shelldisp, F_core = +k2 * shelldisp.
                @test isapprox(atoms.arrays[:shellforces][i, k], -k2i * u[i, k]; rtol = 1e-12)
                @test isapprox(atoms.arrays[:forces][i, k], +k2i * u[i, k]; rtol = 1e-12)
            end
        end
    end

    @testset "Quartic correction: dr² → 6·k₄/k₂ correction in force slope" begin
        prms_q = (
            k2 = FMD.FPPREC[10.0, 30.0, 20.0],
            k4 = FMD.FPPREC[1.0, 2.0, 3.0],
        )
        a = deepcopy_atoms(atoms)
        reset!(a)
        Calculators.calculate!(Calculators.Spring(), a, prms_q)

        # E = ½ k₂ Δ² + (1/24) k₄ Δ⁴
        u = a.arrays[:shelldisp]; e = a.arrays[:elemid]
        E_expected = sum(prms_q[:k2][e[i]] * sum(u[i, :].^2) / 2 +
                         prms_q[:k4][e[i]] * sum(u[i, :].^2)^2 / 24
                         for i in 1:length(a))
        @test isapprox(a.info[:potential_energy][1], E_expected; rtol = 1e-12)
    end

    @testset "Quartic force = -∂E/∂shelldisp (FD)" begin
        # The harmonic shellforce is closed-form-checked above; with k₄ ≠ 0
        # the force is non-linear in Δ, so verify the full (k₂ + k₄) force
        # against a finite difference of the energy.  Amplify the shell
        # displacements so the quartic term is a sizeable fraction of the
        # force (otherwise the FD only exercises the harmonic part).
        prms_q = (k2 = FMD.FPPREC[10.0, 30.0, 20.0],
                  k4 = FMD.FPPREC[50.0, 80.0, 30.0])
        a = deepcopy_atoms(atoms)
        a.arrays[:shelldisp] .*= 3
        calc! = x -> Calculators.calculate!(Calculators.Spring(), x, prms_q)
        assert_shellforce_matches_finite_diff(calc!, a; h = 1e-6,
                                              atom_indices = 1:length(a),
                                              axes = (1, 2, 3),
                                              rtol = 1e-6, atol = 1e-10,
                                              label = "Spring quartic shellforce (FD)")
    end
end
