# test/calculators/test_external_field.jl
using Test

@testset "ExternalField" begin
    atoms = make_perovskite_5(; a_ang = 4.0)
    nelem = 3
    qc = FMD.FPPREC[+2.0, +4.0, -2.0]
    qs = FMD.FPPREC[-0.5, -1.0, +0.5]

    @testset "Force = q · E_ext" begin
        E_ext = FMD.FPPREC[0.1, -0.2, 0.3]
        prms = (qc = qc, qs = qs, efield = E_ext)
        reset!(atoms)
        Calculators.calculate!(Calculators.ExternalField(), atoms, prms)
        for i in 1:length(atoms)
            si = atoms.arrays[:elemid][i]
            for k in 1:3
                @test isapprox(atoms.arrays[:forces][i, k], qc[si] * E_ext[k]; rtol = 1e-12)
                @test isapprox(atoms.arrays[:shellforces][i, k], qs[si] * E_ext[k]; rtol = 1e-12)
            end
        end
    end

    @testset "Zero field ⇒ zero force" begin
        prms = (qc = qc, qs = qs, efield = zeros(FMD.FPPREC, 3))
        reset!(atoms)
        Calculators.calculate!(Calculators.ExternalField(), atoms, prms)
        @test all(==(0), atoms.arrays[:forces])
        @test all(==(0), atoms.arrays[:shellforces])
    end
end
