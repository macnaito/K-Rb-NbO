# test/gpu/test_ewald_gpu.jl
#
# CPU vs GPU parity for the *real-space* part of the Ewald sum,
# i.e. `CoulombBuckingham` evaluated against a NeighborList.
# Reciprocal-space sums on GPU go through PME (Ewald.jl:116
# the reciprocal kernel (_ewald_structure_factor!/_ewald_forces!) is CPU-only,
# see test_ewald_pme_equivalence.jl
# and test_pme_gpu.jl for the long-range coverage).
#
using Test
using CUDA

function coubuck_cpu(atoms, prms)
    a = deepcopy_atoms(atoms)
    nl = neighborlist(a, prms[:rc], 0.0)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, prms)
    return (a.info[:potential_energy][1], copy(a.arrays[:forces]),
            copy(a.info[:virial]))
end

function coubuck_gpu(atoms, prms)
    a = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
    p = arrtyp_prms(CuArray, prms)
    nl = neighborlist(a, p[:rc], 0.0)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, p)
    return (Array(a.info[:potential_energy])[1],
            Array(a.arrays[:forces]),
            Array(a.info[:virial]))
end

@testset "CPU vs GPU CoulombBuckingham parity (NaCl 64)" begin
    atoms = make_nacl_64()
    prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                       rc_factor = 0.45)

    E_c, F_c, V_c = coubuck_cpu(atoms, prms)
    E_g, F_g, V_g = coubuck_gpu(atoms, prms)

    @test isapprox(E_c, E_g; rtol = 1e-5)
    @test isapprox(F_c, F_g; rtol = 1e-5, atol = 1e-8)
    @test isapprox(V_c, V_g; rtol = 1e-5, atol = 1e-8)
end
