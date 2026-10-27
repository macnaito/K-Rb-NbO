# test/gpu/test_xlbomd_gpu.jl
#
# Shadow XL-BOMD on the GPU backend (Issue #149). The shadow step is
# backend-generic by construction — one KA kernel for the closed-form
# minimizer, broadcasts for the auxiliary propagation, and the already
# GPU-tested analytic HVP for the Hessian correction — so this pins the
# parity rather than a separate implementation. Tolerance band as in the
# other GPU parity tests (rtol 1e-6 / atol 1e-10): atomic reassociation
# plus CUFFT vs FFTW, two orders of headroom.

using Test
using CUDA
using Random
using LinearAlgebra

@testset "shadow XL-BOMD (GPU)" begin
    Random.seed!(61)
    atoms_cpu = make_nacl_8()
    atoms_cpu.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:pos]))
    atoms_cpu.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:shelldisp]))
    base = nacl_params(atoms_cpu; alpha = 0.30, with_shell = true)
    prms = (; base..., pme_meshsize = FMD.FPPREC(0.5),
            a = FMD.FPPREC[0.0 5.0; 5.0 8.0],
            rho = FMD.FPPREC[1.0 0.9; 0.9 0.8],
            c = FMD.FPPREC[0.0 2.0; 2.0 4.0],
            k4 = FMD.FPPREC[300.0, 500.0])   # exercise the cubic Newton branch
    N = length(atoms_cpu)
    shat_cpu = atoms_cpu.arrays[:shelldisp] .+
               0.01 .* randn(FMD.FPPREC, size(atoms_cpu.arrays[:shelldisp]))

    # CPU reference.
    a = deepcopy_atoms(atoms_cpu)
    nl = neighborlist(a, prms[:rc], prms[:skin])
    pme = Calculators.PMEInfo(a.info[:lattice], prms)
    a.arrays[:peratom_virial] = zeros(FMD.FPPREC, N, 6)
    Eref, devref = FMD.Apps.xl_shadow_forces!(a, nl, pme, prms, copy(shat_cpu))
    Fref = copy(a.arrays[:forces])
    S1ref = copy(a.arrays[:shelldisp])
    Wref = copy(Array(a.info[:virial]))
    PWref = copy(a.arrays[:peratom_virial])

    @testset "shadow force / energy / minimizer / virial parity" begin
        g = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        g.arrays[:peratom_virial] = CUDA.zeros(FMD.FPPREC, N, 6)
        p = arrtyp_prms(CuArray, prms)
        # Naive list keeps the CPU row order, so no :id permutation is
        # needed for the comparison.
        nlg = neighborlist(g, p[:rc], p[:skin])
        pmeg = Calculators.PMEInfo(g.info[:lattice], p)
        E, dev = FMD.Apps.xl_shadow_forces!(g, nlg, pmeg, p, CuArray(shat_cpu))

        @test maximum(abs, Fref) > 1e-3          # non-vacuous fixture
        @test devref > 1e-6                      # the correction term is live
        fscale = max(one(FMD.FPPREC), maximum(abs, Fref))
        @test maximum(abs, Array(g.arrays[:forces]) .- Fref) < 1e-6 * fscale + 1e-10
        @test maximum(abs, Array(g.arrays[:shelldisp]) .- S1ref) < 1e-6 + 1e-10
        @test isapprox(E, Eref; rtol = 1e-8, atol = 1e-10)
        @test isapprox(dev, devref; rtol = 1e-6, atol = 1e-10)
        # Shadow virial (Issue #151): the pair-kernel accumulation, the
        # spring/intra closed forms, the grid reduction and the assembly
        # kernel are all backend-generic, so this pins the chain.
        wscale = max(one(FMD.FPPREC), maximum(abs, Wref))
        @test maximum(abs, Wref) > 1e-4          # non-vacuous
        @test maximum(abs, Array(g.info[:virial]) .- Wref) < 1e-6 * wscale
        @test maximum(abs, Array(g.arrays[:peratom_virial]) .- PWref) <
              1e-6 * wscale
    end

    # Chebyshev kernel (Issue #152): the direction is a chain of hvp! calls
    # plus broadcasts and one masking kernel, so this pins the backend
    # parity of the chain rather than a separate implementation.
    @testset "Chebyshev kernel direction parity" begin
        kc = FMD.Apps.cheb_kernel_coefs(3, 0.4, 6.0)
        c = deepcopy_atoms(atoms_cpu)
        c.arrays[:xl_dir] = zero(c.arrays[:shelldisp])
        nlc = neighborlist(c, prms[:rc], prms[:skin])
        pmec = Calculators.PMEInfo(c.info[:lattice], prms)
        FMD.Apps.xl_shadow_forces!(c, nlc, pmec, prms, copy(shat_cpu);
                                   kernel_coefs = kc)
        Dref = copy(c.arrays[:xl_dir])

        g = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        g.arrays[:xl_dir] = zero(g.arrays[:shelldisp])
        p = arrtyp_prms(CuArray, prms)
        nlg = neighborlist(g, p[:rc], p[:skin])
        pmeg = Calculators.PMEInfo(g.info[:lattice], p)
        FMD.Apps.xl_shadow_forces!(g, nlg, pmeg, p, CuArray(shat_cpu);
                                   kernel_coefs = kc)

        @test maximum(abs, Dref) > 1e-4                  # non-vacuous
        @test maximum(abs, Array(g.arrays[:xl_dir]) .- Dref) <
              1e-6 * maximum(abs, Dref) + 1e-10
        # The kernel must not touch the forces (same values as the
        # spring-kernel run above, which used the identical ŝ).
        @test maximum(abs, c.arrays[:forces] .- Fref) < 1e-10 * max(1.0, maximum(abs, Fref))
    end

    # Surface term (Issue #181). The shadow step now carries a whole-system
    # term whose reduction runs on the device and whose virial derivative
    # is assembled host-side, so the parity statement has to cover the
    # shadow force, energy and virial with the term switched on.
    @testset "surface-term shadow step parity" begin
        proto = make_nacl_8()
        # |q_s| = 1 (qc + qs = ±0.8, neutral); the block scales as q_s².
        sp = (; prms..., qc = FMD.FPPREC[1.8, -1.8], qs = FMD.FPPREC[-1.0, 1.0])
        R = [0.8 -0.6 0.0; 0.6 0.8 0.0; 0.0 0.0 1.0]
        # Gauge = the unrattled prototype: with the run frame as its own
        # gauge δM = 0 and the whole term vanishes.
        mkst() = Calculators.SurfaceTerm(atoms_cpu, sp[:qc], sp[:qs];
            eps_out = R * Diagonal([1.5, 3.0, 8.0]) * R',
            shape = [1.0, 1.4, 0.7], gauge = proto)

        c = deepcopy_atoms(atoms_cpu)
        pc = (; sp..., surface_term = mkst())
        nlc = neighborlist(c, pc[:rc], pc[:skin])
        pmec = Calculators.PMEInfo(c.info[:lattice], pc)
        Ec, devc = FMD.Apps.xl_shadow_forces!(c, nlc, pmec, pc, copy(shat_cpu))
        Fc = copy(c.arrays[:forces])
        Wc = copy(Array(c.info[:virial]))

        g = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        pg = (; arrtyp_prms(CuArray, sp)..., surface_term = mkst())
        nlg = neighborlist(g, pg[:rc], pg[:skin])
        pmeg = Calculators.PMEInfo(g.info[:lattice], pg)
        E, dev = FMD.Apps.xl_shadow_forces!(g, nlg, pmeg, pg, CuArray(shat_cpu))

        # Non-vacuous: the same fixture without the term differs.
        b = deepcopy_atoms(atoms_cpu)
        nlb = neighborlist(b, sp[:rc], sp[:skin])
        FMD.Apps.xl_shadow_forces!(b, nlb,
            Calculators.PMEInfo(b.info[:lattice], sp), sp, copy(shat_cpu))
        fscale = max(one(FMD.FPPREC), maximum(abs, Fc))
        @test maximum(abs, Fc .- b.arrays[:forces]) > 1e-6 * fscale
        @test maximum(abs, Wc .- Array(b.info[:virial])) > 1e-6

        @test maximum(abs, Array(g.arrays[:forces]) .- Fc) < 1e-6 * fscale + 1e-10
        @test isapprox(E, Ec; rtol = 1e-8, atol = 1e-10)
        @test isapprox(dev, devc; rtol = 1e-6, atol = 1e-10)
        @test maximum(abs, Array(g.info[:virial]) .- Wc) <
              1e-6 * max(one(FMD.FPPREC), maximum(abs, Wc))
    end

    @testset "driver: XL steps on the device cost no CG iterations" begin
        dt = FMD.FPPREC(1.0 / UNITS[:fs])
        mktempdir() do dir
            p = (; prms..., backend = :cuda,
                 elements = Dict("Na" => 1, "Cl" => 2),
                 skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(1e-8),
                 cgmaxiter = 200, nsteps = 6, timestep = dt,
                 thermostat = :none, barostat = :none,
                 initial_temperature = 0.0, target_temperature = 0.0,
                 output_interval = 100,
                 shell_dynamics = :xlbomd, xl_warmup = 3,
                 output_dir = dir)
            out = FMD.Apps.npt!(deepcopy_atoms(atoms_cpu), p)
            @test all(isfinite, Array(out.arrays[:pos]))
            ncg = [parse(Int, last(split(l)))
                   for l in eachline(joinpath(dir, "log.txt"))]
            @test length(ncg) == 6
            @test all(==(0), ncg[4:end])
        end
    end
end
