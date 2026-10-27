# test/gpu/test_pme_gpu.jl
#
# PME on the GPU backend (backend-generic since #51 — the CPU twin lives in
# test/calculators/test_pme.jl).  Verify:
#   (a) PMEInfo + calculate! runs on the GPU.
#   (b) PME reciprocal force = -∇E and virial = -∂E/∂ε at TIGHT tolerance,
#       evaluated on the reciprocal part ALONE (no CoulombBuckingham) so the
#       k-space gather force and presfactor stress are pinned directly — not
#       diluted by the larger real-space contribution.  These are the
#       analytic derivative of the very same discrete mesh energy, so they
#       agree to ~FD truncation (the reciprocal probe shows ~1e-9).
#   (c) the full production force (CB real + PME) = -∇(total energy).
#
# All fixtures are ASYMMETRIC (displaced cores) so the forces/stress are
# genuinely non-zero — the symmetric rocksalt force is at the noise floor
# (the #9 blind spot, see test_ewald_pme_equivalence.jl).

using Test
using CUDA
using Random
using LinearAlgebra

# reciprocal + self ONLY (no CoulombBuckingham).
function pme_recip_eval(atoms_cpu, prms)
    a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
    p = arrtyp_prms(CuArray, prms)
    pme = Calculators.PMEInfo(a.info[:lattice], p)
    reset!(a)
    Calculators.calculate!(pme, a, p)
    return (Array(a.info[:potential_energy])[1],
            Array(a.arrays[:forces]),
            Array(a.info[:virial]))
end
pme_recip_E(atoms_cpu, prms) = pme_recip_eval(atoms_cpu, prms)[1]

# full CoulombBuckingham (real) + PME (reciprocal + self).
function cb_pme_eval(atoms_cpu, prms)
    a = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
    p = arrtyp_prms(CuArray, prms)
    nl = neighborlist(a, p[:rc], 0.0)
    pme = Calculators.PMEInfo(a.info[:lattice], p)
    reset!(a)
    Calculators.calculate!(Calculators.CoulombBuckingham(), a, nl, p)
    Calculators.calculate!(pme, a, p)
    return (Array(a.info[:potential_energy])[1], Array(a.arrays[:forces]))
end
cb_pme_E(atoms_cpu, prms) = cb_pme_eval(atoms_cpu, prms)[1]

# FFT grid size PME would pick for `cell` (mirrors PME.jl construction), used
# to guard the strain FD against a grid-size jump between ±h.
function pme_fftsize(cell, prms)
    c = Array(cell)
    sides = (norm(c[1, :]), norm(c[2, :]), norm(c[3, :]))
    return max.(ceil.(Int, sides ./ prms[:pme_meshsize]), 4)   # 4 = BSPORDER (Issue #53)
end

@testset "PME (GPU)" begin

    @testset "PMEInfo construction + calculate! does not throw" begin
        atoms_cpu = make_nacl_8()
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
        @test_nowarn cb_pme_E(atoms_cpu, prms)
    end

    @testset "PME reciprocal force = -∇E (self-consistency, FD)" begin
        Random.seed!(5)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        _, F_ana, _ = pme_recip_eval(atoms_cpu, prms)
        @test maximum(abs, F_ana) > 1e-2

        ilattice = inv(Array(atoms_cpu.info[:lattice]))
        h = 1e-5
        for i in 1:length(atoms_cpu), axis in 1:3
            δcart = zeros(FMD.FPPREC, 3); δcart[axis] = h
            δfrac = vec(reshape(δcart, 1, 3) * ilattice)
            ap = deepcopy_atoms(atoms_cpu); ap.arrays[:pos][i, :] .+= δfrac
            am = deepcopy_atoms(atoms_cpu); am.arrays[:pos][i, :] .-= δfrac
            num = -(pme_recip_E(ap, prms) - pme_recip_E(am, prms)) / (2h)
            (abs(F_ana[i, axis]) < 1e-5 && abs(num) < 1e-5) && continue
            @test isapprox(F_ana[i, axis], num; rtol = 1e-4, atol = 1e-6)
        end
    end

    # Shared strain-FD driver for the virial self-consistency checks (used by
    # the neutral and charged-cell testsets below).
    function assert_pme_virial_fd(atoms_cpu, prms; h = 1e-5, rtol = 2e-4, atol = 1e-6)
        _, _, W = pme_recip_eval(atoms_cpu, prms)
        lat0 = copy(Array(atoms_cpu.info[:lattice]))
        sd0  = copy(Array(atoms_cpu.arrays[:shelldisp]))
        voigt = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
        for (k, (α, β)) in enumerate(voigt)
            D⁺ = Matrix{FMD.FPPREC}(I, 3, 3); D⁺[α, β] += h
            D⁻ = Matrix{FMD.FPPREC}(I, 3, 3); D⁻[α, β] -= h
            # guard: identical grid at ±h (else the FD is meaningless)
            @assert pme_fftsize(lat0 * D⁺', prms) == pme_fftsize(lat0 * D⁻', prms)
            mk(D) = begin
                ac = deepcopy_atoms(atoms_cpu)
                ac.info[:lattice]     .= lat0 * D'
                ac.arrays[:shelldisp] .= sd0  * D'
                pme_recip_E(ac, prms)
            end
            num = -(mk(D⁺) - mk(D⁻)) / (2h)
            (abs(W[k]) < 1e-6 && abs(num) < 1e-6) && continue
            @test isapprox(W[k], num; rtol = rtol, atol = atol)
        end
    end

    @testset "PME reciprocal virial = -∂E/∂ε (self-consistency, FD)" begin
        Random.seed!(6)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        # 0.137 Å keeps side/mesh well away from an integer, so the FFT grid
        # does not jump between ±h.  (Hitting the ceil boundary corrupts the
        # strain FD — that artifact, not a PME stress error, is what an
        # earlier draft mistook for a ~0.6% discrepancy.)
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.137 / UNITS[:Å]))
        assert_pme_virial_fd(atoms_cpu, prms)
    end

    @testset "PME reciprocal virial, charged cell (Issue #15, FD)" begin
        Random.seed!(8)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        # Non-neutral cell, both charge sets: Σqc = +0.4, Σqs = +0.4 ⇒
        # Q = +0.8 e.  The charged-cell correction shifts each diagonal
        # virial component by E_corr = -πQ²/(2α²V) ≈ -9e-3 — well above the
        # FD tolerance, so an inconsistent (energy-only / wrong-sign)
        # correction fails here.
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                qc = FMD.FPPREC[+1.0, -0.9],
                qs = FMD.FPPREC[-0.2, +0.3],
                pme_meshsize = FMD.FPPREC(0.137 / UNITS[:Å]))
        assert_pme_virial_fd(atoms_cpu, prms)
    end

    @testset "PME + CoulombBuckingham force = -∇E (total, FD)" begin
        Random.seed!(7)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        _, F_ana = cb_pme_eval(atoms_cpu, prms)
        @test maximum(abs, F_ana) > 1e-2
        @test maximum(abs, vec(sum(F_ana, dims = 1))) < 1e-4   # Newton III (B-spline/FFT)

        ilattice = inv(Array(atoms_cpu.info[:lattice]))
        h = 2e-4
        for i in 1:length(atoms_cpu), axis in 1:3
            δcart = zeros(FMD.FPPREC, 3); δcart[axis] = h
            δfrac = vec(reshape(δcart, 1, 3) * ilattice)
            ap = deepcopy_atoms(atoms_cpu); ap.arrays[:pos][i, :] .+= δfrac
            am = deepcopy_atoms(atoms_cpu); am.arrays[:pos][i, :] .-= δfrac
            num = -(cb_pme_E(ap, prms) - cb_pme_E(am, prms)) / (2h)
            (abs(F_ana[i, axis]) < 1e-4 && abs(num) < 1e-4) && continue
            @test isapprox(F_ana[i, axis], num; rtol = 1e-3, atol = 1e-4)
        end
    end

    # GPU twin of the CPU OnlyShellForce check (Issue #157). The CPU test
    # asserts bit-identity; here the tolerance band is the usual GPU one,
    # because kernel_spread!'s atomic accumulation fixes no summation
    # order and the two paths build the charge grid in separate launches.
    # The shell forces must also match the CPU reference, which is what
    # rules out a backend-specific divergence in the skipped work.
    @testset "OnlyShellForce reciprocal path, CPU ↔ GPU (Issue #157)" begin
        Random.seed!(157)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.02 .* randn(size(atoms_cpu.arrays[:pos]))
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]))
        atoms_cpu.arrays[:shelldisp] .= 0.05 .* randn(size(atoms_cpu.arrays[:shelldisp]))

        ref = deepcopy_atoms(atoms_cpu)
        reset!(ref)
        Calculators.calculate!(Calculators.PMEInfo(ref.info[:lattice], prms), ref, prms)
        Fs_ref = copy(ref.arrays[:shellforces])
        @test maximum(abs, Fs_ref) > 1e-4                          # non-vacuous

        p = arrtyp_prms(CuArray, prms)
        gfull = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        reset!(gfull)
        pme = Calculators.PMEInfo(gfull.info[:lattice], p)
        Calculators.calculate!(pme, gfull, p)

        gonly = Cores.change_arrtyp(deepcopy_atoms(atoms_cpu), CuArray)
        reset!(gonly)
        Calculators.calculate!(pme, Calculators.OnlyShellForce(), gonly, p)

        scale = max(one(FMD.FPPREC), maximum(abs, Fs_ref))
        @test maximum(abs, Array(gonly.arrays[:shellforces]) .-
                           Array(gfull.arrays[:shellforces])) < 1e-9 * scale
        @test maximum(abs, Array(gonly.arrays[:shellforces]) .- Fs_ref) < 1e-9 * scale
        # Contract: energy and virial stay untouched on the device too.
        @test Array(gonly.info[:potential_energy]) == zeros(FMD.FPPREC, 1)
        @test Array(gonly.info[:virial]) == zeros(FMD.FPPREC, 6)
    end

    @testset "z-slab solve ≡ replicated solve at P = 1 (CUDA, Issue #193)" begin
        # GPU twin of the CPU force_dd check: exercises the CUFFT region
        # plans (2D rfft/brfft over dims (1,2), in-place 1D fft/bfft over
        # dim 3 — the "advanced layout" path) and the device transpose
        # pack/unpack kernels. Serial collectives are device copies.
        d = Cores.SerialDomain()
        cell = CuArray(FMD.FPPREC.([12.0 0 0; 0 10.0 0; 0 0 11.0]))
        prms = (; alpha = FMD.FPPREC(0.35), pme_meshsize = FMD.FPPREC(0.9))
        pme  = Calculators.PMEInfo(cell, prms)
        pmed = Calculators.PMEInfo(cell, prms, d; force_dd = true)
        @test pmed.dd !== nothing
        @test Array(pmed.dd.expfactor_dd) == Array(pme.expfactor)
        @test Array(pmed.dd.presfactor_dd) == Array(pme.presfactor)

        Random.seed!(193)
        grid_h = FMD.FPPREC.(randn(pme.fftsize) .+ 0.1)   # nonzero mean = charged
        ge_ref, v_ref = Calculators._pme_solve(pme, CuArray(grid_h))
        ge_dd, v_dd = Calculators._pme_solve_dd!(pmed, d, CuArray(grid_h))
        gescale = maximum(abs, Array(ge_ref))
        @test maximum(abs, Array(ge_dd) .- Array(ge_ref)) < 1e-12 * gescale
        @test maximum(abs, v_dd .- v_ref) < 1e-12 * maximum(abs, v_ref)

        gc_dd = Calculators._pme_convolve_dd!(pmed, d, CuArray(grid_h))
        @test Array(gc_dd) == Array(ge_dd)   # structural sharing within the dd path

        # cellupdate! with a device cell: the hook must take the backend
        # from the slab buffers — cellupdate! rebinds `cell` to its host
        # copy before the hook runs (regression: CPU-launch on device
        # factors).
        cell2 = CuArray(FMD.FPPREC.(1.02 .* Array(cell)))
        Calculators.cellupdate!(pmed, cell2, prms)
        pme2 = Calculators.PMEInfo(cell2, prms)
        @test Array(pmed.dd.expfactor_dd) == Array(pme2.expfactor)
    end
end
