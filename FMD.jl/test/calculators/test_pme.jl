# test/calculators/test_pme.jl
#
# CPU twin of test/gpu/test_pme_gpu.jl (thin version).  PME is
# backend-generic since #51, so the FD self-consistency of the reciprocal
# force / virial and the charged-cell correction are pinned on the CPU in
# the default profile; the GPU file keeps the same checks on CuArray under
# FMD_TEST_GPU=1.  Fixture design notes (asymmetric cells, mesh choice away
# from a ceil boundary) are documented in the GPU twin.

using Test
using Random
using LinearAlgebra

# reciprocal + self ONLY (no CoulombBuckingham), on the CPU backend.
function _pmecpu_recip_eval(atoms_cpu, prms)
    a = deepcopy_atoms(atoms_cpu)
    pme = Calculators.PMEInfo(a.info[:lattice], prms)
    reset!(a)
    Calculators.calculate!(pme, a, prms)
    return (a.info[:potential_energy][1], copy(a.arrays[:forces]),
            copy(a.info[:virial]))
end
_pmecpu_recip_E(atoms_cpu, prms) = _pmecpu_recip_eval(atoms_cpu, prms)[1]

# FFT grid size PME would pick for `cell` (via the production sizing
# function — a deliberate call, not a re-derivation: the tests here guard
# the strain FD against a grid-size jump between ±h, and the guard must
# see the same grid the solver builds, smooth rounding included).
function _pmecpu_fftsize(cell, prms)
    c = Array(cell)
    sides = (norm(c[1, :]), norm(c[2, :]), norm(c[3, :]))
    return Calculators.pme_fftsize(sides..., prms[:pme_meshsize])
end

@testset "PME (CPU)" begin

    @testset "PMEInfo construction + calculate! on the CPU backend" begin
        atoms_cpu = make_nacl_8()
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
        @test_nowarn _pmecpu_recip_E(atoms_cpu, prms)
    end

    @testset "fftsize lower bound = BSPORDER (Issue #53)" begin
        # The spread/gather wrap is single-period: the smallest wrapped
        # index is fftsize - (BSPORDER-1).  The old lower bound BSPORDER-1
        # (= 3) therefore made grid index 0 reachable — an OOB write hidden
        # by `inbounds = true` (atom 1 sits at frac (0,0,0), which hits it).
        Random.seed!(53)
        atoms_cpu = make_nacl_8()
        # Perturb so forces are nonzero (the perfect rocksalt lattice has
        # F ≡ 0 by symmetry, which would make the Newton III check vacuous).
        # 0.03 keeps atoms near frac (0,0,0) at grid index 0, preserving
        # the old code's OOB reach.
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        side = norm(Array(atoms_cpu.info[:lattice])[1, :])
        # meshsize > side/3 ⇒ unclamped ceil(side/meshsize) ≤ 3 ⇒ the
        # lower bound is what determines the grid.
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(side / 2.5))
        pme = Calculators.PMEInfo(atoms_cpu.info[:lattice], prms)
        @test pme.fftsize == (4, 4, 4)
        # Exercise spread/gather on the minimal grid; run the suite with
        # --check-bounds=yes to surface any OOB despite inbounds=true.
        # The correctness pin is F = -∇E of the discretized model (exact
        # at any grid size) — NOT Newton III: the smooth-PME net force is
        # an interpolation artifact that only vanishes with grid refinement
        # (~22% of max|F| on this minimal grid).
        E, F_ana, _ = _pmecpu_recip_eval(atoms_cpu, prms)
        @test isfinite(E)
        @test maximum(abs, F_ana) > 1e-3   # non-trivial forces
        ilattice = inv(Array(atoms_cpu.info[:lattice]))
        h = 1e-5
        for i in 1:length(atoms_cpu), axis in 1:3
            δcart = zeros(FMD.FPPREC, 3); δcart[axis] = h
            δfrac = vec(reshape(δcart, 1, 3) * ilattice)
            ap = deepcopy_atoms(atoms_cpu); ap.arrays[:pos][i, :] .+= δfrac
            am = deepcopy_atoms(atoms_cpu); am.arrays[:pos][i, :] .-= δfrac
            num = -(_pmecpu_recip_E(ap, prms) - _pmecpu_recip_E(am, prms)) / (2h)
            (abs(F_ana[i, axis]) < 1e-5 && abs(num) < 1e-5) && continue
            @test isapprox(F_ana[i, axis], num; rtol = 1e-4, atol = 1e-6)
        end
    end

    @testset "PME reciprocal force = -∇E (self-consistency, FD)" begin
        Random.seed!(5)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.15 / UNITS[:Å]))

        _, F_ana, _ = _pmecpu_recip_eval(atoms_cpu, prms)
        @test maximum(abs, F_ana) > 1e-2
        @test maximum(abs, vec(sum(F_ana, dims = 1))) < 1e-4   # Newton III (B-spline/FFT)

        ilattice = inv(Array(atoms_cpu.info[:lattice]))
        h = 1e-5
        for i in 1:length(atoms_cpu), axis in 1:3
            δcart = zeros(FMD.FPPREC, 3); δcart[axis] = h
            δfrac = vec(reshape(δcart, 1, 3) * ilattice)
            ap = deepcopy_atoms(atoms_cpu); ap.arrays[:pos][i, :] .+= δfrac
            am = deepcopy_atoms(atoms_cpu); am.arrays[:pos][i, :] .-= δfrac
            num = -(_pmecpu_recip_E(ap, prms) - _pmecpu_recip_E(am, prms)) / (2h)
            (abs(F_ana[i, axis]) < 1e-5 && abs(num) < 1e-5) && continue
            @test isapprox(F_ana[i, axis], num; rtol = 1e-4, atol = 1e-6)
        end
    end

    # Shared strain-FD driver for the virial self-consistency checks.
    function _assert_pmecpu_virial_fd(atoms_cpu, prms; h = 1e-5, rtol = 2e-4, atol = 1e-6)
        _, _, W = _pmecpu_recip_eval(atoms_cpu, prms)
        lat0 = copy(Array(atoms_cpu.info[:lattice]))
        sd0  = copy(Array(atoms_cpu.arrays[:shelldisp]))
        voigt = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
        for (k, (α, β)) in enumerate(voigt)
            D⁺ = Matrix{FMD.FPPREC}(I, 3, 3); D⁺[α, β] += h
            D⁻ = Matrix{FMD.FPPREC}(I, 3, 3); D⁻[α, β] -= h
            # guard: identical grid at ±h (else the FD is meaningless)
            @assert _pmecpu_fftsize(lat0 * D⁺', prms) == _pmecpu_fftsize(lat0 * D⁻', prms)
            mk(D) = begin
                ac = deepcopy_atoms(atoms_cpu)
                ac.info[:lattice]     .= lat0 * D'
                ac.arrays[:shelldisp] .= sd0  * D'
                _pmecpu_recip_E(ac, prms)
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
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45)...,
                pme_meshsize = FMD.FPPREC(0.137 / UNITS[:Å]))
        _assert_pmecpu_virial_fd(atoms_cpu, prms)
    end

    @testset "PME reciprocal virial, charged cell (Issue #15, FD)" begin
        Random.seed!(8)
        atoms_cpu = make_nacl_8()
        atoms_cpu.arrays[:pos] .+= 0.03 .* randn(size(atoms_cpu.arrays[:pos]))
        prms = (; nacl_params(atoms_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                qc = FMD.FPPREC[+1.0, -0.9],
                qs = FMD.FPPREC[-0.2, +0.3],
                pme_meshsize = FMD.FPPREC(0.137 / UNITS[:Å]))
        _assert_pmecpu_virial_fd(atoms_cpu, prms)
    end

    # The Voigt-6 contraction against presfactor is written as one gemv on
    # a (M, 6) reshape (Issue #153). The reshape is where a mistake would
    # be silent — a wrong stride mixes stress components without changing
    # any magnitude — so it is checked against the straightforward
    # component-by-component sum on a random weight grid. The random grid
    # matters: on a physical |Q̂|² the six components are correlated by
    # symmetry, which would let some permutations through.
    @testset "reciprocal virial contraction = 6 slice sums (Issue #153)" begin
        Random.seed!(153)
        atoms = make_nacl_8()
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]))
        pme = Calculators.PMEInfo(atoms.info[:lattice], prms)

        w = randn(FMD.FPPREC, pme.rfftsize...)
        ref = FMD.FPPREC[sum(w .* pme.presfactor[:, :, :, d]) for d in 1:6]
        got = Calculators._pme_presfactor_contract(pme, w)

        @test length(got) == 6
        @test eltype(got) === FMD.FPPREC
        # PMEInfo's fields are untyped, so without the return assertion
        # this (and hvp!'s return type through it) infers as Any.
        @test (@inferred Calculators._pme_presfactor_contract(pme, w)) == got
        scale = maximum(abs, ref)
        @test scale > 1e-8                                  # non-vacuous
        # Every component distinct by a wide margin, so any permutation of
        # the six (the failure mode a wrong reshape produces) is caught.
        @test minimum(abs(ref[i] - ref[j]) for i in 1:6 for j in (i+1):6) > 1e-3 * scale
        @test maximum(abs, got .- ref) < 1e-12 * scale
    end

    # The shell-relaxation inner loop asks the Coulomb solver for shell
    # forces only (Issue #157). The CG reads nothing but
    # arrays[:shellforces], so the property that keeps every :scf
    # trajectory unchanged is bit-identity of THAT array against the full
    # method — everything else the shell-only variant skips is discarded
    # by the next clear!. Bit-identity (not a tolerance) is the right
    # assertion because the variant reuses the same convolution
    # expression, the same gather! call and the same intra kernel; any
    # drift here means the two paths diverged structurally.
    @testset "OnlyShellForce reciprocal path (Issue #157)" begin
        Random.seed!(157)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(size(atoms.arrays[:pos]))
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(size(atoms.arrays[:shelldisp]))
        pme = Calculators.PMEInfo(atoms.info[:lattice], prms)

        full = deepcopy_atoms(atoms)
        reset!(full)
        Calculators.calculate!(pme, full, prms)

        onlyshell = deepcopy_atoms(atoms)
        reset!(onlyshell)
        Calculators.calculate!(pme, Calculators.OnlyShellForce(), onlyshell, prms)

        @test maximum(abs, full.arrays[:shellforces]) > 1e-4        # non-vacuous
        @test onlyshell.arrays[:shellforces] == full.arrays[:shellforces]

        # The OnlyShellForce contract: energy and virial are NOT written.
        # (arrays[:forces] is deliberately excluded — kernel_subtractintraatom!
        # writes both force arrays and only its shell half is meaningful,
        # exactly as the Buckingham OnlyShellForce leaves core forces alone.)
        @test onlyshell.info[:potential_energy] == zero(onlyshell.info[:potential_energy])
        @test onlyshell.info[:virial] == zero(onlyshell.info[:virial])
        @test abs(full.info[:potential_energy][1]) > 1e-4           # non-vacuous

        # Solvers without a specialization fall back to the full method,
        # which is a superset — the shell forces must still agree.
        ew = Calculators.EwaldInfo(atoms.info[:lattice], prms)
        ewfull = deepcopy_atoms(atoms); reset!(ewfull)
        Calculators.calculate!(ew, ewfull, prms)
        ewonly = deepcopy_atoms(atoms); reset!(ewonly)
        Calculators.calculate!(ew, Calculators.OnlyShellForce(), ewonly, prms)
        @test ewonly.arrays[:shellforces] == ewfull.arrays[:shellforces]
    end

    @testset "integer-translation invariance beyond one box (wrap fix)" begin
        # kernel_spread!'s single-period ifelse cannot fold fractional
        # coordinates more than one box outside [0,1); calculate! must
        # wrap them itself. A whole-lattice-vector translation of > 1 box
        # is a physical no-op, so E/F/virial must be identical (up to the
        # frac*n rounding of the wrap) — before the fix this silently
        # wrote out of grid bounds.
        Random.seed!(9)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        prms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                              rc_factor = 0.45, with_shell = true)...,
                pme_meshsize = FMD.FPPREC(0.137 / UNITS[:Å]))
        pme = Calculators.PMEInfo(atoms.info[:lattice], prms)

        clear!(atoms)
        Calculators.calculate!(pme, atoms, prms)
        E0 = Array(atoms.info[:potential_energy])[1]
        V0 = collect(Array(atoms.info[:virial]))
        F0 = copy(Array(atoms.arrays[:forces]))

        shifted = deepcopy_atoms(atoms)
        shifted.arrays[:pos] .+= reshape(FMD.FPPREC[2.0, -3.0, 1.0], 1, 3)
        clear!(shifted)
        fill!(shifted.arrays[:forces], 0)
        fill!(shifted.arrays[:shellforces], 0)
        Calculators.calculate!(pme, shifted, prms)
        E1 = Array(shifted.info[:potential_energy])[1]
        V1 = collect(Array(shifted.info[:virial]))
        F1 = Array(shifted.arrays[:forces])

        @test abs(E1 - E0) < 1e-10 * abs(E0)
        @test maximum(abs, V1 .- V0) < 1e-10 * max(1.0, maximum(abs, V0))
        @test maximum(abs, F1 .- F0) < 1e-10 * max(1.0, maximum(abs, F0))
    end

    @testset "_slab_partition (Issue #193)" begin
        for (n, P) in ((7, 3), (12, 4), (4, 6), (61, 5), (1, 1))
            counts, offs = Calculators._slab_partition(n, P)
            @test length(counts) == P && length(offs) == P
            @test sum(counts) == n
            @test all(counts .>= 0)
            # Contiguous tiling: offsets are the running sum of counts.
            @test offs == cumsum([0; counts[1:end-1]])
            # Balanced to within one item.
            @test maximum(counts) - minimum(counts) <= 1
        end
        # P > n degenerate case: trailing ranks own zero planes.
        counts, _ = Calculators._slab_partition(4, 6)
        @test counts == [1, 1, 1, 1, 0, 0]
    end

    @testset "z-slab solve ≡ replicated solve at P = 1 (Issue #193)" begin
        # force_dd builds the full slab machinery (partition, 2D+1D plans,
        # transpose pack/unpack, ky-slab factors) on a SerialDomain, where
        # the collectives degrade to single-segment copies — the entire
        # distributed transform runs, only the MPI wire is absent. The 2D+1D
        # split may sum in a different order than the replicated 3D rFFT, so
        # the comparison is scale-relative, not bitwise.
        d = Cores.SerialDomain()
        # Asymmetric cell, all sides distinct, odd/even grid mix.
        cell = FMD.FPPREC.([12.0 0 0; 0 10.0 0; 0 0 11.0])
        prms = (; alpha = FMD.FPPREC(0.35), pme_meshsize = FMD.FPPREC(0.9))
        pme  = Calculators.PMEInfo(cell, prms)
        pmed = Calculators.PMEInfo(cell, prms, d; force_dd = true)
        @test pmed.dd !== nothing
        @test Calculators.PMEInfo(cell, prms, d).dd === nothing   # opt-in only

        # ky-slab factors at P = 1 cover the whole grid: exact equality
        # (same kernel, same global indices).
        @test Array(pmed.dd.expfactor_dd) == Array(pme.expfactor)
        @test Array(pmed.dd.bsp2factor_dd) == Array(pme.bsp2factor)
        @test Array(pmed.dd.presfactor_dd) == Array(pme.presfactor)

        Random.seed!(193)
        # Non-neutral grid (nonzero mean = charged cell): the (0,0,0) mode
        # zeroing must behave identically in both paths.
        grid = FMD.FPPREC.(randn(pme.fftsize) .+ 0.1)
        ge_ref, v_ref = Calculators._pme_solve(pme, copy(grid))
        ge_dd, v_dd = Calculators._pme_solve_dd!(pmed, d, copy(grid))
        gescale = maximum(abs, Array(ge_ref))
        @test maximum(abs, Array(ge_dd) .- Array(ge_ref)) < 1e-12 * gescale
        @test maximum(abs, v_dd .- v_ref) < 1e-12 * maximum(abs, v_ref)

        # Shell-only path: same operator, and within the dd path the full
        # solve's grid is bit-identical by construction (structural sharing
        # of _pme_convolve_dd!).
        gc_ref = Calculators._pme_convolve(pme, copy(grid))
        gc_dd = Calculators._pme_convolve_dd!(pmed, d, copy(grid))
        @test maximum(abs, Array(gc_dd) .- Array(gc_ref)) < 1e-12 * gescale
        @test Array(gc_dd) == Array(ge_dd)

        # Domain guard: the slab state is bound to the domain it was built
        # for — anything else must refuse before touching a collective.
        @test_throws ErrorException Calculators._pme_solve_dd!(pmed, nothing, copy(grid))

        # calculate! wiring of the slab branch (Issue #195): the checks
        # above call _pme_solve_dd! directly, but the production entry is
        # calculate!'s dd branch — pin the full physical outputs (E/V/F/S
        # incl. intra-atom correction and charged-cell path) against the
        # replicated calculate! on the same frame, plus the OnlyShellForce
        # branch (bit-identical to the dd full pass by structural sharing).
        Random.seed!(195)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.03 .* randn(size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(size(atoms.arrays[:shelldisp]))
        aprms = (; nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                               rc_factor = 0.45, with_shell = true)...,
                 pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
        ref = deepcopy_atoms(atoms); reset!(ref)
        Calculators.calculate!(Calculators.PMEInfo(ref.info[:lattice], aprms), ref, aprms)
        dda = deepcopy_atoms(atoms); reset!(dda)
        pme_dda = Calculators.PMEInfo(dda.info[:lattice], aprms, d; force_dd = true)
        @test pme_dda.dd !== nothing
        Calculators.calculate!(pme_dda, dda, aprms)
        E0 = Array(ref.info[:potential_energy])[1]
        E1 = Array(dda.info[:potential_energy])[1]
        @test abs(E1 - E0) < 1e-12 * abs(E0)
        @test maximum(abs, Array(dda.info[:virial]) .- Array(ref.info[:virial])) <
              1e-12 * max(1.0, maximum(abs, Array(ref.info[:virial])))
        @test maximum(abs, Array(dda.arrays[:forces]) .- Array(ref.arrays[:forces])) <
              1e-12 * maximum(abs, Array(ref.arrays[:forces]))
        @test maximum(abs, Array(dda.arrays[:shellforces]) .- Array(ref.arrays[:shellforces])) <
              1e-12 * maximum(abs, Array(ref.arrays[:shellforces]))
        only = deepcopy_atoms(atoms); reset!(only)
        Calculators.calculate!(pme_dda, Calculators.OnlyShellForce(), only, aprms)
        @test Array(only.arrays[:shellforces]) == Array(dda.arrays[:shellforces])

        # cellupdate! refreshes the slab factors alongside the replicated
        # ones (the NPT per-step path).
        cell2 = FMD.FPPREC.(1.02 .* cell)
        pme2  = Calculators.PMEInfo(cell2, prms)
        Calculators.cellupdate!(pmed, cell2, prms)
        @test Array(pmed.dd.expfactor_dd) == Array(pme2.expfactor)
        ge2_ref, v2_ref = Calculators._pme_solve(pme2, copy(grid))
        ge2_dd, v2_dd = Calculators._pme_solve_dd!(pmed, d, copy(grid))
        @test maximum(abs, Array(ge2_dd) .- Array(ge2_ref)) <
              1e-12 * maximum(abs, Array(ge2_ref))
        @test maximum(abs, v2_dd .- v2_ref) < 1e-12 * maximum(abs, v2_ref)
    end
end
