# test/gpu/test_relax_gpu.jl
#
# The FIRE relaxation driver on the GPU backend (Issues #209 / #212).
#
# relax! is backend-generic by construction: it shares the whole force stack
# with npt! (Driver.jl), and its own arithmetic is broadcasts plus three
# reduced scalars. So this pins the parity rather than a separate
# implementation — but "should be generic" is not evidence, and the cell
# path in particular touches the lattice through `conv(arrtyp, ...)` and
# rebuilds the PME accelerator mid-run, neither of which the CPU tests can
# exercise on device arrays.
#
# Why the comparison is not bitwise: GPU atomics reassociate the force sums,
# so two GPU runs of the same input are not bitwise identical either
# (measured noise floor ~1e-16, PR #108). FIRE amplifies that through its
# branch on the sign of P — a step taken 1 ulp differently can flip an
# uphill test near the end and change the iteration count by one.
#
# The two testsets below therefore assert DIFFERENT things, on purpose. With
# a fixed cell the trajectories do not diverge in practice and the structures
# match to ~1e-15, so that is pinned near the reassociation floor (1e-10),
# not at the much looser ftol ball. With the cell relaxing they
# do diverge — the lattice moves through host-side arithmetic driven by a
# device-reduced virial — and the runs stop at different steps inside the
# same ftol ball. Demanding tight agreement there would be asserting the
# wrong invariant, so what is asserted instead is the quantity the app
# exists to produce (the equilibrium lattice constant, bounded by stol) plus
# an independent check that the GPU structure meets the criterion at all.

using Test
using CUDA
using Random
using LinearAlgebra
using FMD.Apps: relax!

@testset "relax! CPU/GPU parity" begin
    function run_backend(backend, atoms, dir; kwargs...)
        base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                           rc_factor = 0.45, with_shell = true, buckingham = true)
        params = (; base...,
                  backend = backend,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  pme_meshsize = FMD.FPPREC(0.25 / UNITS[:Å]),
                  skin = FMD.FPPREC(0.1),
                  cgtol = FMD.FPPREC(1e-8), cgmaxiter = 40,
                  ftol = FMD.FPPREC(1e-4),
                  maxsteps = 200,
                  dt0 = FMD.FPPREC(1.0 / UNITS[:fs]),
                  dt_max = FMD.FPPREC(10.0 / UNITS[:fs]),
                  dmax = FMD.FPPREC(0.1 / UNITS[:Å]),
                  output_interval = 10000,
                  output_dir = dir)
        out, converged = relax!(deepcopy_atoms(atoms), (; params..., kwargs...))
        ids = Array(out.arrays[:id])
        byid(x) = Array(x)[sortperm(ids), :]
        return (; pos = byid(out.arrays[:pos]),
                shelldisp = byid(out.arrays[:shelldisp]),
                lattice = Matrix(Array(out.info[:lattice])),
                converged = converged, n = length(ids))
    end

    @testset "atoms only (cell_dof = :none)" begin
        Random.seed!(31)
        atoms = make_nacl_64()
        atoms.arrays[:pos] .+= (0.05 / UNITS[:Å]) .*
            randn(FMD.FPPREC, size(atoms.arrays[:pos])) ./
            norm(Array(atoms.info[:lattice])[1, :])

        cpu = mktempdir(d -> run_backend(:cpu, atoms, d))
        gpu = mktempdir(d -> run_backend(:cuda, atoms, d))

        @test gpu.n == cpu.n == 64
        @test gpu.converged == cpu.converged
        @test cpu.converged                       # non-vacuous: it really relaxed
        dpos = maximum(abs, gpu.pos .- cpu.pos)
        dsd = maximum(abs, gpu.shelldisp .- cpu.shelldisp)
        @info "relax_gpu deviations vs CPU (atoms only)" dpos dsd
        # Fractional coordinates. Measured 1.5e-15 / 6.6e-17: with a fixed
        # cell the two trajectories do not diverge at all, so this is pinned
        # near the reassociation floor rather than at the ftol ball (~5e-5
        # fractional). A loose bound here would pass a real regression that
        # merely degraded agreement to 1e-9; five orders of margin over the
        # measurement is enough to absorb a different GPU or CUDA version.
        @test dpos < 1e-10
        @test dsd < 1e-10
        # Fixed cell on both backends.
        @test gpu.lattice == cpu.lattice
    end

    @testset "cell relaxation (cell_dof = :iso)" begin
        # Starts off-equilibrium so the cell genuinely moves: the lattice
        # update runs through `conv(arrtyp, I + deps)` on the device, and the
        # edge-length PME rebuild can fire mid-run.
        atoms = make_nacl_64(; a_bohr = 5.45 / UNITS[:Å])
        kw = (; cell_dof = :iso, stol = FMD.FPPREC(1e-7),
              target_stress = zeros(FMD.FPPREC, 6), emax = FMD.FPPREC(1e-3),
              maxsteps = 400)

        cpu = mktempdir(d -> run_backend(:cpu, atoms, d; kw...))
        gpu = mktempdir(d -> run_backend(:cuda, atoms, d; kw...))

        @test gpu.converged == cpu.converged
        @test cpu.converged
        # The cell actually moved (otherwise this is the fixed-cell test again).
        @test !isapprox(cpu.lattice, Matrix(Array(atoms.info[:lattice])); rtol = 1e-6)

        # Unlike the fixed-cell case above, this one is NOT near-bitwise, and
        # asserting that it should be would be asserting the wrong thing.
        # The lattice is updated through host-side 3x3 arithmetic driven by
        # `stress`, which comes off a device reduction — so the two runs
        # diverge at the reassociation level and FIRE amplifies it through
        # its branch on the sign of P and its dt schedule. Measured: the two
        # stop at different steps (55 vs 52) with final max|F| of 7.4e-5 and
        # 3.1e-6, i.e. at different points INSIDE the same ftol ball.
        #
        # What is physically meaningful, and what this app exists to produce,
        # is the equilibrium lattice constant. Its cross-backend spread is
        # bounded by the stress criterion itself, so the bound is DERIVED
        # rather than chosen: each run stops within Δε ≲ stol/C₁₁ of the true
        # zero-stress strain, so two runs differ by at most twice that. C₁₁
        # is taken as 3B with B ≈ 25 GPa for this fixture — a lower bound on
        # the stiffness, hence an upper bound on the spread. Measured: 2.7e-5
        # against a bound of 7.8e-5.
        C11 = 3 * 25.0 / UNITS[:GPa]          # Ha/Bohr³, deliberately soft
        bound = 2 * kw.stol / C11
        a_cpu = norm(cpu.lattice[1, :])
        a_gpu = norm(gpu.lattice[1, :])
        dlat = maximum(abs, gpu.lattice .- cpu.lattice)
        dpos = maximum(abs, gpu.pos .- cpu.pos)
        da = abs(a_gpu - a_cpu) / a_cpu
        @info "relax_gpu deviations vs CPU (cell_dof = :iso)" dlat dpos da bound
        @test da < bound

        # The atoms are only pinned to the ftol ball, so instead of comparing
        # them to the CPU run, check the GPU structure against the criterion
        # directly — re-evaluated on the host, with the shell CG driven far
        # tighter than the run used.
        probe = deepcopy_atoms(atoms)
        # gpu.pos is id-sorted; the fixture's rows are too, but say so rather
        # than depend on it silently.
        @assert issorted(Array(atoms.arrays[:id]))
        probe.arrays[:pos] .= gpu.pos
        probe.info[:lattice] = FMD.FPPREC.(gpu.lattice)
        probe.arrays[:shelldisp] .= gpu.shelldisp
        reset!(probe)
        pbase = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-8,
                            rc_factor = 0.45, with_shell = true, buckingham = true)
        Calculators.calculate!(Calculators.CoreShell(), probe,
                               (; pbase..., cgtol = FMD.FPPREC(1e-10),
                                  cgmaxiter = 60, skin = FMD.FPPREC(0.0)))
        @test maximum(abs, probe.arrays[:forces]) < 1e-4
        # The app's own cell criterion under :iso is the projected residual,
        # |tr σ|/3 — not every component. (The deviatoric components are held
        # by the constraint; requiring them to vanish would be requiring the
        # constraint to do nothing.)
        sig = stress(probe, false)
        @test abs(sum(sig[1:3])) / 3 < 10 * kw.stol
        # Separately: the full tensor stays small, which is a statement about
        # PME-vs-Ewald stress agreement (~0.03 GPa) rather than about stol.
        @test maximum(abs, sig) < 1e-6

        # :iso must hold the cell cubic on the device too — exactly, since
        # the hydrostatic increment is a multiple of I.
        L = gpu.lattice
        @test maximum(abs, L .- Diagonal(diag(L))) == 0
        @test L[1, 1] == L[2, 2] == L[3, 3]
    end
end
