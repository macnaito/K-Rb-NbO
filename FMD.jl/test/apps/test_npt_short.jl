# test/apps/test_npt_short.jl
#
# Short NPT-like dynamics smoke on the GPU.  Mirrors the production NPT
# pipeline (NPT.jl): atoms on CuArray, long-range electrostatics via
# PMEInfo (NOT EwaldInfo — the Ewald reciprocal kernel is CPU-only),
# CoreShell SCF on the GPU, and a velocity-Verlet half-step to confirm
# kinetic energy stays finite.

using Test
using Random
using CUDA
using LinearAlgebra

@testset "NPT short smoke (GPU + PME + CoreShell)" begin
    Random.seed!(0)

    atoms = make_nacl_64()
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    base = (; base...,
            efield = FMD.FPPREC[1e-3, 0, 0],
            pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))

    atoms_gpu = Cores.change_arrtyp(atoms, CuArray)
    atoms_gpu.arrays[:peratom_virial] = CuArray(zeros(FMD.FPPREC, length(atoms), 6))
    base_gpu = arrtyp_prms(CuArray, base)

    # Initialize velocities and zero out the COM velocity.
    randn!(atoms_gpu.arrays[:velo])
    @. atoms_gpu.arrays[:velo] *= 1e-4
    com = vec(sum(atoms_gpu.arrays[:velo] .* atoms_gpu.arrays[:masses], dims = 1)) ./
          sum(atoms_gpu.arrays[:masses])
    atoms_gpu.arrays[:velo] .-= reshape(com, 1, 3)

    com0 = vec(sum(atoms_gpu.arrays[:velo] .* atoms_gpu.arrays[:masses], dims = 1)) ./
           sum(atoms_gpu.arrays[:masses])
    @test all(abs.(com0) .< 1e-10)

    # CoreShell on GPU must use PMEInfo for the long-range part (not the
    # default 3-arg form, which would fall through to EwaldInfo /
    # the CPU-only Ewald reciprocal kernel).
    nl = neighborlist(atoms_gpu, base_gpu[:rc], base_gpu[:skin])
    pme = Calculators.PMEInfo(atoms_gpu.info[:lattice], base_gpu)
    Calculators.calculate!(Calculators.CoreShell(), atoms_gpu, nl, pme, base_gpu)
    @test all(isfinite, Array(atoms_gpu.arrays[:forces]))
    @test all(isfinite, Array(atoms_gpu.info[:potential_energy]))

    # Tiny velocity Verlet half-step — verify kinetic energy stays bounded.
    dt = FMD.FPPREC(0.1)
    K0 = kinetic_energy(atoms_gpu)
    @. atoms_gpu.arrays[:velo] += atoms_gpu.arrays[:forces] /
                                  atoms_gpu.arrays[:masses] * dt / 2
    K1 = kinetic_energy(atoms_gpu)
    @test isfinite(K0)
    @test isfinite(K1)
    @test K1 < 1e6 * (K0 + 1e-12)
end

@testset "production list builders on GPU (CellList + Sub, ghost format)" begin
    # Mirrors npt!'s actual builder sequence (CellList → parent list → Sub
    # derivation) on the device, which the naive-list smoke above does not
    # touch. The fixture keeps info[:species]; the CellList re-sort must
    # work with it present (regression for Issue #54).
    Random.seed!(1)
    a_cpu = make_nacl_64()
    # Rattle: on the ideal rocksalt every force is zero by symmetry and the
    # cross-path comparison below would be vacuous (and blind to pair loss
    # near rc). Displaced positions give |F| = O(1e-2) and a real CG run.
    a_cpu.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(a_cpu.arrays[:pos]))
    prms = (; nacl_params(a_cpu; alpha = 0.30, ewald_tolerance = 1e-6,
                          rc_factor = 0.45, with_shell = true)...,
            pme_meshsize = FMD.FPPREC(0.30 / UNITS[:Å]))
    p = arrtyp_prms(CuArray, prms)
    # Non-zero skin: with the fixture default skin = 0 the half-skin
    # criterion `dr < skin/2` is unsatisfiable and isreusable is always
    # false, which would make the rebuild-mechanics checks vacuous.
    sk = FMD.FPPREC(0.1)

    a = Cores.change_arrtyp(deepcopy_atoms(a_cpu), CuArray)
    cl, a, _ = CellList(a, p[:rc] * (1 + sk))
    nlist = neighborlist(a, cl, p[:rc], sk)
    sub = neighborlist(a, nlist, p[:rc] * FMD.FPPREC(0.6), sk)
    @test sub.ghosts === nlist.ghosts   # Sub derivation shares the parent Ghosts

    pme = Calculators.PMEInfo(a.info[:lattice], p)
    reset!(a)
    Calculators.calculate!(Calculators.CoreShell(), a, nlist, pme, p)
    F_cell = Array(a.arrays[:forces])
    @test all(isfinite, F_cell)

    # naive-path reference in the original atom order, mapped via :id (the
    # CellList re-sorts the atoms).
    b = Cores.change_arrtyp(deepcopy_atoms(a_cpu), CuArray)
    nlb = neighborlist(b, p[:rc], sk)
    pme_b = Calculators.PMEInfo(b.info[:lattice], p)
    reset!(b)
    Calculators.calculate!(Calculators.CoreShell(), b, nlb, pme_b, p)
    F_ref = Array(b.arrays[:forces])
    ids = Array(a.arrays[:id])
    # Shell-CG converged forces: iterate sequences differ in FP between the
    # two list layouts, so the agreement scale is set by cgtol, not eps.
    @test maximum(abs, F_ref) > 1e-3   # non-vacuous (rattled fixture)
    @test maximum(abs, F_cell .- F_ref[ids, :]) < 1e-4 * max(one(FMD.FPPREC), maximum(abs, F_ref))

    # Sub-filter content on the device: must equal a directly built list at
    # the child cutoff (symmetry-independent, catches stencil/ghost-shell
    # truncation that the force comparison could miss).
    @test ghost_pairmap(a, sub) ==
          ghost_pairmap(a, neighborlist(a, p[:rc] * FMD.FPPREC(0.6), sk))

    # Rebuild mechanics: displacing an atom beyond skin/2 must invalidate
    # the list, and a fresh build must be reusable again.
    # (diagonal-lattice fixture: a Cartesian x-displacement maps to a pure
    # fractional x-shift)
    disp_frac = FMD.FPPREC(1.1) * FMD.FPPREC(nlist.skin) / Array(a.info[:lattice])[1, 1]
    CUDA.@allowscalar a.arrays[:pos][1, 1] += disp_frac
    @test !isreusable(nlist, a)
    cl2, a, _ = CellList(a, p[:rc] * (1 + sk))
    nlist2 = neighborlist(a, cl2, p[:rc], sk)
    @test isreusable(nlist2, a)
end

@testset "npt! driver CPU↔GPU parity (2 steps) + device eltype (Issue #39)" begin
    # The device copy of the MLIP must keep FPPREC (cu gave Float32).
    splp_dev = FMD.Apps._resolve_mlip(
        (; splp = Calculators.SplineParam(FMD.FPPREC(1.0), FMD.FPPREC(0.8), 2, 4, 4)),
        CuArray)
    @test splp_dev.c1 isa CuArray{FMD.FPPREC}

    # Same input, both backends, 2 driver steps: MTTK propagation, shell
    # CG, rebuild bookkeeping, and the (deterministic under
    # initial_temperature = 0) integration must agree. Rows are matched
    # via :id — the CellList re-sort order may differ between backends.
    Random.seed!(7)
    a0 = make_nacl_64()
    a0.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(a0.arrays[:pos]))
    base = nacl_params(a0; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    splp = Calculators.SplineParam(FMD.FPPREC(0.6 * base[:rc]),
                                   FMD.FPPREC(0.4 * base[:rc]), 2, 4, 4)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    common = (; base...,
              splp = splp,
              elements = Dict("Na" => 1, "Cl" => 2),
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              nsteps = 2,
              timestep = dt,
              tau_t = 20dt,
              tau_p = 500dt,
              initial_temperature = 0.0,
              target_temperature = 10.0,
              target_stress = zeros(FMD.FPPREC, 6),
              output_interval = 10)
    out_cpu = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; common..., backend = :cpu, output_dir = d))
    end
    out_gpu = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; common..., backend = :cuda, output_dir = d))
    end

    P = sortperm(Array(out_cpu.arrays[:id]))
    G = sortperm(Array(out_gpu.arrays[:id]))
    fc = Array(out_cpu.arrays[:forces])
    fg = Array(out_gpu.arrays[:forces])
    # Forces at shell-CG convergence: the iterate sequences differ in FP
    # between backends, so the agreement scale is cgtol, not eps.
    @test maximum(abs, fc) > 1e-3     # non-vacuous (rattled fixture)
    @test maximum(abs, fc[P, :] .- fg[G, :]) <
          1e-4 * max(one(FMD.FPPREC), maximum(abs, fc))
    pc = Array(out_cpu.arrays[:pos])
    pg = Array(out_gpu.arrays[:pos])
    @test maximum(abs, pc[P, :] .- pg[G, :]) < 1e-6
    @test maximum(abs, Matrix(out_cpu.info[:lattice]) .-
                       Matrix(out_gpu.info[:lattice])) < 1e-6

    # Host-drawn, :id-keyed initial velocities (Issue #34) are bit-identical
    # across backends for the same seed (nsteps = 0 stops right after init).
    seeded = (; common..., initial_temperature = FMD.FPPREC(50.0),
              seed = 3, nsteps = 0)
    vc = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; seeded..., backend = :cpu, output_dir = d))
    end
    vg = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; seeded..., backend = :cuda, output_dir = d))
    end
    Pv = sortperm(Array(vc.arrays[:id]))
    Gv = sortperm(Array(vg.arrays[:id]))
    @test Array(vc.arrays[:velo])[Pv, :] == Array(vg.arrays[:velo])[Gv, :]

    # CSVR parity (Issue #138): the rescaling noise is drawn host-side
    # from the seed-derived stream, so the stochastic thermostat must not
    # introduce backend divergence beyond the shell-CG scale. Fixed cell
    # (barostat = :none): the lattice comes back bitwise identical on
    # BOTH backends.
    csvr = (; common..., thermostat = :csvr, barostat = :none,
            initial_temperature = FMD.FPPREC(50.0), seed = 3,
            target_temperature = 50.0)
    cc = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; csvr..., backend = :cpu, output_dir = d))
    end
    cg = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; csvr..., backend = :cuda, output_dir = d))
    end
    Pc = sortperm(Array(cc.arrays[:id]))
    Gc = sortperm(Array(cg.arrays[:id]))
    vcc = Array(cc.arrays[:velo])
    vcg = Array(cg.arrays[:velo])
    @test maximum(abs, vcc) > 0            # non-vacuous (seeded + rescaled)
    @test maximum(abs, vcc[Pc, :] .- vcg[Gc, :]) < 1e-4 * maximum(abs, vcc)
    @test maximum(abs, Array(cc.arrays[:pos])[Pc, :] .-
                       Array(cg.arrays[:pos])[Gc, :]) < 1e-6
    @test Matrix(cc.info[:lattice]) == Matrix(cg.info[:lattice]) ==
          Matrix(a0.info[:lattice])

    # SCR parity (Issue #140): the stochastic cell update (host-side μ,
    # seed-derived noise) must not open a backend gap — same tolerances
    # as the MTTK case, and the two lattices must agree to the same
    # 1e-6 scale (they are equal host-Float64 products applied to the
    # same start, but the CG-scale force differences feed back into the
    # stress → μ chain, so bitwise equality is not expected).
    scr = (; common..., barostat = :scr,
           compressibility = UNITS[:GPa] / 24.0,
           initial_temperature = FMD.FPPREC(50.0), seed = 3,
           target_temperature = 50.0)
    sc = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; scr..., backend = :cpu, output_dir = d))
    end
    sg = mktempdir() do d
        FMD.Apps.npt!(deepcopy_atoms(a0), (; scr..., backend = :cuda, output_dir = d))
    end
    Ps = sortperm(Array(sc.arrays[:id]))
    Gs = sortperm(Array(sg.arrays[:id]))
    @test Matrix(sc.info[:lattice]) != Matrix(a0.info[:lattice])  # cell moved
    @test maximum(abs, Matrix(sc.info[:lattice]) .-
                       Matrix(sg.info[:lattice])) < 1e-6
    @test maximum(abs, Array(sc.arrays[:pos])[Ps, :] .-
                       Array(sg.arrays[:pos])[Gs, :]) < 1e-6
end

@testset "polmap output on the cuda backend matches the CPU run (Issue #220)" begin
    # The per-step map is one group_p (device atomics) + a device copy of
    # the gauge table; the file the root writes must agree with the CPU
    # run at the trajectory-parity scale of the testset above.
    Random.seed!(11)
    proto = octant_groupid!(make_nacl_64(), (2, 2, 2))
    a0 = deepcopy_atoms(proto)
    a0.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(a0.arrays[:pos]))
    base = nacl_params(a0; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    gauge = Calculators.PolarizationGauge(proto, (2, 2, 2), base[:qc], base[:qs])
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    common = (; base...,
              elements = Dict("Na" => 1, "Cl" => 2),
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              nsteps = 2,
              timestep = dt,
              tau_t = 20dt,
              tau_p = 500dt,
              initial_temperature = 0.0,
              target_temperature = 10.0,
              target_stress = zeros(FMD.FPPREC, 6),
              output_interval = 10,
              polarization_gauge = gauge,
              polmap_interval = 1)
    maps = Dict{Symbol,Any}()
    for be in (:cpu, :cuda)
        mktempdir() do d
            FMD.Apps.npt!(deepcopy_atoms(a0), (; common..., backend = be, output_dir = d))
            @test isfile(joinpath(d, "polmap_1.txt"))
            @test isfile(joinpath(d, "polmap_2.txt"))
            maps[be] = Calculators.read_polmap(joinpath(d, "polmap_2.txt"))
        end
    end
    @test all(isfinite, maps[:cuda].P)
    @test maximum(abs, maps[:cpu].P) > 1e-6     # non-vacuous (rattled octants)
    @test maps[:cuda].step == 2
    @test isapprox(maps[:cuda].P, maps[:cpu].P; rtol = 1e-4, atol = 1e-8)
    @test isapprox(maps[:cuda].lattice, maps[:cpu].lattice; rtol = 1e-7)
end
