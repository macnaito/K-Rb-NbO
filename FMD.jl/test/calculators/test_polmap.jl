# test/calculators/test_polmap.jl
#
# Local polarization map persistence (Issue #220): write_polmap /
# read_polmap round trip, the on-disk unit and ordering conventions, the
# per-step (device-table) form of group_p_gauged, and the identity that
# ties the map to the reported dipole_phys.
#
# Reuses make_binary_supercell (test_umbrella.jl) and map_fixture /
# MAP_QC / MAP_QS (test_map_umbrella.jl), which runtests.jl includes first.

using Test
using LinearAlgebra
using JLD2
using FMD.Calculators: write_polmap, read_polmap, group_p_gauged, PolarizationGauge,
                       save_target_map, load_target_map, calc_dipole_phys!,
                       total_mfrac_ref

# Deterministic pseudo-random values in (-1, 1) (no RNG stream dependence).
_pm_val(n) = 2 * (n * 0.7548776662466927 - floor(n * 0.7548776662466927)) - 1

@testset "write_polmap / read_polmap round trip and on-disk conventions" begin
    gridsize = (2, 3, 4)
    ng = prod(gridsize)
    P = FMD.FPPREC[0.05 * _pm_val(3i + d) for i in 1:ng, d in 1:3]
    # A sheared cell, so the lattice rows are actually distinguished.
    L = FMD.FPPREC[10.0 0.0 0.0; 1.0 11.0 0.0; 0.5 -0.7 12.0]
    mktempdir() do dir
        path = joinpath(dir, "polmap_7.txt")
        @test write_polmap(path, P, gridsize, L; step = 7, source = "step_7.xyz") == path
        r = read_polmap(path)
        @test r.gridsize == gridsize
        @test r.step == 7
        @test r.source == "step_7.xyz"
        @test isapprox(r.P, P; rtol = 1e-9, atol = 1e-12)   # %.10e on disk
        @test isapprox(r.lattice, L; rtol = 1e-9)

        # Unit pin: the file holds e/Å² and Å (the xyz convention), not a.u.
        lines = readlines(path)
        rows = [split(l) for l in lines if !startswith(l, "#")]
        @test length(rows) == ng
        @test all(length(c) == 6 for c in rows)            # np.loadtxt-shaped
        @test parse(Float64, rows[5][4]) ≈ P[5, 1] / UNITS[:Å]^2 rtol = 1e-9
        @test parse(Float64, rows[ng][6]) ≈ P[ng, 3] / UNITS[:Å]^2 rtol = 1e-9
        lat = parse.(Float64, split(lines[3])[3:end])
        @test lat ≈ vec(permutedims(L)) .* UNITS[:Å] rtol = 1e-9   # a1x a1y a1z a2x ...

        # Row order is the flat groupid convention (iz fastest).
        for (i, c) in enumerate(rows)
            gx, gy, gz = Calculators._decode_g(i, gridsize[2], gridsize[3])
            @test parse.(Int, c[1:3]) == [gx, gy, gz]
        end
        # ... and the grid really is (Nx, Ny, Nz), not transposed.
        @test parse.(Int, rows[2][1:3]) == [0, 0, 1]
        @test parse.(Int, rows[gridsize[3] + 1][1:3]) == [0, 1, 0]

        # The header fields are optional.
        path0 = joinpath(dir, "bare.txt")
        write_polmap(path0, P, gridsize, L)
        r0 = read_polmap(path0)
        @test r0.step === nothing
        @test r0.source === nothing
        @test r0.P == r.P

        # Shape / argument guards.
        @test_throws "expected ($ng, 3)" write_polmap(joinpath(dir, "x.txt"),
                                                      P[1:end-1, :], gridsize, L)
        @test_throws "must not contain whitespace" write_polmap(joinpath(dir, "x.txt"),
                                                                P, gridsize, L;
                                                                source = "a b")
    end
end

@testset "read_polmap refuses a permuted, truncated or foreign file" begin
    gridsize = (2, 2, 2)
    ng = 8
    P = FMD.FPPREC[0.01 * (i + d) for i in 1:ng, d in 1:3]
    L = Matrix{FMD.FPPREC}(I, 3, 3) .* 8
    mktempdir() do dir
        path = joinpath(dir, "p.txt")
        write_polmap(path, P, gridsize, L)
        lines = readlines(path)
        head = lines[1:5]
        data = lines[6:end]
        @test length(data) == ng
        wr(name, ls) = (write(joinpath(dir, name), join(ls, "\n") * "\n"); joinpath(dir, name))

        @test_throws "flat groupid order" read_polmap(wr("swap.txt",
            vcat(head, data[vcat([2, 1], 3:ng)])))
        @test_throws "expected $ng" read_polmap(wr("short.txt", vcat(head, data[1:end-1])))
        @test_throws "more than $ng" read_polmap(wr("long.txt", vcat(head, data, data[1:1])))
        @test_throws "not a polmap file" read_polmap(wr("foreign.txt",
            vcat(["# something else"], lines[2:end])))
        bad = copy(lines)
        bad[1] = "# FMD polmap format=99"
        @test_throws "polmap format 99" read_polmap(wr("fmt.txt", bad))
        # Comment lines and blank lines between rows are tolerated (a user
        # annotating a file must not break it).
        @test isapprox(read_polmap(wr("annot.txt", vcat(head, data[1:3], ["# note", ""], data[4:end]))).P,
                       P; rtol = 1e-9)
    end
end

@testset "map ⇄ dipole_phys identity; per-step form; target-map bridge" begin
    gridsize = (3, 2, 2)
    ng = prod(gridsize)
    proto, atoms = map_fixture(gridsize)
    gauge = PolarizationGauge(proto, gridsize, MAP_QC, MAP_QS)
    Pg = group_p_gauged(atoms, gauge, MAP_QC, MAP_QS)
    @test size(Pg) == (ng, 3)
    @test maximum(abs, Pg) > 1e-5           # non-vacuous: the fixture is polarised

    # Σ_g P_g V_g is the whole-cell physical dipole against the same gauge —
    # an identity between the map and the dipole_phys column of log.txt,
    # computed by code that shares nothing with group_p.
    Vg = Cores.volume(atoms) / ng
    M_map = vec(sum(Pg, dims = 1)) .* Vg
    M_phys = calc_dipole_phys!(atoms, (; qc = MAP_QC, qs = MAP_QS), total_mfrac_ref(gauge))
    @test maximum(abs, M_phys) > 1e-4
    @test isapprox(M_map, M_phys; rtol = 1e-10, atol = 1e-12)

    # Gauge covariance: the gauge frame itself, under a pure cell
    # deformation with static fractional coordinates, stays at P_g = 0
    # (the reference co-transforms with the cell) — and a rigid
    # translation of the whole (neutral-group) system leaves P_g alone.
    zero_frame = deepcopy_atoms(proto)
    zero_frame.info[:lattice] = proto.info[:lattice] * FMD.FPPREC[1.03 0.02 0; 0 0.97 0.01; 0 0 1.05]
    @test maximum(abs, group_p_gauged(zero_frame, gauge, MAP_QC, MAP_QS)) < 1e-13
    shifted = deepcopy_atoms(atoms)
    shifted.arrays[:pos] .+= FMD.FPPREC[0.13 -0.07 0.21]
    @test isapprox(group_p_gauged(shifted, gauge, MAP_QC, MAP_QS), Pg;
                   rtol = 1e-10, atol = 1e-14)

    # The per-step (device-table) form the MD driver calls is the same map.
    Pg2 = group_p_gauged(atoms, Matrix(gauge.M_frac_ref), MAP_QC, MAP_QS)
    @test isapprox(Pg2, Pg; rtol = 1e-12, atol = 1e-15)
    @test_throws "must be (ng, 3)" group_p_gauged(atoms, Matrix(gauge.M_frac_ref)', MAP_QC, MAP_QS)

    # A read-back map is in the units save_target_map expects.
    mktempdir() do dir
        write_polmap(joinpath(dir, "m.txt"), Pg, gridsize, atoms.info[:lattice]; step = 3)
        back = read_polmap(joinpath(dir, "m.txt"))
        @test back.step == 3
        @test isapprox(back.lattice, atoms.info[:lattice]; rtol = 1e-9)
        save_target_map(joinpath(dir, "t.jld2"), back.P, gauge)
        @test isapprox(load_target_map(joinpath(dir, "t.jld2"), gauge), Pg;
                       rtol = 1e-9, atol = 1e-12)
    end
end
