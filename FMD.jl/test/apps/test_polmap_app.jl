# test/apps/test_polmap_app.jl
#
# Local polarization map output (Issue #220): the in-loop npt! writer
# (params[:polmap_interval]), the post-process app[:PolMap], the
# equivalence of the two on the same snapshots, and the validation of
# both entry points.

using Test
using Random
using LinearAlgebra
using FMD.Apps: npt!
using FMD.Calculators: PolarizationGauge, read_polmap
using FMD.Cores: writexyz

const _PM_ELEMENTS = Dict("Na" => 1, "Cl" => 2)

# NaCl-64 in (2,2,2) octant groups (one neutral conventional cell each),
# rattled AFTER grouping; the gauge is the unrattled prototype so the map
# is non-zero. Params are the CPU smoke's with the map output added.
function _polmap_fixture()
    Random.seed!(11)
    proto = octant_groupid!(make_nacl_64(), (2, 2, 2))
    atoms = deepcopy_atoms(proto)
    atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    base = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    gauge = PolarizationGauge(proto, (2, 2, 2), base[:qc], base[:qs])
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    params = (; base...,
              backend = :cpu,
              elements = _PM_ELEMENTS,
              pme_meshsize = FMD.FPPREC(0.5 / UNITS[:Å]),
              skin = FMD.FPPREC(0.1),
              cgtol = FMD.FPPREC(2e-5),
              nsteps = 4,
              timestep = dt,
              tau_t = 20dt,
              tau_p = 500dt,
              initial_temperature = 0.0,   # keep the ambient RNG out (Issue #34)
              target_temperature = 10.0,
              target_stress = zeros(FMD.FPPREC, 6),
              output_interval = 2,
              polarization_gauge = gauge,
              polmap_interval = 2)
    return atoms, params, gauge
end

# validate_params error message, or nothing when valid.
function _pm_verr(job, params)
    try
        FMD.validate_params(job, params)
        return nothing
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror, e)
    end
end

@testset "npt! writes polmap_<n>.txt at polmap_interval" begin
    atoms, params, gauge = _polmap_fixture()
    mktempdir() do dir
        Test.collect_test_logs() do
            npt!(deepcopy_atoms(atoms), (; params..., output_dir = dir))
        end
        @test isfile(joinpath(dir, "polmap_2.txt"))
        @test isfile(joinpath(dir, "polmap_4.txt"))
        @test !isfile(joinpath(dir, "polmap_1.txt"))
        @test !isfile(joinpath(dir, "polmap_3.txt"))
        log = [parse.(Float64, split(l)) for l in eachline(joinpath(dir, "log.txt"))]
        for s in (2, 4)
            r = read_polmap(joinpath(dir, "polmap_$(s).txt"))
            @test r.step == s
            @test r.gridsize == (2, 2, 2)
            @test all(isfinite, r.P)
            @test maximum(abs, r.P) > 1e-6        # the rattle polarised the octants
            # The map's lattice is that step's cell (log.txt cols 2-10,
            # column-major, Å).
            @test isapprox(vec(r.lattice) .* UNITS[:Å], log[s][2:10]; rtol = 1e-7)
            # Σ_g P_g V_g equals the gauge-fixed dipole_phys of the same
            # step (log.txt cols 14-16, e·Å) — same gauge, independent
            # code path (calc_dipole_phys!). Both are %.8e/%.10e on disk.
            Vg = abs(det(r.lattice)) / 8
            M = vec(sum(r.P, dims = 1)) .* Vg .* UNITS[:Å]
            @test maximum(abs, log[s][14:16]) > 1e-4
            @test isapprox(M, log[s][14:16]; rtol = 1e-6, atol = 1e-8)
        end
    end
end

@testset "app[:PolMap] reproduces the in-loop maps from step_<n>.xyz" begin
    atoms, params, gauge = _polmap_fixture()
    mktempdir() do dir
        Test.collect_test_logs() do
            npt!(deepcopy_atoms(atoms), (; params..., output_dir = dir))
        end
        outdir = joinpath(dir, "post")
        pp = (; input = [joinpath(dir, "step_2.xyz"), joinpath(dir, "step_4.xyz")],
                output_dir = outdir,
                polarization_gauge = gauge,
                elements = _PM_ELEMENTS,
                qc = params[:qc], qs = params[:qs],
                # inert keys a `pot...` splat carries along
                k2 = params[:k2], rc = params[:rc], qref = 0.0)
        pp = (; FMD.default_params(Val(:PolMap), pp)..., pp...)
        @test FMD.validate_params(:PolMap, pp) === nothing
        @test FMD.app[:PolMap](pp) == 0
        for s in (2, 4)
            a = read_polmap(joinpath(dir, "polmap_$(s).txt"))
            b = read_polmap(joinpath(outdir, "polmap_$(s).txt"))
            @test b.step == s
            @test b.source == "step_$(s).xyz"
            @test b.gridsize == a.gridsize
            # The xyz snapshot rounds positions / lattice to %.8e (Å), so
            # the agreement scale is that round trip, not fp eps.
            @test isapprox(b.P, a.P; rtol = 1e-6, atol = 1e-9)
            @test isapprox(b.lattice, a.lattice; rtol = 1e-7)
        end
    end
end

@testset "app[:PolMap] naming, single-path input, validation" begin
    atoms, params, gauge = _polmap_fixture()
    prm = (; elements = _PM_ELEMENTS)
    mktempdir() do dir
        a = deepcopy_atoms(atoms)
        open(joinpath(dir, "single.xyz"), "w") do io
            writexyz(io, a, prm)
        end
        open(joinpath(dir, "two.xyz"), "w") do io
            writexyz(io, a, prm)
            writexyz(io, a, prm)
        end
        out = joinpath(dir, "post")
        pp = (; input = [joinpath(dir, "single.xyz"), joinpath(dir, "two.xyz")],
                output_dir = out, polarization_gauge = gauge,
                elements = _PM_ELEMENTS, qc = params[:qc], qs = params[:qs])
        @test FMD.app[:PolMap](pp) == 0
        @test isfile(joinpath(out, "polmap_single.txt"))
        @test isfile(joinpath(out, "polmap_two_1.txt"))
        @test isfile(joinpath(out, "polmap_two_2.txt"))
        r1 = read_polmap(joinpath(out, "polmap_single.txt"))
        @test r1.step === nothing
        @test r1.source == "single.xyz"
        @test read_polmap(joinpath(out, "polmap_two_1.txt")).P == r1.P
        @test read_polmap(joinpath(out, "polmap_two_2.txt")).P == r1.P
        # The map of the written frame is the map of the in-memory frame
        # (writexyz/readxyz round trip, %.8e).
        direct = Calculators.group_p_gauged(a, gauge, params[:qc], params[:qs])
        @test isapprox(r1.P, direct; rtol = 1e-6, atol = 1e-9)

        # A single path (not a Vector) is accepted.
        @test FMD.app[:PolMap]((; pp..., input = joinpath(dir, "single.xyz"),
                                  output_dir = joinpath(dir, "post2"))) == 0
        @test isfile(joinpath(dir, "post2", "polmap_single.txt"))

        # An existing map is never replaced silently: refused by default,
        # replaced only with overwrite = true.
        @test_throws "already exists" FMD.app[:PolMap](
            (; pp..., input = joinpath(dir, "single.xyz"), output_dir = joinpath(dir, "post2")))
        @test FMD.app[:PolMap]((; pp..., input = joinpath(dir, "single.xyz"),
                                  output_dir = joinpath(dir, "post2"), overwrite = true)) == 0
        @test read_polmap(joinpath(dir, "post2", "polmap_single.txt")).P == r1.P
        msg = _pm_verr(:PolMap, (; pp..., overwrite = 1))
        @test msg !== nothing && occursin("overwrite must be a Bool", msg)

        # Two inputs that collide on the output name are refused.
        mkpath(joinpath(dir, "b"))
        cp(joinpath(dir, "single.xyz"), joinpath(dir, "b", "single.xyz"))
        @test_throws "map to the same output" FMD.app[:PolMap](
            (; pp..., input = [joinpath(dir, "single.xyz"), joinpath(dir, "b", "single.xyz")],
               output_dir = joinpath(dir, "post3")))

        # Validation at the boundary (ArgumentError from validate_params).
        msg = _pm_verr(:PolMap, (; pp..., input = joinpath(dir, "missing.xyz")))
        @test msg !== nothing && occursin("input file(s) not found", msg)
        msg = _pm_verr(:PolMap, (; pp..., input = String[]))
        @test msg !== nothing && occursin("input is empty", msg)
        msg = _pm_verr(:PolMap, (; pp..., input = 3))
        @test msg !== nothing && occursin("extxyz path or a Vector", msg)
        msg = _pm_verr(:PolMap, (; pp..., polarization_gauge = "gauge.jld2"))
        @test msg !== nothing && occursin("must be a PolarizationGauge", msg)
        msg = _pm_verr(:PolMap, Base.structdiff(pp, NamedTuple{(:polarization_gauge,)}))
        @test msg !== nothing && occursin("missing required key(s): :polarization_gauge", msg)
        msg = _pm_verr(:PolMap, (; pp..., bogus = 1))
        @test msg !== nothing && occursin("unknown key :bogus", msg)
        msg = _pm_verr(:PolMap, (; pp..., qc = params[:qc][1:1]))
        @test msg !== nothing && occursin("length(qc)", msg)
        @test_throws ArgumentError FMD.app[:PolMap]((; pp..., bogus = 1))

        # A frame whose grouping differs from the gauge is refused by the
        # gauge check, not silently mapped onto the wrong grid.
        c = deepcopy_atoms(atoms)
        c.arrays[:groupid] .= 1
        open(joinpath(dir, "regrouped.xyz"), "w") do io
            writexyz(io, c, prm)
        end
        @test_throws "gauge frame" FMD.app[:PolMap](
            (; pp..., input = joinpath(dir, "regrouped.xyz"),
               output_dir = joinpath(dir, "post4")))
    end
end

@testset "polmap_interval validation (:NPTMD and direct npt!)" begin
    atoms, params, gauge = _polmap_fixture()
    # Direct npt! path: checked at entry, before the force stack is built.
    p_nogauge = Base.structdiff(params, NamedTuple{(:polarization_gauge,)})
    mktempdir() do dir
        @test_throws "requires params[:polarization_gauge]" npt!(
            deepcopy_atoms(atoms), (; p_nogauge..., output_dir = dir))
        @test_throws "positive Integer" npt!(
            deepcopy_atoms(atoms), (; params..., polmap_interval = 0, output_dir = dir))
        @test_throws "positive Integer" npt!(
            deepcopy_atoms(atoms), (; params..., polmap_interval = 2.0, output_dir = dir))
        @test isempty(filter(f -> startswith(f, "polmap_"), readdir(dir)))
    end

    # App path: validate_params reports the same conditions.
    pv = (; params..., input = @__FILE__, output_dir = "out")
    pv = (; FMD.default_params(Val(:NPTMD), pv)..., pv...)
    @test FMD.validate_params(:NPTMD, pv) === nothing
    msg = _pm_verr(:NPTMD, Base.structdiff(pv, NamedTuple{(:polarization_gauge,)}))
    @test msg !== nothing && occursin("polmap_interval requires :polarization_gauge", msg)
    msg = _pm_verr(:NPTMD, (; pv..., polmap_interval = 0))
    @test msg !== nothing && occursin("polmap_interval must be a positive Integer", msg)
    msg = _pm_verr(:NPTMD, (; pv..., polmap_interval = 2.0))
    @test msg !== nothing && occursin("polmap_interval must be a positive Integer", msg)
    # Without the interval the gauge alone is still fine (dipole_phys only).
    @test FMD.validate_params(:NPTMD,
        Base.structdiff(pv, NamedTuple{(:polmap_interval,)})) === nothing
end
