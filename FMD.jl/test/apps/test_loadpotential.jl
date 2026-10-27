# test/apps/test_loadpotential.jl
#
# loadpotential — validated JLD2 loading. Fixtures are written with
# jldsave into a temp dir, mirroring exactly what the fit apps save
# (FitShellModel.jl / FitACE.jl / FitSpline.jl jldsave calls).

using Test
using JLD2

function _lp_err(path)
    try
        FMD.loadpotential(path)
        return nothing
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror, e)
    end
end

@testset "loadpotential" begin
    mktempdir() do dir
        elements = Dict("Na" => 1, "Cl" => 2)
        rc = 9.0 / UNITS[:Å]
        shell7 = (qref = 1.0, qc = [2.0, 1.0], qs = [-1.0, -2.0],
                  k2 = [1.0, 1.0], k4 = [0.1, 0.1],
                  eta_c = [1.0, 1.0], eta_s = [0.5, 0.5])
        shell8 = (; shell7..., rc = rc)
        acep = ACEParam(1.0 / UNITS[:Å], 0.8 * rc, 2, 4, 4, 3, 3)
        splp = SplineParam(0.8 * rc, 0.7 * rc, 2, 4, 4)
        prov = (fit_timestamp = "test", dataset = "synth.xyz")

        _save(name; kw...) = (f = joinpath(dir, name); jldsave(f; kw...); f)

        # --- valid schemas -------------------------------------------------
        shellfile = _save("ShellParam.jld2"; shell = shell8)
        pot_shell = FMD.loadpotential(shellfile)
        @test Set(keys(pot_shell)) == Set([:qref, :qc, :qs, :k2, :k4, :eta_c, :eta_s, :rc])
        @test pot_shell.qc == shell7.qc && pot_shell.rc == rc

        # shell-only + provenance is also a valid ShellParam variant
        f = _save("ShellParam_prov.jld2"; shell = shell8, provenance = prov)
        @test FMD.loadpotential(f).qc == shell7.qc

        acefile = _save("ACEP.jld2"; acep = acep, shell = shell7,
                        elements = elements, rc = rc, provenance = prov)
        pot_ace = FMD.loadpotential(acefile)
        @test Set(keys(pot_ace)) == Set([:qref, :qc, :qs, :k2, :k4, :eta_c, :eta_s,
                                         :acep, :elements, :rc])
        @test !haskey(pot_ace, :provenance)   # never injected into params
        @test pot_ace.acep isa ACEParam
        @test pot_ace.elements == elements

        splfile = _save("SplineP.jld2"; splp = splp, shell = shell7,
                        elements = elements, rc = rc, provenance = prov)
        pot_spl = FMD.loadpotential(splfile)
        @test pot_spl.splp isa SplineParam
        @test !haskey(pot_spl, :provenance)

        # --- integration: loader output passes the params validator -------
        # All three example pipelines: ACEP → NPTMD, SplineP → NPTMD,
        # ShellParam (8-field splat) → FitACE / FitSpline.
        npt_knobs = (; input = @__FILE__, output_dir = "out",
                     pme_meshsize = 2.0, timestep = 41.0,
                     tau_t = 820.0, tau_p = 20500.0,
                     target_temperature = 300.0, initial_temperature = 300.0,
                     target_stress = zeros(6), output_interval = 10)
        for pot in (pot_ace, pot_spl)
            p = (; pot..., npt_knobs...)
            p = (; FMD.default_params(Val(:NPTMD), p)..., p...)
            @test FMD.validate_params(:NPTMD, p) === nothing
        end

        fit_base = (; elements, dataset = @__FILE__, traindata_ratio = 0.9,
                    method_weight = (test = 1.0,))
        p = (; pot_shell..., fit_base..., weight_force = 1.0, weight_stress = 1.0,
             rigid = (c0 = 1e-8,), ace_rmax = 5.0 / UNITS[:Å])
        p = (; FMD.default_params(Val(:FitACE), p)..., p...)
        @test FMD.validate_params(:FitACE, p) === nothing

        p = (; pot_shell..., fit_base...)
        p = (; FMD.default_params(Val(:FitSpline), p)..., p...)
        @test FMD.validate_params(:FitSpline, p) === nothing

        # --- optional NLH baseline block (Issue #124) ---------------------
        # The baseline is stored *inside* the potential file, so a potential
        # fitted with NLH always carries it into MD. Absence means "the fit
        # used no baseline", which is why the key is optional rather than
        # required.
        nlh = Calculators.NLHParam(elements, 0.5 * rc)
        nlhfile = _save("SplineP_nlh.jld2"; splp = splp, shell = shell7,
                        elements = elements, rc = rc,
                        nlh = Calculators.nlh_saved(nlh), provenance = prov)
        pot_nlh = FMD.loadpotential(nlhfile)
        @test pot_nlh.nlh isa NLHParam
        @test pot_nlh.nlh.rc == nlh.rc
        @test pot_nlh.nlh.a == nlh.a && pot_nlh.nlh.b == nlh.b
        # and it survives the params validator on the way into NPTMD
        let p = (; pot_nlh..., npt_knobs...)
            p = (; FMD.default_params(Val(:NPTMD), p)..., p...)
            @test FMD.validate_params(:NPTMD, p) === nothing
        end

        # A baseline cutoff beyond rc would silently drop neighbours (the MD
        # list is derived from the rc parent).
        f = _save("nlh_big_rc.jld2"; splp = splp, shell = shell7,
                  elements = elements, rc = rc,
                  nlh = Calculators.nlh_saved(Calculators.NLHParam(elements, 1.5 * rc)))
        msg = _lp_err(f)
        @test msg !== nothing && occursin("must be <= rc", msg)

        # Corrupted coefficients must not install a silently wrong baseline.
        let saved = Calculators.nlh_saved(nlh), a = copy(saved.a)
            a[1, 1, 2] += 0.1                       # asymmetric in the pair
            f = _save("nlh_bad.jld2"; splp = splp, shell = shell7,
                      elements = elements, rc = rc, nlh = (; saved..., a = a))
            msg = _lp_err(f)
            @test msg !== nothing && occursin("\"nlh\" is invalid", msg)
        end

        f = _save("nlh_wrong_type.jld2"; splp = splp, shell = shell7,
                  elements = elements, rc = rc, nlh = "not a NamedTuple")
        msg = _lp_err(f)
        @test msg !== nothing && occursin("\"nlh\" must be a NamedTuple", msg)

        # Species-count mismatch against this file's own `elements`.
        let saved = Calculators.nlh_saved(Calculators.NLHParam(Dict("Na" => 1), 0.5 * rc))
            f = _save("nlh_species.jld2"; splp = splp, shell = shell7,
                      elements = elements, rc = rc, nlh = saved)
            msg = _lp_err(f)
            @test msg !== nothing && occursin("covers 1 species", msg)
        end

        # --- schema violations --------------------------------------------
        f = _save("missing_qs.jld2"; acep = acep,
                  shell = Base.structdiff(shell7, NamedTuple{(:qs,)}),
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("missing field(s): qs", msg)

        f = _save("extra_field.jld2"; shell = (; shell8..., qz = 1.0))
        msg = _lp_err(f)
        @test msg !== nothing && occursin("unexpected field(s): qz", msg)

        f = _save("no_elements.jld2"; acep = acep, shell = shell7, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("missing top-level key(s): elements", msg)

        f = _save("len_mismatch.jld2"; acep = acep,
                  shell = (; shell7..., qc = [1.0, 2.0, 3.0]),
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("share one length", msg)

        # consistent shell lengths that do NOT match elements (separate branch)
        shell3 = (qref = 1.0, qc = [2.0, 1.0, 1.0], qs = [-1.0, -1.0, -1.0],
                  k2 = ones(3), k4 = fill(0.1, 3), eta_c = ones(3), eta_s = fill(0.5, 3))
        f = _save("nelem_mismatch.jld2"; acep = acep, shell = shell3,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("does not match length(elements) = 2", msg)

        # MLIP fitted for a different species count than elements
        acep3 = ACEParam(1.0 / UNITS[:Å], 0.8 * rc, 3, 4, 4, 3, 3)
        f = _save("species_mismatch.jld2"; acep = acep3, shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("fitted for 3 species but elements has 2", msg)

        # in-file self-contradiction: MLIP cutoff beyond the stored rc
        acep_big = ACEParam(1.0 / UNITS[:Å], 1.5 * rc, 2, 4, 4, 3, 3)
        f = _save("cutoff_violation.jld2"; acep = acep_big, shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("silently drops neighbors", msg)

        # multiple violations reported together (species + shell length)
        f = _save("two_problems.jld2"; acep = acep3, shell = shell3,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("2 problems", msg)

        # field type violations must be ArgumentError, not raw MethodError
        f = _save("qc_string.jld2"; acep = acep, shell = (; shell7..., qc = "ab"),
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("must be Real vectors", msg) && occursin(":qc", msg)

        f = _save("qc_nothing.jld2"; shell = (; shell8..., qc = nothing))
        msg = _lp_err(f)
        @test msg !== nothing && occursin("must be Real vectors", msg)

        # rc range (top-level and inside the 8-field shell)
        f = _save("bad_rc.jld2"; acep = acep, shell = shell7,
                  elements = elements, rc = -1.0)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("\"rc\" must be a positive Real", msg)

        f = _save("bad_shell_rc.jld2"; shell = (; shell7..., rc = 0.0))
        msg = _lp_err(f)
        @test msg !== nothing && occursin("shell.rc must be a positive Real", msg)

        # elements id range / type
        f = _save("bad_ids.jld2"; acep = acep, shell = shell7,
                  elements = Dict("Na" => 1, "Cl" => 3), rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("permutation of 1:2", msg)

        f = _save("elements_vector.jld2"; acep = acep, shell = shell7,
                  elements = ["Na", "Cl"], rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("\"elements\" must be a Dict", msg)

        # shell / MLIP type violations
        f = _save("shell_dict.jld2"; shell = Dict(:qref => 1.0))
        msg = _lp_err(f)
        @test msg !== nothing && occursin("\"shell\" must be a NamedTuple", msg)

        f = _save("wrong_type.jld2"; acep = "not an ACEParam", shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("must be an ACEParam", msg)

        f = _save("wrong_splp.jld2"; splp = "nope", shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("must be a SplineParam", msg)

        # c3 symmetry contract (Issue #63): a hand-built asymmetric c3
        # would silently give enumeration-order-dependent energies
        Random.seed!(63)
        spasym = SplineParam(0.8 * rc, 0.7 * rc, 2, 4, 4)
        spasym.c3 .= randn(FMD.FPPREC, size(spasym.c3))
        f = _save("asym_c3.jld2"; splp = spasym, shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("c3 ∈ im(P)", msg)
        # the documented projection makes the same coefficients loadable —
        # and the projection survives the JLD2 round trip bit-exactly
        Calculators._spl_symmetrize_c3!(spasym.c3)
        f = _save("sym_c3.jld2"; splp = spasym, shell = shell7,
                  elements = elements, rc = rc)
        @test FMD.loadpotential(f).splp isa SplineParam

        # exactly one MLIP per file
        f = _save("both_mlips.jld2"; acep = acep, splp = splp, shell = shell7,
                  elements = elements, rc = rc)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("exactly one MLIP", msg)

        # unknown schema / not a JLD2 file / missing file
        f = _save("unrelated.jld2"; foo = 1)
        msg = _lp_err(f)
        @test msg !== nothing && occursin("unrecognized schema", msg)

        f = joinpath(dir, "corrupt.jld2")
        write(f, "this is not a jld2 file")
        msg = _lp_err(f)
        @test msg !== nothing && occursin("failed to read as JLD2", msg)

        @test_throws ArgumentError FMD.loadpotential(joinpath(dir, "no_such.jld2"))
    end
end
