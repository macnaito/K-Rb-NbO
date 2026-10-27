# test/apps/test_param_validation.jl
#
# validate_params — the schema gate at every FMD.app entry point.
# Pure CPU, no MPI / GPU: validate_params is a pure function of the
# params NamedTuple (plus isfile checks against paths in this repo).

using Test

# Remove keys from a NamedTuple (to simulate a missing required key).
_drop(p::NamedTuple, ks::Symbol...) = Base.structdiff(p, NamedTuple{ks})

# Run validate_params and return the error message (or nothing if valid).
function _verr(job, params)
    try
        FMD.validate_params(job, params)
        return nothing
    catch e
        e isa ArgumentError || rethrow()
        return sprint(showerror, e)
    end
end

const _VAL_ELEMENTS = Dict("Na" => 1, "Cl" => 2)
const _VAL_RC = 9.0 / UNITS[:Å]

# Small CPU MLIPs. Cutoffs deliberately < rc so the fixtures validate.
_val_acep() = ACEParam(1.0 / UNITS[:Å], 0.8 * _VAL_RC, 2, 4, 4, 3, 3)
_val_splp() = SplineParam(0.8 * _VAL_RC, 0.7 * _VAL_RC, 2, 4, 4)

# Valid merged params per job. `@__FILE__` stands in for input/dataset
# paths: validation only checks existence, never reads the file.
function _val_nptmd(; kw...)
    p = (;
        elements = _VAL_ELEMENTS,
        acep = _val_acep(),
        rc = _VAL_RC,
        input = @__FILE__,
        output_dir = "out",
        pme_meshsize = 2.0,
        timestep = 41.0,
        tau_t = 820.0,
        tau_p = 20500.0,
        target_temperature = 300.0,
        initial_temperature = 300.0,
        target_stress = zeros(6),
        output_interval = 10,
    )
    return (; FMD.default_params(Val(:NPTMD), p)..., p..., kw...)
end

function _val_fitace(; kw...)
    p = (;
        elements = _VAL_ELEMENTS,
        dataset = @__FILE__,
        traindata_ratio = 0.9,
        method_weight = (test = 1.0,),
        weight_force = 1.0,
        weight_stress = 1.0,
        rigid = (cp = 1e-8, c0 = 1e-8),
        ace_rmax = 5.0 / UNITS[:Å],
    )
    return (; FMD.default_params(Val(:FitACE), p)..., p..., kw...)
end

function _val_fitspline(; kw...)
    p = (;
        elements = _VAL_ELEMENTS,
        dataset = @__FILE__,
        traindata_ratio = 0.9,
        method_weight = (test = 1.0,),
    )
    return (; FMD.default_params(Val(:FitSpline), p)..., p..., kw...)
end

function _val_fitsm(; kw...)
    p = (;
        elements = _VAL_ELEMENTS,
        dataset = @__FILE__,
        formal_charges = [1.0, -1.0],
        opt_weights = (bec = 10.0, eps = 1.0),
        qref = 1.0,
        qc = [2.0, 1.0],
        k2 = [1.0, 1.0],
        eta_c = fill(1.0, 2),
        eta_s = fill(1.0, 2),
        lb = (qref = 0.5, qc = [0.1, 0.1], k2 = [0.8, 0.8],
              eta_c = fill(0.02 / UNITS[:Å], 2), eta_s = fill(0.02 / UNITS[:Å], 2)),
        ub = (qref = 1.5, qc = [10.0, 10.0], k2 = [5.0, 5.0],
              eta_c = fill(4.0 / UNITS[:Å], 2), eta_s = fill(4.0 / UNITS[:Å], 2)),
    )
    return (; FMD.default_params(Val(:FitShellModel), p)..., p..., kw...)
end

@testset "validate_params" begin
    @testset "valid fixtures pass" begin
        @test FMD.validate_params(:NPTMD, _val_nptmd()) === nothing
        p_splp = (; _drop(_val_nptmd(), :acep)..., splp = _val_splp())
        @test FMD.validate_params(:NPTMD, p_splp) === nothing
        @test FMD.validate_params(:FitACE, _val_fitace()) === nothing
        @test FMD.validate_params(:FitSpline, _val_fitspline()) === nothing
        @test FMD.validate_params(:FitShellModel, _val_fitsm()) === nothing
    end

    @testset "unknown job" begin
        msg = _verr(:NoSuchJob, (; elements = _VAL_ELEMENTS))
        @test msg !== nothing
        @test occursin("unknown job", msg) && occursin("NPTMD", msg)
    end

    @testset "typo → did-you-mean" begin
        p = _drop(_val_nptmd(), :timestep)
        msg = _verr(:NPTMD, (; p..., timestap = 41.0))
        @test msg !== nothing
        @test occursin("unknown key :timestap", msg)
        @test occursin("did you mean :timestep?", msg)
        @test occursin("missing required key(s): :timestep", msg)
    end

    @testset "all violations reported at once" begin
        # (:target_stress is conditionally required since #136, so a truly
        # unconditional key stands in for the missing-required phase.)
        p = _drop(_val_nptmd(), :pme_meshsize)
        msg = _verr(:NPTMD, (; p..., timestep = 0.0, ewald_tolerence = 1e-6))
        @test msg !== nothing
        @test occursin("3 problems", msg)
        @test occursin("unknown key :ewald_tolerence", msg)
        @test occursin("missing required key(s): :pme_meshsize", msg)
        @test occursin("timestep must be > 0", msg)
    end

    @testset "merge-forgotten hint" begin
        # Raw caller params without the default_params merge: many keys
        # missing at once triggers the hint.
        msg = _verr(:NPTMD, (;
            elements = _VAL_ELEMENTS, acep = _val_acep(), input = @__FILE__,
            output_dir = "out", pme_meshsize = 2.0, timestep = 41.0,
            tau_t = 820.0, tau_p = 20500.0,
            target_temperature = 300.0, initial_temperature = 300.0,
            target_stress = zeros(6), output_interval = 10,
        ))
        @test msg !== nothing
        @test occursin("missing required key(s)", msg)
        @test occursin("did you forget", msg)
        @test occursin("default_params(Val(:NPTMD)", msg)
    end

    @testset "acep/splp exclusivity" begin
        msg = _verr(:NPTMD, _val_nptmd(; splp = _val_splp()))
        @test msg !== nothing && occursin("at most one of :acep / :splp", msg) && occursin("both", msg)

        # No MLIP at all is legal (Issue #133): the pure shell model.
        @test FMD.validate_params(:NPTMD, _drop(_val_nptmd(), :acep)) === nothing
    end

    @testset "MLIP cutoff > rc" begin
        big = ACEParam(1.0 / UNITS[:Å], 1.5 * _VAL_RC, 2, 4, 4, 3, 3)
        msg = _verr(:NPTMD, _val_nptmd(; acep = big))
        @test msg !== nothing && occursin("silently drops neighbors", msg)

        bigsp = SplineParam(1.5 * _VAL_RC, 0.7 * _VAL_RC, 2, 4, 4)
        p = _drop(_val_nptmd(), :acep)
        msg = _verr(:NPTMD, (; p..., splp = bigsp))
        @test msg !== nothing && occursin("splp.rmax2/rmax3", msg)
    end

    @testset "splp c3 symmetry contract (Issue #63)" begin
        # A generic random c3 is not (j,k)-swap symmetric — the 3-body
        # kernel would return enumeration-order-dependent energies.
        Random.seed!(63)
        p = _drop(_val_nptmd(), :acep)
        sp = _val_splp()
        sp.c3 .= randn(FMD.FPPREC, size(sp.c3))
        msg = _verr(:NPTMD, (; p..., splp = sp))
        @test msg !== nothing && occursin("c3 ∈ im(P)", msg) &&
              occursin("_spl_symmetrize_c3!", msg)
        # After the documented projection the same object validates.
        Calculators._spl_symmetrize_c3!(sp.c3)
        @test FMD.validate_params(:NPTMD, (; p..., splp = sp)) === nothing
    end

    @testset "shape violations" begin
        msg = _verr(:NPTMD, _val_nptmd(; target_stress = zeros(5)))
        @test msg !== nothing && occursin("Voigt-6", msg)

        msg = _verr(:NPTMD, _val_nptmd(; efield = [0.0, 0.0]))
        @test msg !== nothing && occursin("efield must be a 3-vector", msg)

        msg = _verr(:NPTMD, _val_nptmd(; qc = zeros(3), a = zeros(2, 3)))
        @test msg !== nothing
        @test occursin("length(qc) = 3 must equal nelem = 2", msg)
        @test occursin("size(a) = (2, 3) must be (nelem, nelem)", msg)
    end

    @testset "value-range violations" begin
        msg = _verr(:NPTMD, _val_nptmd(; output_interval = 0))
        @test msg !== nothing && occursin("output_interval must be >= 1", msg)

        msg = _verr(:NPTMD, _val_nptmd(; rho = zeros(2, 2), eta_c = [1.0, 0.0]))
        @test msg !== nothing
        @test occursin("all(rho .> 0)", msg)
        @test occursin("all(eta_c .> 0)", msg)

        # erfcinv domain: rc * ewald_tolerance must stay in (0, 2)
        msg = _verr(:NPTMD, _val_nptmd(; ewald_tolerance = 1.0))
        @test msg !== nothing && occursin("erfcinv domain", msg)
    end

    @testset "thermostat/barostat selection (Issue #136)" begin
        # Unknown method symbols are rejected, not deferred (:berendsen
        # will never exist — wrong ensemble, see Issue #116).
        msg = _verr(:NPTMD, _val_nptmd(; thermostat = :berendsen))
        @test msg !== nothing && occursin("thermostat must be :nh, :csvr or :none", msg)
        msg = _verr(:NPTMD, _val_nptmd(; barostat = :berendsen))
        @test msg !== nothing && occursin("barostat must be :mttk, :scr or :none", msg)

        # SCR (Issue #140): legal with :compressibility (+ τ/stress/seed
        # from the fixture); κ is required, only legal with :scr, and > 0;
        # an unthermostatted SCR run is rejected outright.
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(; barostat = :scr, compressibility = 1.0)) === nothing
        msg = _verr(:NPTMD, _val_nptmd(; barostat = :scr))
        @test msg !== nothing && occursin("barostat = :scr requires :compressibility", msg)
        msg = _verr(:NPTMD, _val_nptmd(; compressibility = 1.0))
        @test msg !== nothing && occursin("only read by barostat = :scr", msg)
        msg = _verr(:NPTMD, _val_nptmd(; barostat = :scr, compressibility = -1.0))
        @test msg !== nothing && occursin("compressibility must be > 0", msg)
        p = _drop(_val_nptmd(; barostat = :scr, compressibility = 1.0), :tau_t)
        msg = _verr(:NPTMD, (; p..., thermostat = :none))
        @test msg !== nothing && occursin("barostat = :scr requires a thermostat", msg)

        # CSVR (Issue #138): legal with :tau_t + target >= 1 K; below-1K
        # targets (incl. ramps) are rejected — the shared kT floor
        # max(T, 1 K) would silently thermostat them to 1 K.
        @test FMD.validate_params(:NPTMD, _val_nptmd(; thermostat = :csvr)) === nothing
        msg = _verr(:NPTMD, _drop(_val_nptmd(; thermostat = :csvr), :tau_t))
        @test msg !== nothing && occursin("thermostat = :csvr requires :tau_t", msg)
        msg = _verr(:NPTMD, _val_nptmd(; thermostat = :csvr,
                                        target_temperature = 0.5))
        @test msg !== nothing &&
              occursin("thermostat = :csvr requires target_temperature >= 1 K", msg)
        msg = _verr(:NPTMD, _val_nptmd(; thermostat = :csvr,
                                        final_temperature = 0.5))
        @test msg !== nothing &&
              occursin("thermostat = :csvr requires final_temperature >= 1 K", msg)

        # Value-conditional keys: the active method demands its τ (and the
        # barostat its target stress)…
        msg = _verr(:NPTMD, _drop(_val_nptmd(), :tau_t))
        @test msg !== nothing && occursin("thermostat = :nh requires :tau_t", msg)
        msg = _verr(:NPTMD, _drop(_val_nptmd(), :tau_p))
        @test msg !== nothing && occursin("barostat = :mttk requires :tau_p", msg)
        msg = _verr(:NPTMD, _drop(_val_nptmd(), :target_stress))
        @test msg !== nothing && occursin("barostat = :mttk requires :target_stress", msg)

        # …while :none needs none of them: fixed-cell NVT drops :tau_p and
        # :target_stress, full NVE drops :tau_t as well.
        p = _drop(_val_nptmd(), :tau_p, :target_stress)
        @test FMD.validate_params(:NPTMD, (; p..., barostat = :none)) === nothing
        p = _drop(_val_nptmd(), :tau_t, :tau_p, :target_stress)
        @test FMD.validate_params(:NPTMD,
            (; p..., thermostat = :none, barostat = :none)) === nothing

        # Positivity of the τ's (replaces the pre-#136 mttk_* checks).
        msg = _verr(:NPTMD, _val_nptmd(; tau_t = 0.0))
        @test msg !== nothing && occursin("tau_t must be > 0", msg)
        msg = _verr(:NPTMD, _val_nptmd(; tau_p = -1.0))
        @test msg !== nothing && occursin("tau_p must be > 0", msg)
    end

    @testset "inert keys from JLD2 splat are accepted" begin
        # qref / dataset_idx ride along via `_pot["shell"]...` or fit-input
        # reuse; they are not read by NPTMD but must not be rejected.
        @test FMD.validate_params(:NPTMD, _val_nptmd(; qref = 1.0, dataset_idx = :)) === nothing
    end

    @testset "FitACE: ace_rmax / ace_nk / rigid" begin
        msg = _verr(:FitACE, _drop(_val_fitace(), :ace_rmax))
        @test msg !== nothing && occursin("either :acep (warm start) or :ace_rmax", msg)

        # dead keys removed from default_params must now be rejected
        msg = _verr(:FitACE, _val_fitace(; ace_rmax1 = 5.0))
        @test msg !== nothing && occursin("unknown key :ace_rmax1", msg)

        msg = _verr(:FitACE, _val_fitace(;
            rigid = (c99 = 1.0,), ace_nk = (4, 5, 3, 3), traindata_ratio = 1.5))
        @test msg !== nothing
        @test occursin("unknown sub-key :c99 in params[:rigid]", msg)
        @test occursin("n0 >= n1 >= n2 >= n3 >= 3", msg)
        @test occursin("traindata_ratio must be in (0, 1]", msg)
    end

    @testset "FitACE: cg_preconditioner" begin
        allrigid = (; cp = 1e-8, c0 = 1.0, c00 = 1.0, c11 = 1.0, c22 = 1.0,
                      c33 = 1.0, c011 = 1.0, c022 = 1.0, c033 = 1.0)
        @test FMD.validate_params(:FitACE,
            _val_fitace(; cg_preconditioner = :blockchol_si, rigid = allrigid)) === nothing
        @test FMD.validate_params(:FitACE,
            _val_fitace(; cg_preconditioner = :none)) === nothing

        msg = _verr(:FitACE, _val_fitace(; cg_preconditioner = :jacobi))
        @test msg !== nothing && occursin("must be :none or :blockchol_si", msg)

        # :blockchol_si with the 2-field fixture rigid: every other field is
        # missing and must be named in the error
        msg = _verr(:FitACE, _val_fitace(; cg_preconditioner = :blockchol_si))
        @test msg !== nothing && occursin("rigid[f] > 0", msg)
        @test occursin("c033", msg)

        # zero entries are as bad as missing ones
        msg = _verr(:FitACE, _val_fitace(; cg_preconditioner = :blockchol_si,
            rigid = (; allrigid..., c11 = 0.0)))
        @test msg !== nothing && occursin("c11", msg)

        # typo hits the unknown-key / did-you-mean path
        msg = _verr(:FitACE, _val_fitace(; cg_precondtioner = :blockchol_si))
        @test msg !== nothing && occursin("unknown key :cg_precondtioner", msg)
    end

    @testset "FitSpline: sub-key typos are rejected" begin
        # rigid_curv typos were silently ignored before the validator.
        msg = _verr(:FitSpline, _val_fitspline(; rigid_curv = (c3_t = 1.0,)))
        @test msg !== nothing
        @test occursin("unknown sub-key :c3_t in params[:rigid_curv]", msg)
        # :c3_r and :c3_θ are equidistant; sorted candidates make :c3_r win
        @test occursin("did you mean :c3_r?", msg)
    end

    @testset "FitSpline: value ranges" begin
        msg = _verr(:FitSpline, _val_fitspline(; spline_nbasis3 = 3))
        @test msg !== nothing && occursin("must both be >= 4", msg)
    end

    @testset "nlh baseline cross-field checks" begin
        # The MD path derives the NLH neighbour list from the rc (Coulomb)
        # parent, so a baseline cutoff beyond rc would silently drop pairs.
        els = _VAL_ELEMENTS
        rc = _val_fitspline()[:rc]
        ok  = Calculators.NLHParam(els, 0.5 * rc)
        big = Calculators.NLHParam(els, 1.5 * rc)
        @test FMD.validate_params(:FitSpline, _val_fitspline(; nlh = ok)) === nothing
        @test FMD.validate_params(:FitACE, _val_fitace(; nlh = ok)) === nothing

        msg = _verr(:FitSpline, _val_fitspline(; nlh = big))
        @test msg !== nothing && occursin("must be <= rc", msg)

        msg = _verr(:FitSpline, _val_fitspline(; nlh = "not an NLHParam"))
        @test msg !== nothing && occursin("nlh must be an NLHParam", msg)

        # Species-count mismatch reads the wrong nuclear charges silently.
        one_species = Calculators.NLHParam(Dict("Na" => 1), 0.5 * rc)
        msg = _verr(:FitSpline, _val_fitspline(; nlh = one_species))
        @test msg !== nothing && occursin("was built for 1 species", msg)
    end

    @testset "FitSpline: the removed :zbl pin is rejected, not ignored" begin
        # Issue #160 removed the ZBL wall pin. An input script still passing
        # `zbl = (...)` must fail loudly — silently ignoring it would leave
        # the user believing a short-range wall is being fitted.
        msg = _verr(:FitSpline, _val_fitspline(; zbl = (λ = 1.0,)))
        @test msg !== nothing && occursin("unknown key :zbl", msg)
    end

    @testset "FitShellModel: dynamic lb keys" begin
        # keys(lb) are legitimate top-level keys (opt! injects them).
        @test FMD.validate_params(:FitShellModel, _val_fitsm()) === nothing

        # k4 added as an optimization variable: allowed because k4 is a
        # top-level key (from defaults) and appears in both lb and ub.
        p = _val_fitsm(;
            lb = (; _val_fitsm()[:lb]..., k4 = [0.0, 0.0]),
            ub = (; _val_fitsm()[:ub]..., k4 = [1.0, 1.0]),
            k4 = [0.5, 0.5])
        @test FMD.validate_params(:FitShellModel, p) === nothing
    end

    @testset "FitShellModel: lb/ub violations" begin
        base = _val_fitsm()

        # keys(lb) != keys(ub)
        msg = _verr(:FitShellModel, (; base..., ub = _drop(base[:ub], :k2)))
        @test msg !== nothing && occursin("keys(lb) and keys(ub) must match", msg)

        # lb key without an initial value: drop is impossible for qc
        # (required), so use a synthetic optimization variable name.
        msg = _verr(:FitShellModel, (; base...,
            lb = (; base[:lb]..., myvar = 0.0),
            ub = (; base[:ub]..., myvar = 1.0)))
        @test msg !== nothing && occursin(":myvar in lb has no initial value", msg)

        # lb >= ub
        msg = _verr(:FitShellModel, (; base..., lb = (; base[:lb]..., qref = 2.0)))
        @test msg !== nothing && occursin("lb.qref must be elementwise < ub.qref", msg)

        # initial value on the boundary (atan box transform diverges)
        msg = _verr(:FitShellModel, (; base..., qref = 1.5))
        @test msg !== nothing && occursin("strictly inside", msg)

        # opt_weights sub-keys
        msg = _verr(:FitShellModel, (; base..., opt_weights = (bec = 10.0,)))
        @test msg !== nothing && occursin("both :bec and :eps", msg)
    end

    @testset "elements checked before default_params" begin
        # Non-contiguous species ids previously hit BoundsError inside
        # default_params(Val(:FitSpline)) (Z[id] indexing).
        bad = (; elements = Dict("Na" => 1, "Cl" => 3))
        for job in (:NPTMD, :FitACE, :FitSpline, :FitShellModel)
            @test_throws ArgumentError FMD.default_params(Val(job), bad)
        end
        msg = _verr(:NPTMD, (; _drop(_val_nptmd(), :elements)..., elements = Dict("Na" => 1, "Cl" => 3)))
        @test msg !== nothing && occursin("permutation of 1:2", msg)

        # Missing elements entirely: explicit message, not KeyError.
        @test_throws ArgumentError FMD.default_params(Val(:NPTMD), (; foo = 1))
    end

    @testset "re-validation after param override is idempotent" begin
        # The umbrella-sampling scripts call app[:NPTMD] twice, overriding
        # a few keys in between without re-merging defaults.
        p = _val_nptmd()
        p2 = (; p..., input = @__FILE__, output_interval = 1000, nsteps = 10_000,
              final_temperature = 300.0, initial_temperature = 300.0,
              target_temperature = 300.0)
        @test FMD.validate_params(:NPTMD, p2) === nothing
    end

    @testset "app entry points call validate_params" begin
        # The core contract: validation runs as the first statement of
        # every app closure, before any side effect (readxyz / mkpath /
        # MPI.Init). A bogus key must therefore throw from app[:Job]
        # itself, not only from a direct validate_params call.
        for job in (:NPTMD, :FitACE, :FitSpline, :FitShellModel, :PolMap)
            @test_throws ArgumentError FMD.app[job]((; elements = _VAL_ELEMENTS, bogus = 1))
        end
    end

    @testset "_resolve_mlip exclusivity (direct npt! path)" begin
        @test_throws ErrorException FMD.Apps._resolve_mlip((; acep = _val_acep(), splp = _val_splp()), Array)
        # No MLIP is legal (Issue #133): resolves to nothing — the pure
        # shell model.
        @test FMD.Apps._resolve_mlip((;), Array) === nothing
    end

    @testset "_resolve_mlip / conv preserve eltype (Issue #39)" begin
        # cu() silently downcast Float64 coefficients to Float32; the
        # adapt / arrtyp-constructor replacements must keep FPPREC.
        @test eltype(FMD.Apps._resolve_mlip((; splp = _val_splp()), Array).c1) === FMD.FPPREC
        @test eltype(FMD.Apps._resolve_mlip((; acep = _val_acep()), Array).c0) === FMD.FPPREC
        @test eltype(FMD.Apps.conv(Array, rand(Float64, 3))) === FMD.FPPREC
    end

    @testset ":backend value and umbrella backend consistency" begin
        @test FMD.validate_params(:NPTMD, _val_nptmd(backend = :cpu)) === nothing
        @test FMD.validate_params(:NPTMD, _val_nptmd(backend = :cuda)) === nothing
        @test_throws ArgumentError FMD.validate_params(:NPTMD, _val_nptmd(backend = :gpu))

        # The umbrella bakes its backend into its arrays at construction;
        # validation must reject a mismatch with params[:backend] instead
        # of failing deep inside the force loop.
        a = make_nacl_8()
        # Groups must be charge neutral (PolarizationGauge rejects charged
        # ones), so pair each Na with a Cl: atoms 1..4 are Na, 5..8 Cl.
        a.arrays[:groupid] = vcat(1:4, 1:4)
        pu = Calculators.PolarizationUmbrella(
            a, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (1, 2, 2),
            FMD.FPPREC[1.0, -1.0], zeros(FMD.FPPREC, 2); gauge = a)
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(backend = :cpu, polarization_umbrella = pu)) === nothing
        # CPU-built umbrella under the default (:cuda) backend must fail.
        @test_throws ArgumentError FMD.validate_params(:NPTMD,
            _val_nptmd(polarization_umbrella = pu))
    end

    @testset ":surface_term validation (Issue #179)" begin
        a = make_nacl_8()
        qc = FMD.FPPREC[1.0, -1.0]
        qs = zeros(FMD.FPPREC, 2)
        st = Calculators.SurfaceTerm(a, qc, qs; eps_out = 3.0, gauge = a)
        @test FMD.validate_params(:NPTMD, _val_nptmd(surface_term = st)) === nothing

        # Wrong type fails at the boundary, not as a MethodError mid-run.
        msg = _verr(:NPTMD, _val_nptmd(surface_term = "not a term"))
        @test msg !== nothing && occursin("surface_term must be a SurfaceTerm", msg)

        # Shadow XL-BOMD is accepted since Issue #181 gave the
        # depolarization term its analytic HVP — the umbrella, whose bias
        # still has none, is the one that stays rejected.
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(surface_term = st, shell_dynamics = :xlbomd)) === nothing
        a.arrays[:groupid] = vcat(1:4, 1:4)   # neutral Na+Cl pairs
        pu_xl = Calculators.PolarizationUmbrella(
            a, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (1, 2, 2), qc, qs; gauge = a)
        msg = _verr(:NPTMD, _val_nptmd(backend = :cpu,
                                       polarization_umbrella = pu_xl,
                                       shell_dynamics = :xlbomd))
        @test msg !== nothing &&
              occursin("does not support :polarization_umbrella", msg)

        # Gauge consistency with a declared polarization_gauge: same frame
        # passes, a shifted frame's gauge is a different zero-point.
        pg = Calculators.PolarizationGauge(a, (1, 2, 2), qc, qs)
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(surface_term = st, polarization_gauge = pg)) === nothing
        shifted = deepcopy_atoms(a)
        shifted.arrays[:pos][1, 1] += 0.1
        pg_shift = Calculators.PolarizationGauge(shifted, (1, 2, 2), qc, qs)
        msg = _verr(:NPTMD, _val_nptmd(surface_term = st,
                                       polarization_gauge = pg_shift))
        @test msg !== nothing && occursin("gauge baked into surface_term", msg)

        # ... and with an umbrella: both from the same gauge passes, a
        # surface term on a shifted gauge is rejected.
        pu = Calculators.PolarizationUmbrella(
            a, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (1, 2, 2), qc, qs;
            gauge = pg)
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(backend = :cpu, polarization_umbrella = pu,
                       surface_term = st)) === nothing
        st_shift = Calculators.SurfaceTerm(shifted, qc, qs;
                                           eps_out = 3.0, gauge = shifted)
        msg = _verr(:NPTMD, _val_nptmd(backend = :cpu,
                                       polarization_umbrella = pu,
                                       surface_term = st_shift))
        @test msg !== nothing && occursin("different zero-points", msg)
    end

    @testset ":mpi opt-in validation" begin
        @test FMD.validate_params(:NPTMD, _val_nptmd(mpi = true, backend = :cpu)) === nothing
        @test FMD.validate_params(:NPTMD, _val_nptmd(mpi = false)) === nothing
        # GPU domain decomposition is a valid combination (Phase 2); the
        # CUDA-aware transport is probed at npt! entry, not here.
        @test FMD.validate_params(:NPTMD,
            _val_nptmd(mpi = true, backend = :cuda)) === nothing
        # Non-Bool values are rejected loudly.
        @test_throws ArgumentError FMD.validate_params(:NPTMD, _val_nptmd(mpi = 1))
    end

    @testset "build_mlip_workspace / apply_mlip! dispatch (npt! path)" begin
        # The CPU smoke drives npt! only through the Spline path, so the
        # ACE-side MLIP plumbing — workspace construction and the 4-arg
        # apply_mlip! dispatch — is exercised here directly on a small CPU
        # system.
        Random.seed!(35)
        a = make_perovskite_5(; a_ang = 4.0)
        a.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(a.arrays[:pos]))
        acep = ACEParam(FMD.FPPREC(1.0 / UNITS[:Å]), FMD.FPPREC(4.0 / UNITS[:Å]),
                        3, 4, 4, 3, 3)
        for f in fieldnames(typeof(acep))
            x = getfield(acep, f)
            x isa AbstractArray || continue
            x .= randn(FMD.FPPREC, size(x)) .* 1e-3
        end

        ws = FMD.Apps.build_mlip_workspace(a, (; acep = acep))
        @test ws isa Calculators.ACEWorkspace
        @test FMD.Apps.build_mlip_workspace(a, (; splp = _val_splp())) === nothing

        # ACE: the 4-arg apply_mlip! must equal a direct calculate!.
        nl = neighborlist(a, acep.rmax, 0.0)
        a1 = deepcopy_atoms(a); reset!(a1)
        FMD.Apps.apply_mlip!(a1, acep, (nl,), ws)
        a2 = deepcopy_atoms(a); reset!(a2)
        Calculators.calculate!(acep, a2, nl)
        @test abs(a2.info[:potential_energy][1]) > 1e-8
        @test isapprox(a1.info[:potential_energy][1], a2.info[:potential_energy][1];
                       rtol = 1e-10)
        @test isapprox(a1.arrays[:forces], a2.arrays[:forces];
                       rtol = 1e-10, atol = 1e-14)

        # Spline: 2 lists + no workspace must hit the ::Nothing method.
        splp = SplineParam(FMD.FPPREC(4.0 / UNITS[:Å]), FMD.FPPREC(3.0 / UNITS[:Å]),
                           3, 6, 5)
        splp.c1 .= randn(FMD.FPPREC, size(splp.c1)) .* 1e-3
        splp.c2 .= randn(FMD.FPPREC, size(splp.c2)) .* 1e-3
        splp.c3 .= randn(FMD.FPPREC, size(splp.c3)) .* 1e-3
        nl2 = neighborlist(a, splp.rmax2, 0.0)
        nl3 = neighborlist(a, splp.rmax3, 0.0)
        a3 = deepcopy_atoms(a); reset!(a3)
        FMD.Apps.apply_mlip!(a3, splp, (nl2, nl3), nothing)
        @test isfinite(a3.info[:potential_energy][1])
    end

    @testset "elements edge cases" begin
        # empty Dict must not slip through (Int[] == 1:0 is true)
        @test_throws ArgumentError FMD.default_params(Val(:NPTMD), (; elements = Dict{String,Int}()))
        msg = _verr(:NPTMD, (; _drop(_val_nptmd(), :elements)..., elements = Dict{String,Int}()))
        @test msg !== nothing && occursin("must not be empty", msg)

        # non-Integer species ids ([1.0, 2.0] == 1:2 is also true)
        msg = _verr(:NPTMD, (; _drop(_val_nptmd(), :elements)...,
                             elements = Dict("Na" => 1.0, "Cl" => 2.0)))
        @test msg !== nothing && occursin("Integer species ids", msg)
    end

    @testset "ewald_tolerance upper bound" begin
        # At tol >= 1 the k-space cutoff collapses to 0 and the
        # reciprocal-space Coulomb term silently vanishes.
        msg = _verr(:FitSpline, _val_fitspline(; ewald_tolerance = 1.0))
        @test msg !== nothing && occursin("ewald_tolerance must lie in (0, 1)", msg)
    end

    @testset "MLIP species-count consistency" begin
        acep3 = ACEParam(1.0 / UNITS[:Å], 0.8 * _VAL_RC, 3, 4, 4, 3, 3)
        msg = _verr(:NPTMD, _val_nptmd(; acep = acep3))
        @test msg !== nothing && occursin("fitted for 3 species but elements has 2", msg)

        p = (; _drop(_val_fitace(), :ace_rmax)..., acep = acep3)
        msg = _verr(:FitACE, p)
        @test msg !== nothing && occursin("fitted for 3 species", msg)
    end

    @testset "FitACE warm start is exclusive with ace_rmax" begin
        # warm start alone is valid
        p = (; _drop(_val_fitace(), :ace_rmax)..., acep = _val_acep())
        @test FMD.validate_params(:FitACE, p) === nothing

        # both given: ace_rmin/ace_rmax/ace_nk would be silently ignored
        msg = _verr(:FitACE, _val_fitace(; acep = _val_acep()))
        @test msg !== nothing && occursin("drop one to make the intent explicit", msg)
    end

    @testset "negative loss / regulariser weights" begin
        msg = _verr(:FitSpline, _val_fitspline(; weight_force = -1.0,
                                               rigid = (c1 = -1e-8, c2 = 1.0, c3 = 1.0)))
        @test msg !== nothing
        @test occursin("weight_force must be >= 0", msg)
        @test occursin("rigid entries must be nonnegative", msg)

        msg = _verr(:FitShellModel, (; _val_fitsm()..., opt_weights = (bec = -1.0, eps = 1.0)))
        @test msg !== nothing && occursin("opt_weights must be nonnegative", msg)

        msg = _verr(:FitShellModel, (; _val_fitsm()..., opt_weights = (bec = 0.0, eps = 0.0)))
        @test msg !== nothing && occursin("at least one of opt_weights", msg)
    end

    @testset "qs cannot be an optimization variable" begin
        base = _val_fitsm()
        msg = _verr(:FitShellModel, (; base...,
            lb = (; base[:lb]..., qs = [-2.0, -2.0]),
            ub = (; base[:ub]..., qs = [0.0, 0.0])))
        @test msg !== nothing && occursin("qs cannot be an optimization variable", msg)
    end

    @testset "non-NamedTuple params" begin
        @test_throws ArgumentError FMD.validate_params(:NPTMD, Dict(:elements => _VAL_ELEMENTS))
    end

    @testset "input/dataset existence" begin
        msg = _verr(:NPTMD, _val_nptmd(; input = "no_such_file.xyz"))
        @test msg !== nothing && occursin("input file not found", msg)

        msg = _verr(:FitSpline, _val_fitspline(; dataset = "no_such_file.xyz"))
        @test msg !== nothing && occursin("dataset file not found", msg)
    end
end
