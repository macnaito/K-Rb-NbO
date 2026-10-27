# ParamSpec.jl — single source of truth for the params schema of every
# FMD.app job.
#
# `validate_params(job, params)` runs at the top of each `app[:Job]`
# closure, on the *merged* params (defaults + caller overrides), before
# any side effect (file IO, MPI.Init, GPU allocation). A typo'd key,
# a missing required key, or a physically inconsistent combination is
# reported as one ArgumentError listing every violation at once.
#
# Direct calls to the lower-level drivers (`npt!`, `fitting!`, `opt!`)
# bypass validation on purpose — tests and advanced users construct
# partial params for those.

using StringDistances

export validate_params

"""
    ParamSpec

Validation spec for one job's `params` NamedTuple.

- `required` — keys that must exist in the *merged* params (keys supplied
  by `default_params` are listed here too).
- `optional` — keys that may exist. Also whitelists inert keys that a
  JLD2 splat or an input-script reuse legitimately introduces
  (e.g. `qref`).
- `subkeys` — allowed sub-keys for NamedTuple-valued entries whose keys
  are accessed via `getfield`/`get` downstream (a typo'd sub-key would
  otherwise be silently ignored).
- `checks` — cross-field constraints; each function maps `params` to
  `nothing` (pass) or an error message `String`. Checks must return
  `nothing` when their prerequisite keys are absent — the missing-key
  report covers those.
"""
struct ParamSpec
    required::Vector{Symbol}
    optional::Vector{Symbol}
    subkeys::Dict{Symbol,Vector{Symbol}}
    checks::Vector{Function}
end

# Evaluate `f(params)` only when all prerequisite keys are present.
_when(ks::Tuple, f) = p -> all(haskey(p, k) for k in ks) ? f(p) : nothing

# Key groups shared by every job (core-shell + Buckingham evaluator).
const SHELL_KEYS = Symbol[:qc, :qs, :k2, :k4, :eta_c, :eta_s]
const BUCK_KEYS = Symbol[:a, :rho, :b, :c, :e]
# :nsteps is NOT here: it means "run exactly this many iterations" for the
# MD and fitting jobs, but :Relax runs until converged and caps with
# :maxsteps instead. A job that needs :nsteps lists it itself (Issue #209).
const COMMON_KEYS = Symbol[:elements, :rc, :skin, :cgtol, :cgmaxiter,
                           :efield,
                           SHELL_KEYS..., BUCK_KEYS...]
# Ewald-path accuracy keys (Issue #216). Required by the fitting apps,
# whose Coulomb runs through EwaldInfo: there `ewald_tolerance` moves α
# and kcut together, so tightening it genuinely buys accuracy. The PME
# apps (:NPTMD / :Relax) list them as OPTIONAL instead — α is
# auto-determined by error-estimate minimization at fixed
# (:rc, :pme_meshsize) unless an explicit :alpha overrides it, and a
# leftover :ewald_tolerance is warned about and ignored (against a fixed
# grid it moved only α, making it a non-monotone accuracy knob).
const EWALD_KEYS = Symbol[:ewald_tolerance, :alpha]

"""
    _elements_error(params) -> Union{Nothing,String}

`:elements` must exist and its values must be a permutation of `1:nelem`:
every per-element array is sized by `length(elements)` and indexed by the
species id, so a gap (e.g. `Dict("K" => 1, "O" => 5)`) turns into an
out-of-bounds access deep inside a kernel.
"""
function _elements_error(params)
    haskey(params, :elements) ||
        return "missing :elements (Dict element symbol => species id); " *
               "default_params also needs it to size per-element arrays"
    isempty(params[:elements]) && return "elements must not be empty"
    ids = sort!(collect(values(params[:elements])))
    all(x -> x isa Integer, ids) ||
        return "values(elements) must be Integer species ids (got $ids)"
    ids == 1:length(ids) ||
        return "values(elements) must be a permutation of 1:$(length(ids)) (got $ids)"
    return nothing
end

"""
    _assert_elements(params)

Throw `ArgumentError` unless [`_elements_error`](@ref) passes. Called at
the top of every `default_params` method: the defaults are sized by
`length(elements)` and every downstream kernel indexes them by the species
id `setelemid!` assigns, so a gap in the id set is an out-of-bounds access
deep inside a kernel rather than an error at the input boundary.
"""
function _assert_elements(params)
    msg = _elements_error(params)
    msg === nothing || throw(ArgumentError(msg))
    return nothing
end

# "did you mean :timestep?" suffix for an unknown key, or "" when no
# candidate is close enough (Levenshtein distance ≤ max(1, length÷3)).
# Candidates are sorted so equidistant ties break deterministically.
function _didyoumean(key::Symbol, candidates)
    s = String(key)
    best, bestd = nothing, typemax(Int)
    for c in sort!(collect(candidates))
        d = Levenshtein()(s, String(c))
        d < bestd && ((best, bestd) = (c, d))
    end
    (best === nothing || bestd > max(1, length(s) ÷ 3)) && return ""
    return " — did you mean :$best?"
end

function _report(job::Symbol, errs::Vector{String})
    n = length(errs)
    return "params validation failed for FMD.app[:$job] ($n problem$(n == 1 ? "" : "s")):\n" *
           join(("  $i. $(errs[i])" for i in 1:n), "\n")
end

# --------------------------------------------------------------------------
# Shared cross-field checks
# --------------------------------------------------------------------------

function _shape_checks()
    checks = Function[]
    for k in SHELL_KEYS
        push!(checks, _when((:elements, k), p -> length(p[k]) == length(p[:elements]) ? nothing :
            "length($k) = $(length(p[k])) must equal nelem = $(length(p[:elements]))"))
    end
    for k in (:a, :rho, :b, :c)
        push!(checks, _when((:elements, k), p -> begin
            n = length(p[:elements])
            size(p[k]) == (n, n) ? nothing :
                "size($k) = $(size(p[k])) must be (nelem, nelem) = ($n, $n)"
        end))
    end
    # rho divides r, eta_c/eta_s divide charges — zero means NaN forces.
    push!(checks, _when((:rho,), p -> all(>(0), p[:rho]) ? nothing :
        "all(rho .> 0) is required (rho is a divisor in the Buckingham kernel)"))
    push!(checks, _when((:eta_c,), p -> all(>(0), p[:eta_c]) ? nothing : "all(eta_c .> 0) is required"))
    push!(checks, _when((:eta_s,), p -> all(>(0), p[:eta_s]) ? nothing : "all(eta_s .> 0) is required"))
    push!(checks, _when((:rc,), p -> p[:rc] > 0 ? nothing : "rc must be > 0"))
    push!(checks, _when((:skin,), p -> p[:skin] >= 0 ? nothing : "skin must be >= 0"))
    # ewaldcoeff computes erfcinv(rc * ewald_tolerance): argument must be in (0, 2).
    push!(checks, _when((:rc, :ewald_tolerance), p -> 0 < p[:rc]*p[:ewald_tolerance] < 2 ? nothing :
        "rc*ewald_tolerance = $(p[:rc]*p[:ewald_tolerance]) must lie in (0, 2) (erfcinv domain in ewaldcoeff)"))
    # kcut = 2α·sqrt(-log(tol)): at tol >= 1 the reciprocal-space k-set
    # becomes empty and the k-space Coulomb term silently vanishes.
    push!(checks, _when((:ewald_tolerance,), p -> 0 < p[:ewald_tolerance] < 1 ? nothing :
        "ewald_tolerance must lie in (0, 1) (at 1 the reciprocal-space cutoff " *
        "kcut becomes 0 and the k-space Coulomb term silently vanishes)"))
    push!(checks, _when((:efield,), p -> length(p[:efield]) == 3 ? nothing :
        "efield must be a 3-vector (got length $(length(p[:efield])))"))
    push!(checks, _when((:cgtol,), p -> p[:cgtol] > 0 ? nothing : "cgtol must be > 0"))
    push!(checks, _when((:cgmaxiter,), p -> p[:cgmaxiter] >= 1 ? nothing : "cgmaxiter must be >= 1"))
    push!(checks, _when((:nsteps,), p -> p[:nsteps] >= 1 ? nothing : "nsteps must be >= 1"))
    return checks
end

_fit_common_checks() = Function[
    _when((:dataset,), p -> isfile(p[:dataset]) ? nothing : "dataset file not found: $(p[:dataset])"),
    _when((:traindata_ratio,), p -> 0 < p[:traindata_ratio] <= 1 ? nothing :
        "traindata_ratio must be in (0, 1] (got $(p[:traindata_ratio]))"),
    _when((:weight_force,), p -> p[:weight_force] >= 0 ? nothing :
        "weight_force must be >= 0 (a negative loss weight makes the normal equations indefinite)"),
    _when((:weight_stress,), p -> p[:weight_stress] >= 0 ? nothing :
        "weight_stress must be >= 0 (a negative loss weight makes the normal equations indefinite)"),
    p -> _nonneg_subvals(p, :rigid),
]

# --------------------------------------------------------------------------
# Multi-line checks
#
# Named functions instead of inline closures: `begin ... end` blocks
# cannot appear inside the typed `Function[...]` literals below (inside
# `T[...]` the parser treats begin/end as index keywords).
# --------------------------------------------------------------------------

function _check_acep_cutoff(p)
    (haskey(p, :acep) && haskey(p, :rc)) || return nothing
    p[:acep] isa ACEParam || return "acep must be an ACEParam (got $(typeof(p[:acep])))"
    p[:acep].rmax <= p[:rc] && return nothing
    return "acep.rmax ($(round(p[:acep].rmax*UNITS[:Å]; digits = 3)) Å) must be <= rc " *
           "($(round(p[:rc]*UNITS[:Å]; digits = 3)) Å): MLIP neighbor lists are derived " *
           "from the rc list, so a larger MLIP cutoff silently drops neighbors"
end

function _check_splp_cutoff(p)
    (haskey(p, :splp) && haskey(p, :rc)) || return nothing
    p[:splp] isa SplineParam || return "splp must be a SplineParam (got $(typeof(p[:splp])))"
    max(p[:splp].rmax2, p[:splp].rmax3) <= p[:rc] && return nothing
    return "splp.rmax2/rmax3 ($(round(max(p[:splp].rmax2, p[:splp].rmax3)*UNITS[:Å]; digits = 3)) Å) " *
           "must be <= rc ($(round(p[:rc]*UNITS[:Å]; digits = 3)) Å): MLIP neighbor lists are " *
           "derived from the rc list, so a larger MLIP cutoff silently drops neighbors"
end

function _check_nlh_cutoff(p)
    (haskey(p, :nlh) && haskey(p, :rc)) || return nothing
    p[:nlh] isa NLHParam || return "nlh must be an NLHParam (got $(typeof(p[:nlh])))"
    p[:nlh].rc <= p[:rc] && return nothing
    return "nlh.rc ($(round(p[:nlh].rc*UNITS[:Å]; digits = 3)) Å) must be <= rc " *
           "($(round(p[:rc]*UNITS[:Å]; digits = 3)) Å): the NLH neighbor list is derived " *
           "from the rc list, so a larger baseline cutoff silently drops neighbors"
end

function _check_nlh_species(p)
    (haskey(p, :nlh) && haskey(p, :elements) && p[:nlh] isa NLHParam) || return nothing
    n = length(p[:nlh].Z)
    n == length(p[:elements]) && return nothing
    return "nlh was built for $n species but elements has $(length(p[:elements])) — " *
           "a species-count mismatch reads the wrong nuclear charges without any error"
end

function _check_acep_species(p)
    (haskey(p, :acep) && haskey(p, :elements) && p[:acep] isa ACEParam) || return nothing
    n = length(p[:acep].cp)
    n == length(p[:elements]) && return nothing
    return "acep was fitted for $n species but elements has $(length(p[:elements])) — " *
           "a species-count mismatch reads wrong coefficients without any error"
end

function _check_splp_species(p)
    (haskey(p, :splp) && haskey(p, :elements) && p[:splp] isa SplineParam) || return nothing
    n = size(p[:splp].c2, 2)
    n == length(p[:elements]) && return nothing
    return "splp was fitted for $n species but elements has $(length(p[:elements])) — " *
           "a species-count mismatch reads wrong coefficients without any error"
end

# The 3-body kernel reads c3 at raw (sj, sk) with no canonicalization, so
# the energy is invariant under neighbor-enumeration reordering ONLY for
# c3 ∈ im(P) (the (j,k)-swap-symmetric subspace, Issue #1). The fit path
# enforces this via projection; a hand-built or corrupted file does not,
# and silently returns enumeration-order-dependent energies (Issue #63:
# ~10% macro E difference measured on an asymmetric c3), so refuse to run.
function _check_splp_c3sym(p)
    (haskey(p, :splp) && p[:splp] isa SplineParam) || return nothing
    c3 = p[:splp].c3
    # Shape gate first: SplineParam's fields are fully parametric, so a
    # malformed file passes `isa SplineParam` — it must surface as a
    # collected message, not a MethodError/ArgumentError mid-check.
    c3 isa AbstractArray{<:Real,6} ||
        return "splp.c3 must be a 6-d Real array (nb, nb, nb, ns, ns, ns) " *
               "(got $(typeof(c3)))"
    nb = size(c3, 1)
    ns = size(c3, 4)
    size(c3) == (nb, nb, nb, ns, ns, ns) ||
        return "splp.c3 axes must be (nb, nb, nb, ns, ns, ns) (got $(size(c3)))"
    ns == size(p[:splp].c2, 2) ||
        return "splp.c3 species axes ($ns) do not match splp.c2 " *
               "($(size(p[:splp].c2, 2)))"
    dev = Calculators._spl_c3_asymmetry(c3)
    iszero(dev) && return nothing
    rel = dev / maximum(abs, c3)   # dev > 0 ⇒ maximum > 0
    return "splp.c3 violates the symmetry contract c3 ∈ im(P) " *
           "(max |c3[a,b,θ,sa,sb,si] - c3[b,a,θ,sb,sa,si]| = $dev, " *
           "$(rel) relative to max |c3|): the 3-body kernel reads c3 without " *
           "canonicalization, so an asymmetric c3 gives " *
           "neighbor-enumeration-order-dependent energies. If the asymmetry is " *
           "intentional noise, project deliberately with " *
           "FMD.Calculators._spl_symmetrize_c3!(splp.c3); otherwise fix the file"
end

# All values of a NamedTuple-valued key must be nonnegative Reals
# (negative loss/regulariser weights make the CG normal equations
# indefinite while the operator is declared isposdef — silent garbage).
function _nonneg_subvals(p, key)
    haskey(p, key) || return nothing
    nt = p[key]
    nt isa NamedTuple || return nothing   # reported by the sub-key whitelist
    bad = [k for k in keys(nt) if !(nt[k] isa Real) || nt[k] < 0]
    isempty(bad) && return nothing
    return "$key entries must be nonnegative Reals (offending: " *
           join((":$k" for k in bad), ", ") * ")"
end

function _check_ace_warmstart(p)
    if haskey(p, :acep) && haskey(p, :ace_rmax)
        return "both :acep (warm start) and :ace_rmax were provided — ace_rmin/ace_rmax/ace_nk " *
               "are ignored when acep is given; drop one to make the intent explicit"
    end
    (haskey(p, :acep) || haskey(p, :ace_rmax)) && return nothing
    return "either :acep (warm start) or :ace_rmax must be provided " *
           "(no default — the MLIP cutoff is a deliberate physical choice)"
end

function _check_ace_nk(p)
    haskey(p, :ace_nk) || return nothing
    nk = p[:ace_nk]
    length(nk) == 4 || return "ace_nk must have 4 entries (n0, n1, n2, n3) (got $(length(nk)))"
    (nk[1] >= nk[2] >= nk[3] >= nk[4] >= 3) && return nothing
    return "ace_nk must satisfy n0 >= n1 >= n2 >= n3 >= 3 (got $nk)"
end

# blockchol-si solves each block as live-column Cholesky ⊕ Γ_field·I on the
# structurally dead lower-triangle coefficients (zero columns of the design
# matrix); Γ_field = 0 would leave M⁻¹ undefined on the dead directions and
# make the live sub-block's regularity data-dependent.
function _check_blockchol_rigid(p)
    haskey(p, :cg_preconditioner) || return nothing
    p[:cg_preconditioner] === :blockchol_si || return nothing
    haskey(p, :rigid) || return nothing   # :rigid missing is reported as a required-key error
    bad = [f for f in (:cp, :c0, :c00, :c11, :c22, :c33, :c011, :c022, :c033)
           if !haskey(p[:rigid], f) || !(p[:rigid][f] > 0)]
    isempty(bad) && return nothing
    return "cg_preconditioner = :blockchol_si requires rigid[f] > 0 for every " *
           "coefficient field (missing or nonpositive: $(join(bad, ", ")))"
end

function _check_opt_weights(p)
    haskey(p, :opt_weights) || return nothing
    ow = p[:opt_weights]
    ow isa NamedTuple || return "opt_weights must be a NamedTuple (bec = ..., eps = ...)"
    (haskey(ow, :bec) && haskey(ow, :eps)) || return "opt_weights needs both :bec and :eps"
    (ow.bec >= 0 && ow.eps >= 0) || return "opt_weights must be nonnegative"
    (ow.bec > 0 || ow.eps > 0) || return "at least one of opt_weights.bec / opt_weights.eps must be > 0"
    return nothing
end

function _check_lb_ub(p)
    (haskey(p, :lb) && haskey(p, :ub)) || return nothing
    lb, ub = p[:lb], p[:ub]
    (lb isa NamedTuple && ub isa NamedTuple) || return "lb and ub must be NamedTuples"
    Set(keys(lb)) == Set(keys(ub)) ||
        return "keys(lb) and keys(ub) must match (lb: $(collect(keys(lb))), ub: $(collect(keys(ub))))"
    msgs = String[]
    # qs is derived (qs = shell_charges(...)) after every candidate
    # injection, so optimizing it is a silent no-op.
    :qs in keys(lb) &&
        push!(msgs, "qs cannot be an optimization variable — opt! overwrites qs via shell_charges after every candidate injection")
    for k in keys(lb)
        if !haskey(p, k)
            push!(msgs, "optimization variable :$k in lb has no initial value in params")
            continue
        end
        if !(size(lb[k]) == size(ub[k]) == size(p[k]))
            push!(msgs, "size mismatch for :$k among lb / ub / initial value")
            continue
        end
        all(lb[k] .< ub[k]) || push!(msgs, "lb.$k must be elementwise < ub.$k")
        # from_box_atan maps the open box to R; a boundary (or outside)
        # initial value lands on ±Inf / gets silently clamped.
        all(lb[k] .< p[k] .< ub[k]) ||
            push!(msgs, "initial params[:$k] must lie strictly inside (lb.$k, ub.$k)")
    end
    return isempty(msgs) ? nothing : join(msgs, "; ")
end

# NPTMD control selection (Issue #136): allowed method sets and the
# value-conditional time-constant requirements. :scr is NOT accepted
# until its implementing PR lands (Issue #116) — accepting a symbol that
# silently falls back to another method would be a placeholder. The
# :thermostat/:barostat keys themselves are required (supplied by
# default_params), so absence is covered by the missing-key report and
# this check may assume presence-or-skip per key.
function _check_control_params(p)
    msgs = String[]
    if haskey(p, :thermostat) && !(p[:thermostat] in (:nh, :csvr, :none))
        push!(msgs, "thermostat must be :nh, :csvr or :none (got $(repr(p[:thermostat])))")
    end
    if haskey(p, :barostat) && !(p[:barostat] in (:mttk, :scr, :none))
        push!(msgs, "barostat must be :mttk, :scr or :none (got $(repr(p[:barostat])))")
    end
    # Defaults here mirror the npt! builders (_build_thermostat /
    # _build_barostat read :nh / :mttk when the key is absent) so the
    # validator judges the same configuration the driver would run.
    th = get(p, :thermostat, :nh)
    if th in (:nh, :csvr) && !haskey(p, :tau_t)
        push!(msgs, "thermostat = $(repr(th)) requires :tau_t")
    end
    if th === :csvr && haskey(p, :target_temperature) && !(p[:target_temperature] >= 1.0)
        push!(msgs, "thermostat = :csvr requires target_temperature >= 1 K " *
                    "(the shared kT floor max(T, 1 K) would silently " *
                    "thermostat any lower target to 1 K)")
    end
    if th === :csvr && haskey(p, :final_temperature) &&
       0 < p[:final_temperature] < 1.0
        push!(msgs, "thermostat = :csvr requires final_temperature >= 1 K " *
                    "when ramping (same kT-floor hazard as target_temperature)")
    end
    ba = get(p, :barostat, :mttk)
    if ba in (:mttk, :scr)
        haskey(p, :tau_p) || push!(msgs, "barostat = $(repr(ba)) requires :tau_p")
        haskey(p, :target_stress) ||
            push!(msgs, "barostat = $(repr(ba)) requires :target_stress")
    end
    if ba === :scr
        haskey(p, :compressibility) ||
            push!(msgs, "barostat = :scr requires :compressibility " *
                        "(κ_T estimate, inverse-pressure a.u. — only sets " *
                        "the effective relaxation time)")
        if th === :none
            push!(msgs, "barostat = :scr requires a thermostat " *
                        "(thermostat = :none samples no well-defined " *
                        "ensemble with an overdamped stochastic barostat)")
        end
    end
    if haskey(p, :compressibility)
        ba === :scr || push!(msgs, ":compressibility is only read by " *
                                   "barostat = :scr — drop it (got barostat = $(repr(ba)))")
        p[:compressibility] > 0 || push!(msgs, "compressibility must be > 0")
    end
    haskey(p, :tau_t) && !(p[:tau_t] > 0) && push!(msgs, "tau_t must be > 0")
    haskey(p, :tau_p) && !(p[:tau_p] > 0) && push!(msgs, "tau_p must be > 0")

    # Shell dynamics (Issue #149). The default :scf is the historical
    # per-step relaxation; :xlbomd is the opt-in shadow XL-BOMD path, which
    # runs without the umbrella bias. Barostats are supported since the
    # shadow virial landed (Issue #151), and :surface_term since its
    # analytic HVP landed (Issue #181) — the spectral guards that
    # combination needs live in xl_init!, which sees the charges.
    sd = get(p, :shell_dynamics, :scf)
    if haskey(p, :shell_dynamics) && !(sd in (:scf, :xlbomd))
        push!(msgs, "shell_dynamics must be :scf or :xlbomd (got $(repr(sd)))")
    end
    if sd === :xlbomd
        haskey(p, :polarization_umbrella) &&
            push!(msgs, "shell_dynamics = :xlbomd does not support " *
                        ":polarization_umbrella (no analytic HVP for the bias)")
        haskey(p, :xl_kappa) && !(p[:xl_kappa] > 0) &&
            push!(msgs, "xl_kappa must be > 0")
        haskey(p, :xl_warmup) && !(p[:xl_warmup] >= 1) &&
            push!(msgs, "xl_warmup must be >= 1 (the auxiliary field is seeded " *
                        "from a relaxed state)")
        haskey(p, :xl_max_deviation) && !(p[:xl_max_deviation] > 0) &&
            push!(msgs, "xl_max_deviation must be > 0")
        # Kernel preconditioning of the auxiliary propagation (Issue #152).
        kern = get(p, :xl_kernel, :spring)
        haskey(p, :xl_kernel) && !(kern in (:spring, :chebyshev)) &&
            push!(msgs, "xl_kernel must be :spring or :chebyshev (got $(repr(kern)))")
        for k in (:xl_kernel_order, :xl_kernel_spectrum)
            haskey(p, k) && kern !== :chebyshev &&
                push!(msgs, ":$k is only read by xl_kernel = :chebyshev — " *
                            "drop it (got xl_kernel = $(repr(kern)))")
        end
        haskey(p, :xl_kernel_order) &&
            !(p[:xl_kernel_order] isa Integer && p[:xl_kernel_order] >= 2) &&
            push!(msgs, "xl_kernel_order must be an Integer >= 2 (order 2 is the " *
                        "cheapest useful polynomial: one extra Hessian-vector product)")
        if haskey(p, :xl_kernel_spectrum)
            sp = p[:xl_kernel_spectrum]
            length(sp) == 2 && all(x -> x isa Real, sp) && 0 < sp[1] < sp[2] ||
                push!(msgs, "xl_kernel_spectrum must be (a, b) with 0 < a < b, " *
                            "bounding the spectrum of spring⁻¹·H_ss")
        end
    else
        for k in (:xl_kappa, :xl_warmup, :xl_max_deviation,
                  :xl_kernel, :xl_kernel_order, :xl_kernel_spectrum)
            haskey(p, k) &&
                push!(msgs, ":$k is only read by shell_dynamics = :xlbomd — " *
                            "drop it (got shell_dynamics = $(repr(sd)))")
        end
    end
    return isempty(msgs) ? nothing : join(msgs, "; ")
end

# --------------------------------------------------------------------------
# Job specs
# --------------------------------------------------------------------------

const SPECS = Dict{Symbol,ParamSpec}()

SPECS[:NPTMD] = ParamSpec(
    [:input, :output_dir, :pme_meshsize,
     :initial_temperature, :target_temperature,
     :timestep, :thermostat, :barostat, :output_interval, :seed,
     :nsteps, COMMON_KEYS...],
    # EWALD_KEYS are optional here (Issue #216, see their definition):
    # :alpha present = explicit override of the auto-determined α,
    # :ewald_tolerance present = legacy input, warned about and ignored.
    # :qref rides in via the JLD2 shell splat and is not read by NPTMD;
    # :dataset_idx covers fit-input reuse. Both are inert but legitimate.
    # :tau_t / :tau_p / :target_stress are conditionally required
    # (value-dependent on the :thermostat / :barostat selection —
    # _check_control_params), so they live here rather than in the
    # unconditional required list: a fixed-cell (barostat = :none) run
    # needs no target stress and no barostat time constant.
    [EWALD_KEYS...,
     :acep, :splp, :nlh, :final_temperature, :polarization_umbrella,
     :polarization_gauge, :surface_term,
     # Local polarization map output (Issue #220): every :polmap_interval
     # steps, against :polarization_gauge (required with it — no
     # starting-frame fallback, see the check below).
     :polmap_interval,
     :qref, :dataset_idx, :backend, :mpi, :tau_t, :tau_p, :target_stress,
     :compressibility,
     # PME reciprocal-space z-slab decomposition (Issue #195): default on
     # under MPI with P > 1; :pme_dd => false forces the replicated solve
     # (debug / non-CUDA-aware segmented collectives).
     :pme_dd,
     # Listed so the check below can reject it with a reason. Without this
     # the unknown-key report fires first and says only "unknown key
     # :cell_dof", which reads as a typo rather than an unsupported feature.
     :cell_dof,
     # Shell dynamics (Issue #149): :shell_dynamics selects the per-step
     # shell path (:scf default / :xlbomd), the :xl_* knobs are read by
     # :xlbomd only — all conditionally checked in _check_control_params.
     # Deliberately not in default_params: adding keys there would change
     # the NamedTuple shape the precompile workload mirrors.
     :shell_dynamics, :xl_kappa, :xl_warmup, :xl_max_deviation,
     :xl_kernel, :xl_kernel_order, :xl_kernel_spectrum],
    Dict{Symbol,Vector{Symbol}}(),
    Function[
        # No MLIP at all is a supported configuration: the pure shell model
        # (Coulomb + Buckingham + Spring). Only "both" is rejected — there
        # is no principled way to choose between them.
        p -> !(haskey(p, :acep) && haskey(p, :splp)) ? nothing :
            "at most one of :acep / :splp may be provided (got both); " *
            "omit both to run the pure shell model",
        _when((:backend,), p -> p[:backend] in (:cuda, :cpu) ? nothing :
            "backend must be :cuda or :cpu (got $(repr(p[:backend])))"),
        _when((:seed,), p -> p[:seed] isa Integer ? nothing :
            "seed must be an Integer (got $(typeof(p[:seed])))"),
        _when((:mpi,), p -> p[:mpi] isa Bool ? nothing :
            "mpi must be a Bool (got $(typeof(p[:mpi])))"),
        _when((:pme_dd,), p -> p[:pme_dd] isa Bool ? nothing :
            "pme_dd must be a Bool (got $(typeof(p[:pme_dd])))"),
        # The umbrella bakes its backend into its arrays / FFT plans at
        # construction; a mismatch with the npt! backend would fail deep in
        # the force loop (or worse) instead of here.
        _when((:polarization_umbrella,), function (p)
            want_host = get(p, :backend, :cuda) === :cpu
            is_host = p[:polarization_umbrella].Ghat isa Array
            return want_host == is_host ? nothing :
                "polarization_umbrella was built for the " *
                (is_host ? "CPU" : "GPU") * " backend but params[:backend] " *
                (haskey(p, :backend) ? "= $(repr(p[:backend]))" : "defaults to :cuda") *
                " — rebuild it with backend = " *
                (want_host ? "KernelAbstractions.CPU()" : "CUDA.CUDABackend()")
        end),
        # A wrong type here would surface as a MethodError deep inside the
        # dipole reporting, after the run has already started.
        _when((:polarization_gauge,),
              p -> p[:polarization_gauge] isa Calculators.PolarizationGauge ? nothing :
                   "polarization_gauge must be a PolarizationGauge (got " *
                   "$(typeof(p[:polarization_gauge]))) — build it from the " *
                   "prototype frame or load_gauge(path)"),
        _when((:surface_term,),
              p -> p[:surface_term] isa Calculators.SurfaceTerm ? nothing :
                   "surface_term must be a SurfaceTerm (got " *
                   "$(typeof(p[:surface_term]))) — build it with " *
                   "SurfaceTerm(atoms, qc, qs; eps_out, gauge)"),
        # Two gauges that disagree would make Bias.log and dipole_phys refer
        # to different zero-points with nothing reporting the discrepancy.
        _when((:polarization_gauge, :polarization_umbrella), function (p)
            mfr_u = Array(p[:polarization_umbrella].M_frac_ref)
            mfr_g = p[:polarization_gauge].M_frac_ref
            # sqrt(eps) rather than a fixed 1e-10: FPPREC is a build-time
            # Preference (1e-10 is below eps in a Float32 build), and a gauge
            # built on the GPU sums group moments with atomics, whose order —
            # and therefore last bits — is not reproducible.
            tol = sqrt(eps(FPPREC))
            size(mfr_u) == size(mfr_g) && isapprox(mfr_u, mfr_g; rtol = tol) ?
                nothing :
                "polarization_gauge and the gauge baked into " *
                "polarization_umbrella differ — the bias and the reported " *
                "dipole_phys would use different zero-points.  Build the " *
                "umbrella with `gauge = params[:polarization_gauge]`"
        end),
        # The surface term measures δM from its own frozen gauge total; a
        # zero-point that disagrees with the declared gauge (or with the
        # umbrella's) would bias the dipole against a different reference
        # than the one being reported/sampled, silently.
        _when((:polarization_gauge, :surface_term), function (p)
            p[:surface_term] isa Calculators.SurfaceTerm || return nothing
            mfr_s = p[:surface_term].M_frac_ref
            mfr_g = Calculators.total_mfrac_ref(p[:polarization_gauge])
            # atol as well as rtol: the TOTAL fractional dipole of a
            # centrosymmetric prototype is legitimately ~0, where a pure
            # rtol comparison of two independently-summed values fails on
            # accumulation noise alone.
            tol = sqrt(eps(FPPREC))
            length(mfr_s) == length(mfr_g) &&
                isapprox(mfr_s, mfr_g; rtol = tol, atol = tol) ? nothing :
                "polarization_gauge and the gauge baked into surface_term " *
                "differ — δM and the reported dipole_phys would use " *
                "different zero-points.  Build the surface term with " *
                "`gauge = params[:polarization_gauge]`"
        end),
        _when((:polarization_umbrella, :surface_term), function (p)
            p[:surface_term] isa Calculators.SurfaceTerm || return nothing
            mfr_s = p[:surface_term].M_frac_ref
            mfr_u = vec(sum(Array(p[:polarization_umbrella].M_frac_ref),
                            dims = 1))
            tol = sqrt(eps(FPPREC))   # atol too — see the gauge check above
            length(mfr_s) == length(mfr_u) &&
                isapprox(mfr_s, mfr_u; rtol = tol, atol = tol) ? nothing :
                "surface_term and polarization_umbrella carry different " *
                "gauges — the two biases would measure the dipole from " *
                "different zero-points.  Build both from the same " *
                "PolarizationGauge"
        end),
        _when((:input,), p -> isfile(p[:input]) ? nothing : "input file not found: $(p[:input])"),
        # `cell_dof` belongs to :Relax. The default unknown-key report would
        # suggest ":cell_factor?" and send the user chasing a typo, so say
        # what is actually true: the barostats are not constrained yet.
        # Wiring it in needs MTTK's cell thermostat target to become
        # ndof*kT (it hard-codes 6*kT, with the matching term in its
        # conserved quantity) and, for SCR, a re-derivation of the
        # fluctuation-dissipation amplitude. See _cell_projector.
        p -> !haskey(p, :cell_dof) ? nothing :
            ":cell_dof is not supported by :NPTMD — the barostats are always " *
            "fully flexible. It is a :Relax key (Issue #212); constraining " *
            "the MD cell additionally requires the barostat DOF count and " *
            "the SCR noise amplitude to follow the constraint",
        # The MLIP neighbor lists are derived from the rc (Coulomb) list;
        # an MLIP cutoff beyond rc silently drops neighbors — wrong physics
        # with no error. This is the single most important check here.
        _check_acep_cutoff,
        _check_splp_cutoff,
        _check_nlh_cutoff,
        _check_acep_species,
        _check_splp_species,
        _check_nlh_species,
        _check_splp_c3sym,
        _when((:target_stress,), p -> length(p[:target_stress]) == 6 ? nothing :
            "target_stress must be Voigt-6 [σxx, σyy, σzz, σyz, σxz, σxy] (got length $(length(p[:target_stress])))"),
        _when((:timestep,), p -> p[:timestep] > 0 ? nothing : "timestep must be > 0"),
        _check_control_params,
        _when((:output_interval,), p -> p[:output_interval] >= 1 ? nothing :
            "output_interval must be >= 1 (0 raises DivideError at `step % output_interval`)"),
        _when((:polmap_interval,), p -> p[:polmap_interval] isa Integer && p[:polmap_interval] >= 1 ?
            nothing :
            "polmap_interval must be a positive Integer (got $(repr(p[:polmap_interval])))"),
        # A map has no starting-frame fallback, unlike dipole_phys: a map
        # measured from wherever the run happened to start is not a
        # polarization, and chained runs would each carry a different zero.
        _when((:polmap_interval,), p -> haskey(p, :polarization_gauge) ? nothing :
            "polmap_interval requires :polarization_gauge — the local polarization " *
            "map is only defined against a declared zero-point (load_gauge of the " *
            "shared gauge file, or PolarizationGauge(prototype, gridsize, qc, qs))"),
        _when((:pme_meshsize,), p -> p[:pme_meshsize] > 0 ? nothing : "pme_meshsize must be > 0"),
        _when((:initial_temperature,), p -> p[:initial_temperature] >= 0 ? nothing :
            "initial_temperature must be >= 0"),
        _when((:target_temperature,), p -> p[:target_temperature] >= 0 ? nothing :
            "target_temperature must be >= 0"),
        _when((:final_temperature,), p -> p[:final_temperature] >= 0 ? nothing :
            "final_temperature must be >= 0 (omit it, or pass 0, to disable the ramp)"),
        _shape_checks()...,
    ],
)

# Post-process local polarization maps (Issue #220). The shell charges and
# :elements normally arrive via the `pot...` splat of loadpotential, whose
# other fields (spring constants, MLIP, cutoff) are inert here — listed as
# optional so the same splat validates without stripping keys.
SPECS[:PolMap] = ParamSpec(
    [:input, :output_dir, :polarization_gauge, :elements, :qc, :qs],
    [:overwrite,
     :qref, :k2, :k4, :eta_c, :eta_s, :rc, :acep, :splp, :nlh, :dataset_idx],
    Dict{Symbol,Vector{Symbol}}(),
    Function[
        _when((:input,), function (p)
            inp = p[:input]
            paths = inp isa AbstractString ? [inp] :
                    (inp isa AbstractVector && all(x -> x isa AbstractString, inp) ?
                     collect(inp) : nothing)
            paths === nothing &&
                return "input must be an extxyz path or a Vector of paths (got $(typeof(inp)))"
            isempty(paths) && return "input is empty — nothing to post-process"
            absent = filter(!isfile, paths)
            return isempty(absent) ? nothing :
                "input file(s) not found: " * join(absent, ", ")
        end),
        _when((:polarization_gauge,),
              p -> p[:polarization_gauge] isa Calculators.PolarizationGauge ? nothing :
                   "polarization_gauge must be a PolarizationGauge (got " *
                   "$(typeof(p[:polarization_gauge]))) — load_gauge(path) of the " *
                   "file the run used, or build it from the prototype frame"),
        _when((:overwrite,), p -> p[:overwrite] isa Bool ? nothing :
            "overwrite must be a Bool (got $(typeof(p[:overwrite])))"),
        _shape_checks()...,
    ],
)

SPECS[:Relax] = ParamSpec(
    [:input, :output_dir, :pme_meshsize, :maxsteps, :ftol,
     :dt0, :dt_max, :dmax, :output_interval,
     COMMON_KEYS...],
    # :qref / :dataset_idx ride in via the JLD2 shell splat and the reuse of
    # a fit input script; inert but legitimate (same as :NPTMD).
    # EWALD_KEYS optional as in :NPTMD (Issue #216): :alpha = override,
    # :ewald_tolerance = legacy, warned and ignored.
    [EWALD_KEYS...,
     :acep, :splp, :nlh, :surface_term, :polarization_gauge,
     :qref, :dataset_idx, :backend, :mpi, :pme_dd,
     :verify_cgtol_factor, :verify_mesh_factor, :ftol_cgtol_margin,
     # Cell degrees of freedom (Issue #212). :stol / :target_stress are
     # conditionally required — a fixed-cell run has no use for either, and
     # relax! demands them when cell_dof != :none.
     :cell_dof, :cell_factor, :stol, :emax, :target_stress, :pme_rebuild_ratio],
    Dict{Symbol,Vector{Symbol}}(),
    Function[
        p -> !(haskey(p, :acep) && haskey(p, :splp)) ? nothing :
            "at most one of :acep / :splp may be provided (got both); " *
            "omit both to run the pure shell model",
        _when((:backend,), p -> p[:backend] in (:cuda, :cpu) ? nothing :
            "backend must be :cuda or :cpu (got $(repr(p[:backend])))"),
        _when((:mpi,), p -> p[:mpi] isa Bool ? nothing :
            "mpi must be a Bool (got $(typeof(p[:mpi])))"),
        _when((:pme_dd,), p -> p[:pme_dd] isa Bool ? nothing :
            "pme_dd must be a Bool (got $(typeof(p[:pme_dd])))"),
        _when((:input,), p -> isfile(p[:input]) ? nothing : "input file not found: $(p[:input])"),
        # Same MLIP-cutoff / species checks as :NPTMD — an MLIP cutoff beyond
        # rc silently drops neighbors, which is wrong physics with no error.
        _check_acep_cutoff,
        _check_splp_cutoff,
        _check_nlh_cutoff,
        _check_acep_species,
        _check_splp_species,
        _check_nlh_species,
        _check_splp_c3sym,
        _when((:surface_term,),
              p -> p[:surface_term] isa Calculators.SurfaceTerm ? nothing :
                   "surface_term must be a SurfaceTerm (got " *
                   "$(typeof(p[:surface_term]))) — build it with " *
                   "SurfaceTerm(atoms, qc, qs; eps_out, gauge)"),
        _when((:polarization_gauge,),
              p -> p[:polarization_gauge] isa Calculators.PolarizationGauge ? nothing :
                   "polarization_gauge must be a PolarizationGauge (got " *
                   "$(typeof(p[:polarization_gauge])))"),
        _when((:maxsteps,), p -> p[:maxsteps] >= 1 ? nothing : "maxsteps must be >= 1"),
        _when((:cell_dof,), p -> p[:cell_dof] in (:none, :iso, :diag, :full) ? nothing :
            "cell_dof must be :none (fixed cell), :iso (hydrostatic only), " *
            ":diag (axes free, no shear) or :full (got $(repr(p[:cell_dof])))"),
        # Conditionally required, value-dependent — hence a check rather than
        # the unconditional required list.
        _when((:cell_dof,), function (p)
            p[:cell_dof] === :none && return nothing
            miss = [k for k in (:stol, :target_stress) if !haskey(p, k)]
            isempty(miss) ? nothing :
                "cell_dof = $(repr(p[:cell_dof])) requires " *
                join((":$k" for k in miss), " and ") *
                " (the cell half of the convergence test needs a stress " *
                "tolerance and a target)"
        end),
        _when((:cell_dof, :efield), function (p)
            p[:cell_dof] === :none && return nothing
            any(!iszero, p[:efield]) || return nothing
            return "efield is non-zero and cell_dof = $(repr(p[:cell_dof])): " *
                   "ExternalField contributes no energy and no virial, so the " *
                   "field's strain derivative is missing from the stress and " *
                   "the cell would relax to the wrong shape. Use " *
                   "cell_dof = :none with a field, or drop the field"
        end),
        _when((:stol,), p -> p[:stol] > 0 ? nothing : "stol must be > 0"),
        _when((:cell_factor,), p -> p[:cell_factor] > 0 ? nothing : "cell_factor must be > 0"),
        _when((:emax,), p -> p[:emax] > 0 ? nothing : "emax must be > 0"),
        _when((:pme_rebuild_ratio,), p -> p[:pme_rebuild_ratio] > 1 ? nothing :
            "pme_rebuild_ratio must be > 1 (it is the edge-length ratio that " *
            "triggers a PME rebuild; <= 1 would rebuild every step)"),
        _when((:target_stress,), p -> length(p[:target_stress]) == 6 ? nothing :
            "target_stress must be Voigt-6 [σxx, σyy, σzz, σyz, σxz, σxy] " *
            "(got length $(length(p[:target_stress])))"),
        _when((:ftol,), p -> p[:ftol] > 0 ? nothing : "ftol must be > 0"),
        _when((:dt0,), p -> p[:dt0] > 0 ? nothing : "dt0 must be > 0"),
        _when((:dt_max,), p -> p[:dt_max] > 0 ? nothing : "dt_max must be > 0"),
        _when((:dt0, :dt_max), p -> p[:dt0] <= p[:dt_max] ? nothing :
            "need dt0 <= dt_max (got $(p[:dt0]) > $(p[:dt_max]))"),
        _when((:dmax,), p -> p[:dmax] > 0 ? nothing : "dmax must be > 0"),
        _when((:output_interval,), p -> p[:output_interval] >= 1 ? nothing :
            "output_interval must be >= 1 (0 raises DivideError at `step % output_interval`)"),
        _when((:pme_meshsize,), p -> p[:pme_meshsize] > 0 ? nothing : "pme_meshsize must be > 0"),
        _when((:verify_cgtol_factor,), p -> 0 < p[:verify_cgtol_factor] < 1 ? nothing :
            "verify_cgtol_factor must lie in (0, 1) — it TIGHTENS cgtol for the " *
            "post-convergence re-check (got $(p[:verify_cgtol_factor]))"),
        _when((:verify_mesh_factor,), p -> 0 < p[:verify_mesh_factor] < 1 ? nothing :
            "verify_mesh_factor must lie in (0, 1) — it TIGHTENS the PME mesh " *
            "spacing for the post-convergence re-check (got $(p[:verify_mesh_factor]))"),
        _when((:ftol_cgtol_margin,), p -> p[:ftol_cgtol_margin] >= 1 ? nothing :
            "ftol_cgtol_margin must be >= 1"),
        # The entry guard relax! applies, reported here so an app-level run
        # fails at the input boundary with every other violation listed.
        _when((:ftol, :cgtol), function (p)
            margin = get(p, :ftol_cgtol_margin, 10.0)
            p[:ftol] >= margin * p[:cgtol] ? nothing :
                "ftol = $(p[:ftol]) is not reachable with cgtol = $(p[:cgtol]): " *
                "the inner shell relaxation leaves an O(cgtol) error in the core " *
                "forces, so the outer criterion must clear it by the " *
                ":ftol_cgtol_margin factor (need ftol >= $(margin * p[:cgtol]))"
        end),
        # :Relax has no :nsteps — reject it with a message that does not send
        # the user chasing a typo of :maxsteps.
        p -> !haskey(p, :nsteps) ? nothing :
            ":nsteps is not a :Relax key — relaxation runs until converged " *
            "and caps the iteration count with :maxsteps",
        _shape_checks()...,
    ],
)

SPECS[:FitACE] = ParamSpec(
    [:dataset, :dataset_idx, :traindata_ratio, :method_weight,
     :weight_force, :weight_stress, :rigid, :ace_rmin, :ace_nk,
     :nsteps, COMMON_KEYS..., EWALD_KEYS...],
    [:acep, :ace_rmax, :qref, :nlh, :cg_preconditioner],
    Dict(:rigid => Symbol[:cp, :c0, :c00, :c11, :c22, :c33, :c011, :c022, :c033]),
    Function[
        _check_ace_warmstart,
        _check_nlh_cutoff,
        _check_nlh_species,
        _when((:acep,), p -> p[:acep] isa ACEParam ? nothing :
            "acep must be an ACEParam (got $(typeof(p[:acep])))"),
        _check_acep_species,
        _when((:ace_rmin, :ace_rmax), p -> p[:ace_rmin] < p[:ace_rmax] ? nothing :
            "need ace_rmin < ace_rmax (got $(p[:ace_rmin]) >= $(p[:ace_rmax]))"),
        _check_ace_nk,
        _when((:cg_preconditioner,), p -> p[:cg_preconditioner] in (:none, :blockchol_si) ? nothing :
            "cg_preconditioner must be :none or :blockchol_si (got $(repr(p[:cg_preconditioner])))"),
        _check_blockchol_rigid,
        _fit_common_checks()...,
        _shape_checks()...,
    ],
)

SPECS[:FitSpline] = ParamSpec(
    [:dataset, :dataset_idx, :traindata_ratio, :method_weight,
     :weight_force, :weight_stress, :rigid, :rigid_curv,
     :spline_rmax2, :spline_rmax3, :spline_nbasis2, :spline_nbasis3,
     :nsteps, COMMON_KEYS..., EWALD_KEYS...],
    [:splp, :qref, :nlh],
    Dict(:rigid => Symbol[:c1, :c2, :c3],
         :rigid_curv => Symbol[:c2, :c3_r, :c3_θ, :c2_pin, :c3_r_pin]),
    Function[
        _check_nlh_cutoff,
        _check_nlh_species,
        _when((:splp,), p -> p[:splp] isa SplineParam ? nothing :
            "splp must be a SplineParam (got $(typeof(p[:splp])))"),
        _check_splp_species,
        p -> _nonneg_subvals(p, :rigid_curv),
        _when((:spline_rmax2, :spline_rmax3), p -> (p[:spline_rmax2] > 0 && p[:spline_rmax3] > 0) ? nothing :
            "spline_rmax2 and spline_rmax3 must be > 0"),
        _when((:spline_nbasis2, :spline_nbasis3), p -> (p[:spline_nbasis2] >= 4 && p[:spline_nbasis3] >= 4) ? nothing :
            "spline_nbasis2 and spline_nbasis3 must both be >= 4 (cubic B-spline minimum)"),
        _fit_common_checks()...,
        _shape_checks()...,
    ],
)

SPECS[:FitShellModel] = ParamSpec(
    [:dataset, :dataset_idx, :formal_charges, :opt_weights, :lb, :ub,
     :qref, :sigma0, :popsize, :data_workers,
     :nsteps, COMMON_KEYS..., EWALD_KEYS...],
    Symbol[],
    Dict(:opt_weights => Symbol[:bec, :eps]),
    Function[
        _when((:dataset,), p -> isfile(p[:dataset]) ? nothing : "dataset file not found: $(p[:dataset])"),
        _check_opt_weights,
        _when((:formal_charges, :elements), p -> length(p[:formal_charges]) == length(p[:elements]) ? nothing :
            "length(formal_charges) = $(length(p[:formal_charges])) must equal nelem = $(length(p[:elements]))"),
        _when((:sigma0,), p -> p[:sigma0] > 0 ? nothing : "sigma0 must be > 0"),
        _when((:popsize,), p -> p[:popsize] >= 1 ? nothing : "popsize must be >= 1"),
        _when((:data_workers,), p -> p[:data_workers] >= 1 ? nothing : "data_workers must be >= 1"),
        _check_lb_ub,
        _shape_checks()...,
    ],
)

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

"""
    validate_params(job::Symbol, params::NamedTuple)

Validate the merged `params` for `FMD.app[job]` against the schema in
`SPECS` and throw a single `ArgumentError` listing **all** violations (unknown
keys with a did-you-mean hint, missing required keys, invalid sub-keys,
cross-field constraint failures). Returns `nothing` when `params` is
valid. Called automatically by every `app[:Job]` closure; call it
directly to pre-check an input script without running the job.
"""
function validate_params(job::Symbol, params::NamedTuple)
    haskey(SPECS, job) || throw(ArgumentError(
        "validate_params: unknown job :$job — available: " *
        join(sort!(collect(keys(SPECS))), ", ")))
    spec = SPECS[job]
    errs = String[]

    # (0) elements first: nearly every other check depends on it.
    msg = _elements_error(params)
    msg === nothing || push!(errs, msg)

    # (1) unknown keys (typos). keys(lb) are legitimate top-level keys for
    # FitShellModel: opt! injects every lb key as an optimization variable.
    allowed = Set{Symbol}(spec.required)
    union!(allowed, spec.optional)
    if job == :FitShellModel && haskey(params, :lb) && params[:lb] isa NamedTuple
        union!(allowed, keys(params[:lb]))
    end
    for k in keys(params)
        k in allowed || push!(errs, "unknown key :$k" * _didyoumean(k, allowed))
    end

    # (2) missing required keys, reported together.
    miss = [k for k in spec.required if !haskey(params, k)]
    if !isempty(miss)
        m = "missing required key(s): " * join((":$k" for k in miss), ", ")
        if length(miss) >= 5
            m *= "\n     hint: did you forget `params = (; FMD.default_params(Val(:$job), params)..., params...)`?"
        end
        push!(errs, m)
    end

    # (3) sub-key whitelists for NamedTuple-valued entries.
    for (k, sub) in spec.subkeys
        haskey(params, k) || continue
        v = params[k]
        if !(v isa NamedTuple)
            push!(errs, "params[:$k] must be a NamedTuple (got $(typeof(v)))")
            continue
        end
        for sk in keys(v)
            sk in sub || push!(errs, "unknown sub-key :$sk in params[:$k]" * _didyoumean(sk, sub))
        end
    end

    # (4) cross-field constraints.
    for chk in spec.checks
        msg = chk(params)
        msg === nothing || push!(errs, msg)
    end

    isempty(errs) || throw(ArgumentError(_report(job, errs)))
    return nothing
end

validate_params(job::Symbol, params) = throw(ArgumentError(
    "validate_params: params must be a NamedTuple (got $(typeof(params))) — " *
    "build it as `params = (; key = value, ...)`"))
