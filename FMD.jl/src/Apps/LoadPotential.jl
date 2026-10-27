# LoadPotential.jl — validated loading of potential JLD2 files.
#
# The fit apps save plain data (NamedTuples / Dicts; the only structs are
# ACEParam/SplineParam) so old files stay readable across code changes.
# `loadpotential` is the read-side counterpart: it checks the file against
# the three known schemas and returns a flat NamedTuple ready to splat
# into `params`. The write side (the three jldsave calls) is deliberately
# left untouched.
#
# Cross-field physics checks (species count, MLIP cutoff <= rc) reuse the
# ParamSpec check functions so the rules live in one place.

using JLD2

export loadpotential

# shell field sets: ShellParam.jld2 stores rc inside the shell tuple,
# ACEP/SplineP store rc at the top level (7-field shell).
const _SHELL_FIELDS_7 = (:qref, :qc, :qs, :k2, :k4, :eta_c, :eta_s)
const _SHELL_FIELDS_8 = (:qref, :qc, :qs, :k2, :k4, :eta_c, :eta_s, :rc)
const _SHELL_VEC_FIELDS = (:qc, :qs, :k2, :k4, :eta_c, :eta_s)

function _shell_errors!(errs, shell, expected; nelem = nothing)
    if !(shell isa NamedTuple)
        push!(errs, "\"shell\" must be a NamedTuple (got $(typeof(shell)))")
        return
    end
    missing_f = setdiff(expected, keys(shell))
    isempty(missing_f) || push!(errs, "\"shell\" is missing field(s): " * join(missing_f, ", "))
    extra_f = setdiff(keys(shell), expected)
    isempty(extra_f) || push!(errs, "\"shell\" has unexpected field(s): " * join(extra_f, ", "))
    isempty(missing_f) || return   # the checks below need the fields

    # type checks first — length arithmetic on a String/nothing would
    # either slip through silently or raise a raw MethodError.
    badtype = [k for k in _SHELL_VEC_FIELDS if !(getfield(shell, k) isa AbstractVector{<:Real})]
    isempty(badtype) || push!(errs, "shell field(s) must be Real vectors: " *
        join((":$k (got $(typeof(getfield(shell, k))))" for k in badtype), ", "))
    shell.qref isa Real || push!(errs, "shell.qref must be a Real (got $(typeof(shell.qref)))")
    if :rc in expected
        (shell.rc isa Real && shell.rc > 0) ||
            push!(errs, "shell.rc must be a positive Real (atomic units, Bohr)")
    end
    isempty(badtype) || return

    lens = [length(getfield(shell, k)) for k in _SHELL_VEC_FIELDS]
    allequal(lens) || push!(errs, "shell per-element vectors must share one length " *
                                  "(qc/qs/k2/k4/eta_c/eta_s lengths: $lens)")
    if nelem !== nothing && allequal(lens) && lens[1] != nelem
        push!(errs, "shell per-element length $(lens[1]) does not match length(elements) = $nelem")
    end
end

function _loadpot_error(path, errs)
    n = length(errs)
    return ArgumentError("loadpotential(\"$path\") failed ($n problem$(n == 1 ? "" : "s")):\n" *
                         join(("  $i. $(errs[i])" for i in 1:n), "\n"))
end

"""
    loadpotential(path::AbstractString) -> NamedTuple

Load a potential JLD2 file written by the fit apps, validate it against
the known schemas, and return a flat `NamedTuple` that splats straight
into `params`:

- `ShellParam.jld2` (from `app[:FitShellModel]`) →
  `(; qref, qc, qs, k2, k4, eta_c, eta_s, rc)`
- `ACEP.jld2` (from `app[:FitACE]`) →
  `(; qref, qc, qs, k2, k4, eta_c, eta_s, acep, elements, rc)`
- `SplineP.jld2` (from `app[:FitSpline]`) → same with `splp`

Validation covers the top-level key set, exact shell field sets, field
types, per-element lengths vs `elements`, `elements` id range, the MLIP
type and species count, `rc > 0`, MLIP cutoff ≤ `rc`, and the Spline c3
symmetry contract `c3 ∈ im(P)` (Issue #63; the physics checks are shared
with [`validate_params`](@ref)). Violations are collected per stage and
thrown together as one `ArgumentError`.

`provenance` is accepted but deliberately **not** returned, so the result
never injects unknown keys into `params`; read it directly with
`JLD2.load(path, "provenance")` when you need the fit metadata. Typical
usage:

```julia
pot = FMD.loadpotential("./ACEP.jld2")
params = (; pot..., input = "input.xyz", output_dir = "out", ...)
params = (; FMD.default_params(Val(:NPTMD), params)..., params...)
FMD.app[:NPTMD](params)
```
"""
function loadpotential(path::AbstractString)
    isfile(path) || throw(ArgumentError("loadpotential: file not found: $path"))
    d = try
        JLD2.load(path)
    catch err
        throw(ArgumentError("loadpotential(\"$path\"): failed to read as JLD2 " *
                            "($(typeof(err))) — is this a JLD2 file?"))
    end
    ks = Set(keys(d))
    errs = String[]

    if "acep" in ks && "splp" in ks
        throw(_loadpot_error(path, ["file contains both \"acep\" and \"splp\" — " *
                                    "a potential file must carry exactly one MLIP"]))
    elseif "acep" in ks || "splp" in ks
        mlip_key = "acep" in ks ? "acep" : "splp"
        expected = Set([mlip_key, "shell", "elements", "rc"])
        missing_k = setdiff(expected, ks)
        isempty(missing_k) || push!(errs, "missing top-level key(s): " * join(sort!(collect(missing_k)), ", "))
        # "nlh" is optional: absent ⇒ the fit used no NLH baseline.
        extra_k = setdiff(ks, expected, Set(["provenance", "nlh"]))
        isempty(extra_k) || push!(errs, "unexpected top-level key(s): " * join(sort!(collect(extra_k)), ", "))
        isempty(errs) || throw(_loadpot_error(path, errs))

        elements = d["elements"]
        elements_ok = elements isa AbstractDict
        if elements_ok
            msg = _elements_error((; elements))
            msg === nothing || (elements_ok = false; push!(errs, msg))
        else
            push!(errs, "\"elements\" must be a Dict element symbol => species id " *
                        "(got $(typeof(elements)))")
        end
        _shell_errors!(errs, d["shell"], _SHELL_FIELDS_7;
                       nelem = elements_ok ? length(elements) : nothing)

        mlip = d[mlip_key]
        rc = d["rc"]
        rc_ok = rc isa Real && rc > 0
        rc_ok || push!(errs, "\"rc\" must be a positive Real (atomic units, Bohr)")
        if mlip_key == "acep" && mlip isa ACEParam
            checks = (_check_acep_species, _check_acep_cutoff)
        elseif mlip_key == "splp" && mlip isa SplineParam
            checks = (_check_splp_species, _check_splp_cutoff, _check_splp_c3sym)
        else
            push!(errs, "\"$mlip_key\" must be " *
                        (mlip_key == "acep" ? "an ACEParam" : "a SplineParam") *
                        " (got $(typeof(mlip)))")
            checks = ()
        end
        # ParamSpec checks are haskey-guarded, so only hand them the
        # pieces that individually passed.
        probe = (; Symbol(mlip_key) => mlip,
                 (elements_ok ? (; elements) : (;))...,
                 (rc_ok ? (; rc) : (;))...)
        for chk in checks
            msg = chk(probe)
            msg === nothing || push!(errs, msg)
        end

        # Optional NLH baseline. Rebuilt through the NLHParam constructor so
        # the physical integrity checks (Σa = 1, no non-decaying term, pair
        # symmetry) run on the stored coefficients too, and checked against
        # this file's own species count and Coulomb cutoff.
        nlh = nothing
        if haskey(d, "nlh")
            saved = d["nlh"]
            if !(saved isa NamedTuple)
                push!(errs, "\"nlh\" must be a NamedTuple (got $(typeof(saved)))")
            else
                try
                    nlh = NLHParam(saved)
                catch err
                    push!(errs, "\"nlh\" is invalid: $(err isa ErrorException ? err.msg : err)")
                end
            end
            if nlh !== nothing
                elements_ok && length(nlh.Z) == length(elements) ||
                    push!(errs, "\"nlh\" covers $(length(nlh.Z)) species but " *
                                "\"elements\" has $(length(elements))")
                msg = _check_nlh_cutoff((; nlh, (rc_ok ? (; rc) : (;))...))
                msg === nothing || push!(errs, msg)
            end
        end
        isempty(errs) || throw(_loadpot_error(path, errs))

        return (; d["shell"]..., Symbol(mlip_key) => mlip,
                elements = elements, rc = rc,
                (nlh === nothing ? (;) : (; nlh))...)
    elseif "shell" in ks && issubset(ks, Set(["shell", "provenance"]))
        _shell_errors!(errs, d["shell"], _SHELL_FIELDS_8)
        isempty(errs) || throw(_loadpot_error(path, errs))
        return (; d["shell"]...)
    else
        throw(ArgumentError(
            "loadpotential(\"$path\"): unrecognized schema — top-level keys " *
            "$(sort!(collect(ks))). Expected {shell[, provenance]} (ShellParam.jld2), " *
            "{acep, shell, elements, rc[, provenance]} (ACEP.jld2), or " *
            "{splp, shell, elements, rc[, provenance]} (SplineP.jld2)."))
    end
end
