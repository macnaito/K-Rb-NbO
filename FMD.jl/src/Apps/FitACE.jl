using Dates
using MPI
using Random
using FMD.Cores
using FMD.Calculators
using JLD2

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:FitACE](params)`
#
# Δ-learning fit of an ACE-style MLIP on top of a fixed core-shell model.
# Symmetric to `FMD.app[:FitSpline]` but with an ACE radial basis
# (`ACEParam`) instead of B-splines. Use whichever potential representation
# best matches the chemistry / data.
#
# MPI: data-parallel; `mpisize` ranks evenly partition the dataset.
#
# Pipeline:
#   readxyz(params[:dataset]) → train/test split → ACEParam(...) →
#   fitting! → ACEP.jld2 → re-evaluate → parity_*.csv
#
# Required params: :dataset, :dataset_idx, :elements, :traindata_ratio,
#                  shell parameters, :method_weight, :weight_force,
#                  :weight_stress, :rigid, :rc, :ace_rmin, :ace_rmax,
#                  :ace_nk
# Defaults via:    `default_params(Val(:FitACE), params)`
# Optional:        :acep (existing ACEParam to warm-start)
# Output:          ACEP.jld2 + parity_force/energy/virial.csv
# See also:        [`ACEParam`](@ref), [`fitting!`](@ref),
#                  [`default_params`](@ref)
# --------------------------------------------------------------------------
app[:FitACE] = function (params)
    validate_params(:FitACE, params)

    MPI.Init()
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpisize = MPI.Comm_size(MPI.COMM_WORLD)
    BLAS.set_num_threads(1)
    Random.seed!(0)

    dataset = readxyz(open(params[:dataset], "r"))[params[:dataset_idx]]
    Random.shuffle!(dataset)

    for i in eachindex(dataset)
        setelemid!(dataset[i], params[:elements])
    end

    Ntrain = floor(Int, length(dataset)*params[:traindata_ratio])
    Ntrain >= 1 || error("FitACE: traindata is empty — length(dataset) = $(length(dataset)), " *
                         "traindata_ratio = $(params[:traindata_ratio])")
    trainrange, testrange = 1:Ntrain, (Ntrain+1):length(dataset)
    traindata = dataset[trainrange]
    testdata = dataset[testrange]

    if :acep in keys(params)
        acep = params[:acep]
    else
        acep = ACEParam(params[:ace_rmin], params[:ace_rmax], length(params[:elements]), params[:ace_nk]...)
    end

    acep = fitting!(acep, deepcopy(traindata), params)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        jldsave("ACEP.jld2";
            acep = acep,
            shell = (
                qref  = get(params, :qref, 1.0),
                qc    = params[:qc],
                qs    = params[:qs],
                k2    = params[:k2],
                k4    = params[:k4],
                eta_c = params[:eta_c],
                eta_s = params[:eta_s],
            ),
            elements = params[:elements],
            # Coulomb-Ewald cutoff (atomic units) — load-bearing for the
            # CoreShell evaluator at MD time. ACE radial cutoffs are inside
            # the `acep` struct itself, so they are not duplicated here.
            # Numerical knobs (skin / ewald_tolerance / cgtol / cgmaxiter)
            # are runtime settings and are deliberately NOT saved.
            rc = params[:rc],
            # The NLH baseline is part of the potential, not a runtime knob —
            # see the same block in FitSpline.jl.
            (haskey(params, :nlh) ? (; nlh = nlh_saved(params[:nlh])) : (;))...,
            provenance = (
                fit_timestamp   = string(now()),
                dataset         = string(params[:dataset]),
                nsteps          = params[:nsteps],
                cg_preconditioner = get(params, :cg_preconditioner, :none),
                weight_force    = params[:weight_force],
                weight_stress   = params[:weight_stress],
                method_weight   = params[:method_weight],
                traindata_ratio = params[:traindata_ratio],
                ace_nk          = params[:ace_nk],
                rigid           = params[:rigid],
            ),
        )
    end
    params = (; params..., acep = acep)

    traindata = traindata[(mpirank+1):mpisize:length(traindata)]
    testdata = testdata[(mpirank+1):mpisize:length(testdata)]

    traindata_pred = deepcopy(traindata)
    for i in eachindex(traindata_pred)
        clear!(traindata_pred[i])
        add_baseline!(traindata_pred[i], params)
        calculate!(acep, traindata_pred[i])
    end

    testdata_pred = deepcopy(testdata)
    for i in eachindex(testdata_pred)
        clear!(testdata_pred[i])
        add_baseline!(testdata_pred[i], params)
        calculate!(acep, testdata_pred[i])
    end

    for irank in 0:(mpisize-1)
        mode = ifelse(irank == 0, "w", "a")
        MPI.Barrier(MPI.COMM_WORLD)
        if irank == mpirank
            open("parity_force.csv", mode) do io
                if mode == "w"
                    println(io, "ref, pred, type(0:train,1:test)")
                end
                for idata in eachindex(traindata)
                    for (ref, pred) in zip(traindata[idata].arrays[:forces], traindata_pred[idata].arrays[:forces])
                        println(io, "$ref, $pred, 0")
                    end
                end
                for idata in eachindex(testdata)
                    for (ref, pred) in zip(testdata[idata].arrays[:forces], testdata_pred[idata].arrays[:forces])
                        println(io, "$ref, $pred, 1")
                    end
                end
            end

            open("parity_energy.csv", mode) do io
                if mode == "w"
                    println(io, "ref, pred, type(0:train,1:test)")
                end
                for idata in eachindex(traindata)
                    println(io, "$(traindata[idata].info[:potential_energy][1]), $(traindata_pred[idata].info[:potential_energy][1]), 0")
                end
                for idata in eachindex(testdata)
                    println(io, "$(testdata[idata].info[:potential_energy][1]), $(testdata_pred[idata].info[:potential_energy][1]), 1")
                end
            end

            open("parity_virial.csv", mode) do io
                if mode == "w"
                    println(io, "ref, pred, type(0:train,1:test)")
                end
                for idata in eachindex(traindata)
                    for (ref, pred) in zip(traindata[idata].info[:virial], traindata_pred[idata].info[:virial])
                        println(io, "$ref, $pred, 0")
                    end
                end
                for idata in eachindex(testdata)
                    for (ref, pred) in zip(testdata[idata].info[:virial], testdata_pred[idata].info[:virial])
                        println(io, "$ref, $pred, 1")
                    end
                end
            end
        end
    end

    MPI.Barrier(MPI.COMM_WORLD)

    MPI.Finalize()
    return 0
end

"""
    default_params(::Val{:FitACE}, params) -> NamedTuple

Default params for `FMD.app[:FitACE]`.

# Defaults

| Key | Default | Meaning |
|---|---|---|
| `nsteps` | `10000` | CG iterations |
| `cg_preconditioner` | `:none` | `:blockchol_si` = per-(field × species) block-Cholesky PCG; requires `rigid[f] > 0` on every coefficient field |
| `dataset_idx` | `:` | Indices of frames to use |
| `cgtol`, `cgmaxiter` | `1e-8`, `100` | Shell relaxation |
| `ewald_tolerance` | `1e-8` | Ewald accuracy |
| `efield` | `[0,0,0]` | External field |
| `skin` | `0.1` | Neighbor list skin |
| `ace_rmin` | `1/Å` | Inner cutoff (smooth onset) |
| `ace_nk` | `(12,6,4,4)` | Chebyshev orders `(n0, n1, n2, n3)` |
| `rc` | `11/Å` | Coulomb cutoff |
| `alpha`, `e` | `0.1`, `1e5` | Ewald α / Buckingham cap |
| Shell, Buckingham, Spring placeholders | as in `FitSpline` | typically loaded from `ShellParam.jld2` |

# Required (caller-provided)

- `:dataset`, `:elements`, `:traindata_ratio`, `:method_weight`,
  `:weight_force`, `:weight_stress`, `:rigid`.
- `:ace_rmax` if no `:acep` is supplied (the `ACEParam` constructor uses
  it together with `:ace_rmin` / `:ace_nk`). There is deliberately no
  default — the MLIP cutoff is a physical choice the caller must make.

See `docs/src/parameters.md` for the full reference.
"""
function default_params(::Val{:FitACE}, params)
    _assert_elements(params)
    nelem = length(params[:elements])
    a = (
        nsteps = 10000,
        cg_preconditioner = :none,
        dataset_idx = :,
        cgtol = 1e-8,
        cgmaxiter = 100,
        ewald_tolerance = 1e-8,
        efield = [0.0, 0.0, 0.0],
        skin = 0.1,
        ace_rmin = 1.0/(UNITS[:Å]),
        ace_nk = (12, 6, 4, 4),

        rc = 11.0/(UNITS[:Å]),
        alpha = 0.1,
        e = 1e5,

        qc = zeros(nelem),
        qs = zeros(nelem),
        k2 = zeros(nelem),
        k4 = zeros(nelem),
        eta_c = fill(1e-5/(UNITS[:Å]), nelem),
        eta_s = fill(1e-5/(UNITS[:Å]), nelem),
        a = zeros(nelem,nelem),
        rho = fill(1e-10, (nelem,nelem)),
        b = zeros(nelem,nelem),
        c = zeros(nelem,nelem),
    )
end
