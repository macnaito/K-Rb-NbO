using Dates
using Random
using LinearAlgebra
using MPI
using FMD.Cores
using FMD.Calculators
using JLD2

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:FitSpline](params)`
#
# Δ-learning fit of a 2- + 3-body spline MLIP on top of a fixed core-shell
# model. The MLIP captures the residual after the Coulomb-Buckingham
# core-shell evaluator subtracts its own contribution, so it is *only*
# meaningful when the shell parameters in `params` are already a fit
# (typically loaded from `ShellParam.jld2`).
#
# MPI: data-parallel; `mpisize` ranks evenly partition the dataset for the
# CG fit inside [`fitting!`](@ref) and for parity-CSV writing.
#
# Pipeline:
#   readxyz(params[:dataset]) → train/test split → SplineParam(...) →
#   fitting! → SplineP.jld2 → re-evaluate train+test → parity_*.csv
#
# Required params: :dataset, :dataset_idx, :elements, :traindata_ratio,
#                  shell parameters (qref/qc/qs/k2/k4/eta_c/eta_s),
#                  :method_weight, :weight_force, :weight_stress, :rigid,
#                  :rc (Coulomb cutoff)
# Defaults via:    `default_params(Val(:FitSpline), params)`
# Optional:        :splp (existing SplineParam to warm-start)
# Output:          SplineP.jld2 + parity_force/energy/virial.csv
# See also:        [`SplineParam`](@ref), [`fitting!`](@ref),
#                  [`default_params`](@ref)
# --------------------------------------------------------------------------
app[:FitSpline] = function (params)
    validate_params(:FitSpline, params)

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
    Ntrain >= 1 || error("FitSpline: traindata is empty — length(dataset) = $(length(dataset)), " *
                         "traindata_ratio = $(params[:traindata_ratio])")
    trainrange, testrange = 1:Ntrain, (Ntrain+1):length(dataset)
    traindata = dataset[trainrange]
    testdata = dataset[testrange]

    if :splp in keys(params)
        splp = params[:splp]
    else
        splp = SplineParam(
            params[:spline_rmax2], params[:spline_rmax3],
            length(params[:elements]),
            params[:spline_nbasis2], params[:spline_nbasis3],
        )
    end

    splp = fitting!(splp, deepcopy(traindata), params)

    if MPI.Comm_rank(MPI.COMM_WORLD) == 0
        jldsave("SplineP.jld2";
            splp = splp,
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
            # CoreShell evaluator at MD time. The 2/3-body spline cutoffs
            # already live inside `splp` itself (`splp.rmax2`, `splp.rmax3`),
            # so we don't duplicate them at the top level. Numerical knobs
            # (skin / ewald_tolerance / cgtol / cgmaxiter) are runtime
            # settings and are deliberately NOT saved.
            rc = params[:rc],
            # The NLH baseline is part of the potential, not a runtime knob:
            # storing the exact coefficients that were subtracted here is what
            # makes "fit with NLH, run MD without it" structurally impossible
            # (loadpotential rebuilds it and NPT applies it unconditionally).
            (haskey(params, :nlh) ? (; nlh = nlh_saved(params[:nlh])) : (;))...,
            provenance = (
                fit_timestamp     = string(now()),
                dataset           = string(params[:dataset]),
                nsteps            = params[:nsteps],
                weight_force      = params[:weight_force],
                weight_stress     = params[:weight_stress],
                method_weight     = params[:method_weight],
                traindata_ratio   = params[:traindata_ratio],
                spline_nbasis2    = params[:spline_nbasis2],
                spline_nbasis3    = params[:spline_nbasis3],
                rigid             = params[:rigid],
                rigid_curv        = get(params, :rigid_curv, nothing),
            ),
        )
    end
    params = (; params..., splp = splp)

    traindata = traindata[(mpirank+1):mpisize:length(traindata)]
    testdata  = testdata[(mpirank+1):mpisize:length(testdata)]

    traindata_pred = deepcopy(traindata)
    for i in eachindex(traindata_pred)
        clear!(traindata_pred[i])
        add_baseline!(traindata_pred[i], params)
        calculate!(splp, traindata_pred[i])
    end

    testdata_pred = deepcopy(testdata)
    for i in eachindex(testdata_pred)
        clear!(testdata_pred[i])
        add_baseline!(testdata_pred[i], params)
        calculate!(splp, testdata_pred[i])
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
    default_params(::Val{:FitSpline}, params) -> NamedTuple

Default params for `FMD.app[:FitSpline]`.

# Defaults

| Key | Default | Meaning |
|---|---|---|
| `nsteps` | `10000` | CG iterations inside [`fitting!`](@ref) |
| `dataset_idx` | `:` | Indices of frames to use |
| `cgtol` | `1e-8` | Linear-response shell-relaxation tolerance |
| `cgmaxiter` | `100` | Linear-response max iterations |
| `ewald_tolerance` | `1e-8` | Ewald accuracy |
| `efield` | `[0,0,0]` | External field |
| `skin` | `0.1` | Neighbor list skin |
| `spline_rmax2` | `6/Å` | 2-body spline cutoff |
| `spline_rmax3` | `5/Å` | 3-body spline cutoff |
| `spline_nbasis2` | `12` | Number of basis fns per pair (2-body) |
| `spline_nbasis3` | `6` | Number of basis fns per triplet (3-body) |
| `weight_force` | `1.0` | Loss weight for forces |
| `weight_stress` | `1.0` | Loss weight for virial |
| `rigid` | `(c1=1e-8, c2=1e0, c3=1e0)` | Magnitude L2 (Tikhonov) regularisers |
| `rigid_curv` | `(c2=0.0, c3_r=0.0, c3_θ=0.0, c2_pin=0.0, c3_r_pin=0.0)` | P-spline curvature penalty (∫ϕ''²) along radial / angular axes; `*_pin` keys add a boundary pin `λ·ϕ(rc)²` so the taper has less work to do |
| `rc` | `11/Å` | Coulomb cutoff (atomic units) |
| `alpha` | `0.1` | Ewald α |
| `e` | `1e5` | Buckingham smooth cap |
| `qc`, `qs`, `k2`, `k4`, `eta_c`, `eta_s` | zeros / `1e-5/Å` | Shell-model placeholders, expected to be overridden by loading `ShellParam.jld2` |
| `a`, `rho`, `b`, `c` | zeros / 1e-10 | Buckingham per pair |

# Required (caller-provided)

- `:dataset` — extxyz file (typically `synth.xyz`)
- `:elements` — element-symbol → species id
- `:traindata_ratio` — fraction of dataset for training (rest is test)
- `:method_weight` — `NamedTuple` weighting per `method` tag in the dataset
  (e.g. `(PBEsol = 1.0, MACE = 0.5)`)

See `docs/src/parameters.md` for the full reference.
"""
function default_params(::Val{:FitSpline}, params)
    _assert_elements(params)
    nelem = length(params[:elements])
    a = (
        nsteps = 10000,
        dataset_idx = :,
        cgtol = 1e-8,
        cgmaxiter = 100,
        ewald_tolerance = 1e-8,
        efield = [0.0, 0.0, 0.0],
        skin = 0.1,

        spline_rmax2 = 6.0/(UNITS[:Å]),
        spline_rmax3 = 5.0/(UNITS[:Å]),
        spline_nbasis2 = 12,
        spline_nbasis3 = 6,

        weight_force = 1.0,
        weight_stress = 1.0,
        rigid      = (c1 = 1e-8, c2 = 1e0, c3 = 1e0),
        rigid_curv = (c2 = 0.0, c3_r = 0.0, c3_θ = 0.0,
                      c2_pin = 0.0, c3_r_pin = 0.0),

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
