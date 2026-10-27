using MPI
using Random
using Statistics
using LinearAlgebra
using FMD.Cores
using FMD.Calculators
using CMAEvolutionStrategy
using ComponentArrays
using JLD2

"""
    shell_charges(p) -> Vector

Derive shell charges `qs` from `p[:qc]`, `p[:formal_charges]`, and `p[:qref]`
using `qs[i] = -(qc[i] - formal_charges[i] * qref)`. Used inside the CMA-ES
loop so that only `qc` (and the global rescaling `qref`) are degrees of freedom
while charge neutrality with the formal charges is enforced exactly.
"""
function shell_charges(p)
    qs = similar(p[:qc])
    for i in eachindex(qs)
        qs[i] = -(p[:qc][i] - p[:formal_charges][i]*p[:qref])
    end
    return qs
end

"""
    calcloss(params, dataset, wZ, wε) -> Float64

Squared error of (BEC, dielectric tensor) over `dataset`, evaluated by
[`linear_response!`](@ref) with the candidate `params`. `wZ` and `wε`
are the inverse-variance weights computed once in [`opt!`](@ref).

Returns `Inf` instead of throwing if a candidate produces NaN/Inf in the
linear response (e.g. divergent `eta_*`); CMA-ES treats this as a rejected
candidate.
"""
function calcloss(params, dataset, wZ, wε)
    try
        SE_Z = 0.0
        SE_eps = 0.0
        for (i, atoms) in enumerate(dataset)
            atoms = deepcopy(atoms)
            linear_response!(atoms, params)
            SE_Z +=   sum(abs2, atoms.arrays[:becex] .- dataset[i].arrays[:becex])
                    + sum(abs2, atoms.arrays[:becey] .- dataset[i].arrays[:becey])
                    + sum(abs2, atoms.arrays[:becez] .- dataset[i].arrays[:becez])
            SE_eps += sum(abs2, atoms.info[:dielectric_tensor] .- dataset[i].info[:dielectric_tensor])
        end

        loss = SE_Z * wZ + SE_eps * wε
        return loss
    catch e
        if (isa(e, ArgumentError) && occursin("matrix contains Infs or NaNs", e.msg)) || (isa(e, DomainError))
            return Inf
        else
            rethrow(e)
        end
    end
end

@inline function to_box_atan(z, L, U)
    L + (U-L)*(0.5 + atan(z)/pi)
end

@inline function from_box_atan(x, L, U; eps=1e-12)
    y = clamp((x-L)/(U-L), eps, 1-eps)
    tan(pi*(y - 0.5))
end


"""
    opt!(dataset, params)

Two-level MPI-parallel CMA-ES driver that fits the shell model parameters
listed in `params[:lb]` / `params[:ub]` against BEC and dielectric tensor
labels in `dataset`.

# MPI layout

Ranks form `n_groups = mpisize ÷ params[:data_workers]` candidate groups, each
with `params[:data_workers]` data ranks. The dataset is sharded across the
data ranks of a group; population members are distributed across groups.
With the default `data_workers = 1` this collapses to pure population
parallelism (one rank per CMA-ES candidate).

# Required `params` keys

- `:lb`, `:ub` — `NamedTuple` lower / upper bounds; the keys present here
  are exactly the optimised variables (e.g. `qref`, `qc`, `k2`, `eta_c`,
  `eta_s`).
- `:opt_weights = (bec = ..., eps = ...)` — relative weights between
  BEC squared error and dielectric tensor squared error. Each is divided
  by the dataset variance internally.
- `:formal_charges` — used by [`shell_charges`](@ref) to enforce neutrality.
- `:popsize`, `:sigma0`, `:nsteps`, `:data_workers`.

# Output

On `mpirank == 0`, writes `ShellParam.jld2` with the best parameters and
calls [`parity_BEC_eps`](@ref) to dump `parity_bec.csv` / `parity_eps.csv`.
"""
function opt!(dataset, params)
    comm    = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(comm)
    mpisize = MPI.Comm_size(comm)
    BLAS.set_num_threads(1)
    Random.seed!(0)

    # Two-level MPI layout: n_groups candidate groups × data_workers data ranks each.
    # data_workers = 1 (default) → pure population parallelism, identical to previous behaviour.
    data_workers = params[:data_workers]
    @assert mpisize % data_workers == 0 "mpisize ($mpisize) must be divisible by data_workers ($data_workers)"
    n_groups  = mpisize ÷ data_workers
    group_id  = mpirank ÷ data_workers   # which candidate group this rank belongs to
    data_rank = mpirank %  data_workers  # this rank's position inside its group
    data_comm = MPI.Comm_split(comm, group_id, data_rank)  # intra-group communicator

    # Normalisation weights from the full dataset (computed before sharding).
    arrx = reduce(vcat, [vec(a.arrays[:becex]) for a in dataset])
    arry = reduce(vcat, [vec(a.arrays[:becey]) for a in dataset])
    arrz = reduce(vcat, [vec(a.arrays[:becez]) for a in dataset])
    arr  = vcat(arrx, arry, arrz)
    σZ = std(arr)*sqrt(length(arr))
    wZ = params[:opt_weights][:bec] / σZ^2

    arr = reduce(vcat, [vec(atoms.info[:dielectric_tensor]) for atoms in dataset])
    σε = std(arr)*sqrt(length(arr))
    wε = params[:opt_weights][:eps] / σε^2

    # Shard the dataset across data_workers ranks within each group.
    local_dataset = dataset[(data_rank+1):data_workers:length(dataset)]

    ks = Tuple(sort(collect(keys(params[:lb]))))
    x0 = ComponentArray(NamedTuple{ks}(params))
    lb = ComponentArray(NamedTuple{ks}(params[:lb]))
    ub = ComponentArray(NamedTuple{ks}(params[:ub]))
    x0 = from_box_atan.(x0, lb, ub)
    x_work = deepcopy(x0)

    function f(x_batch)
        N_batch = size(x_batch, 2)
        loss_batch = zeros(N_batch)
        for i in 1:N_batch
            # Candidate i is handled by the group whose group_id equals i % n_groups.
            # All data_workers in that group evaluate i together (on their local shard),
            # so the intra-group Allreduce is always called collectively within data_comm.
            if i % n_groups == group_id
                x_work .= vec(x_batch[:, i])
                x_work .= to_box_atan.(x_work, lb, ub)
                params_in = (; params..., NamedTuple(x_work)...)
                params_in = (; params_in..., qs = shell_charges(params_in))
                partial = calcloss(params_in, local_dataset, wZ, wε)
                # Sum partial losses across this group's data workers.
                full = MPI.Allreduce(partial, MPI.SUM, data_comm)
                # Only data_rank 0 writes to loss_batch so the subsequent global
                # Allreduce accumulates exactly one contribution per candidate.
                if data_rank == 0
                    loss_batch[i] += full
                end
            end
        end
        # Candidate-dimension reduction: after this, loss_batch is identical on all ranks,
        # which is required for minimize to generate the same x_batch on all ranks.
        MPI.Allreduce!(loss_batch, +, comm)
        return loss_batch
    end

    popsize = max(4 + floor(Int, 3*log(length(x0))), params[:popsize])
    result = minimize(f, x0, params[:sigma0]; popsize = popsize, parallel_evaluation = true, maxiter = params[:nsteps], verbosity = Int(mpirank==0), seed = 0)

    if mpirank == 0
        x_best = xbest(result)
        x_work .= vec(x_best)

        x_work .= to_box_atan.(x_work, lb, ub)
        params = (; params..., NamedTuple(x_work)...)
        params = (; params..., qs = shell_charges(params))

        for key in keys(params[:lb])
            val = params[key]
            println("$key = $val,")
        end

        jldsave("ShellParam.jld2"; shell = (
            qref  = params[:qref],
            qc    = params[:qc],
            qs    = params[:qs],
            k2    = params[:k2],
            k4    = params[:k4],
            eta_c = params[:eta_c],
            eta_s = params[:eta_s],
            rc    = params[:rc],
        ))

        parity_BEC_eps(dataset, params)
    end
end

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:FitShellModel](params)`
#
# Fit a core-shell model (charges, spring constants, Gaussian widths) against
# a dataset of Born effective charges (`becex/becey/becez`) and dielectric
# tensors using CMA-ES. The dataset is typically `dfpt.xyz` produced by
# `dataset/dfpt/`.
#
# Pipeline:
#   readxyz(params[:dataset]) → setelemid! → opt! → ShellParam.jld2
#
# Required params: :dataset, :dataset_idx, :elements, :formal_charges,
#                  :lb, :ub, :opt_weights, :qref (initial)
# Defaults via:    `default_params(Val(:FitShellModel), params)`
# Output:          ShellParam.jld2, parity_bec.csv, parity_eps.csv
# See also:        [`opt!`](@ref), [`default_params`](@ref)
# --------------------------------------------------------------------------
app[:FitShellModel] = function (params)
    validate_params(:FitShellModel, params)

    dataset = readxyz(open(params[:dataset], "r"))[params[:dataset_idx]]

    for i in eachindex(dataset)
        setelemid!(dataset[i], params[:elements])
    end

    opt!(dataset, params)
    return 0
end

"""
    parity_BEC_eps(dataset, params)

Re-evaluate `dataset` with the fitted shell-model `params` and write
`parity_bec.csv` (reference vs predicted BEC components) and
`parity_eps.csv` (reference vs predicted dielectric tensor components)
in the current directory. Intended for visual fit-quality inspection.
"""
function parity_BEC_eps(dataset, params)
    dataset_train = deepcopy(dataset)
    dataset_ref = deepcopy(dataset)

    for i in eachindex(dataset_train)
        linear_response!(dataset_train[i], params)
    end

    becex_train = [vec(atoms.arrays[:becex]) for atoms in dataset_train]
    becey_train = [vec(atoms.arrays[:becey]) for atoms in dataset_train]
    becez_train = [vec(atoms.arrays[:becez]) for atoms in dataset_train]
    eps_train = [vec(atoms.info[:dielectric_tensor]) for atoms in dataset_train]

    becex_ref = [vec(atoms.arrays[:becex]) for atoms in dataset_ref]
    becey_ref = [vec(atoms.arrays[:becey]) for atoms in dataset_ref]
    becez_ref = [vec(atoms.arrays[:becez]) for atoms in dataset_ref]
    eps_ref = [vec(atoms.info[:dielectric_tensor]) for atoms in dataset_ref]

    open("parity_bec.csv", "w") do io
        for i in eachindex(becex_train)
            for j in eachindex(becex_train[i])
                println(io, "$(becex_ref[i][j]),$(becex_train[i][j])")
                println(io, "$(becey_ref[i][j]),$(becey_train[i][j])")
                println(io, "$(becez_ref[i][j]),$(becez_train[i][j])")
            end
        end
    end

    open("parity_eps.csv", "w") do io
        for i in eachindex(eps_train)
            for j in eachindex(eps_train[i])
                println(io, "$(eps_ref[i][j]),$(eps_train[i][j])")
            end
        end
    end
end

"""
    default_params(::Val{:FitShellModel}, params) -> NamedTuple

Default params for `FMD.app[:FitShellModel]`. `params[:elements]`
must be set by the caller — `nelem = length(params[:elements])` decides the
shape of per-element / per-pair arrays.

# Defaults

| Key | Default | Meaning |
|---|---|---|
| `nsteps` | `100` | CMA-ES generations |
| `dataset_idx` | `:` | Indices of frames to use |
| `cgtol` | `1e-8` | Linear-response shell-relaxation tolerance |
| `cgmaxiter` | `100` | Linear-response shell-relaxation max iterations |
| `ewald_tolerance` | `1e-6` | Ewald accuracy used for BEC/ε evaluation |
| `efield` | `[0,0,0]` | External field (atomic units) |
| `skin` | `0.1` | Neighbor list skin |
| `sigma0` | `1.0` | Initial CMA-ES global step size |
| `popsize` | `1` | CMA-ES population size (auto-bumped if too small) |
| `data_workers` | `1` | MPI ranks per candidate (data parallelism) |
| `rc` | `10/Å` | Coulomb / pair cutoff (atomic units) |
| `alpha` | `0.1` | Ewald α (overridden if `ewald_tolerance` triggers `ewaldcoeff`) |
| `e` | `1e5` | Buckingham smooth cap |
| `qc`, `qs` | `zeros(nelem)` | Initial core / shell charges |
| `k2`, `k4` | `zeros(nelem)` | Spring constants |
| `eta_c`, `eta_s` | `1e-10/Å` | Gaussian width of cores / shells |
| `a`, `rho`, `b`, `c` | zeros / 1e-10 | Buckingham `(a, ρ, b, c)` per pair |

`UNITS[:Å]` is the Bohr-radius equivalent in this package's atomic-unit
convention; division by it converts an Ångström into the internal length
scale.

# Required (caller-provided) keys

- `:elements` — `Dict{String,Int}` element-symbol → species id
- `:formal_charges` — `Vector` of formal charges in element order
- `:dataset` — extxyz file path, typically `dfpt.xyz`
- `:opt_weights = (bec = ..., eps = ...)` — relative loss weights
- `:lb`, `:ub` — `NamedTuple` of optimisation bounds (keys = optimised vars)
- `:qref` — initial global charge rescaling

See `docs/src/parameters.md` for the full reference.
"""
function default_params(::Val{:FitShellModel}, params)
    _assert_elements(params)
    nelem = length(params[:elements])
    a = (
        nsteps = 100,
        dataset_idx = :,
        cgtol = 1e-8,
        cgmaxiter = 100,
        ewald_tolerance = 1e-6,
        efield = [0.0, 0.0, 0.0],
        skin = 0.1,
        sigma0 = 1.0,
        popsize = 1,
        data_workers = 1,

        rc = 10.0/(UNITS[:Å]),
        alpha = 0.1,
        e = 1e5,

        qc = zeros(nelem),
        qs = zeros(nelem),
        k2 = zeros(nelem),
        k4 = zeros(nelem),
        eta_c = fill(1e-10/(UNITS[:Å]), nelem),
        eta_s = fill(1e-10/(UNITS[:Å]), nelem),
        a = zeros(nelem,nelem),
        rho = fill(1e-10, (nelem,nelem)),
        b = zeros(nelem,nelem),
        c = zeros(nelem,nelem),
    )
end
