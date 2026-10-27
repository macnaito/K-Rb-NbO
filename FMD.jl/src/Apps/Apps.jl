"""
    FMD.Apps

Top-level workflow entry points. Every job is registered as a closure in
the [`app`](@ref) Dict and called with a single `NamedTuple` of `params`.
Per-job defaults live in [`default_params`](@ref); the canonical input
script pattern is

```julia
params = (; dataset = "...", elements = Dict(...), ...)
params = (; FMD.default_params(Val(:FitShellModel), params)..., params...)
FMD.app[:FitShellModel](params)
```

# Available jobs

| Symbol | Purpose | Output |
|---|---|---|
| `:FitShellModel` | Fit core-shell model to BEC + ε (CMA-ES) | `ShellParam.jld2` |
| `:FitSpline`     | Δ-learn 2/3-body Spline MLIP (CG)         | `SplineP.jld2` |
| `:FitACE`        | Δ-learn ACE MLIP (CG)                     | `ACEP.jld2` |
| `:NPTMD`         | MD with shell + MLIP (selectable thermostat/barostat incl. fixed-cell NVE/NVT; GPU default, `:backend => :cpu` for host) | trajectory + `log.txt` |
| `:Relax`         | FIRE structure relaxation (atoms, optionally the cell) | `relaxed.xyz` + `log.txt` |
| `:PolMap`        | Post-process: local polarization maps from extxyz frames against a declared gauge (CPU) | `polmap_<n>.txt` |

See `docs/src/workflow.md` for how the jobs chain together and
`docs/src/parameters.md` for the complete params reference.
"""
module Apps

using ..FMD.Cores
using ..FMD.Calculators

export app, default_params

"""
    FMD.app :: Dict{Symbol,Function}

Registry of workflow entry points. Each entry takes a single `params`
`NamedTuple` and returns `0` on success. `:Relax` additionally returns `1`
when the relaxation did not converge — a legitimate outcome a batch driver
has to be able to branch on, not a crash. See [`Apps`](@ref) for the list
of available jobs.
"""
app = Dict{Symbol,Function}()

"""
    default_params(::Val{:Job}, params) -> NamedTuple

Per-job default `params` `NamedTuple`. Caller-provided keys override these
defaults via the splatting idiom

```julia
params = (; default_params(Val(:Job), params)..., params...)
```

Methods are defined for `:FitShellModel`, `:FitSpline`, `:FitACE`,
`:NPTMD`, `:Relax`, and `:PolMap`. Some defaults depend on `params[:elements]` (per-element /
per-pair array shapes) so that key must already be present when calling.
"""
function default_params end

include("ParamSpec.jl")
include("FitShellModel.jl")
# Driver skeleton shared by every per-step driver (backend/DD entry, MLIP
# dispatch, the neighbour-list family). Before MDControl.jl, which uses
# `conv` (Issue #207).
include("Driver.jl")
include("MDControl.jl")
# Before NPT.jl: npt! dispatches its per-step force stage through these
# (Issue #149, the opt-in shadow XL-BOMD path).
include("XLBOMD.jl")
include("NPT.jl")
include("Relax.jl")
# After NPT.jl: shares the gauge / group_p machinery, no driver of its own.
include("PolMap.jl")
include("FitACE.jl")
include("FitSpline.jl")
include("LoadPotential.jl")

end
