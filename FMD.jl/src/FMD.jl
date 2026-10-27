"""
    FMD

Top-level module of FMD.jl — Fast Molecular Dynamics for ferroelectric
perovskites. The package is organised in three submodules:

- [`FMD.Cores`](@ref Cores): atomic-unit primitives ([`Atoms`](@ref),
  [`readxyz`](@ref) / [`writexyz`](@ref), [`UNITS`](@ref), neighbour
  lists).
- [`FMD.Calculators`](@ref Calculators): force / energy evaluators
  ([`CoreShell`](@ref), [`SplineParam`](@ref), [`ACEParam`](@ref),
  Ewald / PME, [`PolarizationUmbrella`](@ref)).
- [`FMD.Apps`](@ref Apps): user-facing workflow entry points registered in
  [`FMD.app`](@ref) — `:FitShellModel`, `:FitSpline`, `:FitACE`, `:NPTMD`.

See `docs/src/workflow.md` for the end-to-end pipeline and
`docs/src/parameters.md` for the params reference.

# Configuration via Preferences

The `[FMD]` section of `LocalPreferences.toml` controls build-time
behaviour:

- `precision = "Float32" | "Float64"` — sets [`FPPREC`](@ref). Float32 is
  the default and recommended for GPU MD.
"""
module FMD

export FPPREC

using Preferences

"""
    FMD.FPPREC

Working floating-point precision for the entire package, resolved at
load time from the `precision` Preference under `[FMD]` in
`LocalPreferences.toml`. Defaults to `Float32`. Set
`precision = "Float64"` to compile a double-precision build (slower on
GPU but useful for reference / debugging runs).

```toml
# LocalPreferences.toml
[FMD]
precision = "Float64"
```
"""
const FPPREC = @load_preference("precision", "Float32") |> Symbol === :Float64 ? Float64 : Float32

include("Cores/Cores.jl"); using .Cores
include("Calculators/Calculators.jl"); using .Calculators
include("Apps/Apps.jl"); using .Apps

include("precompile.jl")

end
