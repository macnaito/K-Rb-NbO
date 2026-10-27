"""
    FMD.Cores

Atomic-unit primitives shared by every Calculator and App: the [`Atoms`](@ref)
container, extxyz I/O ([`readxyz`](@ref) / [`writexyz`](@ref)), the unit /
mass tables ([`UNITS`](@ref), [`ATOMIC_MASSES`](@ref)), and the neighbour
list machinery ([`CellList`](@ref), [`neighborlist`](@ref),
[`isreusable`](@ref)). All quantities live in atomic units; multiply by
`UNITS[:Å]` etc. to convert when reading or writing.
"""
module Cores

export Atoms, readxyz, writexyz

include("Atoms.jl")
include("Domain.jl")
include("Migrate.jl")
include("Units.jl")
include("XYZ.jl")
include("Ghosts.jl")
include("NeighborList.jl")
include("NeighborListNaive.jl")
include("NeighborListSub.jl")

end
