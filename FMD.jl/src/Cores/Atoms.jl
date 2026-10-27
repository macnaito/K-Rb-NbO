import Base
import KernelAbstractions
using LinearAlgebra

export clear!, setelemid!, volume, kinetic_energy, Atoms, stress, change_arrtyp

"""
    Atoms{T<:AbstractArray}

Container for an atomistic configuration. The array type `T` is generic so
the same struct can live on the CPU (`Array`) or GPU (`CuArray`); see
[`change_arrtyp`](@ref) to migrate between backends.

# Fields

- `info::Dict{Symbol,Any}` — frame-level scalars and tensors. Common keys:
  - `:lattice` — `3×3` cell matrix (rows = lattice vectors, atomic units)
  - `:potential_energy` — single scalar (atomic units)
  - `:virial` — Voigt-6 vector (atomic units)
  - `:dipole`, `:dipole_phys` — `3`-vectors (atomic units)
  - `:dielectric_tensor` — Voigt-6 vector (used by [`FitShellModel`])
  - `:species` — id → element symbol table (`Vector{String}`, index =
    elemid). Element symbols live only here; per-atom species strings are
    never stored (Issue #54).
- `arrays::Dict{Symbol,T}` — per-atom arrays. Every entry must have isbits
  eltype (enforced by [`change_arrtyp`](@ref)) so device gathers and MPI
  migration stay trivial. Common keys:
  - `:pos`, `:velo`, `:forces`, `:shellpos`, `:shelldisp`, `:shellforces`,
    `:peratom_virial`, `:masses`,
    `:elemid` (`Vector{Int}`, provisional ids from [`readxyz`](@ref),
    rebound to `params[:elements]` by [`setelemid!`](@ref)),
    `:becex`, `:becey`, `:becez` (Born effective charge columns).

The XYZ I/O routines ([`readxyz`](@ref), [`writexyz`](@ref)) populate /
consume these keys. `length(atoms)` is the atom count.

Invariant (Issue #70): whenever two `Atoms` share one `info` Dict while
holding distinct `arrays` (the `getindex` slice pattern), the shared `info`
must carry the `:remap_locked` stamp so [`setelemid!`](@ref) refuses to
desynchronize the family. `getindex` stamps it automatically; code that
wraps an *existing* `info` Dict in a new `Atoms` directly must stamp it
itself.
"""
struct Atoms{T<:AbstractArray}
    info::Dict{Symbol,Any}
    arrays::Dict{Symbol,T}
end

function Base.length(atoms::Atoms)
    return size(atoms.arrays[:pos],1)
end

function KernelAbstractions.get_backend(atoms::Atoms)
    return KernelAbstractions.get_backend(atoms.arrays[:pos])
end

"""
    change_arrtyp(atoms::Atoms, arrtyp) -> Atoms

Migrate every entry of `atoms.info` and `atoms.arrays` to `arrtyp`
(typically `CuArray` to push to the GPU, or `Array` to pull back).
`arrtyp` is required: Cores does not depend on CUDA, so a `CuArray`
default would be an `UndefVarError` here — callers name the target.
`info` entries with non-isbits eltype (`info[:species]`, `info[:method]`)
stay host-side — device kernels never read them. Per-atom arrays must have
isbits eltype: a non-isbits array errors here instead of failing later with
an inscrutable device compile error in the CellList re-sort (Issue #54).
Returns a fresh `Atoms{arrtyp}` — the input is not mutated.

Drops the `:remap_locked` stamp (Issue #70): the result has a fresh `info`
Dict and fresh arrays, shares no mutable state that [`setelemid!`](@ref)
touches with the input's info-sharing family, and is therefore always safe
to remap — `change_arrtyp` is the blessed way to obtain a remappable copy
of a sliced frame. This holds for distributed frames too: the lock is NOT
re-stamped on an `MPIDomain` frame — the fresh copy shares no family, so
remap safety is genuine, not an oversight (design decision of Issue #102).
"""
function change_arrtyp(atoms::Atoms, arrtyp)
    for (key, arr) in atoms.arrays
        isbitstype(eltype(arr)) ||
            error("change_arrtyp: arrays[:$key] has non-isbits eltype $(eltype(arr)); " *
                  "per-atom arrays must be isbits (element symbols belong in the " *
                  "frame-level table info[:species])")
    end
    info = Dict(key => _conv_info(arrtyp, atoms.info[key])
                for key in keys(atoms.info) if key !== :remap_locked)
    arrays = Dict(key => arrtyp(atoms.arrays[key]) for key in keys(atoms.arrays))
    return Atoms{arrtyp}(info, arrays)
end

# info-value conversion for change_arrtyp: isbits-eltype values move to the
# target array type, everything else stays host-side. Domain.jl adds an
# explicit AbstractDomain method (host-side pass-through).
_conv_info(arrtyp, v) = isbitstype(eltype(v)) ? arrtyp(v) : v

"""
    setelemid!(atoms::Atoms, mapping::AbstractDict)

Rebind `atoms.arrays[:elemid]` to the `element_symbol => id` `mapping`
(typically `params[:elements]`). [`readxyz`](@ref) assigns provisional ids
(alphabetical order of the symbols in the file) and records the id → symbol
table in `atoms.info[:species]`; this function remaps every per-atom id
through that table into the `mapping`'s id space and replaces
`atoms.info[:species]` with the mapping-ordered table. After calling, every
Calculator can use `elemid` to index per-element parameter tables (`qc`,
`eta_c`, …). Idempotent on an unlocked frame: remapping again with the same
`mapping` is a no-op (a locked frame errors instead, see below).

Errors (before mutating anything) if the frame contains a symbol that
`mapping` does not cover, if the mapping ids are not `1..n` contiguous, or
if `atoms.info[:species]` is missing.

Structural guard (Issue #70): `info` is shared between an `Atoms` and its
`getindex` slices, but `arrays[:elemid]` is not — remapping any member of
such a sharing family rewrites the shared `info[:species]` table while
leaving every other member's `:elemid` in the old id space, and a later
remap of those members becomes a silent identity map (the table is already
in the new order). To make this unreachable, [`Base.getindex`](@ref)
stamps `info[:remap_locked]` into the shared `info`, and this function
errors whenever the stamp is present. Call `setelemid!` on the pristine
frame right after [`readxyz`](@ref), before any slicing (CellList re-sort,
migration, manual `atoms[idx]`). The lock is
conservative: it also fires when every slice has been discarded and the
remap would in fact be safe — take an independent copy via
[`change_arrtyp`](@ref) (which drops the lock) or re-read the frame in
that case.

The remap is in-place and does not change the identity of
`arrays[:elemid]`, which identity-keyed caches (e.g. the ACEFast
species-sort cache) cannot detect — call this before building neighbour
lists or calculator workspaces, never after force evaluation has started.
"""
function setelemid!(atoms::Atoms, mapping)
    haskey(atoms.info, :remap_locked) &&
        error("setelemid!: this frame's info is shared with getindex slices " *
              "(CellList re-sort, migration, or manual atoms[idx]); " *
              "remapping now would desynchronize :elemid across the sharing family " *
              "(Issue #70). Remap the pristine frame right after readxyz, before any " *
              "slicing, neighbour-list build, or workspace construction; or take an " *
              "independent copy via change_arrtyp, which drops the lock.")
    haskey(atoms.info, :species) ||
        error("setelemid!: atoms.info[:species] (id → symbol table) is missing — " *
              "it is set by readxyz; cannot remap :elemid without it")
    ids = sort!(collect(values(mapping)))
    ids == 1:length(ids) ||
        error("setelemid!: mapping ids must be a permutation of 1:n, got $ids")
    table = atoms.info[:species]
    for s in table
        haskey(mapping, s) ||
            error("setelemid!: species \"$s\" is not covered by the mapping " *
                  "$(sort!(collect(keys(mapping))))")
    end
    newtable = Vector{String}(undef, length(ids))
    for (s, id) in mapping
        newtable[id] = s
    end
    old2new = [mapping[s] for s in table]
    elemid = Array(atoms.arrays[:elemid])
    copyto!(atoms.arrays[:elemid], old2new[elemid])
    atoms.info[:species] = newtable
    return atoms
end

"""
    clear!(atoms::Atoms, keys = (:potential_energy, :virial, :shellforces, :forces))

Zero out the listed keys (whichever exist in `info` or `arrays`). Called at
the start of every force-evaluation block so that the additive `calculate!`
methods see a clean slate. Never include `:domain` (or other non-numeric
info entries) in `keys` — `info` entries are reassigned to `zero(...)`.
"""
function clear!(atoms::Atoms, keys = (:potential_energy, :virial, :shellforces, :forces))
    for key in keys
        if haskey(atoms.arrays, key)
            atoms.arrays[key] .= zero(atoms.arrays[key])
        elseif haskey(atoms.info, key)
            atoms.info[key] = zero(atoms.info[key])
        end
    end
end

"""
    volume(atoms::Atoms) -> Float

Cell volume `(a × b) · c` in atomic units, computed from
`atoms.info[:lattice]`.
"""
function volume(atoms::Atoms)
    lattice = Array(atoms.info[:lattice])
    return dot(lattice[1,:], cross(lattice[2,:], lattice[3,:]))
end

"""
    kinetic_energy(atoms::Atoms) -> Float

Sum of `½ m v²` over atoms (atomic units). Reads `:masses` and `:velo`.
Whole-system under domain decomposition (owned partial sum + Allreduce).
"""
function kinetic_energy(atoms::Atoms)
    ke = length(atoms) == 0 ? zero(FPPREC) :
         FPPREC(sum(atoms.arrays[:masses] .* atoms.arrays[:velo] .^ 2) / 2)
    return allreduce_sum(domain(atoms), ke)
end

"""
    stress(atoms::Atoms, kinetic::Bool = false) -> Vector{Float64}

Voigt-6 stress tensor `[σxx, σyy, σzz, σyz, σxz, σxy]` in atomic units,
formed from `-virial / V`. When `kinetic = true` the per-atom kinetic
contribution `-Σmᵢ vᵢ⊗vᵢ / V` is added (use this in NPT MD); leave it
`false` when computing static stress.
"""
function stress(atoms::Atoms, kinetic = false)
    result = zeros(6)
    if kinetic && length(atoms) > 0
        result[1] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,1] .* atoms.arrays[:velo][:,1])
        result[2] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,2] .* atoms.arrays[:velo][:,2])
        result[3] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,3] .* atoms.arrays[:velo][:,3])
        result[4] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,2] .* atoms.arrays[:velo][:,3])
        result[5] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,3] .* atoms.arrays[:velo][:,1])
        result[6] = -sum(atoms.arrays[:masses] .* atoms.arrays[:velo][:,1] .* atoms.arrays[:velo][:,2])
    end
    if kinetic
        # The kinetic moments are owned partial sums; the virial below is
        # already whole-system under the reduction regime — reduce before
        # subtracting it.
        allreduce_sum!(domain(atoms), result)
    end
    virial = Array(atoms.info[:virial])
    result[1] -= virial[1]
    result[2] -= virial[2]
    result[3] -= virial[3]
    result[4] -= virial[4]
    result[5] -= virial[5]
    result[6] -= virial[6]

    return result ./ volume(atoms)
end

"""
    atoms[idx] -> Atoms

Subset selection: returns a new `Atoms` whose `arrays` are sliced by `idx`
(supports any indexing that `Array` supports — `:`, `Vector{Int}`, ranges,
boolean masks). `info` is shared with the parent (frame-level data is not
copied). Used by the `CellList` re-sort (neighbour-list re-ordering).

Side effect: stamps `info[:remap_locked]` into the shared `info` (once,
`haskey`-guarded, so the per-rebuild CellList re-sort does not re-allocate
it), which makes any later [`setelemid!`](@ref) on the parent or the slice
error — once `:elemid` arrays diverge across an info-sharing family, a
remap of one member silently corrupts the others (Issue #70). The stamp is
a host-side 1-element `Bool` vector and travels with the shared Dict;
[`change_arrtyp`](@ref) drops it, because its result shares no mutable
state with the family and is therefore safe to remap.

Every entry of `atoms.arrays` must have the atom index as its first
dimension with `ndims` 1 or 2; anything else raises an error instead of
being silently dropped from the sliced copy.
"""
function Base.getindex(atoms::Atoms, idx)
    newarrays = typeof(atoms.arrays)()
    for key in keys(atoms.arrays)
        if ndims(atoms.arrays[key]) == 1
            newarrays[key] = atoms.arrays[key][idx]
        elseif ndims(atoms.arrays[key]) == 2
            newarrays[key] = atoms.arrays[key][idx, :]
        else
            error("Atoms getindex: arrays[:$key] has ndims = $(ndims(atoms.arrays[key])); " *
                  "per-atom arrays must have the atom index as first dimension (ndims 1 or 2)")
        end
    end
    haskey(atoms.info, :remap_locked) || (atoms.info[:remap_locked] = [true])
    newatoms = Atoms(atoms.info, newarrays)
    return newatoms
end
