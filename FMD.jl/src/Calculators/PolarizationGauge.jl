using JLD2

export PolarizationGauge, validate_gauge, save_gauge, load_gauge, group_p_gauged

# ---------------------------------------------------------------------------
#  PolarizationGauge  –  explicit zero-point for the polarisation
#
#  The absolute polarisation of a periodic system is defined only modulo the
#  quantum e·L·n/V, and on top of that "which structure counts as P = 0" is a
#  physical choice, not something a code may make silently.  This type makes
#  the second one an input.
#
#  The first ambiguity is already handled by the difference machinery
#  Pg = Pg_raw − Pg_ref(t) and is NOT touched here: `:pos` is never wrapped
#  (CLAUDE.md), so P(t) stays on one continuous branch and the branch
#  arbitrariness — including the half-quantum offset a charge set like
#  KNbO₃'s (+1, +5, −2) carries, and the summation-order dependence of
#  non-neutral sub-units — cancels exactly in the difference.  What was
#  broken before this type existed is only that the reference frame was
#  whatever `Atoms` happened to be passed to the umbrella constructor.
# ---------------------------------------------------------------------------

"""
    PolarizationGauge

Explicit zero-point ("gauge") for per-group polarisation: the fractional
dipole moment of a user-chosen reference structure, plus the metadata
needed to prove that a later frame is the same system on the same
polarisation branch.

Build it from the paraelectric prototype frame you want to call `P = 0`,
persist it with [`save_gauge`](@ref), and pass the *same* gauge to every
umbrella window and to every script that measures a target map — that is
what makes windows and targets mutually comparable.

# Fields

- `M_frac_ref::Matrix{FPPREC}` — `(ng, 3)` per-group fractional dipole of
  the reference frame (from [`group_mfrac`](@ref)).
- `pos_ref::Matrix{FPPREC}` — `(natoms, 3)` fractional coordinates of the
  reference frame, kept for the same-branch check in
  [`validate_gauge`](@ref).
- `gridsize::NTuple{3,Int}` — `(Nx, Ny, Nz)` group lattice.
- `elemid`, `groupid`, `id_ref` — per-atom labels of the reference frame.
  Comparison against a later frame is keyed by `id_ref`, not by row: a
  trajectory snapshot comes back cell-sorted (`writexyz` stores the rows in
  cell order with their `:id` column), so requiring the original row order
  would reject exactly the frames a target map is measured from.
"""
struct PolarizationGauge
    M_frac_ref::Matrix{FPPREC}
    pos_ref::Matrix{FPPREC}
    gridsize::NTuple{3,Int}
    elemid::Vector{Int}
    groupid::Vector{Int}
    id_ref::Vector{Int}

    # Field-level invariants, enforced on EVERY path into the type — the
    # normal constructor, a hand-assembled struct, and load_gauge (a truncated
    # or hand-edited JLD2 would otherwise become a silently wrong reference).
    function PolarizationGauge(M_frac_ref, pos_ref, gridsize, elemid, groupid,
                               id_ref)
        ng = prod(gridsize)
        size(M_frac_ref) == (ng, 3) ||
            error("PolarizationGauge: M_frac_ref is $(size(M_frac_ref)), " *
                  "expected ($ng, 3) for gridsize $gridsize")
        size(pos_ref, 2) == 3 ||
            error("PolarizationGauge: pos_ref must be (natoms, 3), got " *
                  "$(size(pos_ref))")
        n = size(pos_ref, 1)
        length(elemid) == n && length(groupid) == n && length(id_ref) == n ||
            error("PolarizationGauge: elemid/groupid/id have " *
                  "$(length(elemid))/$(length(groupid))/$(length(id_ref)) " *
                  "entries but pos_ref has $n atoms")
        all(g -> 1 <= g <= ng, groupid) ||
            error("PolarizationGauge: groupid outside 1:$ng")
        allunique(id_ref) ||
            error("PolarizationGauge: :id must be unique — it is what pairs " *
                  "a later frame's atoms with the reference")
        return new(M_frac_ref, pos_ref, gridsize, elemid, groupid, id_ref)
    end
end

# Format tag written into the JLD2 file.  Bump when the field set changes.
const _GAUGE_FORMAT = 2

"""
    PolarizationGauge(atoms_ref::Atoms, gridsize, qc_table, qs_table)

Capture the polarisation gauge of `atoms_ref`. Pass the **paraelectric
prototype** frame (the structure you declare to have zero polarisation),
not a thermally rattled snapshot: whatever this frame's group dipoles are
becomes the origin every later measurement is reported against, noise
included.

Errors rather than guessing when the frame cannot serve as a gauge:

- a distributed (`MPIDomain`) frame — the reference is a whole-system
  quantity and each rank would freeze its own;
- `groupid` not covering `1:prod(gridsize)`;
- a group whose charge is not neutral — a charged group's dipole depends on
  the cell origin, so the whole system drifting (a legitimate rigid
  translation) would move the collective variable.
"""
function PolarizationGauge(atoms_ref::Atoms, gridsize, qc_table, qs_table)
    Cores.domain(atoms_ref) isa Cores.MPIDomain &&
        error("PolarizationGauge: build from the full replicated frame BEFORE " *
              "distribute (the gauge is a whole-system reference; a distributed " *
              "frame would freeze a rank-local one)")

    groupid = Vector{Int}(Array(atoms_ref.arrays[:groupid]))
    elemid = Vector{Int}(Array(atoms_ref.arrays[:elemid]))
    Nx, Ny, Nz = Tuple(Int.(gridsize))
    ng = Nx * Ny * Nz
    minimum(groupid) == 1 ||
        error("PolarizationGauge: groupid must start from 1 (got " *
              "$(minimum(groupid)))")
    ng == maximum(groupid) ||
        error("PolarizationGauge: gridsize $((Nx, Ny, Nz)) gives $ng groups but " *
              "groupid runs up to $(maximum(groupid))")

    _assert_group_neutral(groupid, elemid, qc_table, qs_table, ng)

    M_frac_ref = Matrix{FPPREC}(Array(group_mfrac(atoms_ref, qc_table, qs_table)))
    pos_ref = Matrix{FPPREC}(Array(atoms_ref.arrays[:pos]))
    id_ref = Vector{Int}(Array(atoms_ref.arrays[:id]))
    return PolarizationGauge(M_frac_ref, pos_ref, (Nx, Ny, Nz), elemid, groupid,
                             id_ref)
end

# Per-group charge neutrality.  The tolerance is relative to the group's own
# charge scale so it is meaningful for any potential's charge magnitudes;
# a group that is genuinely charged fails by orders of magnitude, not by
# rounding.
function _assert_group_neutral(groupid, elemid, qc_table, qs_table, ng)
    qc = Array(qc_table)
    qs = Array(qs_table)
    net = zeros(FPPREC, ng)
    scale = zeros(FPPREC, ng)
    nper = zeros(Int, ng)
    for i in eachindex(groupid)
        q = FPPREC(qc[elemid[i]]) + FPPREC(qs[elemid[i]])
        net[groupid[i]] += q
        scale[groupid[i]] += abs(q)
        nper[groupid[i]] += 1
    end
    for g in 1:ng
        # An empty group means the group lattice does not tile the cell: the
        # smoothing FFT would then average over a phantom site.  It also
        # slips through the neutrality test below (0 ≤ tol trivially).
        nper[g] > 0 ||
            error("PolarizationGauge: group $g has no atoms — :groupid does " *
                  "not cover 1:$ng, so the group lattice does not tile the " *
                  "cell")
        # Tolerance follows the accumulation error of the sum itself: the
        # working precision (FPPREC is a build-time Preference — a fixed
        # 1e-10 would be below eps in a Float32 build), the number of terms,
        # and the charge scale.  A group that is genuinely charged misses by
        # O(1) relative and fails regardless.
        tol = 8 * nper[g] * eps(FPPREC) * max(scale[g], one(FPPREC))
        abs(net[g]) <= tol && continue
        error("PolarizationGauge: group $g carries a net charge of $(net[g]) " *
              "(tolerance $tol).  A charged group's dipole depends on the cell " *
              "origin, so the collective variable would drift with any rigid " *
              "translation of the system.  Check that :groupid collects whole " *
              "formula units.")
    end
    return nothing
end

"""
    validate_gauge(gauge::PolarizationGauge, atoms::Atoms)

Check that `atoms` is the same system as the frame `gauge` was built from,
and that it sits on the same polarisation branch. Called at the entry of
every gauge consumer; errors on a mismatch rather than returning a subtly
wrong polarisation.

Atoms are paired by `:id`, so a **row-reordered frame is accepted** — the MD
driver hands back cell-sorted rows, and that is precisely the frame a target
map gets measured from. What is rejected is a genuine difference: a different
atom count, an `:id` the gauge frame does not contain, a differing `elemid` or
`groupid` once paired, or a fractional coordinate that moved more than 0.45
from the reference.

Two different scales are involved and they are not interchangeable:

- the **error** threshold is 0.45 of a *supercell* period, because `:pos` is
  fractional in the supercell and the periodic images are generated by the
  supercell vectors. Past half a period the image correspondence flips and
  `P − P_ref` is off by a polarisation quantum. This is the hard guarantee,
  and it is what catches a wrapped `:pos` (a jump of ~1).
- the **warning** threshold is a quarter of a *unit cell*, which in a large
  supercell is far tighter. Ferroelectric displacements are ~10% of a
  lattice constant, so anything approaching a quarter cell means the frame
  is probably not the structure this gauge was meant for — worth saying,
  but not worth refusing, since such a displacement is unambiguous.
"""
function validate_gauge(gauge::PolarizationGauge, atoms::Atoms)
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("validate_gauge: the gauge describes the whole system; validate " *
              "against the full replicated frame BEFORE distribute")

    n = size(gauge.pos_ref, 1)
    length(atoms) == n ||
        error("validate_gauge: frame has $(length(atoms)) atoms, gauge was " *
              "built from $n")
    haskey(atoms.arrays, :groupid) ||
        error("validate_gauge: the frame has no :groupid, so it cannot be " *
              "checked against a gauge (which is a per-group reference).  " *
              "Assign the same grouping the gauge was built with")
    haskey(atoms.arrays, :id) ||
        error("validate_gauge: the frame has no :id, which is what pairs its " *
              "atoms with the gauge frame (rows are not comparable — a frame " *
              "from the MD driver comes back cell-sorted)")

    # Pair the atoms by :id rather than by row.  A frame that has been through
    # the MD driver comes back cell-sorted, so a row-wise comparison would
    # reject the very snapshots a target map gets measured from.
    ids = Vector{Int}(Array(atoms.arrays[:id]))
    row_of = Dict{Int,Int}(id => i for (i, id) in enumerate(ids))
    perm = Vector{Int}(undef, n)
    for (g, id) in enumerate(gauge.id_ref)
        r = get(row_of, id, 0)
        r == 0 && error("validate_gauge: the frame has no atom with :id $id, " *
                        "which the gauge frame does — it is a different system")
        perm[g] = r
    end

    elemid = Vector{Int}(Array(atoms.arrays[:elemid]))[perm]
    groupid = Vector{Int}(Array(atoms.arrays[:groupid]))[perm]
    elemid == gauge.elemid ||
        error("validate_gauge: :elemid differs from the gauge frame (compared " *
              "per :id).  Rebuild the gauge from the same structure, or run " *
              "setelemid! with the same element list")
    groupid == gauge.groupid ||
        error("validate_gauge: :groupid differs from the gauge frame " *
              "(compared per :id)")

    ds = abs.(Array(atoms.arrays[:pos])[perm, :] .- gauge.pos_ref)
    dmax = maximum(ds)                       # supercell fractional
    if dmax > 0.45
        i = argmax(ds)
        error("validate_gauge: gauge-frame atom $(i[1]) (:id $(gauge.id_ref[i[1]])) moved $(round(dmax, digits = 3)) of " *
              "a SUPERCELL period in fractional coordinate $(i[2]) relative to " *
              "the gauge frame.  Past half a period the periodic image is " *
              "ambiguous, so P − P_ref would be off by a polarisation quantum.  " *
              "Either the frame is not the structure this gauge describes, or " *
              ":pos was wrapped somewhere (it must not be — see CLAUDE.md)")
    end
    # Same displacements read on the unit-cell scale, where "large" actually
    # means something physical.  In a 16³ supercell the error threshold above
    # is 7 unit cells away, far too loose to notice a wrong structure.
    ds_uc = ds .* reshape(FPPREC.(collect(gauge.gridsize)), 1, 3)
    duc = maximum(ds_uc)
    if duc > 0.25
        i = argmax(ds_uc)
        @warn "validate_gauge: gauge-frame atom $(i[1]) (:id $(gauge.id_ref[i[1]])) sits $(round(duc, digits = 3)) unit " *
              "cells from the gauge frame in direction $(i[2]).  Ferroelectric " *
              "displacements are ~0.1, so check that this is the intended " *
              "reference structure"
    end
    return nothing
end

"""
    group_p_gauged(atoms::Atoms, gauge::PolarizationGauge, qc_table, qs_table)

Per-group polarisation of `atoms` measured against `gauge`:

```
P_g = P_g^raw − M_frac_ref · L(t) / V_g(t)
```

This is exactly the quantity the umbrella biases, so it is what a target
map must be measured with. Returns an `(ng, 3)` host matrix.

Whole-frame helper: pass the replicated frame, not a distributed one
(the same convention as the gauge constructor).
"""
function group_p_gauged(atoms::Atoms, gauge::PolarizationGauge, qc_table, qs_table)
    validate_gauge(gauge, atoms)
    ng = prod(gauge.gridsize)
    Pg_raw = Array(group_p(atoms, qc_table, qs_table; ng = ng))
    L = Array(atoms.info[:lattice])
    Vg = volume(atoms) / ng
    return Pg_raw .- (gauge.M_frac_ref * L) ./ FPPREC(Vg)
end

"""
    save_gauge(path::AbstractString, gauge::PolarizationGauge)
    load_gauge(path::AbstractString) -> PolarizationGauge

Persist / restore a gauge as JLD2. Sharing one file between every umbrella
window, every restart and the script that measures the target map is what
structurally prevents a gauge mismatch between them.
"""
function save_gauge(path::AbstractString, gauge::PolarizationGauge)
    jldsave(path;
            format = _GAUGE_FORMAT,
            M_frac_ref = gauge.M_frac_ref,
            pos_ref = gauge.pos_ref,
            gridsize = collect(gauge.gridsize),
            elemid = gauge.elemid,
            groupid = gauge.groupid,
            id_ref = gauge.id_ref)
    return path
end

@doc (@doc save_gauge)
function load_gauge(path::AbstractString)
    d = JLD2.load(path)
    haskey(d, "format") ||
        error("load_gauge: $path is not a gauge file (no format tag)")
    d["format"] == _GAUGE_FORMAT ||
        error("load_gauge: $path has gauge format $(d["format"]), this build " *
              "writes $(_GAUGE_FORMAT)")
    return PolarizationGauge(
        Matrix{FPPREC}(d["M_frac_ref"]),
        Matrix{FPPREC}(d["pos_ref"]),
        Tuple(Int.(d["gridsize"])),
        Vector{Int}(d["elemid"]),
        Vector{Int}(d["groupid"]),
        Vector{Int}(d["id_ref"]))
end

"""
    total_mfrac_ref(gauge::PolarizationGauge) -> Vector{FPPREC}

Whole-system fractional dipole of the gauge frame — the per-group table
summed over groups. Feed it to [`calc_dipole_phys!`](@ref) to report
`dipole_phys` against the same zero-point the umbrella biases against,
instead of against whatever frame the run happened to start from.
"""
total_mfrac_ref(gauge::PolarizationGauge) =
    Vector{FPPREC}(vec(sum(gauge.M_frac_ref, dims = 1)))

# Accept either a prepared gauge or a raw reference frame at the call sites
# that take `gauge = ...`.  Passing an `Atoms` is the "I mean this frame"
# shorthand; it still goes through the same constructor checks.
_as_gauge(g::PolarizationGauge, gridsize, qc_table, qs_table) = g
_as_gauge(a::Atoms, gridsize, qc_table, qs_table) =
    PolarizationGauge(a, gridsize, qc_table, qs_table)
