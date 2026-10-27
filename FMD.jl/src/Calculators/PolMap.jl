export write_polmap, read_polmap

using Printf

# ---------------------------------------------------------------------------
#  Local polarisation map — plain-text persistence of the gauged per-group
#  polarisation table that every bias in the PolField family restrains.
#
#  A map is an `(ng, 3)` table in flat groupid order (iz fastest, the
#  ASE-natural convention shared with `_decode_g`), measured against a
#  declared PolarizationGauge.  It is written as a text file that numpy can
#  read with `np.loadtxt` (comment lines start with `#`):
#
#      # FMD polmap format=1 step=500
#      # gridsize 16 16 16  ng 4096  order flat=ix*Ny*Nz+iy*Nz+iz (iz fastest)
#      # lattice(A) a1x a1y a1z a2x a2y a2z a3x a3y a3z
#      # units: P in e/A^2 (1 e/A^2 = 1602.18 uC/cm^2), gauge-referenced
#      # columns: ix iy iz Px Py Pz
#      0 0 0  1.2345678901e-02 ...
#
#  The unit convention follows the xyz files (angstrom on disk, atomic
#  units in memory): `write_polmap` divides by UNITS[:Å]^2, `read_polmap`
#  multiplies back, so a table read here can go straight into
#  `save_target_map` / `PolarizationMapUmbrella`.
#
#  Deliberately NOT stored: the gauge identity.  The gauge is a run-level
#  declaration (params[:polarization_gauge] / the app's gauge argument) and
#  is persisted once by save_gauge; duplicating pos_ref into every frame's
#  map would multiply the trajectory size for no protection — a map file
#  is a diagnostic, and the place where a gauge mismatch is caught is the
#  target-map file (load_target_map), which carries the identity.
# ---------------------------------------------------------------------------

const _POLMAP_FORMAT = 1

# One e/Å² in μC/cm²: e / (1e-16 cm²) = 1.602176634e-3 C/cm² = 1602.18 μC/cm².
const _POLMAP_UC_PER_CM2 = 1.602176634e-19 / 1e-16 * 1e6

"""
    write_polmap(path, Pg, gridsize, lattice; step = nothing, source = nothing)

Write a local polarisation map — the gauged per-group polarisation `Pg`
(`(ng, 3)`, flat groupid order, atomic units as returned by
[`group_p_gauged`](@ref)) — as a plain-text file. On disk `P` is in
**e/Å²** and the cell vectors (rows of `lattice`) in Å, the same
convention as the xyz files; [`read_polmap`](@ref) converts back to atomic
units. Each data row is `ix iy iz Px Py Pz`, one per group, in flat order
(`ix·Ny·Nz + iy·Nz + iz + 1`, iz fastest).

`step` (an MD step) and `source` (an input file name) are recorded in the
first header line when given. Returns `path`.
"""
function write_polmap(path::AbstractString, Pg::AbstractMatrix, gridsize::NTuple{3,Int},
                      lattice::AbstractMatrix;
                      step::Union{Nothing,Integer} = nothing,
                      source::Union{Nothing,AbstractString} = nothing)
    Nx, Ny, Nz = gridsize
    ng = Nx * Ny * Nz
    P = Array(Pg)
    size(P) == (ng, 3) ||
        error("write_polmap: Pg is $(size(P)), expected ($ng, 3) for gridsize $gridsize")
    L = Array(lattice)
    size(L) == (3, 3) || error("write_polmap: lattice must be 3×3 (got $(size(L)))")
    source === nothing || !occursin(r"\s", source) ||
        error("write_polmap: source must not contain whitespace (got $(repr(source)))")

    pscale = 1 / UNITS[:Å]^2
    open(path, "w") do io
        print(io, "# FMD polmap format=", _POLMAP_FORMAT)
        step === nothing || print(io, " step=", Int(step))
        source === nothing || print(io, " source=", source)
        println(io)
        println(io, "# gridsize ", Nx, " ", Ny, " ", Nz, "  ng ", ng,
                "  order flat=ix*Ny*Nz+iy*Nz+iz (iz fastest)")
        print(io, "# lattice(A)")
        for i in 1:3, j in 1:3
            @printf(io, " %.10e", L[i, j] * UNITS[:Å])
        end
        println(io)
        @printf(io, "# units: P in e/A^2 (1 e/A^2 = %.2f uC/cm^2), gauge-referenced (P = 0 at the gauge frame)\n",
                _POLMAP_UC_PER_CM2)
        println(io, "# columns: ix iy iz Px Py Pz")
        for i in 1:ng
            gx, gy, gz = _decode_g(i, Ny, Nz)
            @printf(io, "%d %d %d %.10e %.10e %.10e\n", gx, gy, gz,
                    P[i, 1] * pscale, P[i, 2] * pscale, P[i, 3] * pscale)
        end
    end
    return path
end

"""
    read_polmap(path) -> (; P, gridsize, lattice, step, source)

Read a file written by [`write_polmap`](@ref). `P` is the `(ng, 3)` map in
**atomic units** (the on-disk e/Å² converted back), `lattice` the 3×3 cell
in Bohr (rows = cell vectors), `step` the recorded MD step or `nothing`,
`source` the recorded input name or `nothing`. The row order is verified
against the flat-groupid convention, so a hand-edited or re-sorted file is
refused rather than silently permuted.
"""
function read_polmap(path::AbstractString)
    lines = readlines(path)
    length(lines) >= 3 || error("read_polmap: $path is too short to be a polmap file")
    head = split(lines[1])
    length(head) >= 4 && head[1] == "#" && head[2] == "FMD" && head[3] == "polmap" ||
        error("read_polmap: $path is not a polmap file (header: $(repr(lines[1])))")
    kv = Dict{String,String}()
    for tok in head[4:end]
        m = match(r"^([A-Za-z_]+)=(.*)$", tok)
        m === nothing && error("read_polmap: malformed header token $(repr(tok)) in $path")
        kv[m.captures[1]] = m.captures[2]
    end
    haskey(kv, "format") || error("read_polmap: $path has no format tag")
    fmt = parse(Int, kv["format"])
    fmt == _POLMAP_FORMAT ||
        error("read_polmap: $path has polmap format $fmt, this build reads $(_POLMAP_FORMAT)")
    step = haskey(kv, "step") ? parse(Int, kv["step"]) : nothing
    source = get(kv, "source", nothing)

    g = split(lines[2])
    length(g) >= 6 && g[2] == "gridsize" && g[6] == "ng" ||
        error("read_polmap: malformed gridsize line in $path: $(repr(lines[2]))")
    gridsize = (parse(Int, g[3]), parse(Int, g[4]), parse(Int, g[5]))
    Nx, Ny, Nz = gridsize
    ng = Nx * Ny * Nz
    parse(Int, g[7]) == ng ||
        error("read_polmap: ng $(g[7]) disagrees with gridsize $gridsize in $path")

    l = split(lines[3])
    length(l) == 11 && l[2] == "lattice(A)" ||
        error("read_polmap: malformed lattice line in $path: $(repr(lines[3]))")
    lattice = zeros(FPPREC, 3, 3)
    for i in 1:3, j in 1:3
        lattice[i, j] = parse(FPPREC, l[2 + (i - 1) * 3 + j]) / UNITS[:Å]
    end

    P = zeros(FPPREC, ng, 3)
    n = 0
    for ln in lines[4:end]
        s = strip(ln)
        (isempty(s) || startswith(s, "#")) && continue
        n += 1
        n <= ng || error("read_polmap: $path has more than $ng data rows for gridsize $gridsize")
        cols = split(s)
        length(cols) == 6 ||
            error("read_polmap: data row $n of $path has $(length(cols)) columns, expected 6")
        gx, gy, gz = _decode_g(n, Ny, Nz)
        (parse(Int, cols[1]), parse(Int, cols[2]), parse(Int, cols[3])) == (gx, gy, gz) ||
            error("read_polmap: row $n of $path is group $(cols[1:3]), expected " *
                  "($gx, $gy, $gz) — rows must be in flat groupid order (iz fastest)")
        for d in 1:3
            P[n, d] = parse(FPPREC, cols[3 + d]) * UNITS[:Å]^2
        end
    end
    n == ng || error("read_polmap: $path has $n data rows, expected $ng for gridsize $gridsize")
    return (; P, gridsize, lattice, step, source)
end

"""
    group_p_gauged(atoms::Atoms, M_frac_ref::AbstractMatrix, qc_table, qs_table) -> Matrix

Per-step form of [`group_p_gauged`](@ref): the gauge reference is handed
in as the `(ng, 3)` fractional moment table already on the compute
backend (a device copy of `gauge.M_frac_ref`), and no whole-frame
validation is repeated. Safe under domain decomposition — the group table
is Allreduced inside, so the returned `(ng, 3)` map is the whole-system
value on every rank — and on either backend. This is what the MD driver
calls at every `polmap_interval` step; the `PolarizationGauge` method is
the whole-frame, host-side form for scripts.
"""
function group_p_gauged(atoms::Atoms, M_frac_ref::AbstractMatrix, qc_table, qs_table)
    size(M_frac_ref, 2) == 3 ||
        error("group_p_gauged: M_frac_ref must be (ng, 3) (got $(size(M_frac_ref)))")
    ng = size(M_frac_ref, 1)
    Pg, _ = _polfield_pg(atoms, M_frac_ref, qc_table, qs_table, ng)
    return Pg
end
