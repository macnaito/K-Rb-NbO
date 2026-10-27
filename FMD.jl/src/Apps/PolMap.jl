using ..FMD: FPPREC

# --------------------------------------------------------------------------
# Job entry: `FMD.app[:PolMap](params)`
#
# Post-process local polarization maps (Issue #220): for every frame of
# every extxyz file in params[:input], measure the gauged per-group
# polarization against params[:polarization_gauge] and write it as
# `polmap_<n>.txt` (write_polmap format) under params[:output_dir].
#
# The same quantity npt! writes at every polmap_interval step, so a
# trajectory that was run without the in-loop output — or one whose gauge
# is to be changed after the fact — gets the same files. Host-only: one
# group_p per frame is far cheaper than reading the frame.
#
# Output naming mirrors the MD driver: `step_<n>.xyz` → `polmap_<n>.txt`
# with `step=<n>` in the header, so a post-processed trajectory lines up
# with an in-loop one file for file. Any other name → `polmap_<stem>.txt`;
# a multi-frame file → `polmap_<stem>_<k>.txt` per frame.
#
# Required params: :input (path or Vector of paths), :output_dir,
#                  :polarization_gauge, :elements, :qc, :qs — the
#                  `pot...` splat of loadpotential supplies the last three.
# Optional:        :overwrite (default false) — an existing polmap_<n>.txt
#                  in output_dir is refused unless set.
# Output:          `<output_dir>/polmap_<n>.txt` per frame.
# See also:        [`write_polmap`](@ref), [`read_polmap`](@ref),
#                  [`group_p_gauged`](@ref), [`npt!`](@ref).
# --------------------------------------------------------------------------
app[:PolMap] = function (params)
    validate_params(:PolMap, params)

    gauge = params[:polarization_gauge]
    inp = params[:input]
    paths = inp isa AbstractString ? [String(inp)] : String.(collect(inp))

    mkpath(params[:output_dir])
    overwrite = get(params, :overwrite, false)

    written = Set{String}()
    for path in paths
        frames = open(io -> readxyz(io), path, "r")
        isempty(frames) && error("app[:PolMap]: $path contains no frames")
        for (k, atoms) in enumerate(frames)
            setelemid!(atoms, params[:elements])
            # Whole-frame form: re-validates the frame against the gauge
            # (groupid / elemid / id must match the gauge frame), which is
            # exactly the check a post-processed trajectory needs.
            Pg = group_p_gauged(atoms, gauge, params[:qc], params[:qs])
            name, step = _polmap_outname(path, k, length(frames))
            name in written &&
                error("app[:PolMap]: two inputs map to the same output $name " *
                      "(the second is $path) — post-process them into separate " *
                      "output_dirs")
            push!(written, name)
            out = joinpath(params[:output_dir], name)
            # A map already there is most likely a run's in-loop output (or
            # an earlier post-process against another gauge); replacing it
            # silently would leave nothing saying which gauge it belongs to.
            overwrite || !isfile(out) ||
                error("app[:PolMap]: $out already exists — write into a fresh " *
                      "output_dir, or pass overwrite = true to replace it")
            write_polmap(out, Pg, gauge.gridsize,
                         atoms.info[:lattice];
                         step = step, source = basename(path))
        end
    end

    return 0
end

"""
    _polmap_outname(path, k, nframes) -> (name, step)

Output file name for frame `k` of `nframes` in the extxyz file `path`,
and the MD step it records (or `nothing`). `step_<n>.xyz` (the driver's
snapshot naming) becomes `polmap_<n>.txt` with `step = n`, so a
post-processed trajectory lines up with the in-loop output; any other
single-frame file becomes `polmap_<stem>.txt`; a multi-frame file gets a
per-frame suffix `polmap_<stem>_<k>.txt`.
"""
function _polmap_outname(path::AbstractString, k::Int, nframes::Int)
    stem = first(splitext(basename(path)))
    m = match(r"^step_(\d+)$", stem)
    if nframes == 1
        m === nothing && return ("polmap_$(stem).txt", nothing)
        n = parse(Int, m.captures[1])
        return ("polmap_$(n).txt", n)
    end
    return ("polmap_$(stem)_$(k).txt", nothing)
end

"""
    default_params(::Val{:PolMap}, params) -> NamedTuple

Default params for `FMD.app[:PolMap]`: none — every key is
caller-provided (`input`, `output_dir`, `polarization_gauge`, and the
`elements` / `qc` / `qs` that the `pot...` splat of [`loadpotential`](@ref)
supplies). Defined so the standard `(; default_params(...)..., params...)`
idiom works for every job; it only asserts `:elements`.
"""
function default_params(::Val{:PolMap}, params)
    _assert_elements(params)
    return (;)
end
