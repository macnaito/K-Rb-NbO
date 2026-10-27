using Printf
using ..FMD: FPPREC

export readxyz, writexyz

const XYZTYPE = Dict(
    "I" => Int,
    "R" => FPPREC,
    "S" => String,
)

function readframe(io, unitconversions)
    n_atoms = parse(Int, readline(io))

    info = Dict{Symbol,Any}()
    header = readline(io)
    lattice = parse_lattice(header)
    method = parse_method(header)
    potential_energy = parse_potential_energy(header)
    virial = parse_virial(header)
    dielectric_tensor = parse_dielectric_tensor(header)
    if !isnothing(lattice)
        info[:lattice] = FPPREC.(lattice)
    end
    if !isnothing(virial)
        info[:virial] = FPPREC.(virial)
    else
        info[:virial] = zeros(FPPREC, 6)
    end
    if !isnothing(dielectric_tensor)
        info[:dielectric_tensor] = FPPREC.(dielectric_tensor)
    else
        info[:dielectric_tensor] = zeros(FPPREC, 6)
    end
    if !isnothing(potential_energy)
        info[:potential_energy] = [FPPREC(potential_energy)]
    else
        info[:potential_energy] = [zero(FPPREC)]
    end
    if !isnothing(method)
        info[:method] = [method]
    else
        info[:method] = [:unknown]
    end

    m = match(r"Properties=([^\s]+)", header).captures[1]
    properties = reshape(split(m, ":"),3,:)
    arrays = Dict()
    colidx = Dict()
    ncol = 1
    for i in axes(properties,2)
        key, type, ndim = Symbol(properties[1,i]), XYZTYPE[properties[2,i]], parse(Int, properties[3,i])
        arrays[key] = ndim > 1 ? Matrix{type}(undef, n_atoms, ndim) : Vector{type}(undef, n_atoms)
        colidx[key] = ncol:ncol+ndim-1

        ncol += ndim
    end

    for i in 1:n_atoms
        line = split(readline(io))
        for key in keys(arrays)
            if ndims(arrays[key]) == 2
                arrays[key][i,:] .= eltype(arrays[key]) == String ? line[colidx[key]] : parse.(eltype(arrays[key]), line[colidx[key]])
            else
                arrays[key][i] = eltype(arrays[key]) == String ? line[colidx[key]][1] : parse(eltype(arrays[key]), line[colidx[key]][1])
            end
        end
    end

    for key in keys(unitconversions)
        if key in keys(arrays)
            arrays[key] /= FPPREC(unitconversions[key])
        elseif key in keys(info)
            info[key] /= FPPREC(unitconversions[key])
        end
    end

    :species in keys(arrays) || error("species not found in XYZ file")
    :pos in keys(arrays) || error("pos not found in XYZ file")
    :elemid ∉ keys(arrays) ||
        error("XYZ file has an elemid column; :elemid is derived from the species column " *
              "(provisional alphabetical ids, rebind with setelemid!) — remove the column")

    # :species は per-atom には保持しない（Issue #54: 非 isbits 配列は GPU gather /
    # MPI migration と両立しない）．記号は frame-level の id→記号表 info[:species] に
    # 凍結し，per-atom は :elemid のみ．ここでの id はファイル内記号の辞書順による
    # 暫定値で，params の id 体系へは setelemid!（remap）で束縛する．
    species = arrays[:species]
    delete!(arrays, :species)
    symbols = sort(unique(species))
    sym2id = Dict(s => i for (i, s) in enumerate(symbols))
    arrays[:elemid] = [sym2id[s] for s in species]
    info[:species] = symbols

    if :masses ∉ keys(arrays)
        arrays[:masses] = [FPPREC(ATOMIC_MASSES[s]) for s in species]
    end
    if :forces ∉ keys(arrays)
        arrays[:forces] = zero(arrays[:pos])
    end
    if :velo ∉ keys(arrays)
        arrays[:velo] = zero(arrays[:pos])
    end
    if :id ∉ keys(arrays)
        arrays[:id] = collect(1:n_atoms)
    end
    if :shelldisp ∉ keys(arrays)
        arrays[:shelldisp] = zero(arrays[:pos])
    end
    if :shellforces ∉ keys(arrays)
        arrays[:shellforces] = zero(arrays[:pos])
    end

    info[:lattice] = Array(info[:lattice]')
    arrays[:pos] .= (arrays[:pos] * (info[:lattice] \ I))

    atoms = Atoms{Array}(info, arrays)
    return atoms
end

"""
    readxyz(io,
            unitconversions = (; lattice = UNITS[:Å], pos = UNITS[:Å],
                                  shelldisp = UNITS[:Å],
                                  forces = UNITS[:eV]/UNITS[:Å],
                                  peratomstress = UNITS[:eV]/UNITS[:Å]^3,
                                  potential_energy = UNITS[:eV],
                                  virial = UNITS[:eV])) -> Vector{Atoms}

Parse an extxyz stream into a `Vector{Atoms}` (one entry per frame). Frame
header keywords recognised:

- `Lattice="..."` — `3×3` row-major in input units (Å by default).
- `Properties=species:S:1:pos:R:3:...` — column schema; column types are
  `S` (`String`), `I` (`Int`), `R` (current working precision).
- `potential_energy=` (or its extxyz spelling `energy=`, which is what
  [`writexyz`](@ref) emits), `virial="..."` (Voigt-6 or full 9), `method=`,
  `dielectric_tensor="..."`.

# Unit handling

Quantities are converted *into atomic units* by dividing by the values in
`unitconversions`. Pass an empty `NamedTuple` if your file is already in
atomic units. Internally everything below this layer assumes atomic units.
The default table is the exact inverse of what [`writexyz`](@ref) applies to
every column it emits, so a written frame reads back unchanged (Issue #154).
Two caveats on that symmetry:

- `velo` is not in the table because neither side converts it: velocities
  are written and read as atomic units, unlike every other column.
- `:peratomstress` comes back in atomic units but is *not* the internal
  `:peratom_virial` it was written from — that one is a virial, this one a
  stress (`-virial - Σ mvv`, per volume). Nothing downstream reads it.

The `stress` / `dipole` / `dipole_phys` header keys are write-only
diagnostics and are ignored on read; `virial=` and `dielectric_tensor=` are
read-only (they come from external reference data, not from `writexyz`).

# Species handling

The species column is never stored per-atom (Issue #54): the symbols are
frozen into the frame-level id → symbol table `info[:species]` (alphabetical
order) and `:elemid` holds the corresponding provisional per-atom ids.
Rebind them to a potential's `params[:elements]` id space with
[`setelemid!`](@ref) before any force evaluation.

# Auto-populated arrays

The following keys are added if not present in the file: `:masses` (from
`ATOMIC_MASSES` × the species column), `:forces`, `:velo`, `:id`,
`:shelldisp`, `:shellforces`. Positions are stored as fractional coordinates
(`pos · lattice⁻¹`); convert with `pos * lattice` to recover Cartesians,
as [`writexyz`](@ref) does.
"""
function readxyz(io, unitconversions = (; :lattice => UNITS[:Å], :pos => UNITS[:Å], :shelldisp => UNITS[:Å], :forces => UNITS[:eV]/UNITS[:Å], :peratomstress => UNITS[:eV]/UNITS[:Å]^3, :potential_energy => UNITS[:eV], :virial => UNITS[:eV]))
    dataset = Atoms[]

    while true
        try
            atoms = readframe(io, unitconversions)
            push!(dataset, atoms)
        catch e
            if isa(e, ArgumentError) && occursin("input string is empty or only contains whitespace", e.msg)
                break
            else
                rethrow(e)
            end
        end
    end

    return dataset
end

"""
    writexyz(io::IOStream, atoms::Atoms, params)

Write `atoms` to `io` as one extxyz frame. Always migrates to a CPU `Atoms`
first so it is safe to pass GPU-resident frames.

`params[:elements]` is required to map `elemid` → element symbol. When the
frame carries an id → symbol table (`info[:species]`), it must agree with
`params[:elements]` — a mismatch means `:elemid` is still bound to a
different id space (e.g. the provisional ids of a freshly read frame) and
writing would silently swap element symbols; call [`setelemid!`](@ref)
first.

Output columns: `species`, `pos` (Å), `id`, `shelldisp` (Å), `velo`,
`forces` (eV/Å), `peratomstress` (Voigt-6, eV/Å³), `groupid`.

Header includes `Lattice` (Å), optional `dipole` and `dipole_phys` (Å·e),
total `energy` (eV), and the full-cell `stress` tensor (eV/Å³, Voigt-6
expanded to a 9-component matrix).
"""
function writexyz(io::IOStream, atoms::Atoms, params)
    atoms = change_arrtyp(atoms, Array)

    println(io, length(atoms))
    write_header(io, atoms)
    elements = collect(keys(params[:elements]))[sortperm(collect(values(params[:elements])))]
    if haskey(atoms.info, :species) && atoms.info[:species] != elements
        error("writexyz: params[:elements] order $(elements) does not match " *
              "atoms.info[:species] $(atoms.info[:species]) — :elemid is bound to a " *
              "different id space; call setelemid!(atoms, params[:elements]) first")
    end

    positions = atoms.arrays[:pos] * atoms.info[:lattice]
    if haskey(atoms.arrays, :peratom_virial)
        peratom_stress = -atoms.arrays[:peratom_virial]
        @. peratom_stress[:,1] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,1] .* atoms.arrays[:velo][:,1]
        @. peratom_stress[:,2] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,2] .* atoms.arrays[:velo][:,2]
        @. peratom_stress[:,3] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,3] .* atoms.arrays[:velo][:,3]
        @. peratom_stress[:,4] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,2] .* atoms.arrays[:velo][:,3]
        @. peratom_stress[:,5] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,1] .* atoms.arrays[:velo][:,3]
        @. peratom_stress[:,6] -= atoms.arrays[:masses]*atoms.arrays[:velo][:,1] .* atoms.arrays[:velo][:,2]
        peratom_stress ./= volume(atoms)
    else
        peratom_stress = zeros(length(atoms),6)
    end

    if haskey(atoms.arrays, :groupid)
        groupid = atoms.arrays[:groupid]
    else
        groupid = ones(Int, size(atoms.arrays[:elemid]))
    end

    for i in 1:length(atoms.arrays[:masses])
        @printf(io, "%-2s %15.8e %15.8e %15.8e %15i %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %15.8e %-4s \n",
        elements[atoms.arrays[:elemid][i]],
        positions[i,1] * UNITS[:Å],
        positions[i,2] * UNITS[:Å],
        positions[i,3] * UNITS[:Å],
        atoms.arrays[:id][i],
        atoms.arrays[:shelldisp][i,1] * UNITS[:Å],
        atoms.arrays[:shelldisp][i,2] * UNITS[:Å],
        atoms.arrays[:shelldisp][i,3] * UNITS[:Å],
        atoms.arrays[:velo][i,1],
        atoms.arrays[:velo][i,2],
        atoms.arrays[:velo][i,3],
        atoms.arrays[:forces][i,1] * UNITS[:eV] / UNITS[:Å],
        atoms.arrays[:forces][i,2] * UNITS[:eV] / UNITS[:Å],
        atoms.arrays[:forces][i,3] * UNITS[:eV] / UNITS[:Å],
        peratom_stress[i,1] * UNITS[:eV] / UNITS[:Å]^3,
        peratom_stress[i,2] * UNITS[:eV] / UNITS[:Å]^3,
        peratom_stress[i,3] * UNITS[:eV] / UNITS[:Å]^3,
        peratom_stress[i,4] * UNITS[:eV] / UNITS[:Å]^3,
        peratom_stress[i,5] * UNITS[:eV] / UNITS[:Å]^3,
        peratom_stress[i,6] * UNITS[:eV] / UNITS[:Å]^3,
        groupid[i]
        )
    end
end

# ヘルパー関数
function parse_lattice(header::String)
    m = match(r"Lattice=\"(.*?)\"", header)
    if m !== nothing
        nums = parse.(Float64, split(m.captures[1]))
        return reshape(nums, (3,3))
    end
    nothing
end

# ヘルパー関数
function parse_virial(header::String)
    m = match(r"virial=\"(.*?)\"", header)
    if m !== nothing
        virial = parse.(FPPREC, split(m.captures[1]))
        if length(virial) == 9
            return [virial[1], virial[5], virial[9], virial[6], virial[3], virial[2]]
        elseif length(virial) == 6
            return virial
        end
    end
    return nothing
end

# ヘルパー関数
function parse_dielectric_tensor(header::String)
    m = match(r"dielectric_tensor=\"(.*?)\"", header)
    if m !== nothing
        dielectric_tensor = parse.(FPPREC, split(m.captures[1]))
        if length(dielectric_tensor) == 9
            return [dielectric_tensor[1], dielectric_tensor[5], dielectric_tensor[9], dielectric_tensor[6], dielectric_tensor[3], dielectric_tensor[2]]
        elseif length(dielectric_tensor) == 6
            return dielectric_tensor
        end
    end
    return nothing
end

# ヘルパー関数
# `energy=` は extxyz 慣行（ASE / OVITO 互換）で write_header が書き出す綴り，
# `potential_energy=` は外部ツール由来のデータセットで見る綴り．両方受ける
# （Issue #154：以前は後者のみで，自分が書いたファイルを読み戻せなかった）．
# 綴り違いが両方ある場合は明示的な `potential_energy=` を優先する．
# キー前の `(?:^|\s)` により `free_energy=` のような別キーの部分一致は拾わず，
# かつ行頭のキー（extxyz のコメント行はキーの順序が自由）も取りこぼさない．
# `=` 直後の `\s*` は，Issue #154 以前の write_header が書いた
# `energy= 1.5e+00`（幅指定のパディング）を restart で読み戻すための許容．
# 値が空のヘッダでは次のトークンを拾って parse が例外になる（黙って 0 にしない）．
function parse_potential_energy(header::String)
    for pat in (r"(?:^|\s)potential_energy=\s*([^\s]+)", r"(?:^|\s)energy=\s*([^\s]+)")
        m = match(pat, header)
        if !isnothing(m)
            return parse(FPPREC, replace(m.captures[1], "\"" => ""))
        end
    end
    return nothing
end

# ヘルパー関数
function parse_method(header::String)
    m = match(r"\smethod=([^\s]+)", header)
    if !isnothing(m)
        return Symbol(replace(m.captures[1], "\"" => ""))
    end
    return nothing
end

function write_header(io::IO, atoms::Atoms)
    # セル情報
    lattice = atoms.info[:lattice]'
    @printf(io, "Lattice=\"")
    for i in 1:9
        @printf(io, "%15.8e ", lattice[i] * UNITS[:Å])
    end

    # プロパティ定義
    @printf(io, "\" Properties=species:S:1:pos:R:3:id:I:1:shelldisp:R:3:velo:R:3:forces:R:3:peratomstress:R:6:groupid:I:1")

    # 周期境界条件
    @printf(io, " pbc=\"T T T\"")

    if haskey(atoms.info, :dipole)
        dipole = atoms.info[:dipole] .* UNITS[:Å]
        @printf(io, " dipole=\"%15.8e %15.8e %15.8e\"", dipole[1], dipole[2], dipole[3])
    end
    if haskey(atoms.info, :dipole_phys)
        dphys = atoms.info[:dipole_phys] .* UNITS[:Å]
        @printf(io, " dipole_phys=\"%15.8e %15.8e %15.8e\"", dphys[1], dphys[2], dphys[3])
    end

    # エネルギー（裸の key=value は幅指定でパディングしない — `%15.8e` だと
    # `energy= 1.5e+00` のように `=` の直後に空白が入り，空白区切りで
    # key=value を切る extxyz パーサ（readxyz 自身も含む）が読めない．Issue #154）
    @printf(io, " energy=%.8e", sum(atoms.info[:potential_energy]) * UNITS[:eV])

    # 応力テンソル
    @printf(io, " stress=\"")

    st = stress(atoms)
    st_mat = [st[1] st[6] st[5];
              st[6] st[2] st[4];
              st[5] st[4] st[3]]
    for s in (st_mat...,)
        @printf(io, "%15.8e ", s * UNITS[:eV] / UNITS[:Å]^3)
    end
    @printf(io, "\"\n")
end
