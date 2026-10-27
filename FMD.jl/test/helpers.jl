# test/helpers.jl
#
# Test fixtures + reusable assertions for FMD.jl.
# Loaded by runtests.jl after `using FMD, FMD.Cores, FMD.Calculators`.
#
# Conventions:
#   * Atoms.info[:lattice] is in Bohr (atomic units), rows = lattice vectors.
#   * Atoms.arrays[:pos] is fractional in [0,1).
#   * Energies are in Hartree internally; multiply by UNITS[:eV] for eV.
#   * Buckingham fields a/b/c may be zero, but rho must be > 0
#     (it appears as 1/rho in the kernel and as a divisor of r in exp(-r/rho)).

using LinearAlgebra
using Random

# --------------------------------------------------------------------
# Atoms factories
# --------------------------------------------------------------------

"""
    make_atoms(lattice_bohr, frac_pos, elemid; species=nothing, masses=nothing,
               velo=nothing, shelldisp=nothing)

Construct an `Atoms{Array}` populated with all keys that any Calculator
in this package may inspect.  All scalar arrays are filled with zeros
unless an explicit value is provided.

`species` is the id → element symbol table (index = elemid), stored
frame-level as `info[:species]` — per-atom species strings are never stored
(Issue #54).
"""
function make_atoms(lattice_bohr::AbstractMatrix, frac_pos::AbstractMatrix,
                    elemid::AbstractVector{<:Integer};
                    species::Union{Nothing,AbstractVector{String}} = nothing,
                    masses::Union{Nothing,AbstractVector} = nothing,
                    velo::Union{Nothing,AbstractMatrix} = nothing,
                    shelldisp::Union{Nothing,AbstractMatrix} = nothing)
    N = size(frac_pos, 1)
    @assert size(frac_pos, 2) == 3
    @assert length(elemid) == N
    @assert size(lattice_bohr) == (3, 3)

    info = Dict{Symbol,Any}(
        :lattice => Matrix{FMD.FPPREC}(lattice_bohr),
        :potential_energy => FMD.FPPREC[0],
        :virial => zeros(FMD.FPPREC, 6),
        :dielectric_tensor => zeros(FMD.FPPREC, 6),
        :method => Symbol[:test],
    )

    arrays = Dict{Symbol,Array}(
        :pos => Matrix{FMD.FPPREC}(frac_pos),
        :elemid => Vector{Int}(elemid),
        :forces => zeros(FMD.FPPREC, N, 3),
        :shellforces => zeros(FMD.FPPREC, N, 3),
        :shelldisp => shelldisp === nothing ? zeros(FMD.FPPREC, N, 3) : Matrix{FMD.FPPREC}(shelldisp),
        :velo => velo === nothing ? zeros(FMD.FPPREC, N, 3) : Matrix{FMD.FPPREC}(velo),
        :id => collect(1:N),
        :masses => masses === nothing ? ones(FMD.FPPREC, N) : Vector{FMD.FPPREC}(masses),
    )
    if species !== nothing
        info[:species] = Vector{String}(species)
    end
    return Atoms{Array}(info, arrays)
end

# --------------------------------------------------------------------
# NaCl rocksalt: lattice constant a = 5.64 Å (close to experiment).
# 8-atom conventional cell: Na at (0,0,0)+FCC, Cl at (1/2,1/2,1/2)+FCC.
# elemid: 1 = Na, 2 = Cl.
# --------------------------------------------------------------------

const NACL_A_ANG = 5.64
const NACL_A_BOHR = NACL_A_ANG / UNITS[:Å]
const NACL_R0_BOHR = NACL_A_BOHR / 2  # nearest-neighbor distance

function make_nacl_8(; a_bohr::Real = NACL_A_BOHR)
    lat = Matrix{FMD.FPPREC}(I, 3, 3) .* a_bohr
    # 4 Na (FCC) + 4 Cl (FCC offset by (1/2,1/2,1/2))
    na = [
        0.0 0.0 0.0;
        0.0 0.5 0.5;
        0.5 0.0 0.5;
        0.5 0.5 0.0;
    ]
    cl = [
        0.5 0.5 0.5;
        0.5 0.0 0.0;
        0.0 0.5 0.0;
        0.0 0.0 0.5;
    ]
    pos = vcat(na, cl)
    elemid = vcat(fill(1, 4), fill(2, 4))
    species = ["Na", "Cl"]  # id → symbol table
    mass_by_id = FMD.FPPREC[22.99 * UNITS[:dalton], 35.45 * UNITS[:dalton]]
    masses = mass_by_id[elemid]
    return make_atoms(lat, pos, elemid; species=species, masses=masses)
end

"""
    make_nacl_64(; a_bohr = NACL_A_BOHR)

2×2×2 replication of the 8-atom NaCl conventional cubic cell — 64 atoms
(32 Na + 32 Cl, equivalently 32 NaCl pairs) in a cubic cell of side 2a.
The conventional cell has 8 atoms; a 2×2×2 supercell in conventional
coordinates therefore contains 8·8 = 64 atoms.  This is a slightly
larger system than the user's "32-atom 2×2×2" intuition but matches the
crystallographic reality and is what Ewald sees as a representative
periodic NaCl supercell.
"""
function make_nacl_64(; a_bohr::Real = NACL_A_BOHR)
    base = make_nacl_8(; a_bohr = a_bohr)
    base_pos = base.arrays[:pos]
    base_elemid = base.arrays[:elemid]
    base_masses = base.arrays[:masses]
    Natoms = size(base_pos, 1) * 8  # 8 cells

    pos = Matrix{FMD.FPPREC}(undef, Natoms, 3)
    elemid = Vector{Int}(undef, Natoms)
    masses = Vector{FMD.FPPREC}(undef, Natoms)
    k = 0
    for ix in 0:1, iy in 0:1, iz in 0:1
        for i in axes(base_pos, 1)
            k += 1
            pos[k, 1] = (base_pos[i, 1] + ix) / 2
            pos[k, 2] = (base_pos[i, 2] + iy) / 2
            pos[k, 3] = (base_pos[i, 3] + iz) / 2
            elemid[k] = base_elemid[i]
            masses[k] = base_masses[i]
        end
    end
    @assert k == Natoms
    lat = Matrix{FMD.FPPREC}(I, 3, 3) .* (2 * a_bohr)
    return make_atoms(lat, pos, elemid; species=base.info[:species], masses=masses)
end

"""
    nacl_params(atoms; alpha=0.30, ewald_tolerance=1e-8, rc_factor=0.45,
                with_shell=false, buckingham=false)

Return a NamedTuple of Coulomb+Buckingham parameters where the Buckingham
contribution is identically zero (a=b=c=0, rho=1) so that the only
contribution is the Coulomb sum.  α is in 1/Bohr; rc is set to a fixed
fraction of the cell's shortest lattice spacing to stay within the
minimum-image convention.

With `with_shell=true`, non-zero shell charges and spring constants are
returned so CoreShell SCF + the shell-only kernel are exercised.

With `buckingham=true`, a Born–Mayer repulsion is added.  Needed by every
test that relaxes or asserts something about the equilibrium structure:
with a ≡ 0 there is no short-range wall, so rocksalt is not a minimum and
the structure collapses instead of relaxing.  Calibrated so that the ideal
`make_nacl_64` at `with_shell=true` sits at its own E(a) minimum — see the
body for the cutoff requirement it comes with.
"""
function nacl_params(atoms::Atoms;
                     alpha::Real = 0.30,
                     ewald_tolerance::Real = 1e-8,
                     rc_factor::Real = 0.45,
                     with_shell::Bool = false,
                     buckingham::Bool = false)
    nelem = 2
    lat = atoms.info[:lattice]
    L = minimum(norm(lat[i, :]) for i in 1:3)
    rc = rc_factor * L
    qc = FMD.FPPREC[+1.0, -1.0]  # Na+, Cl-
    if with_shell
        # Choose qs so the total charge per cell is zero
        # (qc + qs = +0.8 / -0.8 ⇒ Σ qc = Σ qs = 0).  This is required
        # for the BEC acoustic sum rule and for Ewald to be well-defined.
        qs = FMD.FPPREC[-0.2, +0.2]
        k2 = FMD.FPPREC[10.0, 20.0]
    else
        qs = zeros(FMD.FPPREC, nelem)
        k2 = zeros(FMD.FPPREC, nelem)
    end
    eta = fill(FMD.FPPREC(1e-3), nelem)
    # Born–Mayer repulsion.  Without it the rocksalt structure is NOT a
    # minimum — bare Coulomb + springs has no short-range wall and the cell
    # collapses — so any test that relaxes, or that asserts something about
    # the equilibrium structure, must pass `buckingham = true`.
    #
    # Born–Mayer form A·exp(-r/ρ) with the usual ordering (anion–anion
    # repulsion strongest, cation–cation weakest).  A and ρ are CALIBRATED,
    # not guessed: with `with_shell = true` (effective ion charge ±0.8) they
    # put the E(a) minimum of the ideal rocksalt at exactly the
    # `NACL_A_ANG = 5.64` Å the fixtures use, with a residual σ of
    # -0.16 GPa.  So an unrattled `make_nacl_64` is, to grid resolution, its
    # own equilibrium structure — which is what makes it usable as a
    # relaxation target with an independently known answer.
    #
    # Requires a cutoff that actually reaches the nearest neighbours at
    # a/2 = 5.33 Bohr.  `make_nacl_64` (L = 2a) at rc_factor = 0.45 does;
    # `make_nacl_8` (L = a) does NOT — there rc = 0.45a < a/2 and the whole
    # repulsion is cut off, leaving bare Coulomb with no minimum at all.
    #
    # A test fixture, not a production force field.
    a_buck = zeros(FMD.FPPREC, nelem, nelem)
    rho_buck = ones(FMD.FPPREC, nelem, nelem)
    if buckingham
        a_buck[1, 1] = FMD.FPPREC(10.25)   # Na-Na
        a_buck[1, 2] = FMD.FPPREC(20.5)    # Na-Cl
        a_buck[2, 1] = FMD.FPPREC(20.5)
        a_buck[2, 2] = FMD.FPPREC(41.0)    # Cl-Cl
        rho_buck .= FMD.FPPREC(0.62)
    end
    return (
        alpha = FMD.FPPREC(alpha),
        ewald_tolerance = FMD.FPPREC(ewald_tolerance),
        rc = FMD.FPPREC(rc),
        qc = qc,
        qs = qs,
        eta_c = eta,
        eta_s = eta,
        a = a_buck,
        b = zeros(FMD.FPPREC, nelem, nelem),
        rho = rho_buck,
        c = zeros(FMD.FPPREC, nelem, nelem),
        e = FMD.FPPREC(1.0),
        k2 = k2,
        k4 = zeros(FMD.FPPREC, nelem),
        efield = zeros(FMD.FPPREC, 3),
        cgtol = FMD.FPPREC(1e-6),
        cgmaxiter = 30,
        skin = FMD.FPPREC(0.0),
    )
end

# --------------------------------------------------------------------
# Cubic perovskite ABO3 — 5 atoms in the primitive cell.
# Used as a small fixture for shell-model / spline / ACE smoke.
# --------------------------------------------------------------------

function make_perovskite_5(; a_ang::Real = 4.0,
                           A::String = "Sr",
                           B::String = "Ti",
                           A_mass::Real = 87.62,
                           B_mass::Real = 47.87,
                           O_mass::Real = 15.999)
    a_bohr = a_ang / UNITS[:Å]
    lat = Matrix{FMD.FPPREC}(I, 3, 3) .* a_bohr
    pos = [
        0.0 0.0 0.0;       # A
        0.5 0.5 0.5;       # B
        0.0 0.5 0.5;       # O
        0.5 0.0 0.5;       # O
        0.5 0.5 0.0;       # O
    ]
    elemid = [1, 2, 3, 3, 3]
    species = [A, B, "O"]  # id → symbol table
    masses = [FMD.FPPREC(A_mass * UNITS[:dalton]),
              FMD.FPPREC(B_mass * UNITS[:dalton]),
              FMD.FPPREC(O_mass * UNITS[:dalton]),
              FMD.FPPREC(O_mass * UNITS[:dalton]),
              FMD.FPPREC(O_mass * UNITS[:dalton])]
    return make_atoms(lat, pos, elemid; species=species, masses=masses)
end

"""
    replicate_atoms(atoms, nrep; shuffle_order = false) -> Atoms{Array}

`nrep = (nx, ny, nz)` supercell replication of `atoms` along its lattice
vectors (fractional positions are compressed accordingly, the lattice rows
are scaled). With `shuffle_order = true` the atom order is randomly permuted
(global RNG), so tests exercise layouts that are neither species- nor
cell-sorted.
"""
function replicate_atoms(atoms::Atoms{Array}, nrep::NTuple{3,Int};
                         shuffle_order::Bool = false)
    @assert all(>=(1), nrep)
    # Only pos/elemid/masses are replicated (the frame-level info[:species]
    # table is carried over as-is); refuse to silently drop non-zero
    # per-atom state (make_atoms would reset it to zero).
    for k in (:velo, :shelldisp)
        haskey(atoms.arrays, k) && !all(iszero, atoms.arrays[k]) &&
            error("replicate_atoms: non-zero $k would be dropped (not replicated)")
    end
    base_pos = atoms.arrays[:pos]
    base_elemid = atoms.arrays[:elemid]
    N0 = size(base_pos, 1)
    nx, ny, nz = nrep
    N = N0 * nx * ny * nz

    pos = Matrix{FMD.FPPREC}(undef, N, 3)
    elemid = Vector{Int}(undef, N)
    masses = Vector{FMD.FPPREC}(undef, N)
    k = 0
    for ix in 0:(nx-1), iy in 0:(ny-1), iz in 0:(nz-1), i in 1:N0
        k += 1
        pos[k, 1] = (base_pos[i, 1] + ix) / nx
        pos[k, 2] = (base_pos[i, 2] + iy) / ny
        pos[k, 3] = (base_pos[i, 3] + iz) / nz
        elemid[k] = base_elemid[i]
        masses[k] = atoms.arrays[:masses][i]
    end

    if shuffle_order
        perm = randperm(N)
        pos = pos[perm, :]
        elemid = elemid[perm]
        masses = masses[perm]
    end

    lat = Matrix{FMD.FPPREC}(atoms.info[:lattice])
    lat = vcat((lat[1:1, :] .* nx), (lat[2:2, :] .* ny), (lat[3:3, :] .* nz))

    return make_atoms(lat, pos, elemid;
                      species = get(atoms.info, :species, nothing),
                      masses = masses)
end

"""
    perovskite_params(; nelem=3, alpha=0.20, rc_factor=0.45)

Generic shell-model + Buckingham params for the 5-atom perovskite, with all
Buckingham coefficients zero (Coulomb-only) and small but non-zero shell
charges and spring constants so CoreShell exercise paths trigger.
"""
function perovskite_params(atoms::Atoms;
                           alpha::Real = 0.20,
                           ewald_tolerance::Real = 1e-8,
                           rc_factor::Real = 0.45,
                           with_shell::Bool = false)
    lat = atoms.info[:lattice]
    L = minimum(norm(lat[i, :]) for i in 1:3)
    rc = rc_factor * L
    nelem = 3
    qc = FMD.FPPREC[+2.0, +4.0, -2.0]
    if with_shell
        qs = FMD.FPPREC[-0.5, -1.0, +0.5]
        k2 = FMD.FPPREC[10.0, 30.0, 20.0]
    else
        qs = zeros(FMD.FPPREC, nelem)
        k2 = zeros(FMD.FPPREC, nelem)
    end
    eta = fill(FMD.FPPREC(1e-3), nelem)
    return (
        alpha = FMD.FPPREC(alpha),
        ewald_tolerance = FMD.FPPREC(ewald_tolerance),
        rc = FMD.FPPREC(rc),
        qc = qc,
        qs = qs,
        eta_c = eta,
        eta_s = eta,
        a = zeros(FMD.FPPREC, nelem, nelem),
        b = zeros(FMD.FPPREC, nelem, nelem),
        rho = ones(FMD.FPPREC, nelem, nelem),
        c = zeros(FMD.FPPREC, nelem, nelem),
        e = FMD.FPPREC(1.0),
        k2 = k2,
        k4 = zeros(FMD.FPPREC, nelem),
        efield = zeros(FMD.FPPREC, 3),
        cgtol = FMD.FPPREC(1e-6),
        cgmaxiter = 30,
        skin = FMD.FPPREC(0.0),
    )
end

# --------------------------------------------------------------------
# Re-evaluate a Calculator on a fresh copy of `atoms`.
# Returned tuple is in atomic units (Hartree, Bohr, Hartree/Bohr).
# --------------------------------------------------------------------

"""
    reset!(atoms)

Zero out energy/virial/forces/shellforces accumulators in place.
"""
function reset!(atoms::Atoms)
    Cores.clear!(atoms,
                 (:potential_energy, :virial, :forces, :shellforces))
    return atoms
end

# Apply a closure `calc!(atoms)` after resetting and return (E, F).
function evaluate(calc!, atoms::Atoms)
    reset!(atoms)
    calc!(atoms)
    return atoms.info[:potential_energy][1], copy(atoms.arrays[:forces])
end

# --------------------------------------------------------------------
# Deep copy that survives the `Atoms` Dict-of-Array structure.
# --------------------------------------------------------------------

deepcopy_atoms(a::Atoms) =
    Atoms{Array}(deepcopy(a.info), Dict{Symbol,Array}(k => copy(v) for (k, v) in a.arrays))

# --------------------------------------------------------------------
# Convert all array-typed entries of a NamedTuple-like params bundle to
# `arrtyp`.  Used by GPU tests: NPT.jl line 92 does the same trick to
# move per-element arrays (qc, qs, k2, k4, efield, a, b, rho, c, ...)
# onto the device alongside `Atoms{CuArray}`.
# --------------------------------------------------------------------
function arrtyp_prms(arrtyp, prms::NamedTuple)
    pairs = []
    for (k, v) in zip(keys(prms), values(prms))
        if v isa AbstractArray
            push!(pairs, k => arrtyp(v))
        else
            push!(pairs, k => v)
        end
    end
    return NamedTuple(pairs)
end

# --------------------------------------------------------------------
# Canonical pair maps (shared by core / GPU / apps / MPI tests).
#
# A list is reduced to a canonical form so different builders can be
# compared exactly: a Dict keyed by the ordered (id_i, id_owner_of_j)
# pair, holding the sorted list of image separations Δr = r_j_image − r_i
# (Cartesian, rounded to 1e-7 — far above the ~1e-13 FP jitter between
# computation paths, far below any physical separation).
# --------------------------------------------------------------------

const _PairMap = Dict{Tuple{Int,Int},Vector{NTuple{3,Float64}}}

_canon(Δ) = (round(Float64(Δ[1]); digits = 7),
             round(Float64(Δ[2]); digits = 7),
             round(Float64(Δ[3]); digits = 7))

# Ghost-based lists: neighbours are extended row indices; the owner
# identity comes from `src` (serial: owner row is local) or from
# `dd.ghost_id` (domain decomposition: the owner lives on another rank).
function ghost_pairmap(atoms, nl)
    g = nl.ghosts
    ids = Array(atoms.arrays[:id])
    xc = Array(g.xcart)
    list = Array(nl.list)
    src = Array(g.src)
    out = _PairMap()
    # NOTE: explicit nesting — in the comma form `for i in ..., k in ...` a
    # `break` would exit BOTH loops and silently truncate the map at the
    # first zero-padded row.
    for i in 1:g.n_owned
        for k in axes(list, 2)
            j = list[i, k]
            j < 1 && break
            owner_id = if j <= g.n_owned
                ids[Int(j)]
            elseif g.dd === nothing
                ids[Int(src[j-g.n_owned])]
            else
                g.dd.ghost_id[j-g.n_owned]
            end
            Δ = _canon((xc[j, 1] - xc[i, 1], xc[j, 2] - xc[i, 2], xc[j, 3] - xc[i, 3]))
            push!(get!(out, (ids[i], owner_id), NTuple{3,Float64}[]), Δ)
        end
    end
    foreach(sort!, values(out))
    return out
end

# --------------------------------------------------------------------
# Finite-difference HVP oracles (Issue #147). These are the pre-#147
# `jvp!` / `dF_du_dirderiv!` implementations, kept verbatim as the
# independent verification oracle for the analytic `hvp!`: central
# difference of the full shell/core force stacks along a direction v.
# They perturb and restore `:shelldisp` and leave the force arrays at
# the last perturbed state — evaluate analytic products BEFORE calling
# these on a shared fixture.
# --------------------------------------------------------------------

function fd_jvp!(out, v, atoms::Atoms, nl, coulomb, prms::NamedTuple;
                 delta::Real = 1e-5)
    u = atoms.arrays[:shelldisp]
    sf = atoms.arrays[:shellforces]

    @. u += delta * v
    Calculators.shell_forces!(atoms, nl, coulomb, prms)
    fplus = copy(sf)

    @. u -= 2delta * v
    Calculators.shell_forces!(atoms, nl, coulomb, prms)
    fminus = copy(sf)

    @. u += delta * v

    @. out = (fplus - fminus) / (2delta)
    return out
end

function fd_dFdu_dirderiv!(out, v, atoms::Atoms, nl, coulomb, prms::NamedTuple;
                           delta::Real = 1e-5)
    u = atoms.arrays[:shelldisp]
    F = atoms.arrays[:forces]

    @. u += delta * v
    Calculators.core_forces!(atoms, nl, coulomb, prms)
    Fplus = copy(F)

    @. u -= 2delta * v
    Calculators.core_forces!(atoms, nl, coulomb, prms)
    Fminus = copy(F)

    @. u += delta * v

    @. out = (Fplus - Fminus) / (2delta)
    return out
end

# --------------------------------------------------------------------
# Group lattice for the polarisation-map / gauge fixtures
# --------------------------------------------------------------------

"""
    octant_groupid!(atoms, reps = (2, 2, 2)) -> atoms

Assign `atoms.arrays[:groupid]` by spatial brick of the wrapped fractional
coordinate, flat ASE-natural index `ix·Ny·Nz + iy·Nz + iz + 1` (iz
fastest) — the group-lattice convention of the polarisation biases and
[`write_polmap`](@ref). Assign BEFORE rattling: on the ideal NaCl fixtures
half the atoms sit exactly on a brick boundary, and a rattle would scatter
them across the faces and leave charged groups (which PolarizationGauge
rejects). On `make_nacl_64` with `(2, 2, 2)` every brick is one neutral
conventional cell (4 Na + 4 Cl).
"""
function octant_groupid!(atoms::Atoms, reps::NTuple{3,Int} = (2, 2, 2))
    Nx, Ny, Nz = reps
    w = Cores.wrapfrac.(atoms.arrays[:pos])
    gx = min.(Int.(floor.(w[:, 1] .* Nx)), Nx - 1)
    gy = min.(Int.(floor.(w[:, 2] .* Ny)), Ny - 1)
    gz = min.(Int.(floor.(w[:, 3] .* Nz)), Nz - 1)
    atoms.arrays[:groupid] = gx .* (Ny * Nz) .+ gy .* Nz .+ gz .+ 1
    return atoms
end
