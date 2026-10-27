using KernelAbstractions
using Adapt
using ..FMD: Atoms, FPPREC
using ..FMD.Cores

export NLHParam, nlh_saved, calculate!

# ===========================================================================
# NLH (Nordlund–Lehtola–Hobler) screened-nuclear repulsion — Δ-learning
# baseline for the short range the MLIP training data never reaches.
#
#   V_ij(r) = Z_i Z_j / r · φ_ij(r),   φ_ij(r) = Σ_k a_k exp(-b_k r)
#
# in atomic units (e²/4πε₀ = 1, r in Bohr, V in Hartree). The screening
# function is refitted per element pair against all-electron DFT dimer
# curves — see K. Nordlund, S. Lehtola, G. Hobler, Phys. Rev. A 111,
# 032818 (2025); coefficients from doi:10.5281/zenodo.14172633 (CC-BY 4.0),
# bundled as `data/nlh/nlh_coeffs.dat`.
#
# Two design decisions that the rest of the code depends on:
#
# 1. **Core–core.** The term is a nucleus–nucleus interaction, so it reads
#    core positions only and writes `arrays[:forces]` only. It therefore
#    contributes exactly zero to the shell-displacement gradient, which
#    means the shell CG inner loop (CoreShell.jl), the analytic HVP
#    (HVP.jl, Issue #147) and the shadow XL-BOMD functional (Issue #150,
#    #156) are all untouched — there is no `OnlyShellForce` variant and
#    none is needed. It also makes the Δ-learning subtract→add round trip
#    exact independently of where the shells relax to.
#
# 2. **Shifted force.** The truncation is `ϕ(r) − ϕ(rc) − (r−rc)ϕ′(rc)`,
#    the same scheme CoulombBuckingham uses, so both E and F are continuous
#    at `rc`. The cutoff form is part of the potential's *definition*: fit
#    and MD must use the identical expression, and changing it later would
#    invalidate every potential fitted before the change. The linear term
#    it adds is trivially representable in the MLIP's own 2-body basis, so
#    it costs no accuracy in the Δ-learning residual.
# ===========================================================================

const NLH_NTERM = 3          # exponentials per pair in the published fit
const NLH_MAXZ  = 92         # table coverage
const NLH_TABLE_ID = "NLH-2025 zenodo:10.5281/zenodo.14172633 nlh_coeffs.dat"

nlh_default_table() =
    normpath(joinpath(@__DIR__, "..", "..", "data", "nlh", "nlh_coeffs.dat"))

"""
    NLHParam{FP,TA,TM,TZ}

Per-species-pair NLH screened-nuclear repulsion, ready for
[`calculate!`](@ref). Build it from element symbols with
`NLHParam(elements, rc)`; `Adapt.@adapt_structure` moves it to the GPU.

# Fields

- `rc::FP` — cutoff (Bohr). Shifted-force truncation, see the file header.
- `Z` — `(ns,)` atomic numbers, species-id indexed. Kept for provenance and
  serialisation; the kernel reads `zz` instead.
- `zz` — `(ns, ns)` products `Z_a · Z_b` in `FPPREC` (kernel input).
- `a`, `b` — `(NLH_NTERM, ns, ns)` screening coefficients, symmetric in the
  species pair. `a` is dimensionless with `Σ_k a_k = 1`; `b` is in Bohr⁻¹
  (the published table is Å⁻¹ — converted at load).
- `phi_rc`, `dphi_rc` — `(ns, ns)` cutoff constants `ϕ(rc)` and `ϕ′(rc)` of
  the shifted-force scheme. Precomputed because they are per-species-pair
  constants: evaluating them inside the neighbour loop would triple the
  `exp` count per pair (same trick as `dϕrc_ss` in CoulombBuckingham).
"""
struct NLHParam{FP, TA, TM, TZ}
    rc::FP
    Z::TZ
    zz::TM
    a::TA
    b::TA
    phi_rc::TM
    dphi_rc::TM
end

Adapt.@adapt_structure NLHParam

# ---- table loading ------------------------------------------------- #

# Parse `nlh_coeffs.dat` into (Z1,Z2) => (a, b) with Z1 ≤ Z2, b still in Å⁻¹.
# Only the requested pairs are materialised.
function _nlh_read_table(path, want::Set{Tuple{Int,Int}})
    isfile(path) || error("NLHParam: coefficient table not found: $path")
    found = Dict{Tuple{Int,Int},NTuple{2,NTuple{NLH_NTERM,Float64}}}()
    for line in eachline(path)
        s = strip(line)
        (isempty(s) || startswith(s, "#")) && continue
        f = split(s)
        length(f) >= 2 + 2*NLH_NTERM || error(
            "NLHParam: malformed row in $path (expected ≥ $(2 + 2*NLH_NTERM) " *
            "columns, got $(length(f))): $s")
        key = (parse(Int, f[1]), parse(Int, f[2]))
        key in want || continue
        a = ntuple(k -> parse(Float64, f[1 + 2k]), NLH_NTERM)
        b = ntuple(k -> parse(Float64, f[2 + 2k]), NLH_NTERM)
        found[key] = (a, b)
    end
    return found
end

# Physical integrity of one pair's screening function. Both checks are hard
# errors: a screening function that does not satisfy them is not a usable MD
# baseline, and silently using it would corrupt every fit built on top.
#
# NOTE on coverage: `b_k == 0` catches only the exactly non-decaying case.
# The table also contains terms with small but non-zero `b_k` (16 pairs have
# one below 1 Å⁻¹, down to 0.047 Å⁻¹ = a 21 Å decay length), which are the
# same extrapolation artefact seen through a finite window rather than an
# infinite one. Those are not rejected — they are legal, published fits and
# the residual MLIP absorbs a smooth mid-range background without trouble —
# but `_nlh_warn_soft_tail` reports the actual ϕ(rc) so the choice is never
# silent.
function _nlh_check_pair(Z1, Z2, a, b)
    # φ(0) = Σ a_k = 1 ⇔ V → Z₁Z₂/r (unscreened Coulomb) as r → 0.
    s = sum(a)
    isapprox(s, 1.0; atol = 1e-4) || error(
        "NLHParam: screening coefficients for Z = ($Z1, $Z2) sum to $s, not 1 " *
        "— V(r) would not approach the unscreened Coulomb limit Z₁Z₂/r as r → 0. " *
        "The bundled table is corrupt or was replaced.")
    # A term with b = 0 never decays: φ(∞) = a_k ≠ 0 leaves an unscreened
    # Coulomb tail (e.g. O–Na: 1.27 eV at 5 Å, 0.63 eV at 10 Å). 24 of the
    # 4278 published pairs are like this — an artefact of fitting only the
    # strongly repulsive region (the table quotes its error for V > 10 eV).
    for k in 1:NLH_NTERM
        (a[k] != 0 && b[k] == 0) && error(
            "NLHParam: the published screening function for Z = ($Z1, $Z2) has a " *
            "non-decaying term (a[$k] = $(a[k]) with b[$k] = 0), so " *
            "V(r) → $(a[k])·Z₁Z₂/r instead of decaying. The NLH fit is only " *
            "validated in the strongly repulsive short range, and this tail is an " *
            "extrapolation artefact — it cannot be used as an MD baseline.")
    end
    return nothing
end

"""
    NLHParam(elements::AbstractDict, rc; table = <bundled>) -> NLHParam

Build the NLH baseline for the species in `elements` (element symbol →
species id, the same mapping the rest of `params` uses) with cutoff `rc`
in Bohr.

Every pair is validated at load (unknown symbol, `Z > 92`, missing table
row, species ids that are not a permutation of `1:ns`, `Σ_k a_k ≠ 1`, or an
exactly non-decaying `b_k = 0` term are all errors — this never falls back
to zero), and the screening rates are converted from the table's Å⁻¹ to
Bohr⁻¹.

A pair whose screening merely decays *slowly* is accepted but warned about,
with the actual `ϕ(rc)` printed: the published fit is only validated in the
strongly repulsive short range, so a baseline that is still eV-scale at its
own cutoff is a modelling choice, not a bug to reject blindly.
"""
function NLHParam(elements::AbstractDict, rc; table = nlh_default_table())
    rc > 0 || error("NLHParam: rc must be > 0 (got $rc)")
    ns = length(elements)
    Z = zeros(Int, ns)
    for (sym, id) in elements
        haskey(ATOMIC_NUMBERS, sym) || error(
            "NLHParam: element symbol \"$sym\" is not in ATOMIC_NUMBERS, so its " *
            "nuclear charge is unknown")
        1 <= id <= ns || error(
            "NLHParam: species id $id for \"$sym\" is outside 1:$ns")
        Z[id] = ATOMIC_NUMBERS[sym]
        Z[id] <= NLH_MAXZ || error(
            "NLHParam: \"$sym\" has Z = $(Z[id]) > $NLH_MAXZ — outside the " *
            "published NLH table")
    end
    # A duplicated or skipped species id leaves a hole here, and Z = 0 would
    # go on to look up a nonexistent table row with a confusing message.
    all(>(0), Z) || error(
        "NLHParam: values(elements) must be a permutation of 1:$ns " *
        "(got $(sort!(collect(values(elements)))))")

    want = Set((min(Z[sa], Z[sb]), max(Z[sa], Z[sb])) for sb in 1:ns for sa in 1:sb)
    tab = _nlh_read_table(table, want)
    missing_pairs = [k for k in want if !haskey(tab, k)]
    isempty(missing_pairs) || error(
        "NLHParam: no coefficients for Z pair(s) " *
        join(("($(k[1]), $(k[2]))" for k in sort(missing_pairs)), ", ") *
        " in $table")

    a = zeros(FPPREC, NLH_NTERM, ns, ns)
    b = zeros(FPPREC, NLH_NTERM, ns, ns)
    zz = zeros(FPPREC, ns, ns)
    for sb in 1:ns, sa in 1:ns
        key = (min(Z[sa], Z[sb]), max(Z[sa], Z[sb]))
        ak, bk = tab[key]
        _nlh_check_pair(key[1], key[2], ak, bk)
        zz[sa, sb] = FPPREC(Z[sa] * Z[sb])
        for k in 1:NLH_NTERM
            a[k, sa, sb] = FPPREC(ak[k])
            # table is Å⁻¹; b·r must be dimensionless with r in Bohr
            b[k, sa, sb] = FPPREC(bk[k] * UNITS[:Å])
        end
    end
    phi_rc, dphi_rc = _nlh_cutoff_tables(FPPREC(rc), zz, a, b)
    _nlh_warn_soft_tail(Z, FPPREC(rc), phi_rc)
    return NLHParam(FPPREC(rc), Z, zz, a, b, phi_rc, dphi_rc)
end

# A short-range baseline should be negligible where it is cut off. When it
# is not, the pair's screening function still carries a slowly decaying
# term at rc, and the residual MLIP has to absorb an eV-scale smooth
# background over the whole fitted range. That is a modelling choice the
# user must make knowingly, so report the number rather than guess a
# threshold to reject at (`b_k == 0`, the exactly non-decaying case, is a
# hard error in `_nlh_check_pair`).
const NLH_SOFT_TAIL_WARN = 1e-2 / 27.21138625   # 10 meV in Hartree

function _nlh_warn_soft_tail(Z, rc, phi_rc)
    ns = length(Z)
    loud = [(Z[sa], Z[sb], phi_rc[sa, sb]) for sb in 1:ns for sa in 1:sb
            if abs(phi_rc[sa, sb]) > NLH_SOFT_TAIL_WARN]
    isempty(loud) && return nothing
    @warn "NLHParam: the baseline is not negligible at its own cutoff for " *
          "$(length(loud)) pair(s) — the published screening function still has a " *
          "slowly decaying term there. Shifted-force removes the value and slope at " *
          "rc, but the residual MLIP must absorb the remaining mid-range background; " *
          "consider a larger rc or dropping the baseline for these pairs." *
          join(("\n  Z = ($z1, $z2): ϕ(rc) = $(round(v * 27.21138625; digits = 4)) eV"
                for (z1, z2, v) in loud)) *
          "\n  (rc = $(round(rc * 0.529177210903; digits = 3)) Å)"
    return nothing
end

# ϕ(rc) and ϕ′(rc) per species pair — the shifted-force constants. Built
# from the same `nlh_phi` / `nlh_dphi` the kernel uses, so the truncation is
# defined by one expression only.
function _nlh_cutoff_tables(rc, zz, a, b)
    ns = size(zz, 1)
    phi_rc  = zeros(FPPREC, ns, ns)
    dphi_rc = zeros(FPPREC, ns, ns)
    for sb in 1:ns, sa in 1:ns
        phi_rc[sa, sb]  = nlh_phi(rc, zz[sa, sb], a[1, sa, sb], b[1, sa, sb],
                                  a[2, sa, sb], b[2, sa, sb], a[3, sa, sb], b[3, sa, sb])
        dphi_rc[sa, sb] = nlh_dphi(rc, zz[sa, sb], a[1, sa, sb], b[1, sa, sb],
                                   a[2, sa, sb], b[2, sa, sb], a[3, sa, sb], b[3, sa, sb])
    end
    return phi_rc, dphi_rc
end

# ---- serialisation -------------------------------------------------- #

"""
    nlh_saved(nlh::NLHParam) -> NamedTuple

Plain-data view `(; rc, Z, a, b, table_id)` for the potential JLD2 written
by the fit apps. The **coefficients the fit actually subtracted** are
stored, not a pointer to the bundled table, so replacing or updating
`data/nlh/nlh_coeffs.dat` later can never silently change what MD adds
back. `zz` is omitted (it is `Z_a·Z_b`, rebuilt on load).
"""
nlh_saved(nlh::NLHParam) = (; rc       = FPPREC(nlh.rc),
                            Z        = Vector{Int}(Array(nlh.Z)),
                            a        = Array(nlh.a),
                            b        = Array(nlh.b),
                            table_id = NLH_TABLE_ID)

"""
    NLHParam(saved::NamedTuple) -> NLHParam

Rebuild from the `nlh` block of a potential JLD2 (see [`nlh_saved`](@ref)).
Re-runs every physical integrity check on the stored coefficients, so a
hand-edited or truncated file cannot install a silently wrong baseline.

The advisory `ϕ(rc)` warning is deliberately *not* repeated here: this path
reproduces a baseline the fit already committed to, and the only way to act
on the warning at load time would be to refit.
"""
function NLHParam(saved::NamedTuple)
    for k in (:rc, :Z, :a, :b)
        haskey(saved, k) || error("NLHParam: saved block is missing field :$k")
    end
    rc = saved.rc
    (rc isa Real && rc > 0) || error("NLHParam: saved rc must be a positive Real (got $rc)")
    Z = saved.Z
    ns = length(Z)
    ns >= 1 || error("NLHParam: saved Z is empty")
    all(z -> 1 <= z <= NLH_MAXZ, Z) || error(
        "NLHParam: saved Z must all lie in 1:$NLH_MAXZ (got $Z)")
    size(saved.a) == (NLH_NTERM, ns, ns) && size(saved.b) == (NLH_NTERM, ns, ns) || error(
        "NLHParam: saved a/b must both be ($NLH_NTERM, $ns, $ns) for $ns species " *
        "(got $(size(saved.a)) and $(size(saved.b)))")
    all(isfinite, saved.a) && all(isfinite, saved.b) || error(
        "NLHParam: saved coefficients contain non-finite entries")

    a = Array{FPPREC}(saved.a)
    b = Array{FPPREC}(saved.b)
    zz = zeros(FPPREC, ns, ns)
    for sb in 1:ns, sa in 1:ns
        # The pair potential is symmetric; an asymmetric table would make
        # the energy depend on neighbour-list ordering.
        for k in 1:NLH_NTERM
            (a[k, sa, sb] == a[k, sb, sa] && b[k, sa, sb] == b[k, sb, sa]) || error(
                "NLHParam: saved coefficients are not symmetric in the species pair " *
                "at (k, sa, sb) = ($k, $sa, $sb)")
        end
        _nlh_check_pair(Z[sa], Z[sb],
                        ntuple(k -> a[k, sa, sb], NLH_NTERM),
                        ntuple(k -> b[k, sa, sb], NLH_NTERM))
        zz[sa, sb] = FPPREC(Z[sa] * Z[sb])
    end
    return NLHParam(FPPREC(rc), Vector{Int}(Z), zz, a, b,
                    _nlh_cutoff_tables(FPPREC(rc), zz, a, b)...)
end

# ---- pair function -------------------------------------------------- #
#
# Written against a plain coefficient tuple so the kernel, the shifted-force
# constants, and the tests all evaluate literally the same expression.

@inline @fastmath function nlh_phi(r, zz, a1, b1, a2, b2, a3, b3)
    s = a1*exp(-b1*r) + a2*exp(-b2*r) + a3*exp(-b3*r)
    return zz * s / r
end

@inline @fastmath function nlh_dphi(r, zz, a1, b1, a2, b2, a3, b3)
    s  = a1*exp(-b1*r) + a2*exp(-b2*r) + a3*exp(-b3*r)
    ds = -a1*b1*exp(-b1*r) - a2*b2*exp(-b2*r) - a3*b3*exp(-b3*r)
    return zz * (r*ds - s) / (r*r)
end

# ---- kernel --------------------------------------------------------- #
#
# Ghost format (#57): reads the extended arrays of `nlist.ghosts`, so pair
# vectors are plain Cartesian differences. Pure gather — writes only at the
# owned row I, so no fold-back. Accumulation convention matches
# kernel_gcb!: half the pair energy and half the pair virial per ordered
# pair, the full force.

@kernel inbounds = true function kernel_nlh!(
    potential_energies, forces, virials,
    @Const(xcart), @Const(elemid), @Const(a), @Const(b), @Const(zz),
    @Const(phi_rc), @Const(dphi_rc), rc, @Const(nlist))

    I = @index(Global)

    sᵢ = elemid[I]
    xi, yi, zi = xcart[I,1], xcart[I,2], xcart[I,3]
    Uᵢ = potential_energies[I]
    Fᵢ₁, Fᵢ₂, Fᵢ₃ = forces[I,1], forces[I,2], forces[I,3]
    vᵢ₁, vᵢ₂, vᵢ₃ = virials[I,1], virials[I,2], virials[I,3]
    vᵢ₄, vᵢ₅, vᵢ₆ = virials[I,4], virials[I,5], virials[I,6]

    for jlist in axes(nlist, 2)
        j = nlist[I, jlist]
        if j < 1
            break
        end
        sⱼ = elemid[j]

        rx = xi - xcart[j,1]
        ry = yi - xcart[j,2]
        rz = zi - xcart[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)

        zzᵢⱼ = zz[sᵢ,sⱼ]
        a1 = a[1,sᵢ,sⱼ]; b1 = b[1,sᵢ,sⱼ]
        a2 = a[2,sᵢ,sⱼ]; b2 = b[2,sᵢ,sⱼ]
        a3 = a[3,sᵢ,sⱼ]; b3 = b[3,sᵢ,sⱼ]

        ϕrc  = phi_rc[sᵢ,sⱼ]
        dϕrc = dphi_rc[sᵢ,sⱼ]
        ϕ = nlh_phi(r, zzᵢⱼ, a1, b1, a2, b2, a3, b3) - ϕrc - (r - rc) * dϕrc
        dϕdr_r = (nlh_dphi(r, zzᵢⱼ, a1, b1, a2, b2, a3, b3) - dϕrc) / r

        cutoff = ifelse((r > rc), zero(rc), one(rc))
        ϕ *= cutoff
        dϕdr_r *= cutoff

        Uᵢ += FPPREC(0.5)*ϕ
        Fᵢ₁ -= dϕdr_r * rx
        Fᵢ₂ -= dϕdr_r * ry
        Fᵢ₃ -= dϕdr_r * rz
        vᵢ₁ -= FPPREC(0.5)*dϕdr_r * rx * rx
        vᵢ₂ -= FPPREC(0.5)*dϕdr_r * ry * ry
        vᵢ₃ -= FPPREC(0.5)*dϕdr_r * rz * rz
        vᵢ₄ -= FPPREC(0.5)*dϕdr_r * ry * rz
        vᵢ₅ -= FPPREC(0.5)*dϕdr_r * rx * rz
        vᵢ₆ -= FPPREC(0.5)*dϕdr_r * rx * ry
    end

    potential_energies[I] = Uᵢ
    forces[I,1] = Fᵢ₁
    forces[I,2] = Fᵢ₂
    forces[I,3] = Fᵢ₃
    virials[I,1] = vᵢ₁
    virials[I,2] = vᵢ₂
    virials[I,3] = vᵢ₃
    virials[I,4] = vᵢ₄
    virials[I,5] = vᵢ₅
    virials[I,6] = vᵢ₆
end

"""
    calculate!(nlh::NLHParam, atoms::Atoms)
    calculate!(nlh::NLHParam, atoms::Atoms, nlist::NeighborList)

Add the NLH baseline contribution of `nlh` to
`atoms.info[:potential_energy]`, `atoms.info[:virial]` and
`atoms.arrays[:forces]`. Nothing is written to `arrays[:shellforces]` —
the term is core–core (see the file header), so it is invariant under the
shell displacements and can be evaluated once per step, outside the shell
CG loop.

The single-`atoms` method builds a fresh ghost-format list at `nlh.rc` with
zero skin (the fitting / parity passes, where lists are not reused); the
3-argument method takes a pre-built list and is the production MD path
(`build_nlh_nlist` derives it from the Coulomb-cutoff parent).
"""
function calculate!(nlh::NLHParam, atoms::Atoms)
    return calculate!(nlh, atoms, neighborlist(atoms, nlh.rc, 0.0))
end

function calculate!(nlh::NLHParam, atoms::Atoms, nlist::NeighborList)
    backend = get_backend(atoms)
    N = length(atoms)
    sync!(nlist, atoms)
    g = nlist.ghosts
    g.n_owned == N ||
        error("NLH calculate!: list owned rows do not match the atom count")
    size(nlist.list, 1) == N ||
        error("NLH calculate!: list row count does not match the atom count")

    # An empty rank contributes zeros but must still ride the collective
    # inside add_reduced_ev! — same pattern as Spring / PME / Ewald. The
    # kernel launch and the reductions are not exercised on zero atoms.
    if N == 0
        add_reduced_ev!(atoms, zero(FPPREC), zeros(FPPREC, 6))
        return nothing
    end

    pe_atom = KernelAbstractions.zeros(backend, FPPREC, N)
    virials = KernelAbstractions.zeros(backend, FPPREC, N, 6)

    kernel_nlh!(backend, 512, N)(
        pe_atom, atoms.arrays[:forces], virials,
        g.xcart, g.elemid_ext, nlh.a, nlh.b, nlh.zz,
        nlh.phi_rc, nlh.dphi_rc, nlh.rc, nlist.list,
    )

    add_reduced_ev!(atoms, sum(pe_atom), vec(sum(virials, dims = 1)))
    if haskey(atoms.arrays, :peratom_virial)
        atoms.arrays[:peratom_virial] .+= virials
    end
    return nothing
end
