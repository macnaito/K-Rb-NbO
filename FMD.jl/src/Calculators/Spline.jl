using Statistics
using LinearAlgebra
using MPI
using IterativeSolvers
using LinearMaps
using Adapt
using KernelAbstractions
import AcceleratedKernels as AK
using Atomix: @atomic

export fitting!, SplineParam, calculate!

"""
    SplineParam{FP, T1, T2, T3}

2- + 3-body spline MLIP parameter container. Body-1 (single-particle
shift) `c1`, body-2 `c2`, body-3 `c3` are stored as raw coefficient
arrays so they can live on either CPU or GPU (`Adapt.@adapt_structure`
handles the migration).

# Fields

- `rmax2::FP` — 2-body cutoff (atomic units)
- `rmax3::FP` — 3-body cutoff (atomic units, typically smaller than `rmax2`)
- `c1::T1`   — `(ns,)` per-element constant shift
- `c2::T2`   — `(nb2, ns, ns)` 2-body B-spline coefficients per element pair
- `c3::T3`   — `(nb3, nb3, nb3, ns, ns, ns)` 3-body coefficients

`ns` = number of element species, `nb2` / `nb3` = number of cubic B-spline
basis functions per radial axis. The 3-body cost grows as `nb3³ · ns³`,
so keep `nb3` small (typical `nb3 = 4–6`).

Build with the constructor below, then use [`fitting!`](@ref) to optimise
the coefficients against forces / energies / virials, or [`calculate!`](@ref)
to evaluate the potential.
"""
struct SplineParam{FP, T1, T2, T3}
    rmax2::FP
    rmax3::FP
    c1::T1
    c2::T2
    c3::T3
end

Adapt.@adapt_structure SplineParam

"""
    SplineParam(rmax2, rmax3, ns, nb2, nb3) -> SplineParam

Allocate a fresh `SplineParam` with all coefficients zero. `ns` is the
number of element species (matches `length(params[:elements])`); `nb2`,
`nb3` are the basis counts per axis (each ≥ 4). The resulting object
evaluates to identically zero — call [`fitting!`](@ref) to populate it.
"""
function SplineParam(rmax2::FP, rmax3::FP, ns::Integer, nb2::Integer, nb3::Integer) where {FP}
    @assert rmax2 > zero(rmax2) && rmax3 > zero(rmax3)
    @assert nb2 ≥ 4 && nb3 ≥ 4
    c1 = zeros(FP, ns)
    c2 = zeros(FP, nb2, ns, ns)
    c3 = zeros(FP, nb3, nb3, nb3, ns, ns, ns)
    return SplineParam(rmax2, rmax3, c1, c2, c3)
end

"""
    vec(splp::SplineParam) -> Vector

Flatten `(c1, c2, c3)` into one vector — used internally by the CG fit
to communicate parameters across MPI ranks. Inverse of [`setval!`](@ref).
"""
function Base.vec(splp::SplineParam)
    parts = Vector{Vector{eltype(splp.c1)}}()
    for name in fieldnames(typeof(splp))
        x = getfield(splp, name)
        x isa AbstractArray || continue
        push!(parts, vec(x))
    end
    return isempty(parts) ? eltype(splp.c1)[] : vcat(parts...)
end

"""
    setval!(splp::SplineParam, val::AbstractVector) -> SplineParam
    setval!(splp::SplineParam, val) -> SplineParam

Inverse of [`Base.vec(::SplineParam)`](@ref): copy `val` into `c1`, `c2`,
`c3` (with shape preserved). The scalar form fills every coefficient with
the same value (used to zero out a parameter set quickly).
"""
function setval!(splp::SplineParam, val::AbstractVector)
    offset = 0
    for name in fieldnames(typeof(splp))
        x = getfield(splp, name)
        x isa AbstractArray || continue
        n = length(x)
        @views x .= reshape(val[(offset+1):(offset+n)], size(x))
        offset += n
    end
    @assert offset == length(val) "setval!: length(val)=$(length(val)) but consumed offset=$offset (array fields only)."
    return splp
end

function setval!(splp::SplineParam, val)
    for name in fieldnames(typeof(splp))
        x = getfield(splp, name)
        x isa AbstractArray || continue
        x .= val
    end
    return splp
end

# ---- B-spline basis utilities ------------------------------------- #
#
# Cubic (degree-3) cardinal B-spline. The basis B_n(s) is non-zero on
# s ∈ [n, n+4] and is reproduced via finite differences of (s-n-k)_+^3.
# A 5th-order polynomial taper smoothly drives the value to zero at r=rc.
# Names are prefixed with `_spl_` to avoid clashing with ACE's taper/dtaper
# defined in the same Calculators module.

@inline @fastmath function _spl_b(s, n)
    t0 = max(s - n,     zero(s))^3
    t1 = max(s - n - 1, zero(s))^3
    t2 = max(s - n - 2, zero(s))^3
    t3 = max(s - n - 3, zero(s))^3
    t4 = max(s - n - 4, zero(s))^3
    return (t0 - 4*t1 + 6*t2 - 4*t3 + t4) / 6
end

@inline @fastmath function _spl_db(s, n)
    t0 = max(s - n,     zero(s))^2
    t1 = max(s - n - 1, zero(s))^2
    t2 = max(s - n - 2, zero(s))^2
    t3 = max(s - n - 3, zero(s))^2
    t4 = max(s - n - 4, zero(s))^2
    return (t0 - 4*t1 + 6*t2 - 4*t3 + t4) / 2
end

# Inner radius (as a fraction of rc) where the taper starts kicking in.
# Outside [α·rc, rc] the taper is identically 1; inside [α·rc, rc] it
# smoothly drives spline value/derivatives to zero with C³ continuity at
# both endpoints. 3-body has a small cutoff (~3.5 Å) so we keep α high
# (thin taper band) to preserve expressivity in the active region.
const _SPL_TAPER_FRAC2 = 0.5    # 2-body: taper active in [rc/2, rc]
const _SPL_TAPER_FRAC3 = 0.9    # 3-body: taper active in [0.9·rc, rc]

@inline function _spl_taper(r, rc, r_inner)
    r ≤ r_inner && return one(r)
    r ≥ rc      && return zero(r)
    u = (r - r_inner) / (rc - r_inner)
    u4 = u*u*u*u; u5 = u4*u; u6 = u5*u; u7 = u6*u
    return 20*u7 - 70*u6 + 84*u5 - 35*u4 + 1
end

@inline function _spl_dtaper(r, rc, r_inner)
    (r ≤ r_inner || r ≥ rc) && return zero(r)
    u = (r - r_inner) / (rc - r_inner)
    u3 = u*u*u; u4 = u3*u; u5 = u4*u; u6 = u5*u
    return (140*u6 - 420*u5 + 420*u4 - 140*u3) / (rc - r_inner)
end

# ---- Forward kernels --------------------------------------------- #
#
# `forces`, `pe_atom`, `virials` are out-arguments accumulated in-place.
# Convention matches ACE.jl: virial is packed [xx, yy, zz, yz, zx, xy].

@kernel inbounds = true function _spl_kernel_2body!(
    forces, pe_atom, virials,
    pos, elemid, lattice,
    nlist, shiftx, shifty, shiftz,
    c2, rc, ::Val{NB},
) where {NB}
    I = @index(Global)
    si = elemid[I]
    x_i = pos[I,1]; y_i = pos[I,2]; z_i = pos[I,3]
    scale = (NB-4)/rc
    r_inner = rc * FPPREC(_SPL_TAPER_FRAC2)

    fx = zero(FPPREC); fy = zero(FPPREC); fz = zero(FPPREC)
    v1 = zero(FPPREC); v2 = zero(FPPREC); v3 = zero(FPPREC)
    v4 = zero(FPPREC); v5 = zero(FPPREC); v6 = zero(FPPREC)
    Ei = zero(FPPREC)

    for jlist in axes(nlist, 2)
        j = nlist[I, jlist]
        if j < 1
            continue
        end
        sftx = shiftx[I, jlist]; sfty = shifty[I, jlist]; sftz = shiftz[I, jlist]
        sx = x_i - pos[j,1] + sftx
        sy = y_i - pos[j,2] + sfty
        sz = z_i - pos[j,3] + sftz
        rx = lattice[1,1]*sx + lattice[2,1]*sy + lattice[3,1]*sz
        ry = lattice[1,2]*sx + lattice[2,2]*sy + lattice[3,2]*sz
        rz = lattice[1,3]*sx + lattice[2,3]*sy + lattice[3,3]*sz
        rij = min(sqrt(rx*rx + ry*ry + rz*rz), rc)

        sj = elemid[j]
        sa = min(si, sj); sb = max(si, sj)

        frac = rij/rc * (NB-4) + 4
        teij = floor(Int, frac); tsij = teij - 3
        Bij = (
            _spl_b(frac, tsij), _spl_b(frac, tsij+1),
            _spl_b(frac, tsij+2), _spl_b(frac, tsij+3),
        )
        DBij = (
            _spl_db(frac, tsij), _spl_db(frac, tsij+1),
            _spl_db(frac, tsij+2), _spl_db(frac, tsij+3),
        )
        tap  = _spl_taper(rij, rc, r_inner)
        dtap = _spl_dtaper(rij, rc, r_inner)
        # Bake taper and scale into bij/dbij (so inner loop is pure積和)
        bij = (tap*Bij[1], tap*Bij[2], tap*Bij[3], tap*Bij[4])
        dbij = (
            scale*tap*DBij[1] + dtap*Bij[1],
            scale*tap*DBij[2] + dtap*Bij[2],
            scale*tap*DBij[3] + dtap*Bij[3],
            scale*tap*DBij[4] + dtap*Bij[4],
        )

        Ei_local     = zero(FPPREC)
        dϕdr_r_local = zero(FPPREC)
        for t in 1:4
            cij = c2[tsij+t-1, sa, sb]
            Ei_local     += cij * bij[t]
            dϕdr_r_local += cij * dbij[t]
        end
        Ei += FPPREC(0.5) * Ei_local
        dϕdr_r_local /= rij

        fx -= dϕdr_r_local * rx
        fy -= dϕdr_r_local * ry
        fz -= dϕdr_r_local * rz
        v1 -= FPPREC(0.5)*dϕdr_r_local * rx * rx
        v2 -= FPPREC(0.5)*dϕdr_r_local * ry * ry
        v3 -= FPPREC(0.5)*dϕdr_r_local * rz * rz
        v4 -= FPPREC(0.5)*dϕdr_r_local * ry * rz
        v5 -= FPPREC(0.5)*dϕdr_r_local * rz * rx
        v6 -= FPPREC(0.5)*dϕdr_r_local * rx * ry
    end

    # Gather-only kernel: every write targets own row I, so the plain
    # (non-atomic) stores are intentional (contrast _spl_kernel_3body!).
    forces[I,1] += fx; forces[I,2] += fy; forces[I,3] += fz
    pe_atom[I] += Ei
    virials[I,1] += v1; virials[I,2] += v2; virials[I,3] += v3
    virials[I,4] += v4; virials[I,5] += v5; virials[I,6] += v6
end

@kernel inbounds = true function _spl_kernel_3body!(
    forces, pe_atom, virials,
    pos, elemid, lattice,
    nlist, shiftx, shifty, shiftz,
    c3, rc, ::Val{NB},
) where {NB}
    I = @index(Global)
    si = elemid[I]
    x_i = pos[I,1]; y_i = pos[I,2]; z_i = pos[I,3]
    scale_r = (NB-4)/rc
    scale_θ = (NB-4) * FPPREC(0.5)
    r_inner = rc * FPPREC(_SPL_TAPER_FRAC3)

    fx_i = zero(FPPREC); fy_i = zero(FPPREC); fz_i = zero(FPPREC)
    v1 = zero(FPPREC); v2 = zero(FPPREC); v3 = zero(FPPREC)
    v4 = zero(FPPREC); v5 = zero(FPPREC); v6 = zero(FPPREC)
    Ei = zero(FPPREC)

    for jlist in axes(nlist, 2)
        j = nlist[I, jlist]
        if j < 1
            break
        end
        sftx = shiftx[I, jlist]; sfty = shifty[I, jlist]; sftz = shiftz[I, jlist]
        sx = x_i - pos[j,1] + sftx
        sy = y_i - pos[j,2] + sfty
        sz = z_i - pos[j,3] + sftz
        rxij = lattice[1,1]*sx + lattice[2,1]*sy + lattice[3,1]*sz
        ryij = lattice[1,2]*sx + lattice[2,2]*sy + lattice[3,2]*sz
        rzij = lattice[1,3]*sx + lattice[2,3]*sy + lattice[3,3]*sz
        rij = min(sqrt(rxij*rxij + ryij*ryij + rzij*rzij), rc)
        sj = elemid[j]

        frac_ij = rij/rc * (NB-4) + 4
        teij = floor(Int, frac_ij); tsij = teij - 3
        Bij_raw = (
            _spl_b(frac_ij, tsij), _spl_b(frac_ij, tsij+1),
            _spl_b(frac_ij, tsij+2), _spl_b(frac_ij, tsij+3),
        )
        DBij_raw = (
            _spl_db(frac_ij, tsij), _spl_db(frac_ij, tsij+1),
            _spl_db(frac_ij, tsij+2), _spl_db(frac_ij, tsij+3),
        )
        tap_ij  = _spl_taper(rij, rc, r_inner)
        dtap_ij = _spl_dtaper(rij, rc, r_inner)
        bij = (
            tap_ij*Bij_raw[1], tap_ij*Bij_raw[2],
            tap_ij*Bij_raw[3], tap_ij*Bij_raw[4],
        )
        dbij = (
            scale_r*tap_ij*DBij_raw[1] + dtap_ij*Bij_raw[1],
            scale_r*tap_ij*DBij_raw[2] + dtap_ij*Bij_raw[2],
            scale_r*tap_ij*DBij_raw[3] + dtap_ij*Bij_raw[3],
            scale_r*tap_ij*DBij_raw[4] + dtap_ij*Bij_raw[4],
        )

        for klist in (jlist+1):size(nlist, 2)
            k = nlist[I, klist]
            if k < 1
                break
            end
            sftx2 = shiftx[I, klist]; sfty2 = shifty[I, klist]; sftz2 = shiftz[I, klist]
            sx2 = x_i - pos[k,1] + sftx2
            sy2 = y_i - pos[k,2] + sfty2
            sz2 = z_i - pos[k,3] + sftz2
            rxik = lattice[1,1]*sx2 + lattice[2,1]*sy2 + lattice[3,1]*sz2
            ryik = lattice[1,2]*sx2 + lattice[2,2]*sy2 + lattice[3,2]*sz2
            rzik = lattice[1,3]*sx2 + lattice[2,3]*sy2 + lattice[3,3]*sz2
            rik = min(sqrt(rxik*rxik + ryik*ryik + rzik*rzik), rc)
            sk = elemid[k]

            frac_ik = rik/rc * (NB-4) + 4
            teik = floor(Int, frac_ik); tsik = teik - 3
            Bik_raw = (
                _spl_b(frac_ik, tsik), _spl_b(frac_ik, tsik+1),
                _spl_b(frac_ik, tsik+2), _spl_b(frac_ik, tsik+3),
            )
            DBik_raw = (
                _spl_db(frac_ik, tsik), _spl_db(frac_ik, tsik+1),
                _spl_db(frac_ik, tsik+2), _spl_db(frac_ik, tsik+3),
            )
            tap_ik  = _spl_taper(rik, rc, r_inner)
            dtap_ik = _spl_dtaper(rik, rc, r_inner)
            bik = (
                tap_ik*Bik_raw[1], tap_ik*Bik_raw[2],
                tap_ik*Bik_raw[3], tap_ik*Bik_raw[4],
            )
            dbik = (
                scale_r*tap_ik*DBik_raw[1] + dtap_ik*Bik_raw[1],
                scale_r*tap_ik*DBik_raw[2] + dtap_ik*Bik_raw[2],
                scale_r*tap_ik*DBik_raw[3] + dtap_ik*Bik_raw[3],
                scale_r*tap_ik*DBik_raw[4] + dtap_ik*Bik_raw[4],
            )

            # cosθ via law of cosines (no rjk B-spline, no rjk cutoff)
            rxjk = rxik - rxij; ryjk = ryik - ryij; rzjk = rzik - rzij
            rjk2 = rxjk*rxjk + ryjk*ryjk + rzjk*rzjk

            cosθ = clamp((rij*rij + rik*rik - rjk2)/(2*rij*rik),
                         -one(FPPREC), one(FPPREC))
            inv_rij = 1/rij; inv_rik = 1/rik
            dcosθ_drij = inv_rik - cosθ*inv_rij
            dcosθ_drik = inv_rij - cosθ*inv_rik

            frac_θ = (cosθ + 1)*FPPREC(0.5)*(NB-4) + 4
            teθ = floor(Int, frac_θ); tsθ = teθ - 3
            tsθ = clamp(tsθ, 1, NB-3)
            bcosθ = (
                _spl_b(frac_θ, tsθ), _spl_b(frac_θ, tsθ+1),
                _spl_b(frac_θ, tsθ+2), _spl_b(frac_θ, tsθ+3),
            )
            dbcosθ_raw = (
                _spl_db(frac_θ, tsθ), _spl_db(frac_θ, tsθ+1),
                _spl_db(frac_θ, tsθ+2), _spl_db(frac_θ, tsθ+3),
            )
            dbcosθ = (
                scale_θ*dbcosθ_raw[1], scale_θ*dbcosθ_raw[2],
                scale_θ*dbcosθ_raw[3], scale_θ*dbcosθ_raw[4],
            )

            # No canonicalization: read c3 at raw (sj, sk). Inversion / (j,k)-swap
            # symmetry is delivered by `_spl_symmetrize_c3!` on the fitter side
            # (Issue #1). Requires `c3[a,b,θ,sj,sk,si] == c3[b,a,θ,sk,sj,si]`.
            a_E  = zero(FPPREC)
            a_b1 = zero(FPPREC)
            a_b2 = zero(FPPREC)
            a_θ  = zero(FPPREC)
            for t3 in 1:4, t2 in 1:4, t1 in 1:4
                c = c3[tsij+t1-1, tsik+t2-1, tsθ+t3-1, sj, sk, si]
                a_E  += c * bij[t1] * bik[t2] * bcosθ[t3]
                a_b1 += c * dbij[t1] * bik[t2] * bcosθ[t3]
                a_b2 += c * bij[t1] * dbik[t2] * bcosθ[t3]
                a_θ  += c * bij[t1] * bik[t2] * dbcosθ[t3]
            end
            Ei += a_E

            dEdrij_local = a_b1
            dEdrik_local = a_b2

            # Combined t-coefficients (tjk uses -dEdcosθ/(rij*rik), avoiding 1/rjk)
            tij = (dEdrij_local + a_θ * dcosθ_drij) * inv_rij
            tik = (dEdrik_local + a_θ * dcosθ_drik) * inv_rik
            tjk = -a_θ * inv_rij * inv_rik

            fij_x = tij*rxij; fij_y = tij*ryij; fij_z = tij*rzij
            fik_x = tik*rxik; fik_y = tik*ryik; fik_z = tik*rzik
            fjk_x = tjk*rxjk; fjk_y = tjk*ryjk; fjk_z = tjk*rzjk

            # Newton 3rd law direct writes
            fx_i -= fij_x + fik_x
            fy_i -= fij_y + fik_y
            fz_i -= fij_z + fik_z
            @atomic forces[j, 1] += fij_x - fjk_x
            @atomic forces[j, 2] += fij_y - fjk_y
            @atomic forces[j, 3] += fij_z - fjk_z
            @atomic forces[k, 1] += fik_x + fjk_x
            @atomic forces[k, 2] += fik_y + fjk_y
            @atomic forces[k, 3] += fik_z + fjk_z

            # Virial: all 3 bonds aggregated to atom-i bucket (no 0.5 — single visit)
            v1 -= fij_x*rxij + fik_x*rxik + fjk_x*rxjk
            v2 -= fij_y*ryij + fik_y*ryik + fjk_y*ryjk
            v3 -= fij_z*rzij + fik_z*rzik + fjk_z*rzjk
            v4 -= fij_y*rzij + fik_y*rzik + fjk_y*rzjk
            v5 -= fij_z*rxij + fik_z*rxik + fjk_z*rxjk
            v6 -= fij_x*ryij + fik_x*ryik + fjk_x*ryjk
        end
    end

    # @atomic: row I is also written by other work-items whose j/k equals
    # this I — a plain read-modify-write here races with those @atomic adds
    # and can lose updates (Issue #64). pe_atom/virials stay non-atomic:
    # only the owning work-item writes them — this relies on ndrange == N
    # with I unique per launch; a tiled/remapped launch would break it.
    @atomic forces[I,1] += fx_i
    @atomic forces[I,2] += fy_i
    @atomic forces[I,3] += fz_i
    pe_atom[I] += Ei
    virials[I,1] += v1
    virials[I,2] += v2
    virials[I,3] += v3
    virials[I,4] += v4
    virials[I,5] += v5
    virials[I,6] += v6
end

# ---- Public calculate! ------------------------------------------- #

"""
    calculate!(splp::SplineParam, atoms::Atoms)
    calculate!(splp::SplineParam, atoms::Atoms,
               nl2::NeighborList, nl3::NeighborList)

Add the 2- + 3-body spline contribution of `splp` to
`atoms.info[:potential_energy]`, `atoms.info[:virial]`, and
`atoms.arrays[:forces]`.

The single-`atoms` method builds fresh (ghost-format) neighbour lists at
`splp.rmax2` and `splp.rmax3` with zero skin — convenient for the fitting /
parity passes where neighbour lists are not reused. The 4-arg method takes
pre-built ghost-format lists and is the production NPT MD path
([`build_mlip_nlists`](@ref FMD.Apps.build_mlip_nlists) constructs them).
"""
function calculate!(splp::SplineParam, atoms::Atoms)
    nl2 = neighborlist(atoms, splp.rmax2, 0.0)
    nl3 = neighborlist(atoms, splp.rmax3, 0.0)
    return calculate!(splp, atoms, nl2, nl3)
end

# Ghost-format method (same geometry-stub reuse as ACE, see ACE.jl): the
# 2-body kernel is pure gather and writes owned rows directly; only the
# 3-body kernel scatters onto neighbours (j, k may be ghost rows), so its
# target is ghosts.fext, folded back afterwards. The per-atom virial is
# aggregated into the i bucket by both kernels, so it stays (N, 6).
function calculate!(splp::SplineParam, atoms::Atoms, nl2::NeighborList, nl3::NeighborList)
    backend = get_backend(atoms)
    N = length(atoms)
    sync!(nl2, atoms)
    nl3.ghosts === nl2.ghosts || sync!(nl3, atoms)
    g3 = nl3.ghosts
    nl2.ghosts.n_owned == N && g3.n_owned == N ||
        error("Spline ghost calculate!: list owned rows do not match the atom count")
    size(nl2.list, 1) == N && size(nl3.list, 1) == N ||
        error("Spline ghost calculate!: list row count does not match the atom count")

    pe_atom  = KernelAbstractions.zeros(backend, FPPREC, N)
    virials  = KernelAbstractions.zeros(backend, FPPREC, N, 6)

    e1 = AK.mapreduce(eid -> splp.c1[eid], +, atoms.arrays[:elemid];
                      init = zero(eltype(splp.c1)))

    NB2 = size(splp.c2, 1)
    _spl_kernel_2body!(backend, 256, N)(
        atoms.arrays[:forces], pe_atom, virials,
        nl2.ghosts.xcart, nl2.ghosts.elemid_ext, IdLattice(),
        nl2.list, ZeroShift(), ZeroShift(), ZeroShift(),
        splp.c2, splp.rmax2, Val(NB2),
    )

    NB3 = size(splp.c3, 1)
    fill!(g3.fext, zero(FPPREC))
    _spl_kernel_3body!(backend, 256, N)(
        g3.fext, pe_atom, virials,
        g3.xcart, g3.elemid_ext, IdLattice(),
        nl3.list, ZeroShift(), ZeroShift(), ZeroShift(),
        splp.c3, splp.rmax3, Val(NB3),
    )

    KernelAbstractions.synchronize(backend)
    fold_ghosts!(atoms.arrays[:forces], g3.fext, g3)

    add_reduced_ev!(atoms, e1 + sum(pe_atom), vec(sum(virials, dims = 1)))
    if haskey(atoms.arrays, :peratom_virial)
        atoms.arrays[:peratom_virial] .+= virials
    end
    return nothing
end

# ---- ACE-compatible apply / adjoint interface --------------------- #
#
# `applyA` evaluates (E, F, V) given a precomputed neighbor argument
# bundle from `makearg_spline`. `applyAT!` accumulates the adjoint of
# (u_E, u_F, u_V) into the parameter-shaped gradient `splp_grad`.
#
# Spline is linear in (c1, c2, c3), so the adjoint kernel mirrors the
# forward kernel: every `+= c * stuff` in `_spl_kernel_*body!` becomes
# `c_grad += u * stuff` here.

function applyA(splp::SplineParam, arg)
    backend = KernelAbstractions.get_backend(arg.g2.xcart)
    N = arg.Nat
    sync_ghosts!(arg.g2, arg.atoms)   # follow in-place position updates
    arg.g3 === arg.g2 || sync_ghosts!(arg.g3, arg.atoms)

    F = KernelAbstractions.zeros(backend, FPPREC, N, 3)
    V = zeros(FPPREC, 6)
    pe_atom = KernelAbstractions.zeros(backend, FPPREC, N)
    virials = KernelAbstractions.zeros(backend, FPPREC, N, 6)

    # 1-body term: sum over OWNED species only (the extended array would
    # double-count ghosts).
    e1 = AK.mapreduce(eid -> splp.c1[eid], +, arg.elemid;
                      init = zero(eltype(splp.c1)))

    NB2 = size(splp.c2, 1)
    _spl_kernel_2body!(backend, 256, N)(
        F, pe_atom, virials,
        arg.g2.xcart, arg.g2.elemid_ext, IdLattice(),
        arg.nlist2, ZeroShift(), ZeroShift(), ZeroShift(),
        splp.c2, splp.rmax2, Val(NB2),
    )
    NB3 = size(splp.c3, 1)
    fill!(arg.g3.fext, zero(FPPREC))
    _spl_kernel_3body!(backend, 256, N)(
        arg.g3.fext, pe_atom, virials,
        arg.g3.xcart, arg.g3.elemid_ext, IdLattice(),
        arg.nlist3, ZeroShift(), ZeroShift(), ZeroShift(),
        splp.c3, splp.rmax3, Val(NB3),
    )
    KernelAbstractions.synchronize(backend)
    fold_ghosts!(F, arg.g3.fext, arg.g3)

    E = e1 + sum(pe_atom)
    V .= vec(sum(virials, dims = 1))
    return E, F, V
end

# ---- Adjoint kernels --------------------------------------------- #

@kernel inbounds = true function _spl_scatter_2body!(
    c2_grad,
    u_E, u_F,
    uV1, uV2, uV3, uV4, uV5, uV6,
    pos, elemid, lattice,
    nlist, shiftx, shifty, shiftz,
    rc, ::Val{NB},
) where {NB}
    I = @index(Global)
    si = elemid[I]
    x_i = pos[I,1]; y_i = pos[I,2]; z_i = pos[I,3]
    scale = (NB-4)/rc
    r_inner = rc * FPPREC(_SPL_TAPER_FRAC2)
    ufx = u_F[I,1]; ufy = u_F[I,2]; ufz = u_F[I,3]

    for jlist in axes(nlist, 2)
        j = nlist[I, jlist]
        if j < 1
            continue
        end
        sftx = shiftx[I, jlist]; sfty = shifty[I, jlist]; sftz = shiftz[I, jlist]
        sx = x_i - pos[j,1] + sftx
        sy = y_i - pos[j,2] + sfty
        sz = z_i - pos[j,3] + sftz
        rx = lattice[1,1]*sx + lattice[2,1]*sy + lattice[3,1]*sz
        ry = lattice[1,2]*sx + lattice[2,2]*sy + lattice[3,2]*sz
        rz = lattice[1,3]*sx + lattice[2,3]*sy + lattice[3,3]*sz
        rij = min(sqrt(rx*rx + ry*ry + rz*rz), rc)

        sj = elemid[j]
        sa = min(si, sj); sb = max(si, sj)

        frac = rij/rc * (NB-4) + 4
        teij = floor(Int, frac); tsij = teij - 3
        Bij = (
            _spl_b(frac, tsij), _spl_b(frac, tsij+1),
            _spl_b(frac, tsij+2), _spl_b(frac, tsij+3),
        )
        DBij = (
            _spl_db(frac, tsij), _spl_db(frac, tsij+1),
            _spl_db(frac, tsij+2), _spl_db(frac, tsij+3),
        )
        tap  = _spl_taper(rij, rc, r_inner)
        dtap = _spl_dtaper(rij, rc, r_inner)
        bij = (tap*Bij[1], tap*Bij[2], tap*Bij[3], tap*Bij[4])
        dbij = (
            scale*tap*DBij[1] + dtap*Bij[1],
            scale*tap*DBij[2] + dtap*Bij[2],
            scale*tap*DBij[3] + dtap*Bij[3],
            scale*tap*DBij[4] + dtap*Bij[4],
        )

        # Hoist geometry-only sensitivity scalar (independent of t)
        inv_rij = 1/rij
        S_F = -(ufx*rx + ufy*ry + ufz*rz) * inv_rij
        S_V = -FPPREC(0.5) * inv_rij * (
            uV1*rx*rx + uV2*ry*ry + uV3*rz*rz +
            uV4*ry*rz + uV5*rz*rx + uV6*rx*ry
        )
        S_d = S_F + S_V    # multiplier of dbij[t]
        S_e = u_E * FPPREC(0.5)   # multiplier of bij[t]

        for t in 1:4
            adj = S_e * bij[t] + S_d * dbij[t]
            @atomic c2_grad[tsij+t-1, sa, sb] += adj
        end
    end
end

@kernel inbounds = true function _spl_scatter_3body!(
    c3_grad,
    u_E, u_F,
    uV1, uV2, uV3, uV4, uV5, uV6,
    pos, elemid, lattice,
    nlist, shiftx, shifty, shiftz,
    rc, ::Val{NB},
) where {NB}
    I = @index(Global)
    si = elemid[I]
    x_i = pos[I,1]; y_i = pos[I,2]; z_i = pos[I,3]
    scale_r = (NB-4)/rc
    scale_θ = (NB-4) * FPPREC(0.5)
    r_inner = rc * FPPREC(_SPL_TAPER_FRAC3)
    ufx_i = u_F[I,1]; ufy_i = u_F[I,2]; ufz_i = u_F[I,3]

    for jlist in axes(nlist, 2)
        j = nlist[I, jlist]
        if j < 1
            break
        end
        sftx = shiftx[I, jlist]; sfty = shifty[I, jlist]; sftz = shiftz[I, jlist]
        sx = x_i - pos[j,1] + sftx
        sy = y_i - pos[j,2] + sfty
        sz = z_i - pos[j,3] + sftz
        rxij = lattice[1,1]*sx + lattice[2,1]*sy + lattice[3,1]*sz
        ryij = lattice[1,2]*sx + lattice[2,2]*sy + lattice[3,2]*sz
        rzij = lattice[1,3]*sx + lattice[2,3]*sy + lattice[3,3]*sz
        rij = min(sqrt(rxij*rxij + ryij*ryij + rzij*rzij), rc)
        sj = elemid[j]
        ufx_j = u_F[j,1]; ufy_j = u_F[j,2]; ufz_j = u_F[j,3]

        frac_ij = rij/rc * (NB-4) + 4
        teij = floor(Int, frac_ij); tsij = teij - 3
        Bij_raw = (
            _spl_b(frac_ij, tsij), _spl_b(frac_ij, tsij+1),
            _spl_b(frac_ij, tsij+2), _spl_b(frac_ij, tsij+3),
        )
        DBij_raw = (
            _spl_db(frac_ij, tsij), _spl_db(frac_ij, tsij+1),
            _spl_db(frac_ij, tsij+2), _spl_db(frac_ij, tsij+3),
        )
        tap_ij  = _spl_taper(rij, rc, r_inner)
        dtap_ij = _spl_dtaper(rij, rc, r_inner)
        bij = (
            tap_ij*Bij_raw[1], tap_ij*Bij_raw[2],
            tap_ij*Bij_raw[3], tap_ij*Bij_raw[4],
        )
        dbij = (
            scale_r*tap_ij*DBij_raw[1] + dtap_ij*Bij_raw[1],
            scale_r*tap_ij*DBij_raw[2] + dtap_ij*Bij_raw[2],
            scale_r*tap_ij*DBij_raw[3] + dtap_ij*Bij_raw[3],
            scale_r*tap_ij*DBij_raw[4] + dtap_ij*Bij_raw[4],
        )

        for klist in (jlist+1):size(nlist, 2)
            k = nlist[I, klist]
            if k < 1
                break
            end
            sftx2 = shiftx[I, klist]; sfty2 = shifty[I, klist]; sftz2 = shiftz[I, klist]
            sx2 = x_i - pos[k,1] + sftx2
            sy2 = y_i - pos[k,2] + sfty2
            sz2 = z_i - pos[k,3] + sftz2
            rxik = lattice[1,1]*sx2 + lattice[2,1]*sy2 + lattice[3,1]*sz2
            ryik = lattice[1,2]*sx2 + lattice[2,2]*sy2 + lattice[3,2]*sz2
            rzik = lattice[1,3]*sx2 + lattice[2,3]*sy2 + lattice[3,3]*sz2
            rik = min(sqrt(rxik*rxik + ryik*ryik + rzik*rzik), rc)
            sk = elemid[k]
            ufx_k = u_F[k,1]; ufy_k = u_F[k,2]; ufz_k = u_F[k,3]

            frac_ik = rik/rc * (NB-4) + 4
            teik = floor(Int, frac_ik); tsik = teik - 3
            Bik_raw = (
                _spl_b(frac_ik, tsik), _spl_b(frac_ik, tsik+1),
                _spl_b(frac_ik, tsik+2), _spl_b(frac_ik, tsik+3),
            )
            DBik_raw = (
                _spl_db(frac_ik, tsik), _spl_db(frac_ik, tsik+1),
                _spl_db(frac_ik, tsik+2), _spl_db(frac_ik, tsik+3),
            )
            tap_ik  = _spl_taper(rik, rc, r_inner)
            dtap_ik = _spl_dtaper(rik, rc, r_inner)
            bik = (
                tap_ik*Bik_raw[1], tap_ik*Bik_raw[2],
                tap_ik*Bik_raw[3], tap_ik*Bik_raw[4],
            )
            dbik = (
                scale_r*tap_ik*DBik_raw[1] + dtap_ik*Bik_raw[1],
                scale_r*tap_ik*DBik_raw[2] + dtap_ik*Bik_raw[2],
                scale_r*tap_ik*DBik_raw[3] + dtap_ik*Bik_raw[3],
                scale_r*tap_ik*DBik_raw[4] + dtap_ik*Bik_raw[4],
            )

            rxjk = rxik - rxij; ryjk = ryik - ryij; rzjk = rzik - rzij
            rjk2 = rxjk*rxjk + ryjk*ryjk + rzjk*rzjk

            cosθ = clamp((rij*rij + rik*rik - rjk2)/(2*rij*rik),
                         -one(FPPREC), one(FPPREC))
            inv_rij = 1/rij; inv_rik = 1/rik
            dcosθ_drij = inv_rik - cosθ*inv_rij
            dcosθ_drik = inv_rij - cosθ*inv_rik

            frac_θ = (cosθ + 1)*FPPREC(0.5)*(NB-4) + 4
            teθ = floor(Int, frac_θ); tsθ = teθ - 3
            tsθ = clamp(tsθ, 1, NB-3)
            bcosθ = (
                _spl_b(frac_θ, tsθ), _spl_b(frac_θ, tsθ+1),
                _spl_b(frac_θ, tsθ+2), _spl_b(frac_θ, tsθ+3),
            )
            dbcosθ_raw = (
                _spl_db(frac_θ, tsθ), _spl_db(frac_θ, tsθ+1),
                _spl_db(frac_θ, tsθ+2), _spl_db(frac_θ, tsθ+3),
            )
            dbcosθ = (
                scale_θ*dbcosθ_raw[1], scale_θ*dbcosθ_raw[2],
                scale_θ*dbcosθ_raw[3], scale_θ*dbcosθ_raw[4],
            )

            # Sensitivity scalars: u·F + u·V projected onto each t-coefficient
            S_tij = (ufx_j - ufx_i)*rxij + (ufy_j - ufy_i)*ryij + (ufz_j - ufz_i)*rzij
            S_tik = (ufx_k - ufx_i)*rxik + (ufy_k - ufy_i)*ryik + (ufz_k - ufz_i)*rzik
            S_tjk = (ufx_k - ufx_j)*rxjk + (ufy_k - ufy_j)*ryjk + (ufz_k - ufz_j)*rzjk
            S_tij -= uV1*rxij*rxij + uV2*ryij*ryij + uV3*rzij*rzij +
                     uV4*ryij*rzij + uV5*rzij*rxij + uV6*rxij*ryij
            S_tik -= uV1*rxik*rxik + uV2*ryik*ryik + uV3*rzik*rzik +
                     uV4*ryik*rzik + uV5*rzik*rxik + uV6*rxik*ryik
            S_tjk -= uV1*rxjk*rxjk + uV2*ryjk*ryjk + uV3*rzjk*rzjk +
                     uV4*ryjk*rzjk + uV5*rzjk*rxjk + uV6*rxjk*ryjk

            # No canonicalization: deposit at raw (sj, sk). Symmetry of the
            # gradient under (j,k)-swap is enforced by `_spl_symmetrize_c3!`
            # on the fitter side after MPI Allreduce (Issue #1).
            G_b1 = S_tij * inv_rij
            G_b2 = S_tik * inv_rik
            G_θ  = S_tij*inv_rij*dcosθ_drij + S_tik*inv_rik*dcosθ_drik -
                   S_tjk*inv_rij*inv_rik

            for t3 in 1:4, t2 in 1:4, t1 in 1:4
                Pe = bij[t1] * bik[t2] * bcosθ[t3]
                Pi = dbij[t1] * bik[t2] * bcosθ[t3]
                Pk = bij[t1] * dbik[t2] * bcosθ[t3]
                Pθ = bij[t1] * bik[t2] * dbcosθ[t3]
                adj = u_E * Pe + G_b1 * Pi + G_b2 * Pk + G_θ * Pθ
                @atomic c3_grad[tsij+t1-1, tsik+t2-1, tsθ+t3-1, sj, sk, si] += adj
            end
        end
    end
end

function applyAT!(splp_grad::SplineParam, arg, u_E, u_F, u_V)
    backend = KernelAbstractions.get_backend(arg.g2.xcart)
    N = arg.Nat
    sync_ghosts!(arg.g2, arg.atoms)
    arg.g3 === arg.g2 || sync_ghosts!(arg.g3, arg.atoms)

    # 1-body: dE/dc1[s] = (number of atoms of species s) — OWNED only.
    elemid_host = arg.elemid isa AbstractArray ? Array(arg.elemid) : arg.elemid
    for s in 1:length(splp_grad.c1)
        splp_grad.c1[s] += u_E * count(==(s), elemid_host)
    end

    # 2-body: the adjoint reads u_F[I] (owned rows) only.
    NB2 = size(splp_grad.c2, 1)
    _spl_scatter_2body!(backend, 256, N)(
        splp_grad.c2,
        u_E, u_F,
        u_V[1], u_V[2], u_V[3], u_V[4], u_V[5], u_V[6],
        arg.g2.xcart, arg.g2.elemid_ext, IdLattice(),
        arg.nlist2, ZeroShift(), ZeroShift(), ZeroShift(),
        splp_grad.rmax2, Val(NB2),
    )

    # 3-body: reads u_F[j] / u_F[k] at extended rows — replicate owners.
    extend_ghosts!(arg.u_F_ext3, u_F, arg.g3)
    NB3 = size(splp_grad.c3, 1)
    _spl_scatter_3body!(backend, 256, N)(
        splp_grad.c3,
        u_E, arg.u_F_ext3,
        u_V[1], u_V[2], u_V[3], u_V[4], u_V[5], u_V[6],
        arg.g3.xcart, arg.g3.elemid_ext, IdLattice(),
        arg.nlist3, ZeroShift(), ZeroShift(), ZeroShift(),
        splp_grad.rmax3, Val(NB3),
    )

    KernelAbstractions.synchronize(backend)
    return splp_grad
end

function makearg_spline(atoms::Atoms, rmax2, rmax3)
    # Ghost selection frozen at build (fitting frames are static); extended
    # positions re-synced at every applyA/applyAT! call, so in-place
    # coordinate updates are followed.
    nl2 = neighborlist(atoms, rmax2, 0.0)
    nl3 = neighborlist(atoms, rmax3, 0.0)
    return (;
        Nat = length(atoms),
        atoms = atoms,
        elemid = atoms.arrays[:elemid],
        g2 = nl2.ghosts, nlist2 = nl2.list,
        g3 = nl3.ghosts, nlist3 = nl3.list,
        u_F_ext3 = KernelAbstractions.zeros(KernelAbstractions.get_backend(atoms), FPPREC, length(nl3.ghosts), 3),
        rmax2 = rmax2, rmax3 = rmax3,
    )
end


# ---- Fitting ----------------------------------------------------- #
#
# Mirrors FitACE's structure: weighted least-squares with a Hessian
# operator (LinearMap) solved by conjugate gradient. Spline is linear
# in (c1, c2, c3), so applyA / applyAT! together implement the same
# normal-equation solve, without ever materializing the design matrix.

# P-spline (Eilers–Marx) curvature penalty. For uniform knots the integral
# ∫(ϕ'')² dr is well approximated by Σ_n (c_{n+1} − 2c_n + c_{n−1})², so the
# Hessian gains λ · D^T D along each B-spline axis. Stencil is pentadiagonal:
#   k = 1     :  c₁ − 2c₂ + c₃
#   k = 2     : −2c₁ + 5c₂ − 4c₃ + c₄
#   3≤k≤N−2  :  c_{k−2} − 4c_{k−1} + 6c_k − 4c_{k+1} + c_{k+2}
#   k = N−1  :  c_{N−3} − 4c_{N−2} + 5c_{N−1} − 2c_N
#   k = N    :  c_{N−2} − 2c_{N−1} + c_N
# Symmetric posdef ⇒ CG keeps `issymmetric=true, isposdef=true`.
function _spl_add_curv_axis1!(g::AbstractArray, c::AbstractArray, λ)
    iszero(λ) && return g
    nb = size(c, 1)
    nb < 3 && return g
    other = length(c) ÷ nb
    g2 = reshape(g, nb, other)
    c2 = reshape(c, nb, other)
    @views @inbounds begin
        g2[1, :] .+= λ .* (c2[1, :] .- 2 .* c2[2, :] .+ c2[3, :])
        if nb ≥ 4
            g2[2, :] .+= λ .* (-2 .* c2[1, :] .+ 5 .* c2[2, :] .- 4 .* c2[3, :] .+ c2[4, :])
        end
        for k in 3:(nb-2)
            g2[k, :] .+= λ .* (c2[k-2, :] .- 4 .* c2[k-1, :] .+ 6 .* c2[k, :] .- 4 .* c2[k+1, :] .+ c2[k+2, :])
        end
        if nb ≥ 4
            g2[nb-1, :] .+= λ .* (c2[nb-3, :] .- 4 .* c2[nb-2, :] .+ 5 .* c2[nb-1, :] .- 2 .* c2[nb, :])
        end
        g2[nb, :] .+= λ .* (c2[nb-2, :] .- 2 .* c2[nb-1, :] .+ c2[nb, :])
    end
    return g
end

# 3-body curvature: split radial (axes 1,2 = rij, rik) and angular (axis 3 = cosθ).
# Permutations (2,1,3,4,5,6) and (3,2,1,4,5,6) are self-inverse so we reuse
# them for the permute-back step.
function _spl_add_curv_3body!(g3::AbstractArray, c3::AbstractArray, λ_r, λ_θ)
    _spl_add_curv_axis1!(g3, c3, λ_r)
    if !iszero(λ_r)
        cP = permutedims(c3, (2, 1, 3, 4, 5, 6))
        gP = zero(cP)
        _spl_add_curv_axis1!(gP, cP, λ_r)
        g3 .+= permutedims(gP, (2, 1, 3, 4, 5, 6))
    end
    if !iszero(λ_θ)
        cP = permutedims(c3, (3, 2, 1, 4, 5, 6))
        gP = zero(cP)
        _spl_add_curv_axis1!(gP, cP, λ_θ)
        g3 .+= permutedims(gP, (3, 2, 1, 4, 5, 6))
    end
    return g3
end

# Boundary pin: penalises ϕ(rc)² with the un-tapered B-spline value at the
# right endpoint, ϕ(rc) = (c[nb-3] + 4·c[nb-2] + c[nb-1]) / 6 (since the
# cubic basis evaluates to (1/6, 4/6, 1/6, 0) there). Pulls the spline value
# at rc toward zero so the taper does not have to swallow a steep remainder
# over its narrow [α·rc, rc] active band. Rank-1 PSD ⇒ CG-safe.
function _spl_add_pin_axis1!(g::AbstractArray, c::AbstractArray, λ)
    iszero(λ) && return g
    nb = size(c, 1)
    nb < 4 && return g
    other = length(c) ÷ nb
    g_flat = reshape(g, nb, other)
    c_flat = reshape(c, nb, other)
    @views @inbounds begin
        # ∂(λ·v²)/∂c[nb-3] = λv/3, ∂/∂c[nb-2] = 4λv/3, ∂/∂c[nb-1] = λv/3
        # with v = (c[nb-3] + 4c[nb-2] + c[nb-1]) / 6
        # ⇒ coef = λ·v/3 = λ·(c[nb-3] + 4c[nb-2] + c[nb-1]) / 18
        coef = (λ / 18) .* (c_flat[nb-3, :] .+ 4 .* c_flat[nb-2, :] .+ c_flat[nb-1, :])
        g_flat[nb-3, :] .+= coef
        g_flat[nb-2, :] .+= 4 .* coef
        g_flat[nb-1, :] .+= coef
        # g_flat[nb, :] unchanged: B(0) = 0 so c[nb] doesn't enter ϕ(rc)
    end
    return g
end

# 3-body boundary pin: apply along radial axes (rij and rik). The cosθ axis
# has no cutoff concept, so no pin there.
function _spl_add_pin_3body!(g3::AbstractArray, c3::AbstractArray, λ_r)
    _spl_add_pin_axis1!(g3, c3, λ_r)
    if !iszero(λ_r)
        cP = permutedims(c3, (2, 1, 3, 4, 5, 6))
        gP = zero(cP)
        _spl_add_pin_axis1!(gP, cP, λ_r)
        g3 .+= permutedims(gP, (2, 1, 3, 4, 5, 6))
    end
    return g3
end

"""
    _spl_symmetrize_c3!(c3) -> c3

Project `c3` onto the involution-symmetric subspace
`c3[a,b,θ,sa,sb,si] = c3[b,a,θ,sb,sa,si]` in place. This is the orthogonal
projection P = ½(I + σ) where σ swaps axes 1↔2 and species indices sa↔sb
together. Idempotent (P² = P) and self-adjoint (P^T = P).

Required by `_spl_kernel_3body!`, which reads `c3[..., sj, sk, si]`
without species canonicalization: under spatial inversion or any neighbor-
list reordering that swaps (j, k), the read picks the σ-image entry. If
`c3 ∈ im(P)` the two entries are equal and the potential is invariant —
see Issue #1.
"""
function _spl_symmetrize_c3!(c3::AbstractArray{<:Any,6})
    nb3 = size(c3, 1)
    nθ  = size(c3, 3)
    ns  = size(c3, 4)
    nsi = size(c3, 6)
    @assert size(c3, 2) == nb3 && size(c3, 5) == ns
    half = eltype(c3)(0.5)
    @inbounds for si in 1:nsi
        # Diagonal slices sa == sb: enforce (axis1 ↔ axis2) symmetry.
        for s in 1:ns, t3 in 1:nθ, b in 1:nb3, a in 1:(b-1)
            m = half * (c3[a, b, t3, s, s, si] + c3[b, a, t3, s, s, si])
            c3[a, b, t3, s, s, si] = m
            c3[b, a, t3, s, s, si] = m
        end
        # Off-diagonal pairs: average (sa,sb) with the σ-image (sb,sa)
        # under (a ↔ b). Skips sa == sb (handled above).
        for sb in 1:ns, sa in 1:(sb-1), t3 in 1:nθ, b in 1:nb3, a in 1:nb3
            m = half * (c3[a, b, t3, sa, sb, si] + c3[b, a, t3, sb, sa, si])
            c3[a, b, t3, sa, sb, si] = m
            c3[b, a, t3, sb, sa, si] = m
        end
    end
    return c3
end

"""
    _spl_c3_asymmetry(c3) -> Real

Maximum absolute violation of the involution symmetry
`c3[a,b,θ,sa,sb,si] == c3[b,a,θ,sb,sa,si]`, i.e. the distance of `c3`
from `im(P)` measured entrywise (see [`_spl_symmetrize_c3!`](@ref)).
Exactly zero for fitter-produced coefficients: the projection writes both
entries of every σ-orbit from the same value, `Hmul!` projects each
operator output and [`fitting!`](@ref) re-projects defensively before
returning, so the saved coefficients are bit-exactly symmetric — and a
JLD2 save/load round trip preserves the bits.

No `@inbounds`: this runs on untrusted (file-loaded) arrays, once per
job start — bounds checks stay on. Callers gate the shape (6-d, axes
1/2 and 4/5 equal) via `_check_splp_c3sym`.
"""
function _spl_c3_asymmetry(c3::AbstractArray{<:Any,6})
    nb3 = size(c3, 1)
    nθ  = size(c3, 3)
    ns  = size(c3, 4)
    nsi = size(c3, 6)
    size(c3, 2) == nb3 && size(c3, 5) == ns ||
        throw(ArgumentError("_spl_c3_asymmetry: axes (1,2) and (4,5) must " *
                            "match pairwise (got size $(size(c3)))"))
    dev = zero(real(float(eltype(c3))))
    for si in 1:nsi, sb in 1:ns, sa in 1:sb, t3 in 1:nθ, b in 1:nb3, a in 1:nb3
        (sa == sb && a >= b) && continue   # diagonal: visit each {a,b} pair once
        dev = max(dev, abs(c3[a, b, t3, sa, sb, si] - c3[b, a, t3, sb, sa, si]))
    end
    return dev
end

"""
    build_normal_system(splp::SplineParam, dataset::Vector{Atoms}, params)
        -> (; Hmul!, y, x0)

Build the MPI-distributed normal system that [`fitting!`](@ref) solves:
`Hmul!(q, v)` applies the projected, regularised normal operator
`P (AᵀA + R) P` to `v`, `y` is the projected RHS `P Aᵀ (E_ref, F_ref,
V_ref)`, and `x0 = vec(splp)` is the CG start vector. Frame partial sums
are sharded across ranks and Allreduced inside, so the returned system
is identical on every rank.

Single source of truth for the fit system — `fitting!` is exactly
`cg!` applied to this. Unexported; also called by the MPI tests to
assert 1-rank vs 2-rank parity of the partial sums at machine precision
without the CG solve on top (Issue #23: on an ill-conditioned system an
unconverged CG amplifies reduction-order roundoff, so solution parity
alone cannot isolate a partial-sum bug).

Collective over `MPI.COMM_WORLD`: every rank must call this — and apply
the returned `Hmul!` the same number of times — or the internal
`Allreduce!` deadlocks. Never guard a call with `if rank == 0`.

Mutates `splp`: projects `c3` onto the involution-symmetric subspace,
and each `Hmul!` application overwrites the coefficients with `v`.
`fitting!` hands `x0` to `cg!` as the in-place solution buffer, so after
a solve it holds the solution, not the start vector.
"""
function build_normal_system(splp::SplineParam, dataset::Vector{Atoms}, params)
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)
    mpisize = MPI.Comm_size(MPI.COMM_WORLD)
    dataset = deepcopy(dataset)

    ωE, ωF, ωS = 1.0, params[:weight_force], params[:weight_stress]
    ∑ω = ωE + ωF + ωS
    ωE, ωF, ωS = ωE/∑ω, ωF/∑ω, ωS/∑ω

    arr = [atoms.info[:potential_energy][1]/length(atoms) for atoms in dataset]
    σE = isnan(std(arr)) ? 1.0 : std(arr)*sqrt(length(arr))
    ωE /= σE^2

    arr = reduce(vcat, [vec(atoms.arrays[:forces]) for atoms in dataset])
    σF = std(arr)*sqrt(length(arr))
    ωF /= σF^2

    arr = reduce(vcat, [vec(atoms.info[:virial])/length(atoms) for atoms in dataset])
    σS = std(arr)*sqrt(length(arr))
    ωS /= σS^2

    dataset = dataset[(mpirank+1):mpisize:length(dataset)]

    # ---- baseline subtraction (core-shell [+ NLH]) -------------------
    subtract_baseline!(dataset, params)

    dataset_args = [makearg_spline(atoms, splp.rmax2, splp.rmax3) for atoms in dataset]

    ufi_grad = deepcopy(splp)
    # Issue #1: project the initial coefficients onto the involution-symmetric
    # subspace so CG iterates start (and therefore stay) in im(P).
    _spl_symmetrize_c3!(splp.c3)
    x0 = vec(splp)
    y = zeros(length(x0))

    function Hmul!(q, v)
        setval!(splp, v)
        setval!(ufi_grad, zero(eltype(v)))

        for idata in eachindex(dataset)
            Nat = length(dataset[idata])
            mname = dataset[idata].info[:method][1]
            w = haskey(params[:method_weight], mname) ? params[:method_weight][mname] : 1.0
            E, F, V = applyA(splp, dataset_args[idata])
            applyAT!(ufi_grad, dataset_args[idata], w*ωE*E/Nat^2, w*ωF*F, w*ωS*V/Nat^2)
        end

        if mpirank == 0
            # magnitude L2 (Tikhonov): keeps CG well-conditioned, pins boundary coeffs
            for key in keys(params[:rigid])
                getfield(ufi_grad, key) .+= params[:rigid][key] .* getfield(splp, key)
            end
            # curvature (P-spline): smooths ϕ shape without bounding magnitude
            if haskey(params, :rigid_curv)
                rc = params[:rigid_curv]
                _spl_add_curv_axis1!(ufi_grad.c2, splp.c2,
                                     get(rc, :c2, zero(eltype(splp.c2))))
                _spl_add_curv_3body!(ufi_grad.c3, splp.c3,
                                     get(rc, :c3_r, zero(eltype(splp.c3))),
                                     get(rc, :c3_θ, zero(eltype(splp.c3))))
                # boundary pin: ϕ(rc) → 0 so the taper has less work in [α·rc, rc]
                _spl_add_pin_axis1!(ufi_grad.c2, splp.c2,
                                    get(rc, :c2_pin, zero(eltype(splp.c2))))
                _spl_add_pin_3body!(ufi_grad.c3, splp.c3,
                                    get(rc, :c3_r_pin, zero(eltype(splp.c3))))
            end
        end

        q .= vec(ufi_grad)
        MPI.Allreduce!(q, MPI.SUM, MPI.COMM_WORLD)
        # Issue #1: each rank's local A^T A v carries asymmetric components
        # from (j,k) iteration order. Project the global gradient once so
        # CG sees an operator that maps im(P) → im(P).
        setval!(ufi_grad, q)
        _spl_symmetrize_c3!(ufi_grad.c3)
        q .= vec(ufi_grad)
        return q
    end

    # ---- RHS: A^T (E_ref, F_ref, V_ref) -----------------------------
    setval!(ufi_grad, zero(eltype(x0)))
    for idata in eachindex(dataset)
        Nat = length(dataset[idata])
        mname = dataset[idata].info[:method][1]
        w = haskey(params[:method_weight], mname) ? params[:method_weight][mname] : 1.0
        applyAT!(ufi_grad, dataset_args[idata],
                 w*ωE*(dataset[idata].info[:potential_energy][1])/Nat^2,
                 w*ωF*(dataset[idata].arrays[:forces]),
                 w*ωS*(dataset[idata].info[:virial])/Nat^2)
    end
    y .= vec(ufi_grad)
    MPI.Allreduce!(y, MPI.SUM, MPI.COMM_WORLD)
    # Issue #1: project the RHS so CG residual r = Py − (PAᵀAP)x stays in im(P).
    setval!(ufi_grad, y)
    _spl_symmetrize_c3!(ufi_grad.c3)
    y .= vec(ufi_grad)

    return (; Hmul!, y, x0)
end

"""
    fitting!(splp::SplineParam, dataset::Vector{Atoms}, params) -> SplineParam

Run the CG Δ-learning fit. For each frame in `dataset`:

1. Subtract the core-shell + Coulomb + ExternalField + Spring contribution
   (computed by [`CoreShell`](@ref) using the shell parameters in `params`)
   from the reference forces / energy / virial.
2. Build a linear system that maps `vec(splp)` to the residual.

Then solve the resulting (regularised) normal equations across MPI ranks
and write the optimised coefficients back into `splp` (mutated in place
and returned).

Required `params`: `:nsteps` (CG max iterations), `:weight_force`,
`:weight_stress`, `:method_weight` (per-method scalar weights),
`:rigid` (magnitude L2 / Tikhonov regularisers `(c1, c2, c3)` — keeps the
CG well-conditioned and pins boundary basis coefficients).

Optional `params[:rigid_curv]`: P-spline curvature penalty along the
B-spline axes — penalises ∫(ϕ'')² dr without bounding coefficient
magnitude. Keys: `:c2` (2-body radial), `:c3_r` (3-body radial; applied
to both rij and rik axes), `:c3_θ` (3-body angular, cosθ). Missing keys
default to 0. Use this when short-range repulsion needs large coefficients
(magnitude penalty hurts) but long-range / sparse-data regions still need
smoothing.

Also accepts boundary-pin keys `:c2_pin` and `:c3_r_pin`, which add
`λ_pin · ϕ_spline(rc)²` along each radial axis. This pulls the un-tapered
spline value at the cutoff toward zero so the taper (active only in
[α·rc, rc]) does not have to swallow a steep remainder. cosθ has no
cutoff, so no `:c3_θ_pin`.
"""
function fitting!(splp::SplineParam, dataset::Vector{Atoms}, params)
    mpirank = MPI.Comm_rank(MPI.COMM_WORLD)

    sys = build_normal_system(splp, dataset, params)
    x = sys.x0

    # ---- Conjugate gradient -----------------------------------------
    # `issymmetric` and `isposdef` describe the operator restricted to im(P)
    # (the involution-symmetric subspace), where it is `P A^T A P + P R P`.
    # CG starts at `x_0 = vec(splp)` and `b = Py`, both in im(P), so all
    # iterates stay there up to roundoff (cleaned up post-CG).
    Aop = LinearMap(sys.Hmul!, length(x); ismutating = true, issymmetric = true, isposdef = true)
    cg!(x, Aop, sys.y; reltol = 1e-12, maxiter = params[:nsteps], verbose = (mpirank == 0))

    setval!(splp, x)
    # Issue #1: defensive re-projection in case floating-point rounding
    # leaked any asymmetric component during CG.
    _spl_symmetrize_c3!(splp.c3)

    if mpirank == 0
        @show mean(abs, splp.c1) mean(abs, splp.c2) mean(abs, splp.c3)
    end

    return splp
end
