# ============================================================================
# Analytic Hessian-vector products (HVP) for the core-shell model
# (Issue #147; derivation: docs/src/theory/coreshell-analytic-hvp.md).
#
# Variable convention (pinned by the force kernels): the independent
# variables are the core and shell POSITIONS (x_c, x_s = x_c + shelldisp).
# `forces` is -∂E/∂x_c at fixed x_s and `shellforces` is -∂E/∂x_s at fixed
# x_c — the Spring reaction written into `forces` (Spring.jl) pins this
# convention: U_spring depends on x_s - x_c alone, yet contributes to both
# arrays with opposite signs. For a direction v in shell-displacement
# space, hvp! evaluates in one pass
#
#   Jv = ∂F_s/∂u · v = -(∂²E/∂x_s∂x_s)·v     (symmetric block)
#   Kv = ∂F_c/∂u · v = -(∂²E/∂x_c∂x_s)·v     (mixed block)
#
# writing nothing into the atoms arrays (pure). Per-term blocks:
#   Spring / intra    closed form per atom, Kv term = -Jv term
#   short-range       pair Hessian B(r) = (ϕ″ - ϕ_eff′/r)·r̂⊗r̂ + (ϕ_eff′/r)·I
#                     Jv: -B_sc·v_i  and  -B_ss·(v_i - v_j);  Kv: +B_cs·v_j
#   reciprocal        via _hvp_coulomb_cache / hvp_coulomb! specialized on
#                     the CoulombInfo subtype
#
# This file is included LAST in Calculators.jl on purpose: EwaldInfo /
# PMEInfo are not visible from CoreShell.jl (include order), so every
# HVP piece that touches them lives here.
# ============================================================================

using KernelAbstractions
import AcceleratedKernels as AK
using LinearAlgebra
using LoopVectorization
using StaticArrays
using Atomix: @atomic
using SpecialFunctions: erf
using ..FMD: Atoms, FPPREC
using ..FMD.Cores

export hvp_workspace, hvp!

"""
    CoreShellHvpWorkspace

Frozen-snapshot workspace for [`hvp!`](@ref): positions, shell
displacements, and the cell are read once at construction
([`hvp_workspace`](@ref)) and the reciprocal-space cache is built for that
snapshot. Apply `hvp!` as many times as needed with different directions
`v` — but if `pos`, `shelldisp`, or `lattice` change, the workspace is
stale and must be rebuilt; `hvp!` enforces this with a snapshot
comparison and errors instead of answering wrongly (a refresh path is
deliberately deferred to the MD integration, Issue #117 P4). The
reciprocal caches carry mutable scratch, so a single workspace must not
run `hvp!` concurrently.
"""
struct CoreShellHvpWorkspace{TC,TCache,TSurf,TK<:AbstractVector,
                             TE<:AbstractMatrix,TS<:AbstractMatrix}
    coulomb::TC
    nl::NeighborList
    cache::TCache
    surf::TSurf     # SurfaceHvpCache, or nothing without :surface_term
    kvals::TK       # Jacobi diagonal for the preconditioned CG: k2 (guarded)
    vext::TE        # (N+G, 3) ghost-extended direction scratch
    Jscratch::TS    # (N, 3) J output for calls that only want Kv
    Wscratch::TS    # (N, 6) virial sink for calls that only want Jv/Kv
    # Frozen-snapshot guard: hvp! refuses to run if any of these moved
    # since construction (a stale reciprocal cache would be silently
    # wrong otherwise — "stop rather than answer wrongly").
    pos_snap::TS
    sdisp_snap::TS
    lat_snap::Matrix{FPPREC}
end

# Specialized in this file for EwaldInfo / PMEInfo; anything else is an
# unsupported Coulomb solver for the analytic HVP.
_hvp_coulomb_cache(coulomb, atoms::Atoms, nl::NeighborList, prms::NamedTuple) =
    error("analytic HVP: no reciprocal-space cache for $(typeof(coulomb))")

"""
    hvp_workspace(atoms, nl, coulomb, prms) -> CoreShellHvpWorkspace

Build the frozen-snapshot workspace for [`hvp!`](@ref) (syncs the ghost
arrays of `nl` once). `coulomb` is an `EwaldInfo` or `PMEInfo` whose
`cellupdate!` matches the current lattice; pass `nothing` to restrict the
HVP to the real-space + spring + intra terms (test harness use).

Domain decomposition is supported through the `PMEInfo` path only
(Issue #149): the reciprocal grid rides the same Allreduce as the force
path. `EwaldInfo` under DD errors — its structure factor has no reduction
wired. Construction and every later `hvp!` are collective on all ranks,
empty ones included.

`prms[:surface_term]` is frozen into the workspace here (its `δM` and
`1/V` belong to this snapshot, like the reciprocal cache);
`prms[:polarization_umbrella]` is still refused — its bias has no
analytic HVP.
"""
function hvp_workspace(atoms::Atoms, nl::NeighborList, coulomb, prms::NamedTuple)
    haskey(prms, :polarization_umbrella) &&
        error("hvp_workspace: the polarization umbrella has no analytic HVP yet " *
              "(Issue #117 P4 defines it consistently with the shadow functional)")
    backend = get_backend(atoms)
    N = length(atoms)
    sync!(nl, atoms)

    kvals = KernelAbstractions.zeros(backend, FPPREC, N)
    k2tbl = prms[:k2]
    # k2 = 0 marks shell-less species: unit diagonal keeps the Jacobi
    # preconditioner well-defined (those rows carry no shell physics).
    N > 0 && AK.map!(e -> begin
                         k = k2tbl[e]
                         ifelse(k > zero(k), k, one(k))
                     end, kvals, atoms.arrays[:elemid])

    vext = KernelAbstractions.zeros(backend, FPPREC, size(nl.ghosts.xcart, 1), 3)
    Jscratch = KernelAbstractions.zeros(backend, FPPREC, N, 3)
    Wscratch = KernelAbstractions.zeros(backend, FPPREC, N, 6)
    cache = coulomb === nothing ? nothing :
            _hvp_coulomb_cache(coulomb, atoms, nl, prms)
    # Collective (the dipole is Allreduced), like the reciprocal cache.
    surf = haskey(prms, :surface_term) ?
           surface_hvp_cache(prms[:surface_term], atoms, prms) : nothing
    return CoreShellHvpWorkspace(coulomb, nl, cache, surf, kvals, vext, Jscratch,
                                 Wscratch,
                                 copy(atoms.arrays[:pos]),
                                 copy(atoms.arrays[:shelldisp]),
                                 Matrix{FPPREC}(Array(atoms.info[:lattice])))
end

"""
    hvp!(Jv, Kv, Wv, v, atoms, ws, prms) -> Wrec

Analytic Hessian-vector product at the workspace snapshot: overwrites
`Jv` with `(∂F_s/∂u)·v` and, unless `Kv === nothing`, `Kv` with
`(∂F_c/∂u)·v`. Reads the atoms arrays only — no force/energy state is
touched. `v` is an owned-row `(N, 3)` matrix; ghost replication happens
internally.

`Wv` is an owned-row `(N, 6)` matrix, overwritten with the per-atom
directional derivative of the Voigt virial along `v` — the strain
counterpart of `Jv`/`Kv` (Issue #151, `docs/src/theory/xlbomd-shell.md`
§4.1), packed and signed exactly like `atoms.arrays[:peratom_virial]`.
It is **not** optional: a barostat driven by a stale virial is silently
wrong, so every call site has to see the term.

The reciprocal-space part has no per-atom attribution — the same
situation as the force path, where `_pme_solve`'s grid virial goes to
`atoms.info[:virial]` only — so it is the RETURN value instead: a host
Voigt-6 vector, already whole-system (the grid is replicated) and
therefore **not** to be summed again under domain decomposition. The
whole-system virial derivative is
`allreduce_sum(sum(Wv, dims = 1)) + Wrec`.

With `prms[:surface_term]` the depolarization block rides both outputs
(Issue #181): its rank-3 contribution to `Jv`/`Kv` is per-atom, its
virial derivative joins `Wrec` under the same replicated regime — see
[`surface_hvp!`](@ref). `prms` must carry the same `SurfaceTerm` the
workspace was built with; anything else is an error rather than a
silently incomplete product.
"""
function hvp!(Jv::AbstractMatrix, Kv::Union{Nothing,AbstractMatrix},
              Wv::AbstractMatrix, v::AbstractMatrix, atoms::Atoms,
              ws::CoreShellHvpWorkspace, prms::NamedTuple)
    N = length(atoms)
    fill!(Jv, zero(FPPREC))
    Kv === nothing || fill!(Kv, zero(FPPREC))
    fill!(Wv, zero(FPPREC))

    # Frozen-snapshot guard (see CoreShellHvpWorkspace): a stale cache
    # answers wrongly and silently, so refuse loudly instead. The
    # comparison is rank-local but its inputs are too, so under DD every
    # rank reaches the same verdict (an empty rank compares empty arrays
    # and passes trivially).
    (atoms.arrays[:pos] == ws.pos_snap &&
     atoms.arrays[:shelldisp] == ws.sdisp_snap &&
     Array(atoms.info[:lattice]) == ws.lat_snap) ||
        error("hvp!: pos/shelldisp/lattice changed since hvp_workspace was " *
              "built — the workspace is a frozen snapshot; rebuild it")

    # The workspace freezes which terms the product contains; a `prms` that
    # disagrees would silently drop (or double) the surface term, so refuse.
    _hvp_check_surface(ws, prms)

    # No early `N == 0` return: under DD the ghost replication is staged
    # point-to-point communication, the reciprocal cache rides a grid
    # Allreduce and the surface block its own, so an empty rank must still
    # walk all three (Issue #149, #181).
    Cores.extend_ghosts!(ws.vext, v, ws.nl.ghosts)
    # Function barrier: the Kv === nothing branch is decided here once, so
    # the kernel launches below are statically typed (the CG inner loop
    # calls this every iteration).
    N > 0 && _hvp_terms!(Jv, Kv, Wv, v, atoms, ws, prms)
    Wrec = ws.cache === nothing ? zeros(FPPREC, 6) :
                                  hvp_coulomb!(Jv, Kv, v, atoms, ws.cache, prms)
    # Surface term (Issue #181): rank-3 closed form, one 3-vector
    # Allreduce, and a virial derivative that — like the reciprocal one —
    # is whole-system and rides the return value, not `Wv`. Called
    # unconditionally on purpose: an empty rank must reach the Allreduce.
    ws.surf === nothing ||
        (Wrec .+= surface_hvp!(Jv, Kv, v, atoms, ws.surf))
    return Wrec
end

# Rank-local comparison of rank-local inputs, so every rank reaches the
# same verdict under DD (an identity check, not a value check: the cache
# is a frozen snapshot of THAT SurfaceTerm's gauge and tensors).
function _hvp_check_surface(ws::CoreShellHvpWorkspace, prms::NamedTuple)
    have = haskey(prms, :surface_term)
    if ws.surf === nothing
        have && error("hvp!: prms carries :surface_term but the workspace " *
                      "was built without it — rebuild hvp_workspace with " *
                      "the same prms (the product would silently omit the " *
                      "depolarization block)")
    elseif !have
        error("hvp!: the workspace was built with :surface_term but prms " *
              "no longer carries it — rebuild hvp_workspace with the same " *
              "prms")
    elseif prms[:surface_term] !== ws.surf.st
        error("hvp!: prms[:surface_term] is a different SurfaceTerm than " *
              "the workspace was built with — rebuild hvp_workspace")
    end
    return nothing
end

function _hvp_terms!(Jv, Kv::AbstractMatrix, Wv, v, atoms::Atoms,
                     ws::CoreShellHvpWorkspace, prms::NamedTuple)
    backend = get_backend(atoms)
    N = length(atoms)
    g = ws.nl.ghosts
    kernel_gcb_hvp!(backend, 512, N)(Jv, Kv, Wv, Val(true), ws.vext,
        g.xcart, g.sdisp_ext, g.elemid_ext,
        prms[:a], prms[:b], prms[:rho], prms[:c], prms[:e],
        prms[:qc], prms[:qs], prms[:eta_c], prms[:eta_s],
        prms[:alpha], prms[:rc], ws.nl.list)
    kernel_hvp_spring_intra!(backend, 512, N)(Jv, Kv, Wv, Val(true), v,
        atoms.arrays[:shelldisp], atoms.arrays[:elemid],
        prms[:k2], prms[:k4], prms[:qc], prms[:qs], prms[:alpha])
    return nothing
end

function _hvp_terms!(Jv, Kv::Nothing, Wv, v, atoms::Atoms,
                     ws::CoreShellHvpWorkspace, prms::NamedTuple)
    backend = get_backend(atoms)
    N = length(atoms)
    g = ws.nl.ghosts
    # Jv rides the unused Kv slot; never written under Val(false).
    kernel_gcb_hvp!(backend, 512, N)(Jv, Jv, Wv, Val(false), ws.vext,
        g.xcart, g.sdisp_ext, g.elemid_ext,
        prms[:a], prms[:b], prms[:rho], prms[:c], prms[:e],
        prms[:qc], prms[:qs], prms[:eta_c], prms[:eta_s],
        prms[:alpha], prms[:rc], ws.nl.list)
    kernel_hvp_spring_intra!(backend, 512, N)(Jv, Jv, Wv, Val(false), v,
        atoms.arrays[:shelldisp], atoms.arrays[:elemid],
        prms[:k2], prms[:k4], prms[:qc], prms[:qs], prms[:alpha])
    return nothing
end

# ---------------------------------------------------------------------------
# Per-atom closed forms: Spring + intra-atom reciprocal correction.
#
# Spring (Spring.jl): F_s = -(k2 + k4|u|²/6)·u, so
#   ∂F_s/∂u·v = -(k2 + k4|u|²/6)·v - (k4/3)(u·v)·u,     Kv term = -Jv term.
#
# Intra (kernel_subtractintraatom!, Ewald.jl): F_s = factor·u with the
# same two branches as the force kernel; both are differentiated exactly
# as implemented. Non-Taylor branch: factor = w·g′(d)/d with
# g(d) = erf(αd)/d, w = qc·qs, and
#   g″(d) = -2α/√π·e^{-(αd)²}·(2α² + 2/d²) + 2·erf(αd)/d³
# (the library erf is exact, so its analytic derivative is too), giving
#   ∂F_s/∂u·v = factor·v + (w·g″(d) - factor)·(u·v)/d²·u.
# Taylor branch (d < 0.1/α): factor = w·(-4α³/(3√π) + 4α⁵/(5√π)·d²), so
#   ∂F_s/∂u·v = factor·v + w·(8α⁵/(5√π))·(u·v)·u.
#
# Virial derivative (Issue #151). Both force kernels write their virial as
# a ±(scalar)·r⊗r with r = -u, so with δ = D_v r = -v,
#   D_v[c·r⊗r] = (D_v c)·u⊗u + c·(v⊗u + u⊗v)
# and `D_v c` is the SAME scalar the force derivative above already needs:
# (k4/3)(u·v) for the spring, `cuu·(u·v)` for the intra branch (that is
# exactly how `cuu` was defined). Written with the sign of the kernel it
# differentiates — `virials -= …` in kernel_spring!, `virials += …` in
# kernel_subtractintraatom!.
# ---------------------------------------------------------------------------
@kernel inbounds = true function kernel_hvp_spring_intra!(Jv, Kv, vir, ::Val{WITHK},
    @Const(v), @Const(shelldisp), @Const(elemid),
    @Const(k2), @Const(k4), @Const(qc), @Const(qs), @Const(alpha)) where {WITHK}

    I = @index(Global)
    si = elemid[I]
    ux, uy, uz = shelldisp[I,1], shelldisp[I,2], shelldisp[I,3]
    vx, vy, vz = v[I,1], v[I,2], v[I,3]
    d2 = ux*ux + uy*uy + uz*uz
    uv = ux*vx + uy*vy + uz*vz

    # Spring
    k2i = k2[si]
    k4i = k4[si]
    c0 = k2i + k4i*d2/6
    tx = -(c0*vx + k4i/3*uv*ux)
    ty = -(c0*vy + k4i/3*uv*uy)
    tz = -(c0*vz + k4i/3*uv*uz)

    dc = k4i/3*uv                       # D_v of the spring's ϕ′/r
    w1 = -(dc*ux*ux + c0*2*vx*ux)
    w2 = -(dc*uy*uy + c0*2*vy*uy)
    w3 = -(dc*uz*uz + c0*2*vz*uz)
    w4 = -(dc*uy*uz + c0*(vy*uz + uy*vz))
    w5 = -(dc*uz*ux + c0*(vz*ux + uz*vx))
    w6 = -(dc*ux*uy + c0*(vx*uy + ux*vy))

    # Intra — branch on the same expression as the force kernel
    # (kernel_subtractintraatom!: dr < 0.1/α) so the two never disagree
    # in the rounding band at the threshold. The Jv/Kv lines are left
    # exactly where they were: reordering them would move the shadow
    # forces in the last bits for no reason.
    w = qc[si]*qs[si]
    tsp = 2/sqrt(FPPREC(pi))
    d = sqrt(d2)
    if d < FPPREC(1e-1)/alpha
        factor = w * (-4*alpha^3/(3*sqrt(FPPREC(pi))) + 4*alpha^5/(5*sqrt(FPPREC(pi)))*d2)
        cuu = w * 8*alpha^5/(5*sqrt(FPPREC(pi)))
        tx += factor*vx + cuu*uv*ux
        ty += factor*vy + cuu*uv*uy
        tz += factor*vz + cuu*uv*uz
        dcf = cuu*uv
        w1 += dcf*ux*ux + factor*2*vx*ux
        w2 += dcf*uy*uy + factor*2*vy*uy
        w3 += dcf*uz*uz + factor*2*vz*uz
        w4 += dcf*uy*uz + factor*(vy*uz + uy*vz)
        w5 += dcf*uz*ux + factor*(vz*ux + uz*vx)
        w6 += dcf*ux*uy + factor*(vx*uy + ux*vy)
    else
        E = exp(-(d*alpha)^2)
        factor = w * (-erf(d*alpha)/d + tsp*alpha*E) / d2
        gpp = -tsp*alpha*E*(2*alpha^2 + 2/d2) + 2*erf(d*alpha)/(d2*d)
        cuu = (w*gpp - factor) / d2
        tx += factor*vx + cuu*uv*ux
        ty += factor*vy + cuu*uv*uy
        tz += factor*vz + cuu*uv*uz
        dcf = cuu*uv
        w1 += dcf*ux*ux + factor*2*vx*ux
        w2 += dcf*uy*uy + factor*2*vy*uy
        w3 += dcf*uz*uz + factor*2*vz*uz
        w4 += dcf*uy*uz + factor*(vy*uz + uy*vz)
        w5 += dcf*uz*ux + factor*(vz*ux + uz*vx)
        w6 += dcf*ux*uy + factor*(vx*uy + ux*vy)
    end

    Jv[I,1] += tx
    Jv[I,2] += ty
    Jv[I,3] += tz
    if WITHK
        Kv[I,1] -= tx
        Kv[I,2] -= ty
        Kv[I,3] -= tz
    end
    vir[I,1] += w1
    vir[I,2] += w2
    vir[I,3] += w3
    vir[I,4] += w4
    vir[I,5] += w5
    vir[I,6] += w6
end

# ---------------------------------------------------------------------------
# Ewald reciprocal-space HVP (CPU reference; the discretization-free
# oracle for the PME path). E_rec = Σ_K ½·kf·(∑cos² + ∑sin²) over the 2N
# core+shell point charges. Differentiating the implemented force
# expression F_a = Σ_K kf·k·(kqrsin_a·∑cos − kqrcos_a·∑sin) along a shell
# direction v, with the per-k dipole factors of the direction
#   T_c(k) = Σ_j q_sj·cos(k·x_sj)·(k·v_j)
#   T_s(k) = Σ_j q_sj·sin(k·x_sj)·(k·v_j)
# (shell sums only — core phases do not move with u):
#   Jv_i += Σ_K kf·k·[ (k·v_i)·(kqrcos_i˄s·∑cos + kqrsin_i˄s·∑sin)
#                      − (kqrsin_i˄s·T_s + kqrcos_i˄s·T_c) ]
#   Kv_i += Σ_K kf·k·[ − (kqrsin_i˄c·T_s + kqrcos_i˄c·T_c) ]
# where ˄s/˄c are the shell (i+N) / core (i) rows of the kqr buffers.
# ---------------------------------------------------------------------------

struct EwaldHvpCache{TW<:NamedTuple,TS,TV,TF}
    ws::TW              # _ewald_workspace of the frozen snapshot
    S::TS               # (NK, 2) structure factor at the snapshot
    kvectors::TV
    kfactors::TF
    kvdot::Vector{FPPREC}
end

function _hvp_coulomb_cache(coulomb::EwaldInfo, atoms::Atoms, nl::NeighborList,
                            prms::NamedTuple)
    get_backend(atoms.info[:lattice]) isa KernelAbstractions.CPU ||
        error("EwaldInfo HVP is CPU-only (like the Ewald reciprocal force " *
              "pass) — use PMEInfo on the GPU backend")
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("EwaldInfo HVP has no domain-decomposition path (its structure " *
              "factor is an owned partial sum with no reduction wired) — " *
              "use PMEInfo under DD")
    arrays = (
        pos_r = atoms.arrays[:pos] * atoms.info[:lattice],
        spos_r = atoms.arrays[:pos] * atoms.info[:lattice] + atoms.arrays[:shelldisp],
        elemid = atoms.arrays[:elemid],
    )
    ws = _ewald_workspace(arrays, prms[:qc], prms[:qs])
    S = zeros(FPPREC, size(coulomb.kvectors, 1), 2)
    _ewald_structure_factor!(S, ws, coulomb.kvectors)
    return EwaldHvpCache(ws, S, coulomb.kvectors, coulomb.kfactors,
                         zeros(FPPREC, ws.N))
end

function hvp_coulomb!(Jv, Kv, v, atoms::Atoms, cache::EwaldHvpCache,
                      prms::NamedTuple)
    ws = cache.ws
    N = ws.N
    Wrec = zeros(FPPREC, 6)
    N == 0 && return Wrec
    alpha = prms[:alpha]
    kqrcos = ws.kqrcos
    kqrsin = ws.kqrsin
    kvd = cache.kvdot
    vx = @view v[:, 1]
    vy = @view v[:, 2]
    vz = @view v[:, 3]
    Jx = zeros(FPPREC, N); Jy = zeros(FPPREC, N); Jz = zeros(FPPREC, N)
    withK = Kv !== nothing
    Kx = zeros(FPPREC, withK ? N : 0)
    Ky = zeros(FPPREC, withK ? N : 0)
    Kz = zeros(FPPREC, withK ? N : 0)

    @inbounds @fastmath for K in eachindex(cache.kfactors)
        kf = cache.kfactors[K]
        kx = cache.kvectors[K,1]; ky = cache.kvectors[K,2]; kz = cache.kvectors[K,3]
        _ewald_fill_kqr!(kqrcos, kqrsin, kx, ky, kz,
                         ws.rcorex, ws.rcorey, ws.rcorez,
                         ws.rshellx, ws.rshelly, ws.rshellz,
                         ws.ccharge, ws.scharge, N)
        ∑cos = cache.S[K, 1]
        ∑sin = cache.S[K, 2]

        @turbo for i in 1:N
            kvd[i] = muladd(kx, vx[i], muladd(ky, vy[i], kz*vz[i]))
        end
        Tc = zero(FPPREC)
        Ts = zero(FPPREC)
        @turbo for i in 1:N
            Tc += kqrcos[i+N]*kvd[i]
            Ts += kqrsin[i+N]*kvd[i]
        end
        @turbo for i in 1:N
            g = kf*(kvd[i]*(kqrcos[i+N]*∑cos + kqrsin[i+N]*∑sin) -
                    (kqrsin[i+N]*Ts + kqrcos[i+N]*Tc))
            Jx[i] += kx*g
            Jy[i] += ky*g
            Jz[i] += kz*g
        end
        # Virial derivative (Issue #151). _ewald_forces! writes the per-k
        # virial as tensor(k)·ϕ with ϕ = ½·kf·(∑cos² + ∑sin²); only the
        # structure factor moves with u, and ∂∑cos/∂λ = -Ts,
        # ∂∑sin/∂λ = +Tc are the sums already formed above.
        dϕ = kf*(∑sin*Tc - ∑cos*Ts)
        k² = kx*kx + ky*ky + kz*kz
        pf = 2*(1/k² + 1/(4*alpha^2))*dϕ
        Wrec[1] += dϕ - pf*kx*kx
        Wrec[2] += dϕ - pf*ky*ky
        Wrec[3] += dϕ - pf*kz*kz
        Wrec[4] -= pf*ky*kz
        Wrec[5] -= pf*kz*kx
        Wrec[6] -= pf*kx*ky
        if withK
            @turbo for i in 1:N
                gc = -kf*(kqrsin[i]*Ts + kqrcos[i]*Tc)
                Kx[i] += kx*gc
                Ky[i] += ky*gc
                Kz[i] += kz*gc
            end
        end
    end

    Jv[:, 1] .+= Jx
    Jv[:, 2] .+= Jy
    Jv[:, 3] .+= Jz
    if withK
        Kv[:, 1] .+= Kx
        Kv[:, 2] .+= Ky
        Kv[:, 3] .+= Kz
    end
    return Wrec
end

# ---------------------------------------------------------------------------
# PME reciprocal-space HVP (dipole-PME path, Toukmaji-style; Issue #147).
#
# The implemented force is F_a = -q_a·Σ_g ∇W_a(g)·φ(g) with φ = G ∗ Q the
# potential grid (`gridenergy`) and W the wrapped B-spline weights.
# Differentiating along the shell direction v:
#
#   Jv_i = -q_si·[ (v_i·∇)∇W_i˄s · φ  +  ∇W_i˄s · φ_v ]      (shell sites)
#   Kv_i = -q_ci·[ ∇W_i˄c · φ_v ]                             (core sites)
#
# with φ_v = G ∗ Q_v and the dipole grid
# Q_v(g) = Σ_j q_sj·(v_j·∇)W_j˄s(g). Derivative chain (matches
# kernel_gather!): frac = r·icell, s_d = frac_d·fftsize_d, so
# (v·∇) = Σ_d vsc_d·∂/∂s_d with vsc = (v·icell) .* fftsize. One HVP is
# one dipole spread + one _pme_convolve + first-derivative gathers, plus
# the second-derivative gather of φ for the shell local term — the only
# place d2weights (piecewise-linear at BSPORDER = 4) enters.
# ---------------------------------------------------------------------------

struct PMEHvpCache{TP,TIC,TICD,TFS,TQ,TG,TW,TGRID,TFGRID,TV}
    pme::TP
    icell::TIC          # SMatrix 3×3 (host, passed by value into kernels)
    icell_dev::TICD     # 3×3 on the compute backend (for v·icell)
    fsrow::TFS          # 1×3 fftsize on the compute backend
    qc::TQ              # per-atom charges
    qs::TQ
    gi_c::TG            # grid indices / weights, frozen at the snapshot
    w_c::TW
    dw_c::TW
    gi_s::TG
    w_s::TW
    dw_s::TW
    d2w_s::TW
    phi::TGRID          # potential grid φ of the snapshot charges
    qhat::TFGRID        # forward transform of the snapshot charge grid
    gridv::TGRID        # dipole-grid scratch (Q_v, then reused per call)
    vsc::TV             # (N, 3) scaled-direction scratch
end

function _hvp_coulomb_cache(coulomb::PMEInfo, atoms::Atoms, nl::NeighborList,
                            prms::NamedTuple)
    backend = get_backend(atoms.info[:lattice])
    dom = Cores.domain(atoms)
    N = length(atoms)

    cell = Array(atoms.info[:lattice])
    icell = SMatrix{3,3,FPPREC}(inv(cell))
    icell_dev = _devcopy(backend, Matrix{FPPREC}(inv(cell)))
    fsrow = _devcopy(backend, permutedims(FPPREC.(collect(coulomb.fftsize))))

    # Same preamble as calculate!(::PMEInfo, ...) — shared, so the wrap
    # convention cannot drift between the force pass and the HVP.
    cpos_r, spos_r, qc, qs = _pme_site_charges(atoms, prms)

    gridcharge = KernelAbstractions.zeros(backend, FPPREC, coulomb.fftsize)
    # An empty DD rank spreads nothing but must still ride the grid
    # reduction and the (replicated) transform below — the same shape as
    # calculate!(::PMEInfo, ...). The per-atom tables it builds are empty.
    gi_c, w_c, dw_c = interpolation_weights(coulomb, cpos_r)
    gi_s, w_s, dw_s, d2w_s = interpolation_weights2(coulomb, spos_r)
    if N > 0
        spread!(gridcharge, coulomb, gi_c, qc, w_c)
        spread!(gridcharge, coulomb, gi_s, qs, w_s)
    end

    dom isa Cores.MPIDomain && Cores.allreduce_sum!(dom, vec(gridcharge))
    # The forward transform is kept (Issue #151): the reciprocal virial
    # depends on the shell displacements only through the charge grid, so
    # its directional derivative is the cross term between this transform
    # and the dipole grid's — no additional FFT anywhere.
    qhat = coulomb.rplan * gridcharge
    phi = _pme_convolve_hat(coulomb, qhat)

    gridv = KernelAbstractions.zeros(backend, FPPREC, coulomb.fftsize)
    vsc = KernelAbstractions.zeros(backend, FPPREC, N, 3)
    return PMEHvpCache(coulomb, icell, icell_dev, fsrow, qc, qs,
                       gi_c, w_c, dw_c, gi_s, w_s, dw_s, d2w_s,
                       phi, qhat, gridv, vsc)
end

@kernel inbounds = true function kernel_spread_dipole!(grid, @Const(grididx),
    @Const(charges), @Const(weights), @Const(dweights), @Const(vsc), fftsize)

    index = @index(Global)
    qᵢ = charges[index]
    v1 = vsc[index,1]; v2 = vsc[index,2]; v3 = vsc[index,3]
    g₁ = grididx[index,1]
    g₂ = grididx[index,2]
    g₃ = grididx[index,3]
    for l3 in 1:BSPORDER, l2 in 1:BSPORDER, l1 in 1:BSPORDER
        gx = g₁ - BSPORDER + l1
        gx += ifelse(gx < 1, fftsize[1], zero(eltype(fftsize)))
        gx -= ifelse(gx > fftsize[1], fftsize[1], zero(eltype(fftsize)))
        gy = g₂ - BSPORDER + l2
        gy += ifelse(gy < 1, fftsize[2], zero(eltype(fftsize)))
        gy -= ifelse(gy > fftsize[2], fftsize[2], zero(eltype(fftsize)))
        gz = g₃ - BSPORDER + l3
        gz += ifelse(gz < 1, fftsize[3], zero(eltype(fftsize)))
        gz -= ifelse(gz > fftsize[3], fftsize[3], zero(eltype(fftsize)))

        w1 = weights[index,1,l1]; w2 = weights[index,2,l2]; w3 = weights[index,3,l3]
        @atomic grid[gx,gy,gz] += qᵢ * (v1*dweights[index,1,l1]*w2*w3 +
                                        v2*w1*dweights[index,2,l2]*w3 +
                                        v3*w1*w2*dweights[index,3,l3])
    end
end

# Gather of the HVP force derivative at one site family. WITHLOCAL=true
# (shell sites): both the φ_v first-derivative term and the φ
# second-derivative local term. WITHLOCAL=false (core sites): φ_v only —
# `phi` and `d2weights` are then unused dummies.
@kernel inbounds = true function kernel_gather_hvp!(out, ::Val{WITHLOCAL},
    @Const(grididx), @Const(q), @Const(phiv), @Const(phi),
    @Const(weights), @Const(dweights), @Const(d2weights), @Const(vsc),
    fftsize, icell) where {WITHLOCAL}

    index = @index(Global)
    qᵢ = q[index]
    fx = zero(FPPREC); fy = zero(FPPREC); fz = zero(FPPREC)
    v1 = vsc[index,1]; v2 = vsc[index,2]; v3 = vsc[index,3]
    g₁ = grididx[index,1]
    g₂ = grididx[index,2]
    g₃ = grididx[index,3]
    for l3 in 1:BSPORDER, l2 in 1:BSPORDER, l1 in 1:BSPORDER
        gx = g₁ - BSPORDER + l1
        gx += ifelse(gx < 1, fftsize[1], zero(eltype(fftsize)))
        gx -= ifelse(gx > fftsize[1], fftsize[1], zero(eltype(fftsize)))
        gy = g₂ - BSPORDER + l2
        gy += ifelse(gy < 1, fftsize[2], zero(eltype(fftsize)))
        gy -= ifelse(gy > fftsize[2], fftsize[2], zero(eltype(fftsize)))
        gz = g₃ - BSPORDER + l3
        gz += ifelse(gz < 1, fftsize[3], zero(eltype(fftsize)))
        gz -= ifelse(gz > fftsize[3], fftsize[3], zero(eltype(fftsize)))

        w1 = weights[index,1,l1]; w2 = weights[index,2,l2]; w3 = weights[index,3,l3]
        d1 = dweights[index,1,l1]; d2 = dweights[index,2,l2]; d3 = dweights[index,3,l3]

        Pv = phiv[gx,gy,gz] * qᵢ
        fx -= Pv * d1*w2*w3 * fftsize[1]
        fy -= Pv * w1*d2*w3 * fftsize[2]
        fz -= Pv * w1*w2*d3 * fftsize[3]

        if WITHLOCAL
            P = phi[gx,gy,gz] * qᵢ
            t1 = v1*d2weights[index,1,l1]*w2*w3 + v2*d1*d2*w3 + v3*d1*w2*d3
            t2 = v1*d1*d2*w3 + v2*w1*d2weights[index,2,l2]*w3 + v3*w1*d2*d3
            t3 = v1*d1*w2*d3 + v2*w1*d2*d3 + v3*w1*w2*d2weights[index,3,l3]
            fx -= P * t1 * fftsize[1]
            fy -= P * t2 * fftsize[2]
            fz -= P * t3 * fftsize[3]
        end
    end
    out[index,1] += fx * icell[1,1] + fy * icell[1,2] + fz * icell[1,3]
    out[index,2] += fx * icell[2,1] + fy * icell[2,2] + fz * icell[2,3]
    out[index,3] += fx * icell[3,1] + fy * icell[3,2] + fz * icell[3,3]
end

function hvp_coulomb!(Jv, Kv, v, atoms::Atoms, cache::PMEHvpCache,
                      prms::NamedTuple)
    backend = get_backend(atoms.info[:lattice])
    dom = Cores.domain(atoms)
    N = length(atoms)
    pme = cache.pme
    fs32 = Int32.(pme.fftsize)

    fill!(cache.gridv, zero(FPPREC))
    if N > 0
        # vsc = (v·icell) .* fftsize — the fractional-space chain of (v·∇).
        mul!(cache.vsc, v, cache.icell_dev)
        cache.vsc .*= cache.fsrow
        kernel_spread_dipole!(backend, 512, N)(cache.gridv, cache.gi_s, cache.qs,
            cache.w_s, cache.dw_s, cache.vsc, fs32)
    end

    # Same reduction regime as the charge grid of the force path: the grid
    # points an atom touches are global, so the owned-atom dipole spreads
    # are summed across ranks before the (replicated) transform. Missing
    # this is silently wrong, not an error — each rank would convolve its
    # own partial dipole grid (Issue #149).
    dom isa Cores.MPIDomain && Cores.allreduce_sum!(dom, vec(cache.gridv))
    qvhat = pme.rplan * cache.gridv
    phiv = _pme_convolve_hat(pme, qvhat)

    # Reciprocal virial derivative (Issue #151). calculate!(::PMEInfo, …)
    # adds +Σ_g bsp2·|Q̂|²·presfactor to info[:virial]; the shell
    # displacements enter through Q̂ alone, so d/dλ replaces |Q̂|² by
    # 2·Re(conj(Q̂)·Q̂_v). Both transforms are already in hand, so this is
    # one grid reduction and no extra FFT. Replicated under DD (the grids
    # were reduced above), hence returned rather than summed again. The
    # contraction is shared with the force path — see
    # _pme_presfactor_contract for why it is a gemv and not six sums.
    re = @. pme.bsp2factor * 2 * real(conj(cache.qhat) * qvhat)
    Wrec = _pme_presfactor_contract(pme, re)

    if N > 0
        kernel_gather_hvp!(backend, 512, N)(Jv, Val(true), cache.gi_s, cache.qs,
            phiv, cache.phi, cache.w_s, cache.dw_s, cache.d2w_s, cache.vsc,
            fs32, cache.icell)
        if Kv !== nothing
            kernel_gather_hvp!(backend, 512, N)(Kv, Val(false), cache.gi_c, cache.qc,
                phiv, cache.phi, cache.w_c, cache.dw_c, cache.dw_c, cache.vsc,
                fs32, cache.icell)
        end
    end
    return Wrec
end

# ---------------------------------------------------------------------------
# Short-range pair Hessian (ghost format, pure gather — same neighbour
# list and cutoff branching as the force kernels in CoulombBuckingham.jl).
# ϕ_eff″ = ϕ″ (the shifted-force constants drop out of the second
# derivative), so no rc-shifted d2ϕ terms appear. With an unnormalized
# pair vector rvec: B·v = (ϕ″ - ϕ_eff′/r)/r²·(rvec·v)·rvec + ϕ_eff′/r·v.
#
# Virial derivative (Issue #151). kernel_gcb! accumulates each pair term
# as `virials -= ½·(ϕ_eff′/r)·r⊗r`; differentiating along v with
# δ = D_v r gives
#   D_v[½·(ϕ_eff′/r)·r⊗r] = ½·[ c1·(r·δ)·r⊗r + (ϕ_eff′/r)·(δ⊗r + r⊗δ) ]
# where c1 = (ϕ″ - ϕ_eff′/r)/r² and (r·δ) are already in registers for
# the Hessian product itself — the added work is the six-component
# accumulation alone. δ is v_i, v_i - v_j and -v_j for the shell-core,
# shell-shell and core-shell blocks respectively (the core-core term does
# not move with u at all and contributes nothing).
# ---------------------------------------------------------------------------
@kernel inbounds = true function kernel_gcb_hvp!(Jv, Kv, vir, ::Val{WITHK}, @Const(vext),
    @Const(xcart), @Const(sdisp), @Const(elemid),
    @Const(A), @Const(B), @Const(rho), @Const(C), @Const(e),
    @Const(qc), @Const(qs), @Const(ηc), @Const(ηs),
    @Const(alpha), @Const(rc), @Const(nlist)) where {WITHK}

    I = @index(Global)
    sᵢ = elemid[I]
    rcore_i = xcart[I,1], xcart[I,2], xcart[I,3]
    rshell_i = sdisp[I,1], sdisp[I,2], sdisp[I,3]
    vix, viy, viz = vext[I,1], vext[I,2], vext[I,3]
    qs_i = qs[sᵢ]; ηs_i = ηs[sᵢ]
    qc_i = qc[sᵢ]; ηc_i = ηc[sᵢ]
    Jx = zero(FPPREC); Jy = zero(FPPREC); Jz = zero(FPPREC)
    Kx = zero(FPPREC); Ky = zero(FPPREC); Kz = zero(FPPREC)
    w1 = zero(FPPREC); w2 = zero(FPPREC); w3 = zero(FPPREC)
    w4 = zero(FPPREC); w5 = zero(FPPREC); w6 = zero(FPPREC)

    for jlist in axes(nlist,2)
        j = nlist[I,jlist]
        if j < 1
            break
        end
        sⱼ = elemid[j]
        rx_cc = rcore_i[1] - xcart[j,1]
        ry_cc = rcore_i[2] - xcart[j,2]
        rz_cc = rcore_i[3] - xcart[j,3]
        qc_j = qc[sⱼ]; qs_j = qs[sⱼ]
        ηc_j = ηc[sⱼ]; ηs_j = ηs[sⱼ]

        # shell(i)-core(j): depends on u_i only → Jv_i -= B_sc·v_i
        rx = rx_cc + rshell_i[1]
        ry = ry_cc + rshell_i[2]
        rz = rz_cc + rshell_i[3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        q = qs_i*qc_j
        ϕp_r = q*(dϕC(r, alpha, ηs_i, ηc_j) - dϕC(rc, alpha, ηs_i, ηc_j))/r
        ϕpp = q*d2ϕC(r, alpha, ηs_i, ηc_j)
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        c1 = cutoff*(ϕpp - ϕp_r)/(r*r)
        ϕp_r *= cutoff
        rv = rx*vix + ry*viy + rz*viz
        Jx -= c1*rv*rx + ϕp_r*vix
        Jy -= c1*rv*ry + ϕp_r*viy
        Jz -= c1*rv*rz + ϕp_r*viz
        cr = FPPREC(0.5)*c1*rv
        pp = FPPREC(0.5)*ϕp_r
        w1 -= cr*rx*rx + pp*2*vix*rx
        w2 -= cr*ry*ry + pp*2*viy*ry
        w3 -= cr*rz*rz + pp*2*viz*rz
        w4 -= cr*ry*rz + pp*(viy*rz + ry*viz)
        w5 -= cr*rz*rx + pp*(viz*rx + rz*vix)
        w6 -= cr*rx*ry + pp*(vix*ry + rx*viy)

        # shell(i)-shell(j): Jv_i -= B_ss·(v_i - v_j)
        Aᵢⱼ = A[sᵢ,sⱼ]; Bᵢⱼ = B[sᵢ,sⱼ]; ρᵢⱼ = rho[sᵢ,sⱼ]; Cᵢⱼ = C[sᵢ,sⱼ]
        rx = rx_cc + rshell_i[1] - sdisp[j,1]
        ry = ry_cc + rshell_i[2] - sdisp[j,2]
        rz = rz_cc + rshell_i[3] - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        q = qs_i*qs_j
        ϕp_r = (q*(dϕC(r, alpha, ηs_i, ηs_j) - dϕC(rc, alpha, ηs_i, ηs_j)) +
                dϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - dϕB(rc, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e))/r
        ϕpp = q*d2ϕC(r, alpha, ηs_i, ηs_j) + d2ϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e)
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        c1 = cutoff*(ϕpp - ϕp_r)/(r*r)
        ϕp_r *= cutoff
        dvx = vix - vext[j,1]
        dvy = viy - vext[j,2]
        dvz = viz - vext[j,3]
        rv = rx*dvx + ry*dvy + rz*dvz
        Jx -= c1*rv*rx + ϕp_r*dvx
        Jy -= c1*rv*ry + ϕp_r*dvy
        Jz -= c1*rv*rz + ϕp_r*dvz
        cr = FPPREC(0.5)*c1*rv
        pp = FPPREC(0.5)*ϕp_r
        w1 -= cr*rx*rx + pp*2*dvx*rx
        w2 -= cr*ry*ry + pp*2*dvy*ry
        w3 -= cr*rz*rz + pp*2*dvz*rz
        w4 -= cr*ry*rz + pp*(dvy*rz + ry*dvz)
        w5 -= cr*rz*rx + pp*(dvz*rx + rz*dvx)
        w6 -= cr*rx*ry + pp*(dvx*ry + rx*dvy)

        # core(i)-shell(j): depends on u_j only → Kv_i += B_cs·v_j.
        # Evaluated even when Kv is not wanted: this pair term still moves
        # with u_j, so its half of the virial derivative is not optional
        # (the per-atom split then matches kernel_gcb!'s in both branches).
        rx = rx_cc - sdisp[j,1]
        ry = ry_cc - sdisp[j,2]
        rz = rz_cc - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        q = qc_i*qs_j
        ϕp_r = q*(dϕC(r, alpha, ηc_i, ηs_j) - dϕC(rc, alpha, ηc_i, ηs_j))/r
        ϕpp = q*d2ϕC(r, alpha, ηc_i, ηs_j)
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        c1 = cutoff*(ϕpp - ϕp_r)/(r*r)
        ϕp_r *= cutoff
        vjx = vext[j,1]; vjy = vext[j,2]; vjz = vext[j,3]
        rv = rx*vjx + ry*vjy + rz*vjz
        if WITHK
            Kx += c1*rv*rx + ϕp_r*vjx
            Ky += c1*rv*ry + ϕp_r*vjy
            Kz += c1*rv*rz + ϕp_r*vjz
        end
        # δ = -v_j here, so (r·δ) = -rv and (δ⊗r + r⊗δ) picks up -v_j:
        # both sign flips are folded into the accumulation below.
        cr = FPPREC(0.5)*c1*rv
        pp = FPPREC(0.5)*ϕp_r
        w1 += cr*rx*rx + pp*2*vjx*rx
        w2 += cr*ry*ry + pp*2*vjy*ry
        w3 += cr*rz*rz + pp*2*vjz*rz
        w4 += cr*ry*rz + pp*(vjy*rz + ry*vjz)
        w5 += cr*rz*rx + pp*(vjz*rx + rz*vjx)
        w6 += cr*rx*ry + pp*(vjx*ry + rx*vjy)
    end

    Jv[I,1] += Jx
    Jv[I,2] += Jy
    Jv[I,3] += Jz
    if WITHK
        Kv[I,1] += Kx
        Kv[I,2] += Ky
        Kv[I,3] += Kz
    end
    vir[I,1] += w1
    vir[I,2] += w2
    vir[I,3] += w3
    vir[I,4] += w4
    vir[I,5] += w5
    vir[I,6] += w6
end

# ---------------------------------------------------------------------------
# Linear response on the analytic HVP (moved here from CoreShell.jl,
# Issue #147: the workspace type dispatches on EwaldInfo/PMEInfo).
# ---------------------------------------------------------------------------

"""
    cg_hessian_free!(x, b, atoms, ws, prms; tol = 1e-8, maxiter = 20)

Solve `H·x = b` with `H = ∂²E/∂x_s∂x_s = -J` (SPD at a stable shell
configuration) by preconditioned CG on the analytic [`hvp!`](@ref)
(J block only — one dipole solve per iteration, no FD δ anywhere). The
Jacobi preconditioner is the spring diagonal `ws.kvals` (k2 dominates the
Hessian diagonal; unit entries guard shell-less species). `x`, `b` are
`(N, 3)`; all three right-hand sides ride one workspace snapshot.
"""
function cg_hessian_free!(x, b, atoms::Atoms, ws::CoreShellHvpWorkspace,
                          prms::NamedTuple; tol::Real = 1e-8, maxiter::Int = 20)
    T = eltype(x)

    r  = copy(b)
    z  = similar(r)
    p  = similar(r)
    Jp = similar(r)
    Hp = similar(r)

    @. z = r / ws.kvals
    copyto!(p, z)

    rz_old = sum(r .* z)
    bnorm  = sqrt(max(sum(b .* b), eps(T)))

    fill!(x, zero(T))

    converged = false
    rnorm = bnorm
    for it in 1:maxiter
        # The virial derivative is not optional (see hvp!); this solve has
        # no use for it, so it lands in the workspace sink.
        hvp!(Jp, nothing, ws.Wscratch, p, atoms, ws, prms)
        @. Hp = -Jp

        pHp = sum(p .* Hp)
        # SPD-breakdown guard, sign-based on purpose: the historical
        # `abs(pHp) < eps(T)` compared a DIMENSIONAL quantity against an
        # absolute epsilon and, with Jacobi scaling (p ~ r/k2), fired long
        # before convergence — stalling the residual at ~1e-9 relative and
        # putting a floor under the BEC acoustic sum rule (Issue #147).
        if pHp <= zero(T)
            break
        end

        α = rz_old / pHp

        @. x = x + α * p
        @. r = r - α * Hp

        rnorm = sqrt(max(sum(r .* r), eps(T)))
        if rnorm / bnorm < tol
            converged = true
            break
        end

        @. z = r / ws.kvals

        rz_new = sum(r .* z)
        β = rz_new / rz_old
        rz_old = rz_new

        @. p = z + β * p
    end

    # An unconverged x still produces plausible-looking downstream
    # response quantities (linear_response! / FitShellModel would digest
    # them silently), so at least say so.
    converged ||
        @warn "cg_hessian_free!: linear solve not converged" relres = rnorm/bnorm tol maxiter maxlog = 5

    return x
end

function linear_response!(atoms::Atoms, nl::NeighborList, coulomb::CoulombInfo,
                          prms::NamedTuple;
                          tol::Real = 1e-8, maxiter::Int = 50)

    @assert get_backend(atoms) isa KernelAbstractions.CPU "linear_response! currently supports only CPU backend."
    # The polarization loops below use rank-local sums and rank-local
    # convergence branching — under DD they would deadlock or produce
    # per-rank dielectric tensors. Fitting drives this through
    # data-parallel MPI (SerialDomain frames), never through DD.
    # (hvp_workspace refuses DD as well.)
    Cores.domain(atoms) isa Cores.MPIDomain &&
        error("linear_response! is not supported under domain decomposition " *
              "(rank-local CG and polarization sums); use a SerialDomain frame")

    # ε = I + (4π/V)Σ q_s·x is the TIN-FOIL (conductor) permittivity: it
    # assumes the shells respond to the applied field alone. With a
    # surface term the shell response x already carries the depolarization
    # screening of the declared external medium, so the same expression
    # would report a boundary-dependent quantity under the name of the
    # material's ε — and the BECs likewise. Refuse rather than relabel.
    haskey(prms, :surface_term) &&
        error("linear_response! does not support :surface_term: the " *
              "dielectric tensor it reports is the conductor-boundary one " *
              "(ε = I + 4π/V·Σ q_s·∂u/∂E), which the depolarization field " *
              "would silently redefine.  Compute the response without the " *
              "surface term, or take the screened response from an MD run")

    clear!(atoms, (:potential_energy, :virial, :shellforces, :forces,
                   :becex, :becey, :becez, :dielectric_tensor))

    # 1) Shell relaxation under zero field
    scfcnt = calculate!(CoreShell(), atoms, nl, coulomb, prms)

    # 2) Freeze the linearization point: one workspace serves all three
    #    field directions and both Hessian blocks. Unlike the FD path this
    #    leaves the relaxed forces in the atoms arrays untouched.
    ws = hvp_workspace(atoms, nl, coulomb, prms)

    N = length(atoms)

    qs = zeros(FPPREC, N)
    qc = zeros(FPPREC, N)
    for i in 1:N
        qs[i] = prms[:qs][atoms.arrays[:elemid][i]]
        qc[i] = prms[:qc][atoms.arrays[:elemid][i]]
    end

    ε = Matrix{FPPREC}(I, 3, 3)

    b = zeros(FPPREC, N, 3)
    x = zeros(FPPREC, N, 3)

    dF_relax = zeros(FPPREC, N, 3)  # (∂F_c/∂u)·xβ

    for β in 1:3
        # --- (A) xβ = ∂u/∂Eβ: H·x = b with b = q_s·eβ (RHS from the
        #     u-linear external-field term) ---
        fill!(b, 0)
        for i in 1:N
            b[i,β] = qs[i]
        end

        cg_hessian_free!(x, b, atoms, ws, prms; tol = tol, maxiter = maxiter)

        # --- (B) clamped-ion permittivity: pvec[α] = Σ_i q_s[i]·x[i,α] ---
        pvec = zeros(FPPREC, 3)
        for α in 1:3
            s = zero(FPPREC)
            for i in 1:N
                s += qs[i] * x[i,α]
            end
            pvec[α] = s
        end
        ε[:,β] .+= (4π / volume(atoms)) .* pvec

        # --- (C) BEC: dF/dEβ = (∂F_c/∂u)·xβ + qc·δ_{αβ} (the mixed block
        #     of the same hvp!; the J output goes to the workspace scratch) ---
        hvp!(ws.Jscratch, dF_relax, ws.Wscratch, x, atoms, ws, prms)

        if β == 1
            atoms.arrays[:becex] .= dF_relax
            for i in 1:N
                atoms.arrays[:becex][i,1] += qc[i]
            end
        elseif β == 2
            atoms.arrays[:becey] .= dF_relax
            for i in 1:N
                atoms.arrays[:becey][i,2] += qc[i]
            end
        else
            atoms.arrays[:becez] .= dF_relax
            for i in 1:N
                atoms.arrays[:becez][i,3] += qc[i]
            end
        end
    end

    atoms.info[:dielectric_tensor] .= [ε[1,1], ε[2,2], ε[3,3], ε[2,3], ε[3,1], ε[1,2]]
    return scfcnt
end
