using LinearAlgebra
using KernelAbstractions
using SpecialFunctions: erf, erfcinv

export CoulombInfo, EwaldInfo, ewaldcoeff, calculate!

"""
    CoulombInfo

Abstract supertype for k-space Coulomb accelerators. Concrete subtypes
[`EwaldInfo`](@ref) (analytic Ewald, suitable for small cells) and
[`PMEInfo`](@ref) (particle-mesh Ewald, recommended for large cells)
share the [`calculate!`](@ref) and [`cellupdate!`](@ref) interface, so the
shell relaxation in [`CoreShell`](@ref) is agnostic to the choice.
"""
abstract type CoulombInfo end

"""
    EwaldInfo <: CoulombInfo

Analytic Ewald sum over reciprocal-lattice vectors with `|k| ≤ kcut`.
Field `numk = (nx, ny, nz)` is the half-box of `k`-points considered;
`kvectors` and `kfactors` cache the surviving (in-cutoff) modes after
[`cellupdate!`](@ref).

Construct with `EwaldInfo(cell, params)`; rebuild the k-table after a
cell change via `cellupdate!(ewald, new_cell, params)`. Parameters used:
`:alpha` (Ewald α — see [`ewaldcoeff`](@ref)), `:ewald_tolerance`.
"""
mutable struct EwaldInfo <: CoulombInfo
    numk::NTuple{3,Int}
    kvectors
    kfactors
end

"""
    ewaldcoeff(params) -> Float

Return the Ewald splitting parameter α that satisfies the real-space
truncation error bound

```
erfc(α · rc) ≈ params[:ewald_tolerance]
```

Pass the result back into `params[:alpha]` (typically by splatting
`(; params..., alpha = ewaldcoeff(params))`) before constructing
[`EwaldInfo`](@ref) or [`PMEInfo`](@ref).
"""
function ewaldcoeff(params)
    return erfcinv(params[:rc]*params[:ewald_tolerance])/params[:rc]
end

@kernel function kernel_ewaldfactors!(isincutoff, kvectors, kfactors, numk, lattice, alpha, kcut)
    I = @index(Global, NTuple)

    i = I[1] - numk[1] - 1
    j = I[2] - numk[2] - 1
    k = I[3] - numk[3] - 1

    ax, ay, az = lattice[1,1], lattice[1,2], lattice[1,3]
    bx, by, bz = lattice[2,1], lattice[2,2], lattice[2,3]
    cx, cy, cz = lattice[3,1], lattice[3,2], lattice[3,3]

    v = ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)

    cross_bc_x = by * cz - bz * cy
    cross_bc_y = bz * cx - bx * cz
    cross_bc_z = bx * cy - by * cx

    cross_ca_x = cy * az - cz * ay
    cross_ca_y = cz * ax - cx * az
    cross_ca_z = cx * ay - cy * ax

    cross_ab_x = ay * bz - az * by
    cross_ab_y = az * bx - ax * bz
    cross_ab_z = ax * by - ay * bx

    kvec_x = (2π * i * cross_bc_x + 2π * j * cross_ca_x + 2π * k * cross_ab_x) / v
    kvec_y = (2π * i * cross_bc_y + 2π * j * cross_ca_y + 2π * k * cross_ab_y) / v
    kvec_z = (2π * i * cross_bc_z + 2π * j * cross_ca_z + 2π * k * cross_ab_z) / v

    k2 = kvec_x^2 + kvec_y^2 + kvec_z^2

    if 0 < k2 && k2 <= kcut^2
        isincutoff[I...] = true
        kvectors[I..., 1] = kvec_x
        kvectors[I..., 2] = kvec_y
        kvectors[I..., 3] = kvec_z
        kfactors[I...] = 4π/v/k2*exp(-k2/(4*alpha^2))
    else
        isincutoff[I...] = false
    end
end

"""
    cellupdate!(ewald::EwaldInfo, cell, params) -> EwaldInfo

Recompute `numk`, `kvectors`, `kfactors` after the cell has changed (e.g.
during NPT MD). Mutates and returns `ewald`. Cheap; called once per MD
step from [`CoreShell`](@ref).
"""
function cellupdate!(ewald::EwaldInfo, cell, params)
    backend = get_backend(cell)
    α = params[:alpha]
    kcut = 2α * sqrt(-log(params[:ewald_tolerance]))
    numk = ((ceil.(Int, [norm(cell[1,:]),norm(cell[2,:]),norm(cell[3,:])] .* kcut ./(2*pi)))...,)

    isincutoff = KernelAbstractions.zeros(backend, Bool, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))
    kvectors = KernelAbstractions.zeros(backend, FPPREC, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1, 3))
    kfactors = KernelAbstractions.zeros(backend, FPPREC, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))

    kernel_ewaldfactors!(backend)(isincutoff, kvectors, kfactors, numk, cell, α, kcut; ndrange = (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))
    kvectors = reshape(kvectors[isincutoff,:], :, 3)
    kfactors = reshape(kfactors[isincutoff], :)

    ewald.numk = (numk...,)
    ewald.kvectors = kvectors
    ewald.kfactors = kfactors

    return ewald
end

function EwaldInfo(cell, params)
    backend = get_backend(cell)
    α = params[:alpha]
    kcut = 2α * sqrt(-log(params[:ewald_tolerance]))
    numk = ((ceil.(Int, [norm(cell[1,:]),norm(cell[2,:]),norm(cell[3,:])] .* kcut ./(2*pi)))...,)

    isincutoff = KernelAbstractions.zeros(backend, Bool, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))
    kvectors = KernelAbstractions.zeros(backend, FPPREC, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1, 3))
    kfactors = KernelAbstractions.zeros(backend, FPPREC, (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))

    kernel_ewaldfactors!(backend)(isincutoff, kvectors, kfactors, numk, cell, α, kcut; ndrange = (2numk[1]+1, 2numk[2]+1, 2numk[3]+1))
    kvectors = reshape(kvectors[isincutoff,:], :, 3)
    kfactors = reshape(kfactors[isincutoff], :)

    ewaldinfo = EwaldInfo((numk...,), kvectors, kfactors)

    return ewaldinfo
end

# Fill the per-atom kqrcos/kqrsin work buffers for one k-vector: cores in
# rows 1:N, shells in N+1:2N. Shared verbatim by the structure-factor pass
# and the force pass so the recomputed values are bit-identical.
@inline function _ewald_fill_kqr!(kqrcos, kqrsin, kvx, kvy, kvz,
                                  rcorex, rcorey, rcorez, rshellx, rshelly, rshellz,
                                  ccharge, scharge, N)
    @turbo for i in 1:N
        θ = muladd(kvx, rcorex[i], muladd(kvy, rcorey[i], kvz*rcorez[i]))
        s, c = sincos(θ)
        q = ccharge[i]
        kqrcos[i] = q * c
        kqrsin[i] = q * s
    end
    @turbo for i in 1:N
        j = i + N
        θ = muladd(kvx, rshellx[i], muladd(kvy, rshelly[i], kvz*rshellz[i]))
        s, c = sincos(θ)
        q = scharge[i]
        kqrcos[j] = q * c
        kqrsin[j] = q * s
    end
    return nothing
end

# Per-call workspace shared by the two passes: the column copies, the
# gathered charges, and the kqr buffers are built once so both passes see
# bit-identical inputs without double allocation.
function _ewald_workspace(arrays::NamedTuple, qc, qs)
    N = length(arrays[:elemid])
    return (N = N,
            kqrcos = zeros(FPPREC, 2N), kqrsin = zeros(FPPREC, 2N),
            rcorex = arrays[:pos_r][:, 1], rcorey = arrays[:pos_r][:, 2],
            rcorez = arrays[:pos_r][:, 3],
            rshellx = arrays[:spos_r][:, 1], rshelly = arrays[:spos_r][:, 2],
            rshellz = arrays[:spos_r][:, 3],
            ccharge = qc[arrays[:elemid]], scharge = qs[arrays[:elemid]])
end

"""
    _ewald_structure_factor!(S, ws, kvectors) -> S

Pass A of the two-pass reciprocal-space sum: fill `S :: (NK, 2)` with the
per-k structure factor `[Σcos Σsin]` summed over the atoms in the
workspace `ws` (cores + shells). Under domain decomposition each rank
sums its owned atoms only — `S` is then an owned partial sum that the
caller must Allreduce **before** the force pass (the forces need the
global S(k); reducing only the energy would leave the forces wrong).
"""
function _ewald_structure_factor!(S, ws::NamedTuple, kvectors)
    if ws.N == 0   # empty rank: the partial sum is zero (vsum / @turbo on
        fill!(S, zero(FPPREC))   # empty ranges is unverified territory)
        return S
    end
    @inbounds @fastmath for K in axes(S, 1)
        kvx = kvectors[K,1]; kvy = kvectors[K,2]; kvz = kvectors[K,3]
        _ewald_fill_kqr!(ws.kqrcos, ws.kqrsin, kvx, kvy, kvz,
                         ws.rcorex, ws.rcorey, ws.rcorez,
                         ws.rshellx, ws.rshelly, ws.rshellz,
                         ws.ccharge, ws.scharge, ws.N)
        S[K, 1] = vsum(ws.kqrcos)
        S[K, 2] = vsum(ws.kqrsin)
    end
    return S
end

"""
    _ewald_forces!(arrays, S, ws, potential_energy, virial, kvectors, kfactors, alpha)

Pass B: with the (globally complete) structure factor `S`, add the
reciprocal-space energy and virial — **replicated quantities** under
domain decomposition (identical on every rank once S is global, so they
are added directly, never through `add_reduced_ev!`; an empty rank still
adds them) — and the k-space forces on the local atoms (per-atom given
global S; no fold needed). The sincos values are recomputed with the
exact operations of pass A, so the serial result is bit-identical to the
historical single-pass kernel.
"""
function _ewald_forces!(arrays::NamedTuple, S, ws::NamedTuple, potential_energy, virial,
                        kvectors, kfactors, alpha)
    # @inbounds below reads S[K, :] for K in eachindex(kfactors): a length
    # mismatch would be a silent out-of-bounds read, not an error.
    size(S, 1) == length(kfactors) == size(kvectors, 1) ||
        error("_ewald_forces!: S has $(size(S, 1)) rows but kfactors/kvectors have " *
              "$(length(kfactors))/$(size(kvectors, 1)) — mismatched k-tables")
    N = ws.N
    kqrcos = ws.kqrcos
    kqrsin = ws.kqrsin

    fx = zero(arrays[:forces][:,1]); fy = zero(arrays[:forces][:,2]); fz = zero(arrays[:forces][:,3])
    sfx = zero(arrays[:shellforces][:,1]); sfy = zero(arrays[:shellforces][:,2]); sfz = zero(arrays[:shellforces][:,3])

    @inbounds @fastmath for K in eachindex(kfactors)
        kf = kfactors[K]
        kvx = kvectors[K,1]; kvy = kvectors[K,2]; kvz = kvectors[K,3]

        if N > 0
            _ewald_fill_kqr!(kqrcos, kqrsin, kvx, kvy, kvz,
                             ws.rcorex, ws.rcorey, ws.rcorez,
                             ws.rshellx, ws.rshelly, ws.rshellz,
                             ws.ccharge, ws.scharge, N)
        end
        ∑cos = S[K, 1]
        ∑sin = S[K, 2]

        ϕ = 0.5*kf*(∑cos*∑cos + ∑sin*∑sin)
        k2 = kvx*kvx + kvy*kvy + kvz*kvz
        potential_energy[1] += ϕ
        virial[1] += (1.0 - 2.0*kvx*kvx * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ
        virial[2] += (1.0 - 2.0*kvy*kvy * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ
        virial[3] += (1.0 - 2.0*kvz*kvz * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ
        virial[4] += (0.0 - 2.0*kvy*kvz * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ
        virial[5] += (0.0 - 2.0*kvz*kvx * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ
        virial[6] += (0.0 - 2.0*kvx*kvy * (1.0 /k2 + 1.0 /(4.0*alpha^2)) )*ϕ

        # Force is −∂E/∂rᵢ with E = ½·kf·(∑cos² + ∑sin²).  Since
        # ∂∑cos/∂rᵢ = −qᵢsin(k·rᵢ)·k and ∂∑sin/∂rᵢ = +qᵢcos(k·rᵢ)·k,
        # the gradient gives  kf·k·(kqrsin·∑cos − kqrcos·∑sin).
        # (kqrcos and kqrsin are *not* interchangeable here: the wrong
        # ordering is even under k→−k and cancels to zero over the
        # inversion-symmetric k-set, silently dropping the whole force.)
        if N > 0
            @turbo for i in 1:N
                fx[i] += kvx * kf*(kqrsin[i]*∑cos - kqrcos[i]*∑sin)
                fy[i] += kvy * kf*(kqrsin[i]*∑cos - kqrcos[i]*∑sin)
                fz[i] += kvz * kf*(kqrsin[i]*∑cos - kqrcos[i]*∑sin)
            end
            @turbo for i in 1:N
                sfx[i] += kvx * kf*(kqrsin[i+N]*∑cos - kqrcos[i+N]*∑sin)
                sfy[i] += kvy * kf*(kqrsin[i+N]*∑cos - kqrcos[i+N]*∑sin)
                sfz[i] += kvz * kf*(kqrsin[i+N]*∑cos - kqrcos[i+N]*∑sin)
            end
        end
    end
    arrays[:forces][:,1] .+= fx
    arrays[:forces][:,2] .+= fy
    arrays[:forces][:,3] .+= fz
    arrays[:shellforces][:,1] .+= sfx
    arrays[:shellforces][:,2] .+= sfy
    arrays[:shellforces][:,3] .+= sfz
end

"""
    chargedcell_correction!(atoms, Q, alpha)

Add the charged-cell correction for a non-neutral cell (total charge `Q`):

    E_corr = -π Q² / (2 α² V)

to `atoms.info[:potential_energy]`, and `E_corr` to each diagonal
component of `atoms.info[:virial]`.  Excluding the k=0 mode implicitly
neutralizes a charged cell with a uniform background but leaves the
energy α-dependent; this term restores a well-defined (α-independent)
energy and NPT pressure.  E_corr depends on positions not at all and on
the cell only through V, so it adds no force and its virial is diagonal,
`W_αβ += δ_αβ·E_corr` (convention: virial = -∂E/∂ε).  Q = 0 ⇒ exact
no-op.

Only the α-dependence is removed: the implicit uniform neutralizing
background remains, so absolute energies / pressures of charged cells
still carry the O(Q²/L) monopole finite-size (Makov–Payne-type) error
inherent to jellium-neutralized PBC (Issue #15).
"""
function chargedcell_correction!(atoms::Atoms, Q, alpha)
    cell = Array(atoms.info[:lattice])
    V = dot(cell[1,:], cross(cell[2,:], cell[3,:]))
    Ecorr = -FPPREC(π) * FPPREC(Q)^2 / (2 * FPPREC(alpha)^2 * V)
    atoms.info[:potential_energy] .+= Ecorr
    atoms.info[:virial][1:3] .+= Ecorr
    return nothing
end

"""
    calculate!(ewaldinfo::EwaldInfo, atoms::Atoms, params)

Add the analytic Ewald reciprocal-space + self-energy contributions to
`atoms.info[:potential_energy]`, `atoms.info[:virial]`,
`atoms.arrays[:forces]`, and `atoms.arrays[:shellforces]`. The real-space
short-range part is handled separately by [`CoulombBuckingham`](@ref).

For a non-neutral cell (Σq ≠ 0) the charged-cell correction is included
via [`chargedcell_correction!`](@ref); note that only the α-dependence is
removed — the monopole finite-size error of the implicit neutralizing
background remains (Issue #15).

`params[:alpha]` (Ewald α — see [`ewaldcoeff`](@ref)) and `params[:qc]`,
`params[:qs]` (per-element charges) are required.
"""
function calculate!(ewaldinfo::EwaldInfo, atoms::Atoms, params)
    backend = get_backend(atoms.info[:lattice])
    # The two-pass reciprocal sum below is deliberately host-only (@turbo
    # scalar loops; reference-grade). GPU frames would die in an obscure
    # scalar-indexing error deep inside — refuse loudly at the entry
    # instead. Reciprocal space on the GPU backend goes through PME.
    backend isa KernelAbstractions.CPU ||
        error("Ewald reciprocal (EwaldInfo) is CPU-only by design — use PME " *
              "(PMEInfo) for reciprocal-space sums on the GPU backend")
    dom = Cores.domain(atoms)

    α = params[:alpha]
    N = length(atoms.arrays[:masses])

    arrays = (
        pos_r = atoms.arrays[:pos] * atoms.info[:lattice],
        spos_r = atoms.arrays[:pos] * atoms.info[:lattice] + atoms.arrays[:shelldisp],
        elemid = atoms.arrays[:elemid],
        forces = atoms.arrays[:forces],
        shellforces = atoms.arrays[:shellforces],
        shelldisp = atoms.arrays[:shelldisp],
    )

    NK = size(ewaldinfo.kvectors, 1)

    # Two-pass reciprocal sum: local S(k) partial sums, one Allreduce,
    # then energy/virial (replicated once S is global — added directly per
    # the reduction regime) and the local-atom forces. Serial results are
    # bit-identical to the historical single-pass kernel.
    ws = _ewald_workspace(arrays, params[:qc], params[:qs])
    S = zeros(FPPREC, NK, 2)
    _ewald_structure_factor!(S, ws, ewaldinfo.kvectors)
    Cores.allreduce_sum!(dom, vec(S))
    _ewald_forces!(arrays, S, ws, atoms.info[:potential_energy], atoms.info[:virial],
                   ewaldinfo.kvectors, ewaldinfo.kfactors, α)

    # Intra (core–shell same-atom) correction: per-owned-atom sums →
    # reduction regime. An empty rank contributes zeros but must still
    # ride the collective inside add_reduced_ev!.
    if N > 0
        potential_energies = KernelAbstractions.zeros(backend, FPPREC, N)
        virials = KernelAbstractions.zeros(backend, FPPREC, N, 6)
        kernel_subtractintraatom!(backend)(arrays, potential_energies, virials, params[:qc], params[:qs], α; ndrange = N)
        add_reduced_ev!(atoms, sum(potential_energies),
                        accumulate(+, virials, dims = 1)[end, :])
    else
        add_reduced_ev!(atoms, zero(FPPREC), zeros(FPPREC, 6))
    end

    # Charged-cell correction (Issue #15) — see chargedcell_correction!.
    # Both the core and the shell charges count toward the cell total. Q is
    # an owned partial sum; the correction itself is replicated (depends
    # only on the global Q and V) and is added directly on every rank.
    qc = params[:qc]; qs = params[:qs]
    Qlocal = N == 0 ? zero(FPPREC) : FPPREC(sum(i -> qc[i] + qs[i], atoms.arrays[:elemid]))
    Q = Cores.allreduce_sum(dom, Qlocal)
    chargedcell_correction!(atoms, Q, α)
end

@kernel function kernel_subtractintraatom!(arrays::NamedTuple, potential_energies, virials, qc, qs, alpha)
    I = @index(Global)
    rx = -arrays[:shelldisp][I,1]
    ry = -arrays[:shelldisp][I,2]
    rz = -arrays[:shelldisp][I,3]
    dr = sqrt(rx*rx + ry*ry + rz*rz)

    qc_i = qc[arrays[:elemid][I]]
    qs_i = qs[arrays[:elemid][I]]
    if dr < FPPREC(1e-1)/alpha
        potential_energies[I] -= qc_i*qs_i * 2/sqrt(FPPREC(pi)) * (alpha - alpha^3/3 * dr^2 + alpha^5/10*dr^4)
        factor = qc_i*qs_i * (-4*alpha^3/(3*sqrt(FPPREC(pi))) + 4*alpha^5/(5*sqrt(FPPREC(pi))) *dr^2 )
    else
        potential_energies[I] -= qc_i*qs_i * erf(dr*alpha)/dr
        factor = qc_i*qs_i * (-erf(dr*alpha)/dr + 2alpha * exp(-(dr*alpha)^2) / sqrt(FPPREC(pi)))/dr^2
    end
    arrays[:forces][I,1] += factor * rx
    arrays[:forces][I,2] += factor * ry
    arrays[:forces][I,3] += factor * rz
    arrays[:shellforces][I,1] -= factor * rx
    arrays[:shellforces][I,2] -= factor * ry
    arrays[:shellforces][I,3] -= factor * rz
    virials[I,1] += factor * rx*rx
    virials[I,2] += factor * ry*ry
    virials[I,3] += factor * rz*rz
    virials[I,4] += factor * ry*rz
    virials[I,5] += factor * rz*rx
    virials[I,6] += factor * rx*ry
end
