using LinearAlgebra
using KernelAbstractions
import AcceleratedKernels as AK
using LoopVectorization
using ..FMD: Atoms, FPPREC
using ..FMD.Cores
using StaticArrays

export calculate!, CoulombBuckingham, OnlyShellForce

"""
    CoulombBuckingham

Real-space short-range part of the Coulomb interaction (smeared Gaussian
charges, complementary to the k-space [`EwaldInfo`](@ref) / [`PMEInfo`](@ref)
contributions) plus the Buckingham repulsion / dispersion. Dispatched
through [`calculate!`](@ref).

Required `params` keys: `:qc`, `:qs`, `:eta_c`, `:eta_s`, `:a`, `:rho`,
`:b`, `:c`, `:e` (Buckingham smooth cap), `:alpha` (Ewald α — pass
[`ewaldcoeff`](@ref)`(params)` if not already in `params`).
"""
struct CoulombBuckingham end

"""
    OnlyShellForce

Marker type passed alongside [`CoulombBuckingham`](@ref) to request that a
real-space pass evaluates *only* the shell-displacement gradient — used
inside the shell CG inner loop where core-position forces and the virial
are not needed yet.
"""
struct OnlyShellForce end

const p1 = 0.406742016006509f0
const p2 = 0.0072279182302319f0
const A = (
    0.316879890481381f0,
    -0.138329314150635f0,
    1.08680830347054f0,
    -1.11694155120396f0,
    1.20644903073232f0,
    -0.393127715207728f0,
    0.0382613542530727f0
)

@inline @fastmath function erfc_approx(x)
    t = 1 / (1 + p1*x + p2*x*x)

    return t * evalpoly(t, A) * exp(-x*x)
end

@inline @fastmath function ϕC(r, alpha, ηi, ηj)
    βi = 1/(2*ηi^2)
    βj = 1/(2*ηj^2)
    β = sqrt(βi*βj/(βi+βj))
    return erfc_approx(alpha*r)/r - erfc_approx(β*r)/r
end

@inline @fastmath function dϕC(r, alpha, ηi, ηj)
    βi = 1/(2*ηi^2)
    βj = 1/(2*ηj^2)
    β = sqrt(βi*βj/(βi+βj))
    return -erfc_approx(alpha*r)/r^2 - 2*alpha/sqrt(FPPREC(π))*exp(-(alpha*r)^2)/r - (-erfc_approx(β*r)/r^2 - 2β/sqrt(FPPREC(π))*exp(-(β*r)^2)/r)
end

@inline @fastmath function ϕB(r, a, b, rho, c, e)
    return a * (1 + b*r) * exp(-r/rho) - c / (r^6 + e/r^6)
end

@inline @fastmath function dϕB(r, a, b, rho, c, e)
    return (-a/rho) * (1 + b*r) * exp(-r/rho) + a * b * exp(-r/rho) + 6*c/(r^6 + e/r^6)^2*(r^5-e/r^7)
end

# ---------------------------------------------------------------------------
# Second derivatives for the analytic HVP (Issue #147). These are the exact
# r-derivatives of the *implemented* first-derivative kernels above — not of
# the mathematical erfc: dϕC substitutes the analytic erfc′ = -2/√π·exp(-x²)
# into the derivative formula while erfc_approx supplies the value, and d2ϕC
# differentiates that mixed expression as written, so FD(dϕC) ↔ d2ϕC
# agreement is limited only by FD truncation, never by the ~1e-7 rational-
# approximation error. The shifted-force scheme needs no rc constants here:
# ϕ_eff′(r) = ϕ′(r) − ϕ′(rc) has ϕ_eff″ = ϕ″.
# Derivation: docs/src/theory/coreshell-analytic-hvp.md §2-3.
# ---------------------------------------------------------------------------

# evalpoly(t, dA) == d/dt evalpoly(t, A). The products are formed in
# FPPREC: at run time evalpoly promotes each Float32 entry of A
# individually, so a Float32 product here would bake in a ~1e-7
# coefficient rounding that the value path never sees.
const dA = ntuple(i -> FPPREC(i) * FPPREC(A[i+1]), length(A) - 1)

@inline @fastmath function derfc_approx(x)
    t = 1 / (1 + p1*x + p2*x*x)
    dt = -(p1 + 2*p2*x) * t * t
    P = evalpoly(t, A)
    dP = evalpoly(t, dA)
    return exp(-x*x) * (dt * (P + t*dP) - 2*x*t*P)
end

@inline @fastmath function d2ϕC(r, alpha, ηi, ηj)
    βi = 1/(2*ηi^2)
    βj = 1/(2*ηj^2)
    β = sqrt(βi*βj/(βi+βj))
    tsp = 2/sqrt(FPPREC(π))
    ga = -alpha*derfc_approx(alpha*r)/r^2 + 2*erfc_approx(alpha*r)/r^3 +
         tsp*alpha*exp(-(alpha*r)^2)*(2*alpha^2 + 1/r^2)
    gb = -β*derfc_approx(β*r)/r^2 + 2*erfc_approx(β*r)/r^3 +
         tsp*β*exp(-(β*r)^2)*(2*β^2 + 1/r^2)
    return ga - gb
end

@inline @fastmath function d2ϕB(r, a, b, rho, c, e)
    ex = exp(-r/rho)
    D = r^6 + e/r^6
    dD6 = r^5 - e/r^7           # D′/6
    return (a/rho^2) * (1 + b*r) * ex - 2*a*b/rho * ex +
           6*c*(5*r^4 + 7*e/r^8)/D^2 - 72*c*dD6^2/D^3
end

# ============================================================================
# Real-space kernels (ghost format, #57): read the extended arrays
# (xcart / sdisp_ext / elemid_ext) of nlist.ghosts, so pair vectors are
# plain Cartesian differences — no shift columns, no per-pair
# fractional→Cartesian matvec. All three kernels are pure gather (writes at
# owned row I only), so no fold-back is needed.
# ============================================================================

@kernel inbounds = true function kernel_gcb!(::CoulombBuckingham,
    potential_energies, forces, shell_forces, virials,
    @Const(xcart), @Const(sdisp), @Const(elemid),
    A, B, rho, C, e, qc, qs,
    ηc, ηs, alpha, rc, @Const(nlist))

    I = @index(Global)

    sᵢ = elemid[I]
    rcore_i = xcart[I,1], xcart[I,2], xcart[I,3]
    rshell_i = sdisp[I,1], sdisp[I,2], sdisp[I,3]
    Uᵢ = potential_energies[I]
    Fcᵢ₁, Fcᵢ₂, Fcᵢ₃ = forces[I,1], forces[I,2], forces[I,3]
    Fsᵢ₁, Fsᵢ₂, Fsᵢ₃ = shell_forces[I,1], shell_forces[I,2], shell_forces[I,3]
    vᵢ₁, vᵢ₂, vᵢ₃, vᵢ₄, vᵢ₅, vᵢ₆ = virials[I,1], virials[I,2], virials[I,3], virials[I,4], virials[I,5], virials[I,6]
    qc_i, qs_i = qc[sᵢ], qs[sᵢ]
    ηc_i, ηs_i = ηc[sᵢ], ηs[sᵢ]
    for jlist in axes(nlist,2)
        j = nlist[I,jlist]
        if j < 1
            break
        end

        sⱼ = elemid[j]
        rx_cc = rcore_i[1] - xcart[j,1]
        ry_cc = rcore_i[2] - xcart[j,2]
        rz_cc = rcore_i[3] - xcart[j,3]
        qc_j = qc[sⱼ]
        qs_j = qs[sⱼ]
        ηc_j = ηc[sⱼ]
        ηs_j = ηs[sⱼ]

        # core-core interaction-------------------------------------
        rx = rx_cc
        ry = ry_cc
        rz = rz_cc
        r = sqrt(rx*rx + ry*ry + rz*rz)
        ϕ = qc_i*qc_j*(ϕC(r, alpha, ηc_i, ηc_j) - ϕC(rc, alpha, ηc_i, ηc_j) - (r-rc)*dϕC(rc, alpha, ηc_i, ηc_j))
        dϕdr_r = qc_i*qc_j*(dϕC(r, alpha, ηc_i, ηc_j) - dϕC(rc, alpha, ηc_i, ηc_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        ϕ *= cutoff
        dϕdr_r *= cutoff
        Uᵢ += FPPREC(0.5)*ϕ
        Fcᵢ₁ -= dϕdr_r * rx
        Fcᵢ₂ -= dϕdr_r * ry
        Fcᵢ₃ -= dϕdr_r * rz
        vᵢ₁ -= FPPREC(0.5)*dϕdr_r * rx * rx
        vᵢ₂ -= FPPREC(0.5)*dϕdr_r * ry * ry
        vᵢ₃ -= FPPREC(0.5)*dϕdr_r * rz * rz
        vᵢ₄ -= FPPREC(0.5)*dϕdr_r * ry * rz
        vᵢ₅ -= FPPREC(0.5)*dϕdr_r * rx * rz
        vᵢ₆ -= FPPREC(0.5)*dϕdr_r * rx * ry

        # core-shell interaction------------------------------------
        rx = rx_cc - sdisp[j,1]
        ry = ry_cc - sdisp[j,2]
        rz = rz_cc - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        ϕ = qc_i*qs_j*(ϕC(r, alpha, ηc_i, ηs_j) - ϕC(rc, alpha, ηc_i, ηs_j) - (r-rc)*dϕC(rc, alpha, ηc_i, ηs_j))
        dϕdr_r = qc_i*qs_j*(dϕC(r, alpha, ηc_i, ηs_j) - dϕC(rc, alpha, ηc_i, ηs_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        ϕ *= cutoff
        dϕdr_r *= cutoff
        Uᵢ += FPPREC(0.5)*ϕ
        Fcᵢ₁ -= dϕdr_r * rx
        Fcᵢ₂ -= dϕdr_r * ry
        Fcᵢ₃ -= dϕdr_r * rz
        vᵢ₁ -= FPPREC(0.5)*dϕdr_r * rx * rx
        vᵢ₂ -= FPPREC(0.5)*dϕdr_r * ry * ry
        vᵢ₃ -= FPPREC(0.5)*dϕdr_r * rz * rz
        vᵢ₄ -= FPPREC(0.5)*dϕdr_r * ry * rz
        vᵢ₅ -= FPPREC(0.5)*dϕdr_r * rx * rz
        vᵢ₆ -= FPPREC(0.5)*dϕdr_r * rx * ry

        # shell-core interaction------------------------------------
        rx = rx_cc + rshell_i[1]
        ry = ry_cc + rshell_i[2]
        rz = rz_cc + rshell_i[3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        ϕ = qs_i*qc_j*(ϕC(r, alpha, ηs_i, ηc_j) - ϕC(rc, alpha, ηs_i, ηc_j) - (r-rc)*dϕC(rc, alpha, ηs_i, ηc_j))
        dϕdr_r = qs_i*qc_j*(dϕC(r, alpha, ηs_i, ηc_j) - dϕC(rc, alpha, ηs_i, ηc_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        ϕ *= cutoff
        dϕdr_r *= cutoff
        Uᵢ += FPPREC(0.5)*ϕ
        Fsᵢ₁ -= dϕdr_r * rx
        Fsᵢ₂ -= dϕdr_r * ry
        Fsᵢ₃ -= dϕdr_r * rz
        vᵢ₁ -= FPPREC(0.5)*dϕdr_r * rx * rx
        vᵢ₂ -= FPPREC(0.5)*dϕdr_r * ry * ry
        vᵢ₃ -= FPPREC(0.5)*dϕdr_r * rz * rz
        vᵢ₄ -= FPPREC(0.5)*dϕdr_r * ry * rz
        vᵢ₅ -= FPPREC(0.5)*dϕdr_r * rx * rz
        vᵢ₆ -= FPPREC(0.5)*dϕdr_r * rx * ry

        # shell-shell interaction-----------------------------------
        Aᵢⱼ = A[sᵢ,sⱼ]
        Bᵢⱼ = B[sᵢ,sⱼ]
        ρᵢⱼ = rho[sᵢ,sⱼ]
        Cᵢⱼ = C[sᵢ,sⱼ]
        rx = rx_cc + rshell_i[1] - sdisp[j,1]
        ry = ry_cc + rshell_i[2] - sdisp[j,2]
        rz = rz_cc + rshell_i[3] - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        ϕ = ϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - ϕB(rc, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - (r-rc)*dϕB(rc, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e)
        dϕdr_r = (dϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - dϕB(rc, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e))/r
        ϕ += qs_i*qs_j*(ϕC(r, alpha, ηs_i, ηs_j) - ϕC(rc, alpha, ηs_i, ηs_j) - (r-rc)*dϕC(rc, alpha, ηs_i, ηs_j))
        dϕdr_r += qs_i*qs_j*(dϕC(r, alpha, ηs_i, ηs_j) - dϕC(rc, alpha, ηs_i, ηs_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        ϕ *= cutoff
        dϕdr_r *= cutoff
        Uᵢ += FPPREC(0.5)*ϕ
        Fsᵢ₁ -= dϕdr_r * rx
        Fsᵢ₂ -= dϕdr_r * ry
        Fsᵢ₃ -= dϕdr_r * rz
        vᵢ₁ -= FPPREC(0.5)*dϕdr_r * rx * rx
        vᵢ₂ -= FPPREC(0.5)*dϕdr_r * ry * ry
        vᵢ₃ -= FPPREC(0.5)*dϕdr_r * rz * rz
        vᵢ₄ -= FPPREC(0.5)*dϕdr_r * ry * rz
        vᵢ₅ -= FPPREC(0.5)*dϕdr_r * rx * rz
        vᵢ₆ -= FPPREC(0.5)*dϕdr_r * rx * ry
    end
    Uᵢ -= alpha /sqrt(FPPREC(π)) * (qc_i^2+qs_i^2)
    potential_energies[I] = Uᵢ
    forces[I,1] = Fcᵢ₁
    forces[I,2] = Fcᵢ₂
    forces[I,3] = Fcᵢ₃
    shell_forces[I,1] = Fsᵢ₁
    shell_forces[I,2] = Fsᵢ₂
    shell_forces[I,3] = Fsᵢ₃
    virials[I,1] = vᵢ₁
    virials[I,2] = vᵢ₂
    virials[I,3] = vᵢ₃
    virials[I,4] = vᵢ₄
    virials[I,5] = vᵢ₅
    virials[I,6] = vᵢ₆
end

@kernel inbounds = true function kernel_gcb_onlyshell!(::CoulombBuckingham,
    shell_forces,
    @Const(xcart), @Const(sdisp), @Const(elemid),
    @Const(A), @Const(B), @Const(rho), @Const(C), @Const(e), @Const(qc), @Const(qs),
    @Const(ηc), @Const(ηs), @Const(alpha), @Const(rc), @Const(nlist))

    I = @index(Global)

    sᵢ = elemid[I]
    rcore_i = xcart[I,1], xcart[I,2], xcart[I,3]
    rshell_i = sdisp[I,1], sdisp[I,2], sdisp[I,3]
    Fsᵢ₁, Fsᵢ₂, Fsᵢ₃ = shell_forces[I,1], shell_forces[I,2], shell_forces[I,3]
    qs_i = qs[sᵢ]
    ηs_i = ηs[sᵢ]
    for jlist in axes(nlist,2)
        j = nlist[I,jlist]
        if j < 1
            break
        end

        sⱼ = elemid[j]
        rx_cc = rcore_i[1] - xcart[j,1]
        ry_cc = rcore_i[2] - xcart[j,2]
        rz_cc = rcore_i[3] - xcart[j,3]
        qc_j = qc[sⱼ]
        qs_j = qs[sⱼ]
        ηc_j = ηc[sⱼ]
        ηs_j = ηs[sⱼ]

        # shell-core interaction------------------------------------
        rx = rx_cc + rshell_i[1]
        ry = ry_cc + rshell_i[2]
        rz = rz_cc + rshell_i[3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        dϕdr_r = qs_i*qc_j*(dϕC(r, alpha, ηs_i, ηc_j) - dϕC(rc, alpha, ηs_i, ηc_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        dϕdr_r *= cutoff
        Fsᵢ₁ -= dϕdr_r * rx
        Fsᵢ₂ -= dϕdr_r * ry
        Fsᵢ₃ -= dϕdr_r * rz

        # shell-shell interaction-----------------------------------
        Aᵢⱼ = A[sᵢ,sⱼ]
        Bᵢⱼ = B[sᵢ,sⱼ]
        ρᵢⱼ = rho[sᵢ,sⱼ]
        Cᵢⱼ = C[sᵢ,sⱼ]
        rx = rx_cc + rshell_i[1] - sdisp[j,1]
        ry = ry_cc + rshell_i[2] - sdisp[j,2]
        rz = rz_cc + rshell_i[3] - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        dϕdr_r = (dϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - dϕB(rc, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e))/r
        dϕdr_r += qs_i*qs_j*(dϕC(r, alpha, ηs_i, ηs_j) - dϕC(rc, alpha, ηs_i, ηs_j))/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        dϕdr_r *= cutoff
        Fsᵢ₁ -= dϕdr_r * rx
        Fsᵢ₂ -= dϕdr_r * ry
        Fsᵢ₃ -= dϕdr_r * rz
    end
    shell_forces[I,1] = Fsᵢ₁
    shell_forces[I,2] = Fsᵢ₂
    shell_forces[I,3] = Fsᵢ₃
end

@kernel inbounds = true function cpukernel_gcb_onlyshell(::CoulombBuckingham,
    shell_forces,
    xcart, sdisp, elemid,
    A, B, rho, C, e, qc, qs,
    ηc, ηs, alpha, rc,
    dϕrc_sc, dϕrc_ss,
    nlist, nneigh)

    I = @index(Global)

    sᵢ = elemid[I]
    rcore_i = xcart[I,1], xcart[I,2], xcart[I,3]
    rshell_i = sdisp[I,1], sdisp[I,2], sdisp[I,3]
    Fsᵢ₁, Fsᵢ₂, Fsᵢ₃ = shell_forces[I,1], shell_forces[I,2], shell_forces[I,3]
    qs_i = qs[sᵢ]
    ηs_i = ηs[sᵢ]

    # @turbo pre-load guard: skip the loop entirely when the list is
    # empty to avoid SIMD pre-loads past the bound.
    if nneigh[I] > 0 && size(nlist, 2) > 0

    @turbo for jlist in 1:nneigh[I]
        j = nlist[I,jlist]

        sⱼ = elemid[j]
        rx_cc = rcore_i[1] - xcart[j,1]
        ry_cc = rcore_i[2] - xcart[j,2]
        rz_cc = rcore_i[3] - xcart[j,3]
        qc_j = qc[sⱼ]
        qs_j = qs[sⱼ]
        ηc_j = ηc[sⱼ]
        ηs_j = ηs[sⱼ]

        # shell-core interaction------------------------------------
        rx = rx_cc + rshell_i[1]
        ry = ry_cc + rshell_i[2]
        rz = rz_cc + rshell_i[3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        dϕdr_r = (qs_i*qc_j*dϕC(r, alpha, ηs_i, ηc_j) - dϕrc_sc[sᵢ,sⱼ])/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        dϕdr_r *= cutoff
        fsc1 = -dϕdr_r * rx
        fsc2 = -dϕdr_r * ry
        fsc3 = -dϕdr_r * rz

        # shell-shell interaction-----------------------------------
        Aᵢⱼ = A[sᵢ,sⱼ]
        Bᵢⱼ = B[sᵢ,sⱼ]
        ρᵢⱼ = rho[sᵢ,sⱼ]
        Cᵢⱼ = C[sᵢ,sⱼ]
        rx = rx_cc + rshell_i[1] - sdisp[j,1]
        ry = ry_cc + rshell_i[2] - sdisp[j,2]
        rz = rz_cc + rshell_i[3] - sdisp[j,3]
        r = sqrt(rx*rx + ry*ry + rz*rz)
        dϕdr_r = (qs_i*qs_j*dϕC(r, alpha, ηs_i, ηs_j) + dϕB(r, Aᵢⱼ, Bᵢⱼ, ρᵢⱼ, Cᵢⱼ, e) - dϕrc_ss[sᵢ,sⱼ])/r
        cutoff = ifelse((r > rc), zero(rc), one(rc))
        dϕdr_r *= cutoff
        fss1 = -dϕdr_r * rx
        fss2 = -dϕdr_r * ry
        fss3 = -dϕdr_r * rz

        # Reductions is limited to once in @turbo loops
        Fsᵢ₁ += (fsc1 + fss1)
        Fsᵢ₂ += (fsc2 + fss2)
        Fsᵢ₃ += (fsc3 + fss3)
    end

    end  # if nneigh[I] > 0 && size(nlist, 2) > 0
    shell_forces[I,1] = Fsᵢ₁
    shell_forces[I,2] = Fsᵢ₂
    shell_forces[I,3] = Fsᵢ₃
end

"""
    calculate!(::CoulombBuckingham, atoms::Atoms, nlist::NeighborList, prms::NamedTuple)

Real-space Coulomb (Gaussian smearing) + Buckingham evaluation over a
ghost-format neighbour list, reading the extended arrays of
`nlist.ghosts` (Cartesian positions with image shifts applied). Syncs
the ghosts itself at entry: **as long as the list is still valid under the
skin criterion** ([`isreusable`](@ref)), positions / shells / lattice may
have changed freely since the list was built. Beyond the skin budget the
list itself is stale (silent pair loss) — the caller must rebuild.
"""
function calculate!(::CoulombBuckingham, atoms::Atoms, nlist::NeighborList, prms::NamedTuple)
    backend = get_backend(atoms)
    arr = atoms.arrays
    sync!(nlist, atoms)
    g = nlist.ghosts

    potential_energies = KernelAbstractions.zeros(backend, FPPREC, length(atoms))
    virials = KernelAbstractions.zeros(backend, FPPREC, length(atoms),6)

    kernel_gcb!(backend, 512, length(atoms))(CoulombBuckingham(),
        potential_energies, arr[:forces], arr[:shellforces], virials,
        g.xcart, g.sdisp_ext, g.elemid_ext,
        prms[:a], prms[:b], prms[:rho], prms[:c], prms[:e], prms[:qc],
        prms[:qs], prms[:eta_c], prms[:eta_s], prms[:alpha], prms[:rc],
        nlist.list
    )

    add_reduced_ev!(atoms, AK.sum(potential_energies), AK.sum(virials, dims = 1)[1, :])

    if haskey(atoms.arrays, :peratom_virial)
        atoms.arrays[:peratom_virial] .+= virials
    end
end

function calculate!(::CoulombBuckingham, ::OnlyShellForce, backend::KernelAbstractions.Backend, atoms::Atoms, nlist::NeighborList, prms::NamedTuple)
    backend = get_backend(atoms)
    arr = atoms.arrays
    sync!(nlist, atoms)
    g = nlist.ghosts

    kernel_gcb_onlyshell!(backend, 512, length(atoms))(CoulombBuckingham(),
        arr[:shellforces],
        g.xcart, g.sdisp_ext, g.elemid_ext,
        prms[:a], prms[:b], prms[:rho], prms[:c], prms[:e], prms[:qc],
        prms[:qs], prms[:eta_c], prms[:eta_s], prms[:alpha], prms[:rc],
        nlist.list
    )
end

function calculate!(::CoulombBuckingham, ::OnlyShellForce, backend::KernelAbstractions.CPU, atoms::Atoms, nlist::NeighborList, prms::NamedTuple)
    arr = atoms.arrays
    sync!(nlist, atoms)
    g = nlist.ghosts

    nneigh = zeros(Int, length(atoms))
    @turbo for i in 1:length(atoms)
        for jlist in axes(nlist.list,2)
            j = nlist.list[i,jlist]
            nneigh[i] += ifelse(j > 0, 1, 0)
        end
    end

    dϕrc_sc = zeros(FPPREC, size(prms[:a]))
    dϕrc_ss = zeros(FPPREC, size(prms[:a]))
    for si in axes(dϕrc_sc,2), sj in axes(dϕrc_sc,1)
        dϕrc_sc[si,sj] = prms[:qs][si]*prms[:qc][sj]*dϕC(prms[:rc], prms[:alpha], prms[:eta_s][si], prms[:eta_c][sj])
        dϕrc_ss[si,sj] = prms[:qs][si]*prms[:qs][sj]*dϕC(prms[:rc], prms[:alpha], prms[:eta_s][si], prms[:eta_s][sj]) + dϕB(prms[:rc], prms[:a][si,sj], prms[:b][si,sj], prms[:rho][si,sj], prms[:c][si,sj], prms[:e])
    end
    dϕrc_sc = SMatrix{size(dϕrc_sc)...}(dϕrc_sc)
    dϕrc_ss = SMatrix{size(dϕrc_sc)...}(dϕrc_ss)

    kernel = cpukernel_gcb_onlyshell(backend, 512, length(atoms))

    kernel(CoulombBuckingham(),
        arr[:shellforces],
        g.xcart, g.sdisp_ext, g.elemid_ext,
        prms[:a], prms[:b], prms[:rho], prms[:c], prms[:e], prms[:qc],
        prms[:qs], prms[:eta_c], prms[:eta_s], prms[:alpha], prms[:rc],
        dϕrc_sc, dϕrc_ss,
        nlist.list, nneigh
    )
end
