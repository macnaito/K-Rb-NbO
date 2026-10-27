# bench/check_xl_spectrum.jl — the spectrum that sets `xl_kernel_spectrum`
# (Issue #152; results recorded in docs/src/theory/xlbomd-shell.md §6.1).
#
# The auxiliary Verlet of shadow XL-BOMD sees the operator
#
#     M = A⁻¹H_ss        (A = spring Hessian, H_ss = -J the full one)
#
# whose spectrum decides both the stability bound (κ·λmax < 4 for the
# default :spring kernel) and the interval a Chebyshev kernel must be
# declared on. This script measures it along a finite-temperature
# trajectory — the spread ALONG the run is the reason a κ frozen at warmup
# is not safe for a non-equilibrium run, and the reason `b` wants a margin.
#
# Measured by Lanczos with full reorthogonalization on the symmetric
# similar operator B = A^{-1/2}H_ss A^{-1/2}; `k2 = 0` species carry no
# shell degree of freedom and are projected out (leaving them in with the
# unit diagonal `hvp_workspace` substitutes injects a spurious eigenvalue).
#
# Run:  JULIA_LOAD_PATH=... julia --project bench/check_xl_spectrum.jl
#
# To measure another potential, replace `_base` / `_cell` below with that
# potential's params and a representative cell.
using Random
using LinearAlgebra
using Printf
using Statistics
using FMD
using FMD: UNITS, FPPREC
using FMD.Cores
using FMD.Calculators
using FMD.Apps: npt!

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
include(joinpath(@__DIR__, "..", "example", "ShellModel-MD", "potential.jl"))

const nelem = 3

_base_raw(atoms) = (; POT...,
    b = zeros(FPPREC, nelem, nelem),
    e = FPPREC(1.0),
    ewald_tolerance = FPPREC(1e-6),
    eta_c = fill(FPPREC(1e-5 / UNITS[:Å]), nelem),
    eta_s = fill(FPPREC(1e-5 / UNITS[:Å]), nelem),
    efield = zeros(FPPREC, 3),
    pme_meshsize = FPPREC(2.0),
    skin = FPPREC(0.2),
    elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3))

_base(atoms) = (; _base_raw(atoms)...,
                alpha = Calculators.ewaldcoeff(_base_raw(atoms)))

function _cell(n; seed = 3, rattle = 0.05)
    Random.seed!(seed)
    a = replicate_atoms(make_perovskite_5(; a_ang = 4.00, A = "Ba", B = "Ti"),
                        (n, n, n))
    a.arrays[:pos] .+= (rattle / (4.0 * n)) .* randn(FPPREC, size(a.arrays[:pos]))
    a.arrays[:shelldisp] .= 0.05 .* randn(FPPREC, size(a.arrays[:shelldisp]))
    return a
end

# Gauge frame for the surface term: the same cell without the rattle.
_proto(n) = replicate_atoms(make_perovskite_5(; a_ang = 4.00, A = "Ba",
                                              B = "Ti"), (n, n, n))

"""
    spectrum(atoms, nl, coulomb, prms; m, seed) -> (lmin, lmax, niter)

Extreme eigenvalues of `A⁻¹H_ss` at the current configuration.
"""
function spectrum(atoms, nl, coulomb, prms; m = 40, seed = 17)
    ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
    k2tbl = Array(prms[:k2])
    k2 = [k2tbl[e] for e in Array(atoms.arrays[:elemid])]
    live = k2 .> 0
    scal = [live[i] ? 1 / sqrt(k2[i]) : zero(FPPREC) for i in eachindex(k2)]

    N = length(atoms)
    Jv = zeros(FPPREC, N, 3)
    tmp = zeros(FPPREC, N, 3)
    function Bmul(v)
        @. tmp = scal * v
        Calculators.hvp!(Jv, nothing, ws.Wscratch, tmp, atoms, ws, prms)
        return @. scal * (-Jv)                       # H_ss = -J
    end

    rng = Random.Xoshiro(seed)
    v = randn(rng, FPPREC, N, 3)
    @. v *= live
    v ./= norm(v)

    V = Matrix{FPPREC}[v]
    α = FPPREC[]
    β = FPPREC[]
    w = Bmul(v)
    push!(α, dot(w, v))
    @. w -= α[end] * v
    for _ in 2:m
        for u in V                                   # full reorthogonalization
            w .-= dot(w, u) .* u
        end
        b = norm(w)
        b < 1e-12 && break
        push!(β, b)
        vnew = w ./ b
        push!(V, vnew)
        w = Bmul(vnew)
        @. w -= β[end] * V[end-1]
        push!(α, dot(w, vnew))
        @. w -= α[end] * vnew
    end
    ev = eigvals(SymTridiagonal(α, β))
    return (lmin = minimum(ev), lmax = maximum(ev), niter = length(α))
end

# --- driver ----------------------------------------------------------------
n = 3
seg = 100
nseg = 8

atoms = _cell(n)
prms = (; _base(atoms)..., cgtol = FPPREC(1e-10), cgmaxiter = 300)
dt = FPPREC(1.0 / UNITS[:fs])
common = (; prms..., backend = :cpu, timestep = dt, nsteps = seg,
          thermostat = :none, barostat = :none,
          target_temperature = 300.0, seed = 7, output_interval = seg)

println("BaTiO3 $(n)^3 = $(length(atoms)) atoms, 1 fs, NVE " *
        "(initial_temperature = 600 K, equipartition halves it)")
println("Lanczos(m = 40) on B = A^{-1/2}H_ss A^{-1/2}, sampled every $seg steps\n")
@printf("%6s %7s %9s %9s %8s %11s %10s %9s\n",
        "step", "T[K]", "lmin", "lmax", "cond", "4/lmax", "lmax+surf",
        "dlam_surf")

results = NamedTuple[]
frame = deepcopy_atoms(atoms)
for s in 1:nseg
    # initial_temperature = 0 leaves :velo untouched, so the segments chain
    # into one trajectory.
    p = (; common..., initial_temperature = (s == 1 ? 600.0 : 0.0))
    T = 0.0
    mktempdir() do dir
        global frame = npt!(frame, (; p..., output_dir = dir))
        rows = [split(l) for l in eachline(joinpath(dir, "log.txt"))]
        T = mean(parse(Float64, r[1]) for r in rows)
    end
    nl = neighborlist(frame, prms[:rc], prms[:skin])
    pme = Calculators.PMEInfo(frame.info[:lattice], prms)
    Calculators.calculate!(Calculators.CoreShell(), frame, nl, pme, prms)
    r = spectrum(frame, nl, pme, prms)
    # Same frame with the depolarization block switched on (Issue #181).
    # A vacuum sphere is the extreme case: T = I/3 is the largest T any
    # (eps_out, shape) produces, and eps_out -> infinity gives T -> 0, the
    # tin-foil boundary the solvers already implement.
    ps = (; prms..., surface_term = Calculators.SurfaceTerm(frame, prms[:qc],
              prms[:qs]; eps_out = 1.0, gauge = _proto(n)))
    rs = spectrum(frame, nl, pme, ps)
    Δ = Calculators.surface_spectral_bump(ps[:surface_term], frame, ps)
    push!(results, (; step = s * seg, T, r..., lmax_s = rs.lmax,
                    lmin_s = rs.lmin, dlam = Δ))
    @printf("%6d %7.1f %9.4f %9.4f %8.1f %11.3f %10.4f %9.4f\n",
            s * seg, T, r.lmin, r.lmax, r.lmax / r.lmin, 4 / r.lmax,
            rs.lmax, Δ)
end

lmaxs = [r.lmax for r in results]
lmins = [r.lmin for r in results]
println()
@printf("lmin : min %.4f  median %.4f  max %.4f\n",
        minimum(lmins), median(lmins), maximum(lmins))
@printf("lmax : min %.4f  median %.4f  max %.4f  (spread %.1f%% along the run)\n",
        minimum(lmaxs), median(lmaxs), maximum(lmaxs),
        100 * (maximum(lmaxs) - minimum(lmaxs)) / median(lmaxs))
@printf("condition number: median %.1f\n", median(lmaxs ./ lmins))
println()
@printf(":spring  stability bound  kappa < 4/lmax = %.3f (worst frame)\n",
        4 / maximum(lmaxs))
@printf(":chebyshev  declare  a <= %.2f  and  b >= %.2f (~2x the worst frame)\n",
        minimum(lmins), 2 * maximum(lmaxs))
# One literal format string: @printf refuses a concatenated one.
@printf("            the shipped default (0.4, 6.0) is this measurement with a %.2fx margin on b\n",
        6.0 / maximum(lmaxs))
println("""
    b carries a margin on purpose: declaring it below the true lmax
    destabilizes the auxiliary field, declaring it too high only weakens
    the preconditioning (theory note §6.1). lmax also moves along the
    run — see the spread above — so the margin has to cover that too.""")

# --- surface term (Issue #181) ---------------------------------------------
lmaxs_s = [r.lmax_s for r in results]
dlams = [r.dlam for r in results]
println()
@printf("with a vacuum-sphere surface term (T = I/3, the extreme case):\n")
@printf("lmax : min %.4f  median %.4f  max %.4f\n",
        minimum(lmaxs_s), median(lmaxs_s), maximum(lmaxs_s))
@printf("measured increase in lmax: median %.4f  max %.4f\n",
        median(lmaxs_s .- lmaxs), maximum(lmaxs_s .- lmaxs))
@printf("closed-form dlam_surf    : median %.4f  max %.4f\n",
        median(dlams), maximum(dlams))
@printf("bracket holds (0 <= increase <= dlam_surf on every frame): %s\n",
        all(0 .<= lmaxs_s .- lmaxs .<= dlams .* (1 + 1e-8)))
@printf(":chebyshev with this term: declare b >= %.2f (~2x the worst frame)\n",
        2 * maximum(lmaxs_s))
println("""
    dlam_surf is what xl_init! refuses against: it is a LOWER bound on the
    full lmax (Weyl), so b < dlam_surf or kappa*dlam_surf >= 4 are wrong
    with certainty.  The measured increase above is the honest number for
    choosing b — it is at most dlam_surf and typically less, because the
    rank-3 block is not aligned with the stiffest existing mode.""")
