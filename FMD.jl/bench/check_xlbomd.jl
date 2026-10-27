# bench/check_xlbomd.jl — shadow XL-BOMD acceptance measurements
# (Issue #149; results recorded in docs/src/theory/xlbomd-shell.md §6 and
# docs/src/parameters.md).
#
# Questions, in the order they matter for an opt-in option:
#   (1) how far do the forces move relative to the relaxed surface — the
#       honest "results can differ" number;
#   (2) does the conserved quantity drift more than the tight-SCF
#       reference over an NVE window;
#   (3) what does it buy in wall-clock at production size;
#   (4) the same as (1) for the virial the barostat is driven by
#       (Issue #151);
#   (5) whether the barostatted invariant is conserved and the cell
#       settles where `:scf` puts it — the dynamical check on (4).
#
# Run:  JULIA_LOAD_PATH=... julia --project bench/check_xlbomd.jl [--gpu]
using Random
using LinearAlgebra
using Printf
using Statistics
using FMD
using FMD: UNITS
using FMD.Cores
using FMD.Calculators
using FMD.Apps: npt!

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
include(joinpath(@__DIR__, "..", "example", "ShellModel-MD", "potential.jl"))

const USE_GPU = "--gpu" in ARGS

nelem = 3
_base(atoms) = (; _base_raw(atoms)...,
    alpha = Calculators.ewaldcoeff(_base_raw(atoms)))

_base_raw(atoms) = (; POT...,
    b = zeros(FMD.FPPREC, nelem, nelem),
    e = FMD.FPPREC(1.0),
    ewald_tolerance = FMD.FPPREC(1e-6),
    eta_c = fill(FMD.FPPREC(1e-5/UNITS[:Å]), nelem),
    eta_s = fill(FMD.FPPREC(1e-5/UNITS[:Å]), nelem),
    efield = zeros(FMD.FPPREC, 3),
    pme_meshsize = FMD.FPPREC(2.0),
    skin = FMD.FPPREC(0.2),
    elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3))

function _cell(n; seed = 3, rattle = 0.05)
    Random.seed!(seed)
    a = replicate_atoms(make_perovskite_5(; a_ang = 4.00, A = "Ba", B = "Ti"),
                        (n, n, n))
    a.arrays[:pos] .+= (rattle / (4.0 * n)) .* randn(FMD.FPPREC, size(a.arrays[:pos]))
    a.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(a.arrays[:shelldisp]))
    return a
end

# --- (1) force deviation of the shadow surface vs the relaxed one -------
#
# Measured ON the shadow trajectory, which is what a user of :xlbomd
# actually integrates: the auxiliary field is carried by its own Verlet
# exactly as in npt!, and at sampled steps the tightly relaxed force is
# computed at the SAME positions for comparison. (Comparing whole
# trajectories would only measure Lyapunov divergence.)
#
# Three metrics, because they weight the modes differently (Issue #152):
# the max-norm force deviation is dominated by the STIFF modes, which the
# default spring kernel already tracks near-optimally, while the
# polarization deviation is carried by the ferroelectric SOFT mode — the
# one a kernel actually improves, and the one ε(ω) is made of.
function force_deviation(n; nsteps = 300, sample = 25, kappa = 0.5,
                         kernel = nothing)
    atoms = _cell(n)
    prms = (; _base(atoms)..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 300)
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    pme = PMEInfo(atoms.info[:lattice], prms)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    κ = FMD.FPPREC(kappa)

    qs = [prms[:qs][e] for e in atoms.arrays[:elemid]]

    Random.seed!(11)
    velo = 3e-4 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    # Seed the auxiliary field from a relaxed state (the warmup npt! does).
    Calculators.calculate!(CoreShell(), atoms, nl, pme, prms)
    shat = copy(atoms.arrays[:shelldisp])
    shatm = copy(shat)
    atoms.arrays[:xl_dir] = zero(atoms.arrays[:shelldisp])
    fmax = Float64[]; frms = Float64[]; dP = Float64[]

    for step in 1:nsteps
        atoms.arrays[:pos] .+= (velo * dt) * (atoms.info[:lattice] \ I)
        if !isreusable(nl, atoms)
            nl = neighborlist(atoms, prms[:rc], prms[:skin])
        end
        # Auxiliary Verlet on the kernel direction of the previous step
        # (= xl_propagate!).
        tmp = 2 .* shat .- shatm .+ κ .* atoms.arrays[:xl_dir]
        shatm = shat
        shat = tmp

        FMD.Apps.xl_shadow_forces!(atoms, nl, pme, prms, shat;
                                   kernel_coefs = kernel)
        F_xl = copy(atoms.arrays[:forces])
        s_xl = copy(atoms.arrays[:shelldisp])
        # Half-step kick/drift is enough to keep the sampling on a
        # physically moving configuration.
        @. velo += F_xl / atoms.arrays[:masses] * dt

        if step % sample == 0
            b = deepcopy_atoms(atoms)
            Calculators.calculate!(CoreShell(), b, nl, pme, prms)
            Fref = b.arrays[:forces]; sref = b.arrays[:shelldisp]
            push!(fmax, maximum(abs, F_xl .- Fref) /
                        max(1e-12, maximum(abs, Fref)))
            push!(frms, norm(F_xl .- Fref) / max(1e-12, norm(Fref)))
            P = vec(sum(qs .* sref, dims = 1))
            push!(dP, norm(vec(sum(qs .* (s_xl .- sref), dims = 1))) /
                      max(1e-12, norm(P)))
        end
    end
    return (fmax = fmax, frms = frms, dP = dP)
end

# --- (2) NVE conserved-quantity drift -----------------------------------
function nve_drift(n, mode; nsteps = 600, warmdiscard = 100)
    atoms = _cell(n)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    common = (; _base(atoms)...,
              backend = USE_GPU ? :cuda : :cpu,
              timestep = dt, nsteps = nsteps,
              thermostat = :none, barostat = :none,
              initial_temperature = 300.0, target_temperature = 300.0,
              seed = 7, output_interval = nsteps)
    extra = mode === :xlbomd ?
        (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300,
           shell_dynamics = :xlbomd, xl_warmup = 3) :
        mode === :xlcheb ?
        (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300,
           shell_dynamics = :xlbomd, xl_warmup = 3, xl_kernel = :chebyshev) :
        mode === :tight ? (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300) :
                          (; cgtol = FMD.FPPREC(2e-5), cgmaxiter = 50)
    mktempdir() do dir
        t = @elapsed npt!(deepcopy_atoms(atoms),
                          (; common..., extra..., output_dir = dir))
        rows = [split(l) for l in eachline(joinpath(dir, "log.txt"))]
        cons = [parse(Float64, r[18]) for r in rows[(warmdiscard+1):end]]
        ncg = [parse(Int, r[end]) for r in rows[(warmdiscard+1):end]]
        ts = collect(1:length(cons)) .* 1e-3          # ps
        tm = mean(ts); cm = mean(cons)
        slope = sum((ts .- tm) .* (cons .- cm)) / sum(abs2, ts .- tm)
        return (drift = slope / length(atoms), ncg = mean(ncg),
                steps_per_s = nsteps / t, T = mean(parse(Float64, r[1]) for r in rows))
    end
end

@printf("backend = %s\n\n", USE_GPU ? "cuda" : "cpu")

println("(1) shadow-vs-relaxed deviation on the shadow trajectory " *
        "(BaTiO3 2x2x2, 300 steps)")
@printf("%-28s %11s %11s %11s %6s\n",
        "kernel", "F max-norm", "F rms", "polarization", "+HVP")
for (label, κ, kc) in
        (("spring (default), κ = 0.5", 0.5, nothing),
         ("chebyshev order 2, κ = 1.8", 1.8, FMD.Apps.cheb_kernel_coefs(2, 0.4, 6.0)),
         ("chebyshev order 3, κ = 1.8", 1.8, FMD.Apps.cheb_kernel_coefs(3, 0.4, 6.0)),
         ("chebyshev order 4, κ = 1.8", 1.8, FMD.Apps.cheb_kernel_coefs(4, 0.4, 6.0)))
    r = force_deviation(2; kappa = κ, kernel = kc)
    @printf("%-28s %11.3e %11.3e %11.3e %6d\n", label,
            median(r.fmax), median(r.frms), median(r.dP),
            kc === nothing ? 0 : length(kc) - 1)
end
println()

println("(2) NVE conserved-quantity drift (BaTiO3 3x3x3, 600 steps, 1 fs)")
@printf("%-10s %14s %10s %10s %10s\n", "mode", "drift[Ha/at/ps]", "ncg", "steps/s", "<T>[K]")
for mode in (:default, :tight, :xlbomd, :xlcheb)
    r = nve_drift(3, mode)
    @printf("%-10s %14.3e %10.2f %10.2f %10.1f\n",
            mode, r.drift, r.ncg, r.steps_per_s, r.T)
end

if USE_GPU
    println("\n(3) production-size throughput (BaTiO3 8x8x8 = 2560 atoms, 200 steps)")
    @printf("%-10s %10s\n", "mode", "steps/s")
    for mode in (:default, :xlbomd, :xlcheb)
        r = nve_drift(8, mode; nsteps = 200, warmdiscard = 50)
        @printf("%-10s %10.2f\n", mode, r.steps_per_s)
    end
end

# --- (4) stress: the shadow virial vs the relaxed one -------------------
#
# Issue #151. Same protocol as (1) — measured ON the shadow trajectory,
# with a tight relaxation at the same positions for reference — but the
# quantity is the virial the barostat is driven by. Two numbers: the full
# Voigt deviation (the shear components are where the shadow surface
# differs most) and the hydrostatic pressure offset in GPa, which is what
# an NPT run converts into a volume.
function stress_deviation(n; nsteps = 300, sample = 25, kappa = 0.5,
                          kernel = nothing)
    atoms = _cell(n)
    prms = (; _base(atoms)..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 300)
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    pme = PMEInfo(atoms.info[:lattice], prms)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    κ = FMD.FPPREC(kappa)
    vol = Cores.volume(atoms)

    Random.seed!(11)
    velo = 3e-4 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    Calculators.calculate!(CoreShell(), atoms, nl, pme, prms)
    shat = copy(atoms.arrays[:shelldisp])
    shatm = copy(shat)
    atoms.arrays[:xl_dir] = zero(atoms.arrays[:shelldisp])
    dW = Float64[]; dWs = Float64[]; dP = Float64[]

    for step in 1:nsteps
        atoms.arrays[:pos] .+= (velo * dt) * (atoms.info[:lattice] \ I)
        if !isreusable(nl, atoms)
            nl = neighborlist(atoms, prms[:rc], prms[:skin])
        end
        tmp = 2 .* shat .- shatm .+ κ .* atoms.arrays[:xl_dir]
        shatm = shat
        shat = tmp

        FMD.Apps.xl_shadow_forces!(atoms, nl, pme, prms, shat;
                                   kernel_coefs = kernel)
        W_xl = copy(Array(atoms.info[:virial]))
        @. velo += atoms.arrays[:forces] / atoms.arrays[:masses] * dt

        if step % sample == 0
            b = deepcopy_atoms(atoms)
            Calculators.calculate!(CoreShell(), b, nl, pme, prms)
            W_rel = Array(b.info[:virial])
            push!(dW, maximum(abs, W_xl .- W_rel) / maximum(abs, W_rel))
            push!(dWs, maximum(abs, W_xl[4:6] .- W_rel[4:6]) /
                       max(maximum(abs, W_rel[4:6]), eps()))
            push!(dP, (sum(W_xl[1:3]) - sum(W_rel[1:3])) / (3vol) * UNITS[:GPa])
        end
    end
    return (dW = dW, dWs = dWs, dP = dP)
end

# --- (5) NPH: the dynamical consistency check ---------------------------
#
# A barostat with no thermostat integrates the MTTK invariant, which
# couples the cell to the virial the same way the nuclei are coupled to
# the force. Its drift is therefore the dynamical counterpart of T1/T9:
# an inconsistent virial pumps the cell and shows up as drift even though
# every individual force is exact. ⟨V⟩ from the same run answers the
# user-facing question — does the cell settle where `:scf` puts it.
function npt_volume(n, mode; nsteps = 600, warmdiscard = 200)
    atoms = _cell(n)
    dt = FMD.FPPREC(1.0 / UNITS[:fs])
    common = (; _base(atoms)...,
              backend = USE_GPU ? :cuda : :cpu,
              timestep = dt, nsteps = nsteps,
              thermostat = :none, barostat = :mttk,
              tau_t = 50dt, tau_p = 500dt,
              target_stress = zeros(FMD.FPPREC, 6),
              initial_temperature = 300.0, target_temperature = 300.0,
              seed = 7, output_interval = nsteps)
    extra = mode === :xlbomd ?
        (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300,
           shell_dynamics = :xlbomd, xl_warmup = 3) :
        mode === :xlcheb ?
        (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300,
           shell_dynamics = :xlbomd, xl_warmup = 3, xl_kernel = :chebyshev) :
        mode === :tight ? (; cgtol = FMD.FPPREC(1e-8), cgmaxiter = 300) :
                          (; cgtol = FMD.FPPREC(2e-5), cgmaxiter = 50)
    mktempdir() do dir
        t = @elapsed npt!(deepcopy_atoms(atoms),
                          (; common..., extra..., output_dir = dir))
        rows = [split(l) for l in eachline(joinpath(dir, "log.txt"))]
        keep = rows[(warmdiscard+1):end]
        vol = [parse(Float64, r[17]) for r in keep]
        cons = [parse(Float64, r[18]) for r in keep]
        T = [parse(Float64, r[1]) for r in keep]
        ts = collect(1:length(cons)) .* 1e-3            # ps
        tm = mean(ts); cm = mean(cons)
        slope = sum((ts .- tm) .* (cons .- cm)) / sum(abs2, ts .- tm)
        return (V = mean(vol), Vsd = std(vol), T = mean(T),
                drift = slope / length(atoms), steps_per_s = nsteps / t)
    end
end

println("\n(4) shadow virial vs the relaxed virial on the shadow trajectory " *
        "(BaTiO3 2x2x2, 300 steps)")
@printf("%-28s %11s %11s %11s\n", "kernel", "Voigt dev", "shear dev", "ΔP [GPa]")
for (label, κ, kc) in
        (("spring (default), κ = 0.5", 0.5, nothing),
         ("chebyshev order 3, κ = 1.8", 1.8,
          FMD.Apps.cheb_kernel_coefs(3, 0.4, 6.0)))
    r = stress_deviation(2; kappa = κ, kernel = kc)
    @printf("%-28s %11.3e %11.3e %+11.4f\n", label,
            median(r.dW), median(r.dWs), mean(r.dP))
end

println("\n(5) NPH with a barostat (BaTiO3 3x3x3, 600 steps, MTTK, no " *
        "thermostat, P = 0; statistics over the last 400)")
@printf("%-10s %14s %10s %10s %14s %10s\n",
        "mode", "<V> [Å³]", "sd", "<T> [K]", "drift[Ha/at/ps]", "steps/s")
for mode in (:default, :tight, :xlbomd, :xlcheb)
    r = npt_volume(3, mode)
    @printf("%-10s %14.3f %10.3f %10.1f %14.3e %10.2f\n",
            mode, r.V, r.Vsd, r.T, r.drift, r.steps_per_s)
end
