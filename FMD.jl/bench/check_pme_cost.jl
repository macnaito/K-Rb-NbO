# bench/check_pme_cost.jl — what the reciprocal path costs per call, and
# what the shell-force-only variant saves (Issue #157; results recorded in
# docs/src/parameters.md and the Issue).
#
# The point of the script is that this cost is invisible in the profile the
# test suite runs. At `precision = "Float64"` the reciprocal path is ~1% of
# an MD step for the BaTiO₃ set below — the rc = 11 Å Buckingham/real-space
# pair loop dominates, and on the GPU double-precision arithmetic runs at
# 1/64 rate, which makes it dominate even harder. In the configuration
# production actually uses (`precision = "Float32"` on the GPU) the real-space
# kernel drops by ~60× while the reciprocal path — latency-bound at these
# grid sizes, not work-bound — barely moves, and becomes 30-40% of a step.
#
# So: run this on a Float32 build to reproduce the recorded numbers. A
# Float64 run is still meaningful (it prints FPPREC and the backend), it
# just measures a regime nobody runs production in.
#
# Switch precision with a LOAD_PATH overlay, never by editing the repo's
# LocalPreferences.toml. An overlay is a directory holding a Project.toml
# whose `[extras]` names FMD (uuid 4196dbc5-4e03-43f4-9488-1228e23e1a38)
# and a LocalPreferences.toml with `[FMD] precision = "Float32"`; put it
# first in JULIA_LOAD_PATH, ahead of the usual entries:
#
#   JULIA_LOAD_PATH="<fp32-overlay>:<mpi-overlay>:@:@v#.#:@stdlib" \
#       julia --project bench/check_pme_cost.jl --gpu
#
# Run:  JULIA_LOAD_PATH=... julia --project bench/check_pme_cost.jl [--gpu]
#
# The `:scf` step is timed from a FIXED off-minimum shell state, restored
# before every repeat, so each repeat runs the same CG iteration count
# (a relaxed fixture converges in zero iterations, which is not an MD step).
# `PERT` sets how far off the minimum; the default lands near the iteration
# counts a production run shows.
using Random
using LinearAlgebra
using Printf
using FMD
using FMD: UNITS, FPPREC
using FMD.Cores
using FMD.Calculators

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
include(joinpath(@__DIR__, "..", "example", "ShellModel-MD", "potential.jl"))

const USE_GPU = "--gpu" in ARGS
USE_GPU && (using CUDA)

nelem = 3
# POT is written in Float64; a Float32 build needs every scalar and array
# in FPPREC or the kernels promote (and `neighborlist` rejects the mix).
_fp(x::AbstractFloat) = FPPREC(x)
_fp(x::AbstractArray{<:AbstractFloat}) = FPPREC.(x)
_fp(x) = x
_conv(nt) = (; (k => _fp(v) for (k, v) in pairs(nt))...)

_base_raw() = _conv((; POT...,
    b = zeros(FPPREC, nelem, nelem),
    e = FPPREC(1.0),
    ewald_tolerance = FPPREC(1e-6),
    eta_c = fill(FPPREC(1e-5 / UNITS[:Å]), nelem),
    eta_s = fill(FPPREC(1e-5 / UNITS[:Å]), nelem),
    efield = zeros(FPPREC, 3),
    pme_meshsize = FPPREC(2.0),
    skin = FPPREC(0.2),
    elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3)))
_base() = (; _base_raw()...,
            alpha = FPPREC(Calculators.ewaldcoeff(_base_raw())))

function _cell(n; seed = 3, rattle = 0.05)
    Random.seed!(seed)
    a = replicate_atoms(make_perovskite_5(; a_ang = 4.00, A = "Ba", B = "Ti"),
                        (n, n, n))
    a.arrays[:pos] .+= (rattle / (4.0 * n)) .* randn(FPPREC, size(a.arrays[:pos]))
    a.arrays[:shelldisp] .= 0.05 .* randn(FPPREC, size(a.arrays[:shelldisp]))
    return a
end

_sync() = USE_GPU ? CUDA.synchronize() : nothing

# Minimum over repeats: the quantity of interest is the cost of the work
# itself, and the minimum is the sample least polluted by other load.
function bench(f, reps = 20)
    f(); _sync()
    t = Inf
    for _ in 1:reps
        s = time_ns(); f(); _sync()
        t = min(t, (time_ns() - s) * 1e-9)
    end
    return t
end

function report(n; pert = FPPREC(parse(Float64, get(ENV, "PERT", "5e-4"))))
    atoms = _cell(n)
    USE_GPU && (atoms = Cores.change_arrtyp(atoms, CuArray))
    prms = (; _base()..., cgtol = FPPREC(2e-5), cgmaxiter = 300)
    USE_GPU && (prms = arrtyp_prms(CuArray, prms))
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    pme = PMEInfo(atoms.info[:lattice], prms)

    Calculators.calculate!(Calculators.CoreShell(), atoms, nl, pme, prms)
    Random.seed!(7)
    p = pert .* randn(FPPREC, size(Array(atoms.arrays[:shelldisp])))
    s0 = copy(atoms.arrays[:shelldisp]) .+ (USE_GPU ? CuArray(p) : p)
    atoms.arrays[:shelldisp] .= s0
    ncg = Calculators.calculate!(Calculators.CoreShell(), atoms, nl, pme, prms)

    t_full  = bench(() -> Calculators.calculate!(pme, atoms, prms))
    t_shell = bench(() -> Calculators.calculate!(pme, Calculators.OnlyShellForce(),
                                                 atoms, prms))
    t_step  = bench(() -> begin
        atoms.arrays[:shelldisp] .= s0
        Calculators.calculate!(Calculators.CoreShell(), atoms, nl, pme, prms)
    end)

    @printf("n=%d  N=%d  grid=%s  FPPREC=%s  backend=%s\n", n, length(atoms),
            string(pme.fftsize), string(FPPREC), USE_GPU ? "cuda" : "cpu")
    @printf("  calculate!(PMEInfo)                  %7.3f ms\n", 1e3t_full)
    @printf("  calculate!(PMEInfo, OnlyShellForce)  %7.3f ms   (%.2fx cheaper)\n",
            1e3t_shell, t_full / t_shell)
    # The CG runs ncg + 1 shell evaluations; the closing core evaluation
    # still takes the full reciprocal call.
    @printf("  :scf step (ncg=%d)                   %7.3f ms  (%.1f steps/s)\n",
            ncg, 1e3t_step, 1 / t_step)
    @printf("    reciprocal share of the step        %5.1f%%  (%d full + %d shell-only calls)\n",
            100 * (t_full + (ncg + 1) * t_shell) / t_step, 1, ncg + 1)
    @printf("    saved by the shell-only variant    %7.3f ms  (would be %.3f ms without)\n",
            1e3 * (ncg + 1) * (t_full - t_shell),
            1e3 * (t_step + (ncg + 1) * (t_full - t_shell)))
    return nothing
end

for n in (3, 8)
    report(n)
end
