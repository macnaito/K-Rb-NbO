# bench/bench_ace.jl
#
# ACE MD-path benchmark: times the production force/energy/virial evaluator
# `calculate!(acep, atoms, nl)` (ACEFast.jl) on synthetic perovskite
# supercells. Self-contained (no dataset files) and BenchmarkTools-free, so
# the same script runs on main and on refactor branches for before/after
# comparison. CPU runs always; GPU runs when `CUDA.functional()`.
#
# Usage (from the repo root):
#   julia --project bench/bench_ace.jl
#
# CPU timings depend on the Julia thread count (KernelAbstractions CPU
# backend); pin it for comparisons, e.g. `julia --project -t 8 ...`.
#
# FP32 vs FP64 (IMPORTANT — measure BOTH):
#   FPPREC is the `[FMD] precision` load-time Preference. Consumer GPUs
#   (e.g. RTX 5070 Ti) run FP64 at 1/64 of FP32 rate, so FP64-only numbers
#   conflate structural speedups with the FP64 ALU ceiling. Switch precision
#   with a Preferences overlay — do NOT edit the repo's LocalPreferences.toml:
#
#     PREFDIR=/tmp/fmd-pref-fp32 && mkdir -p $PREFDIR
#     cat > $PREFDIR/Project.toml <<'EOF'
#     [extras]
#     FMD = "4196dbc5-4e03-43f4-9488-1228e23e1a38"
#     CUDA_Runtime_jll = "76a88914-d11a-5bdc-97e0-2f5a05c973a2"
#     EOF
#     cat > $PREFDIR/LocalPreferences.toml <<'EOF'
#     [FMD]
#     precision = "Float32"
#     EOF
#     # On machines without a local CUDA toolkit (no nvcc), also append:
#     #   [CUDA_Runtime_jll]
#     #   __clear__ = ["version", "local"]
#     JULIA_LOAD_PATH="$PREFDIR:@:@v#.#:@stdlib" julia --project bench/bench_ace.jl
#
#   For the FP64 run on such machines, use an overlay with only the
#   CUDA_Runtime_jll `__clear__` entry (precision stays Float64 from the
#   repo's LocalPreferences.toml).

using FMD
using FMD.Cores
using FMD.Calculators
using KernelAbstractions
using LinearAlgebra
using Random
using Statistics
using Adapt
using CUDA

# ---- configuration (edit here) --------------------------------------------
const NK        = (12, 6, 4, 4)          # (nc0, nc1, nc2, nc3), production-like
const NS        = 3
const RMIN_ANG  = 1.0
const RMAX_ANG  = 6.0
const SKIN      = 0.2                    # neighbor-list skin ratio, as in example/NPT-MD
const NREP_CPU  = [(4, 4, 4), (6, 6, 6)]                  # 320 / 1080 atoms
const NREP_GPU  = [(8, 8, 8), (16, 16, 16), (24, 24, 24)] # 2560 / 20480 / 69120 atoms
const NWARMUP   = 3
const NSAMPLES  = 20
# ---------------------------------------------------------------------------

"""
    make_perovskite_supercell(nrep) -> Atoms{Array}

n×n×n replication of the 5-atom cubic perovskite (SrTiO3-like, a = 4 Å)
used by the ACE tests (test/helpers.jl `make_perovskite_5`), with positions
jittered so forces are non-trivial. Kept inline so the bench is
self-contained.
"""
function make_perovskite_supercell(nrep::NTuple{3,Int})
    a_bohr = 4.0 / FMD.UNITS[:Å]
    base_pos = [
        0.0 0.0 0.0        # Sr
        0.5 0.5 0.5        # Ti
        0.0 0.5 0.5        # O
        0.5 0.0 0.5        # O
        0.5 0.5 0.0        # O
    ]
    base_elemid = [1, 2, 3, 3, 3]

    nx, ny, nz = nrep
    N = 5 * nx * ny * nz
    pos = Matrix{FMD.FPPREC}(undef, N, 3)
    elemid = Vector{Int}(undef, N)
    k = 0
    for ix in 0:(nx-1), iy in 0:(ny-1), iz in 0:(nz-1)
        for i in axes(base_pos, 1)
            k += 1
            pos[k, 1] = (base_pos[i, 1] + ix) / nx
            pos[k, 2] = (base_pos[i, 2] + iy) / ny
            pos[k, 3] = (base_pos[i, 3] + iz) / nz
            elemid[k] = base_elemid[i]
        end
    end
    # Jitter in Float64 then convert, so FP32/FP64 runs see the identical system.
    pos .+= FMD.FPPREC.(0.01 .* randn(N, 3) ./ max(nx, ny, nz))  # break symmetry

    lat = Matrix{FMD.FPPREC}(I, 3, 3)
    lat[1, 1] = nx * a_bohr; lat[2, 2] = ny * a_bohr; lat[3, 3] = nz * a_bohr

    # No :method key on purpose — change_arrtyp would choke on a Symbol array
    # when moving the system to the GPU.
    info = Dict{Symbol,Any}(
        :lattice => lat,
        :potential_energy => FMD.FPPREC[0],
        :virial => zeros(FMD.FPPREC, 6),
    )
    arrays = Dict{Symbol,Array}(
        :pos => pos,
        :elemid => elemid,
        :forces => zeros(FMD.FPPREC, N, 3),
        :shellforces => zeros(FMD.FPPREC, N, 3),
        :shelldisp => zeros(FMD.FPPREC, N, 3),
        :velo => zeros(FMD.FPPREC, N, 3),
        :id => collect(1:N),
        :masses => ones(FMD.FPPREC, N),
    )
    return Atoms{Array}(info, arrays)
end

function make_acep()
    rmin = FMD.FPPREC(RMIN_ANG / FMD.UNITS[:Å])
    rmax = FMD.FPPREC(RMAX_ANG / FMD.UNITS[:Å])
    acep = Calculators.ACEParam(rmin, rmax, NS, NK...)
    for f in fieldnames(typeof(acep))
        x = getfield(acep, f)
        x isa AbstractArray || continue
        # Draw in Float64 then convert: identical coefficients per precision.
        x .= FMD.FPPREC.(randn(size(x)...) .* 1e-3)
    end
    return acep
end

function bench_loop(f!, sync)
    ts = Float64[]
    for i in 1:(NWARMUP + NSAMPLES)
        t = @elapsed begin
            f!()
            sync()
        end
        i > NWARMUP && push!(ts, t)
    end
    return (min = minimum(ts), med = median(ts))
end

fmt_ms(t) = string(round(t * 1e3; digits = 3), " ms")

function bench_backend(label, to_backend, sync, gpu::Bool)
    println("\n=== $label  (FPPREC = $(FMD.FPPREC)) ===")
    for nrep in (gpu ? NREP_GPU : NREP_CPU)
        Random.seed!(0)
        acep = make_acep()
        atoms = to_backend(make_perovskite_supercell(nrep))
        # `adapt` preserves FPPREC. NOTE: this differs from production
        # (`_resolve_mlip` uses `cu(acep)`, which silently downcasts Float64
        # coefficients to Float32) — the FP64 GPU rows here are a *consistent*
        # FP64 baseline, not a bit-replica of the current mixed-precision
        # production path.
        acep_b = gpu ? adapt(CuArray, acep) : acep
        cl, atoms, _ = CellList(atoms, acep_b.rmax * (1 + SKIN))
        nl = neighborlist(atoms, cl, acep_b.rmax, SKIN)

        clear!(atoms)
        calculate!(acep_b, atoms, nl)   # compile + sanity
        sync()
        E = Array(atoms.info[:potential_energy])[1]
        @assert isfinite(E) "non-finite energy E = $E (Nat = $(length(atoms)))"

        alloclabel = gpu ? "dev-alloc/step " : "host-alloc/step "
        r = bench_loop(() -> (clear!(atoms); calculate!(acep_b, atoms, nl)), sync)
        alloc = gpu ? CUDA.@allocated(begin
                    clear!(atoms); calculate!(acep_b, atoms, nl); sync()
                end) :
                @allocated(begin
                    clear!(atoms); calculate!(acep_b, atoms, nl)
                end)
        println(rpad("Nat = $(length(atoms))", 14),
                "  min ", rpad(fmt_ms(r.min), 12),
                "  median ", rpad(fmt_ms(r.med), 12),
                "  ", alloclabel, round(alloc / 2^20; digits = 1), " MiB",
                "  (E = $E)")

        # Persistent-workspace path (production NPT).
        ws = Calculators.ACEWorkspace(get_backend(atoms), acep_b, length(atoms))
        clear!(atoms)
        calculate!(acep_b, atoms, nl, ws)
        sync()
        rw = bench_loop(() -> (clear!(atoms); calculate!(acep_b, atoms, nl, ws)), sync)
        allocw = gpu ? CUDA.@allocated(begin
                    clear!(atoms); calculate!(acep_b, atoms, nl, ws); sync()
                end) :
                @allocated(begin
                    clear!(atoms); calculate!(acep_b, atoms, nl, ws)
                end)
        println(rpad("  +workspace", 14),
                "  min ", rpad(fmt_ms(rw.min), 12),
                "  median ", rpad(fmt_ms(rw.med), 12),
                "  ", alloclabel, round(allocw / 2^20; digits = 1), " MiB")

        GC.gc()
        gpu && CUDA.reclaim()
    end
end

println("Julia $(VERSION), threads = $(Threads.nthreads()), FPPREC = $(FMD.FPPREC)")
println("ACE: ns = $NS, nk = $NK, rmin/rmax = $RMIN_ANG/$RMAX_ANG Å")

bench_backend("CPU (KernelAbstractions)", identity, () -> nothing, false)

if CUDA.functional()
    println("\nGPU: ", CUDA.name(CUDA.device()))
    bench_backend("GPU (CUDA)", a -> change_arrtyp(a, CuArray), () -> CUDA.synchronize(), true)
else
    println("\nGPU: CUDA not functional — GPU bench skipped ",
            "(on this dev machine, run with the CUDA_Runtime_jll __clear__ overlay; see header)")
end
