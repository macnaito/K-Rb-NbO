# test/gpu/test_ace_gpu.jl
#
# ACE CPU-reference vs GPU (CUDABackend) parity. Until this file, ACE had
# zero GPU-executed test coverage (test_ace_spline_gpu.jl covers Spline
# only), while ACEFast.jl is the production NPT GPU path. Establishes the
# GPU correctness baseline ahead of the ACEFast refactor (persistent
# workspace + 3-stage GEMM split); the reference is the hand-derived CPU
# implementation in ACE.jl — the fitting ground truth.
#
# Two systems on purpose:
#   * 40-atom shuffled supercell — elemid neither species- nor cell-sorted,
#     all four Chebyshev orders distinct (n0 > n1 > n2 > n3), so every
#     truncation branch of the nested basis loops runs.
#   * 5-atom primitive cell with rmax > a — self-image pairs (i == j with
#     shift ≠ 0) enter the list: the code path that atomically adds both +f
#     and −f to the same atom index. Their distance is a lattice-vector
#     length (position-independent), so this requires rmax strictly greater
#     than the cell edge; the CPU parity fixture (rmax = a) never hits it.
#
# ACEParam moves to the device via `adapt(CuArray, ·)`, which preserves
# FPPREC. Production `_resolve_mlip` currently uses `cu`, which silently
# downcasts coefficients to Float32 (Issue #39) — parity against the
# Float64 reference requires the adapt route.

using Test
using CUDA
using Adapt
using Random
using LinearAlgebra

# Smallest pair distance in a neighbor list (host arrays). Guards the
# fixtures against jitter pushing a pair close to rmin, where the basis is
# cut without tapering and a CPU/GPU ulp difference could flip pair
# acceptance.
function _min_pair_distance(nl)
    xc = Array(nl.ghosts.xcart)
    list = Array(nl.list)
    rmin_seen = Inf
    for i in 1:nl.ghosts.n_owned
        for jj in axes(list, 2)
            j = list[i, jj]
            j < 1 && break
            r = sqrt((xc[i, 1] - xc[j, 1])^2 + (xc[i, 2] - xc[j, 2])^2 +
                     (xc[i, 3] - xc[j, 3])^2)
            r > 0 && (rmin_seen = min(rmin_seen, r))
        end
    end
    return rmin_seen
end

function _make_test_acep(rmin, rmax; ns = 3, nk = (6, 5, 4, 3))
    # Default: all four orders distinct so the n0 > n1 > n2 > n3 truncations
    # are hit. ns/nk overridable for the warp-kernel dims sweep below.
    acep = Calculators.ACEParam(FMD.FPPREC(rmin), FMD.FPPREC(rmax),
                                ns, nk...)
    for f in fieldnames(typeof(acep))
        x = getfield(acep, f)
        x isa AbstractArray || continue
        x .= randn(FMD.FPPREC, size(x)) .* 1e-3
    end
    return acep
end

# Shared CPU-reference vs GPU comparison for one configuration.
function _ace_gpu_parity(atoms, acep)
    nl = neighborlist(atoms, acep.rmax, 0.0)
    @test _min_pair_distance(nl) > 1.2 * acep.rmin

    # CPU reference (ACE.jl, ghost raw form + fold).
    gh = nl.ghosts
    Fext = zeros(FMD.FPPREC, length(gh), 3)
    V_ref = zeros(FMD.FPPREC, 6)
    E_ref = Calculators.calculate!(acep, Fext, V_ref, gh, nl.list)
    F_ref = zeros(FMD.FPPREC, length(atoms), 3)
    Cores.fold_ghosts!(F_ref, Fext, gh)
    @test maximum(abs, F_ref) > 1e-3   # non-trivial forces being compared
    @test abs(E_ref) > 1e-6
    @test maximum(abs, V_ref) > 1e-8

    # GPU run (ACEFast KernelAbstractions path on CUDABackend), with the
    # production-path :peratom_virial branch enabled.
    acep_gpu = adapt(CuArray, acep)
    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
    reset!(a_gpu)
    a_gpu.arrays[:peratom_virial] = CUDA.zeros(FMD.FPPREC, length(atoms), 6)
    nl_gpu = neighborlist(a_gpu, acep_gpu.rmax, 0.0)
    Calculators.calculate!(acep_gpu, a_gpu, nl_gpu)
    E_gpu = Array(a_gpu.info[:potential_energy])[1]
    F_gpu = Array(a_gpu.arrays[:forces])
    V_gpu = Array(a_gpu.info[:virial])
    PV_gpu = Array(a_gpu.arrays[:peratom_virial])

    @test isapprox(E_gpu, E_ref; rtol = 1e-6)
    @test isapprox(F_gpu, F_ref; rtol = 1e-6, atol = 1e-10)
    @test isapprox(V_gpu, V_ref; rtol = 1e-6, atol = 1e-10)
    # Per-atom virials must re-sum to the total (production NPT reads them).
    @test isapprox(vec(sum(PV_gpu, dims = 1)), V_ref; rtol = 1e-6, atol = 1e-10)
    # Translation invariance: forces sum to zero.
    @test isapprox(vec(sum(F_gpu, dims = 1)), zeros(FMD.FPPREC, 3); atol = 1e-8)

    # Persistent-workspace path (production NPT): same ws twice — results
    # must match the reference both times (reuse purity on the device).
    ws = Calculators.ACEWorkspace(Cores.get_backend(a_gpu), acep, length(atoms))
    for _ in 1:2
        b_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
        reset!(b_gpu)
        b_gpu.arrays[:peratom_virial] = CUDA.zeros(FMD.FPPREC, length(atoms), 6)
        Calculators.calculate!(acep_gpu, b_gpu, nl_gpu, ws)
        @test isapprox(Array(b_gpu.info[:potential_energy])[1], E_ref; rtol = 1e-6)
        @test isapprox(Array(b_gpu.arrays[:forces]), F_ref; rtol = 1e-6, atol = 1e-10)
        @test isapprox(Array(b_gpu.info[:virial]), V_ref; rtol = 1e-6, atol = 1e-10)
        @test isapprox(vec(sum(Array(b_gpu.arrays[:peratom_virial]), dims = 1)), V_ref;
                       rtol = 1e-6, atol = 1e-10)
    end
    return nl
end

@testset "ACE CPU reference vs GPU parity (E, F, virial)" begin
    rmin = 1.0 / UNITS[:Å]
    rmax = 4.0 / UNITS[:Å]

    @testset "40-atom shuffled supercell" begin
        Random.seed!(41)
        atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (2, 2, 2);
                                shuffle_order = true)
        atoms.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        _ace_gpu_parity(atoms, _make_test_acep(rmin, rmax))
    end

    @testset "5-atom cell (self-image pairs)" begin
        Random.seed!(42)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.025 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        # rmax > a so the position-independent self-image distance (= a)
        # falls inside the strict r < rmax acceptance.
        nl = _ace_gpu_parity(atoms, _make_test_acep(rmin, 4.5 / UNITS[:Å]))
        # The point of this case: self-image pairs must actually be present.
        # In the ghost format a self-image is a ghost row whose OWNER is the
        # atom itself.
        list = Array(nl.list)
        src = Array(nl.ghosts.src)
        n = nl.ghosts.n_owned
        owner(j) = j <= n ? Int(j) : Int(src[j-n])
        has_self = any(owner(list[i, jj]) == i
                       for i in 1:n, jj in axes(list, 2)
                       if list[i, jj] >= 1)
        @test has_self
    end
end

# The CUDA warp-per-atom kernels (ACEFastCUDA.jl, Issue #187) are what
# `calculate!` dispatches to on the GPU, so the parity suite above already
# runs them against the CPU reference. This runs one configuration directly
# against the KernelAbstractions kernels they replaced. The features are
# compared at rtol 1e-12, not `==`: the warp kernel consumes neighbours in
# the same list order with the same expressions (measured bit-identical on
# the dev machine), but FMA fusion of KA-generated vs native CUDA code is a
# compiler detail we do not want the suite to pin across toolchains. The
# forces are compared to summation-order roundoff (the i-side is
# block-reduced).
function _warp_vs_ka_parity(atoms, acep)
    acep_gpu = adapt(CuArray, acep)
    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
    reset!(a_gpu)
    Nat = length(a_gpu)
    nl = neighborlist(a_gpu, acep_gpu.rmax, 0.0)
    backend = Cores.get_backend(a_gpu)
    ws = Calculators.ACEWorkspace(backend, acep, Nat)
    Calculators._refresh_species_order!(ws, a_gpu)
    sync!(nl, a_gpu)
    g = nl.ghosts
    b = ws.buf
    IdL = Calculators.IdLattice()
    ZS = Calculators.ZeroShift()

    # --- KA pipeline, run explicitly on the CUDABackend ---
    Calculators.kernel_ace_features!(backend, 256, Nat)(
        b.S, b.V, b.M, b.N, ws.rowrank, g.xcart, g.elemid_ext, IdL,
        nl.list, ZS, ZS, ZS, ws.rmin, ws.rmax, ws.dims.ns)
    CUDA.synchronize()
    S_ka = Array(b.S); V_ka = Array(b.V); M_ka = Array(b.M); N_ka = Array(b.N)
    Calculators._ace_stageB_generic!(backend, ws)
    fill!(g.fext, zero(FMD.FPPREC)); fill!(g.vext, zero(FMD.FPPREC))
    Calculators.kernel_ace_forces!(backend, 256, Nat)(
        g.fext, g.vext, b.dEdS, b.dEdV, b.dEdM, b.dEdN, ws.rowrank,
        g.xcart, g.elemid_ext, IdL, nl.list, ZS, ZS, ZS,
        ws.rmin, ws.rmax, ws.dims.ns)
    F_ka_d = CUDA.zeros(FMD.FPPREC, Nat, 3)
    PV_ka_d = CUDA.zeros(FMD.FPPREC, Nat, 6)
    Cores.fold_ghosts!(F_ka_d, g.fext, g)
    Cores.fold_ghosts!(PV_ka_d, g.vext, g)
    F_ka = Array(F_ka_d); PV_ka = Array(PV_ka_d)
    @test maximum(abs, F_ka) > 1e-3   # non-trivial comparison

    # --- warp pipeline through the dispatch entry points ---
    fill!(b.S, FMD.FPPREC(NaN)); fill!(b.V, FMD.FPPREC(NaN))
    fill!(b.M, FMD.FPPREC(NaN)); fill!(b.N, FMD.FPPREC(NaN))
    Calculators._ace_features_launch!(backend, ws, g, nl.list)
    CUDA.synchronize()
    # NaN prefill: the warp write-out must cover every slot of every row.
    @test isapprox(Array(b.S), S_ka; rtol = 1e-12, atol = 1e-14, nans = false)
    @test isapprox(Array(b.V), V_ka; rtol = 1e-12, atol = 1e-14, nans = false)
    @test isapprox(Array(b.M), M_ka; rtol = 1e-12, atol = 1e-14, nans = false)
    @test isapprox(Array(b.N), N_ka; rtol = 1e-12, atol = 1e-14, nans = false)

    Calculators._ace_stageB_generic!(backend, ws)
    fill!(g.fext, zero(FMD.FPPREC)); fill!(g.vext, zero(FMD.FPPREC))
    Calculators._ace_forces_launch!(backend, ws, g, nl.list)
    F_w_d = CUDA.zeros(FMD.FPPREC, Nat, 3)
    PV_w_d = CUDA.zeros(FMD.FPPREC, Nat, 6)
    Cores.fold_ghosts!(F_w_d, g.fext, g)
    Cores.fold_ghosts!(PV_w_d, g.vext, g)
    @test isapprox(Array(F_w_d), F_ka; rtol = 1e-10, atol = 1e-13)
    @test isapprox(Array(PV_w_d), PV_ka; rtol = 1e-10, atol = 1e-13)
    return nothing
end

@testset "ACEFast warp kernels vs KA kernels (GPU, Issue #187)" begin
    # Dims sweep over the Val specializations' boundary arithmetic:
    #   ns=3, nk=(6,5,4,3): nslot = 225, MR = 8 (the CPU-parity baseline dims)
    #   ns=1, nk=(12,8,5,3): nslot = 96 — an exact multiple of 32, no dead
    #     lane registers, single species block
    #   ns=2, nk=(5,4,4,3): nslot = 142, MR = 5 with 18 dead registers and an
    #     odd per-species block length
    # elemid is clamped to 1:ns so the reduced-species configs stay valid.
    for (ns, nk) in ((3, (6, 5, 4, 3)), (1, (12, 8, 5, 3)), (2, (5, 4, 4, 3)))
        @testset "ns=$ns nk=$nk" begin
            Random.seed!(43)
            atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (3, 3, 3);
                                    shuffle_order = true)
            atoms.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
            atoms.arrays[:elemid] .= min.(atoms.arrays[:elemid], ns)
            _warp_vs_ka_parity(atoms,
                _make_test_acep(1.0 / UNITS[:Å], 4.5 / UNITS[:Å]; ns, nk))
        end
    end

    # Empty DD rank: the launch guards must no-op on Nat = 0 (a 0-block
    # @cuda launch is invalid). g/list are deliberately `nothing` — the
    # guard must fire before either is touched.
    @testset "empty-rank guard (Nat = 0)" begin
        acep = _make_test_acep(1.0 / UNITS[:Å], 4.0 / UNITS[:Å])
        ws0 = Calculators.ACEWorkspace(CUDA.CUDABackend(), acep, 0)
        @test Calculators._ace_features_launch!(CUDA.CUDABackend(), ws0,
                                                nothing, nothing) === nothing
        @test Calculators._ace_forces_launch!(CUDA.CUDABackend(), ws0,
                                              nothing, nothing) === nothing
    end
end

# Stage B: the CUDA field-fused implementation (Issue #189) against the
# generic KernelAbstractions one, on identical features. B_l and dEdX_l are
# expression- and accumulation-order-identical (compared at fusion-tolerant
# rtol, cf. the features note above); pe changes summation order (S terms in
# a dedicated kernel, field terms block-reduced), so it gets a looser
# roundoff tolerance. NaN prefill proves full output coverage.
function _stageB_fused_vs_generic(atoms, acep)
    acep_gpu = adapt(CuArray, acep)
    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
    reset!(a_gpu)
    Nat = length(a_gpu)
    nl = neighborlist(a_gpu, acep_gpu.rmax, 0.0)
    backend = Cores.get_backend(a_gpu)
    ws = Calculators.ACEWorkspace(backend, acep, Nat)
    Calculators._refresh_species_order!(ws, a_gpu)
    sync!(nl, a_gpu)
    b = ws.buf
    Calculators._ace_features_launch!(backend, ws, nl.ghosts, nl.list)

    Calculators._ace_stageB_generic!(backend, ws)
    CUDA.synchronize()
    outs = (:pe, :dEdS, :dEdV, :dEdM, :dEdN, :B1, :B2, :B3)
    refs = Dict(k => Array(getproperty(b, k)) for k in outs)
    @test maximum(abs, refs[:dEdV]) > 1e-6   # non-trivial comparison

    for k in outs
        fill!(getproperty(b, k), FMD.FPPREC(NaN))
    end
    Calculators._ace_stageB!(backend, ws)    # CUDABackend → field-fused
    CUDA.synchronize()
    for k in (:dEdS, :dEdV, :dEdM, :dEdN, :B1, :B2, :B3)
        @test isapprox(Array(getproperty(b, k)), refs[k];
                       rtol = 1e-12, atol = 1e-14, nans = false)
    end
    @test isapprox(Array(b.pe), refs[:pe]; rtol = 1e-10, atol = 1e-12, nans = false)
    return nothing
end

@testset "ACE stage B fused vs generic (GPU, Issue #189)" begin
    for (ns, nk) in ((3, (6, 5, 4, 3)), (1, (12, 8, 5, 3)), (2, (5, 4, 4, 3)))
        @testset "ns=$ns nk=$nk" begin
            Random.seed!(45)
            atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.0), (3, 3, 3);
                                    shuffle_order = true)
            atoms.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
            atoms.arrays[:elemid] .= min.(atoms.arrays[:elemid], ns)
            _stageB_fused_vs_generic(atoms,
                _make_test_acep(1.0 / UNITS[:Å], 4.5 / UNITS[:Å]; ns, nk))
        end
    end

    # Dims whose field kernel would need > 48 KB of dynamic shared memory
    # (ns=5, nk3=19 ⇒ P3 = 4560, 51.7 KB at Float64) must fall back to the
    # generic stage B — same outputs, one @warn, no error (the model stays
    # runnable, it just misses the fusion speedup).
    @testset "shared-memory over-budget fallback" begin
        Random.seed!(46)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.0125 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        acep = _make_test_acep(1.0 / UNITS[:Å], 4.5 / UNITS[:Å];
                               ns = 5, nk = (19, 19, 19, 19))
        d5 = (; nch3 = 5 * 19, P3 = (5 * 19) * (5 * 19 + 1) ÷ 2)
        @assert (2 * 10 * d5.nch3 + d5.P3 + 1) * sizeof(FMD.FPPREC) > 48 * 1024
        atoms.arrays[:elemid] .= min.(atoms.arrays[:elemid], 5)
        @test_logs (:warn, r"falling back to the generic stage B") match_mode = :any begin
            _stageB_fused_vs_generic(atoms, acep)
        end
    end

    # Empty DD rank: the Nat = 0 guard must fire before any launch.
    @testset "empty-rank guard (Nat = 0)" begin
        acep = _make_test_acep(1.0 / UNITS[:Å], 4.0 / UNITS[:Å])
        ws0 = Calculators.ACEWorkspace(CUDA.CUDABackend(), acep, 0)
        @test Calculators._ace_stageB!(CUDA.CUDABackend(), ws0) === nothing
    end
end

# Degenerate neighbour lists: rmax below the smallest pair distance leaves
# every row empty (maxneigh may even be 0). The warp kernels' parallel
# neighbour count and chunk loop must survive this; energies reduce to the
# per-species constants cp and forces to exactly zero.
@testset "ACEFast warp kernels: empty neighbour lists (GPU)" begin
    Random.seed!(44)
    atoms = make_perovskite_5(; a_ang = 4.0)   # nearest pair ≈ 2 Å
    acep = _make_test_acep(0.3 / UNITS[:Å], 0.9 / UNITS[:Å])
    acep_gpu = adapt(CuArray, acep)
    a_gpu = Cores.change_arrtyp(deepcopy_atoms(atoms), CuArray)
    reset!(a_gpu)
    nl = neighborlist(a_gpu, acep_gpu.rmax, 0.0)
    ws = Calculators.ACEWorkspace(Cores.get_backend(a_gpu), acep, length(a_gpu))
    Calculators.calculate!(acep_gpu, a_gpu, nl, ws)
    E_ref = sum(acep.cp[Array(a_gpu.arrays[:elemid])])
    @test isapprox(Array(a_gpu.info[:potential_energy])[1], E_ref;
                   rtol = 1e-12, atol = 1e-14)
    @test maximum(abs, Array(a_gpu.arrays[:forces])) == 0
end
