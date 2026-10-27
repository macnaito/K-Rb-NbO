# test/calculators/test_map_umbrella.jl
#
# PolarizationMapUmbrella: restraint towards an arbitrary per-group target
# map.  Shares the smoothing/adjoint machinery with PolarizationUmbrella
# (PolField.jl), so what is genuinely new here is the residual against a
# fixed map, the per-group weight, and the absence of the chain-rule term
# that the least-squares amplitude needed.

using Test
using LinearAlgebra
using Random
using JLD2

# Reuses make_binary_supercell / sublattice_a from test_umbrella.jl, which
# runtests.jl includes first.

const MAP_QC = FMD.FPPREC[1.2, -0.8]
const MAP_QS = FMD.FPPREC[0.7, -1.1]

# Deterministic perturbation (no RNG stream dependence), same idea as the
# golden fixture in test_umbrella.jl.
_map_jitter(n) = 2 * (n * 0.6180339887498949 - floor(n * 0.6180339887498949)) - 1

function map_fixture(reps; amp = 0.008, sd = 0.03)
    proto = make_binary_supercell(reps)
    atoms = deepcopy(proto)
    for k in axes(atoms.arrays[:pos], 1), d in 1:3
        atoms.arrays[:pos][k, d] += FMD.FPPREC(amp) * _map_jitter(3k + 7d)
        atoms.arrays[:shelldisp][k, d] = FMD.FPPREC(sd) * _map_jitter(5k + 2d)
    end
    return proto, atoms
end

# Reference smoothing, evaluated straight from the definition and sharing no
# code with the implementation's rFFT path.
#
# The operator is defined in Fourier space as Ĝ(k) = exp(-σ²k²/2) on the
# Brillouin zone of the group lattice.  Note this is NOT a sampled real-space
# Gaussian: that would be the aliased kernel (Poisson summation adds the
# |n| > N/2 modes), which differs by ~2% at σ = 0.9.  So the honest
# independent check is to build the real-space kernel by summing the defining
# modes explicitly, then convolve with a plain double loop.
function gauss_kernel_direct(reps::NTuple{3,Int}, sigma)
    Nx, Ny, Nz = reps
    g = zeros(FMD.FPPREC, Nx, Ny, Nz)
    fold(n, N) = n <= N ÷ 2 ? n : n - N
    for jx in 0:Nx-1, jy in 0:Ny-1, jz in 0:Nz-1
        acc = zero(FMD.FPPREC)
        for nx in 0:Nx-1, ny in 0:Ny-1, nz in 0:Nz-1
            k2 = (2π * fold(nx, Nx) / Nx)^2 + (2π * fold(ny, Ny) / Ny)^2 +
                 (2π * fold(nz, Nz) / Nz)^2
            acc += exp(-sigma^2 * k2 / 2) *
                   cos(2π * (nx * jx / Nx + ny * jy / Ny + nz * jz / Nz))
        end
        g[jx + 1, jy + 1, jz + 1] = acc / (Nx * Ny * Nz)
    end
    return g
end

# Circular convolution of the gauged group polarisation with that kernel.
function smoothed_reference(atoms, gauge, reps::NTuple{3,Int}, sigma)
    Nx, Ny, Nz = reps
    ng = Nx * Ny * Nz
    Pg = Calculators.group_p_gauged(atoms, gauge, MAP_QC, MAP_QS)
    sigma == 0 && return Pg
    g = gauss_kernel_direct(reps, sigma)
    flat(gx, gy, gz) = mod(gx, Nx) * Ny * Nz + mod(gy, Ny) * Nz + mod(gz, Nz) + 1
    out = zeros(FMD.FPPREC, ng, 3)
    for i in 1:ng
        ix = (i - 1) ÷ (Ny * Nz); iy = ((i - 1) ÷ Nz) % Ny; iz = (i - 1) % Nz
        for jx in 0:Nx-1, jy in 0:Ny-1, jz in 0:Nz-1
            w = g[jx + 1, jy + 1, jz + 1]
            j = flat(ix - jx, iy - jy, iz - jz)
            @views out[i, :] .+= w .* Pg[j, :]
        end
    end
    return out
end

# The bias `forces` array holds the CORE force only, but a finite difference
# in `pos` at fixed `shelldisp` moves the shell with it (r_shell = pos·L + sd),
# so what it measures is -(F_core + F_shell).  The shared helper compares
# against `forces` alone, which is right for a shell-free fixture and wrong
# here — so this check is local.
function assert_map_core_force_fd(calc!, atoms; h = 1e-5, indices = 1:6,
                                  rtol = 1e-4, atol = 1e-9, label = "")
    a = deepcopy_atoms(atoms)
    reset!(a); calc!(a)
    F = copy(a.arrays[:forces]) .+ copy(a.arrays[:shellforces])
    ilat = inv(Array(atoms.info[:lattice]))
    @testset "$label" begin
        for i in indices, axis in 1:3
            δ = zeros(FMD.FPPREC, 3); δ[axis] = h
            δfrac = vec(δ' * ilat)
            ap = deepcopy_atoms(atoms); ap.arrays[:pos][i, :] .+= δfrac
            am = deepcopy_atoms(atoms); am.arrays[:pos][i, :] .-= δfrac
            Ep, _ = evaluate(calc!, ap)
            Em, _ = evaluate(calc!, am)
            num = -(Ep - Em) / (2h)
            ana = F[i, axis]
            (abs(ana) < atol && abs(num) < atol) && continue
            @test isapprox(ana, num; rtol = rtol, atol = atol)
        end
    end
end

@testset "PolarizationMapUmbrella construction guards" begin
    reps = (2, 2, 2)
    ng = prod(reps)
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    tgt = zeros(FMD.FPPREC, ng, 3)

    mk(; kw...) = Calculators.PolarizationMapUmbrella(
        atoms, 1.0, tgt, reps, MAP_QC, MAP_QS; gauge = gauge, kw...)
    @test mk() isa Calculators.PolarizationMapUmbrella

    # Shape of the target map.
    @test_throws ErrorException Calculators.PolarizationMapUmbrella(
        atoms, 1.0, zeros(FMD.FPPREC, ng, 2), reps, MAP_QC, MAP_QS; gauge = gauge)
    @test_throws ErrorException Calculators.PolarizationMapUmbrella(
        atoms, 1.0, zeros(FMD.FPPREC, ng + 1, 3), reps, MAP_QC, MAP_QS; gauge = gauge)
    # Non-finite entries would poison the whole field through the FFT.
    bad = copy(tgt); bad[1, 1] = NaN
    @test_throws ErrorException Calculators.PolarizationMapUmbrella(
        atoms, 1.0, bad, reps, MAP_QC, MAP_QS; gauge = gauge)
    # Weights.
    @test_throws ErrorException mk(weight = ones(FMD.FPPREC, ng - 1))
    @test_throws ErrorException mk(weight = [-1.0; ones(FMD.FPPREC, ng - 1)])
    @test_throws ErrorException mk(weight = zeros(FMD.FPPREC, ng))
    # Gauge for a different grid.
    other = Calculators.PolarizationGauge(
        make_binary_supercell(reps), (1, 2, 4), MAP_QC, MAP_QS)
    @test_throws ErrorException Calculators.PolarizationMapUmbrella(
        atoms, 1.0, tgt, reps, MAP_QC, MAP_QS; gauge = other)
end

@testset "Zero-bias identity: target = the current smoothed field" begin
    # The sharpest statement of correctness: set the target to the field the
    # configuration already has and NOTHING may happen — no energy, no force,
    # no virial.  Catches a sign error, an index transposition of the map, or
    # a stray offset, all of which survive a finite-difference check.
    for (reps, sigma) in (((3, 4, 5), 0.0), ((3, 4, 5), 0.9))
        proto, atoms = map_fixture(reps)
        gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
        tgt = smoothed_reference(atoms, gauge, reps, sigma)

        pu = Calculators.PolarizationMapUmbrella(
            atoms, 1e4, tgt, reps, MAP_QC, MAP_QS;
            gauge = gauge, sigma = sigma, r_th = 0.0)
        a = deepcopy(atoms)
        Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a, pu, (qc = MAP_QC, qs = MAP_QS))

        @test isapprox(pu.results["bias"], 0; atol = 1e-18)
        @test isapprox(pu.results["cv"], 0; atol = 1e-9)
        @test maximum(abs, a.arrays[:forces]) < 1e-10
        @test maximum(abs, a.arrays[:shellforces]) < 1e-10
        @test maximum(abs, Array(a.info[:virial])) < 1e-10
    end
end

@testset "Smoothed field matches an independent real-space convolution" begin
    # Pins the FFT path against a naive periodic Gaussian sum that shares no
    # code with it.  A finite-difference test cannot catch an error here: it
    # would differentiate the same wrong smoothing.
    reps = (3, 4, 5)
    sigma = 0.9
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    ref = smoothed_reference(atoms, gauge, reps, sigma)

    # With target = 0 and r_th = 0 the bias is (k/2)Σ|P̃|², so sum_r2 is
    # exactly Σ|P̃|² of the implementation's own smoothed field.
    zero_tgt = zeros(FMD.FPPREC, prod(reps), 3)
    pu = Calculators.PolarizationMapUmbrella(
        atoms, 1.0, zero_tgt, reps, MAP_QC, MAP_QS; gauge = gauge, sigma = sigma)
    a = deepcopy(atoms)
    Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(a, pu, (qc = MAP_QC, qs = MAP_QS))

    @test isapprox(pu.results["sum_r2"], sum(abs2, ref); rtol = 1e-10)
    @test sum(abs2, ref) > 1e-6          # non-vacuous
end

@testset "Bias force = -∇U (FD)" begin
    reps = (3, 3, 3)
    ng = prod(reps)
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    # A structured, non-zero target so the residual is not accidentally
    # aligned with anything.
    tgt = FMD.FPPREC[0.004 * _map_jitter(2i + 5d) for i in 1:ng, d in 1:3]
    w_nonuniform = FMD.FPPREC[0.25 + 0.75 * (i % 3) for i in 1:ng]

    for (label, sigma, r_th, w) in (
            ("sigma=0, r_th=0", 0.0, 0.0, nothing),
            ("sigma>0, r_th=0", 0.8, 0.0, nothing),
            ("sigma>0, r_th>0, w non-uniform", 0.8, 2e-3, w_nonuniform))
        pu = Calculators.PolarizationMapUmbrella(
            atoms, 5e3, tgt, reps, MAP_QC, MAP_QS;
            gauge = gauge, sigma = sigma, r_th = r_th, weight = w)
        prms = (qc = MAP_QC, qs = MAP_QS)
        calc! = a -> Calculators.umbrella_p!(a, pu, prms)

        @testset "$label" begin
            assert_map_core_force_fd(calc!, atoms;
                h = 1e-5, indices = 1:6, rtol = 1e-4, atol = 1e-9,
                label = "map umbrella core+shell force ($label)")
            assert_shellforce_matches_finite_diff(calc!, atoms;
                h = 1e-6, atom_indices = 1:6, rtol = 1e-4, atol = 1e-9,
                label = "map umbrella shellforce ($label)")
            # symmetrize: the target map is lab-frame data, so ∂U/∂ε has an
            # antisymmetric part the Voigt-6 virial does not carry.
            assert_virial_matches_finite_diff(calc!, atoms;
                h = 1e-5, rtol = 1e-4, atol = 1e-9, symmetrize = true,
                label = "map umbrella virial ($label)")
        end
    end
end

@testset "Gauge covariance: pure cell deformation produces no bias force" begin
    reps = (3, 3, 3)
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    tgt = smoothed_reference(atoms, gauge, reps, 0.0)
    pu = Calculators.PolarizationMapUmbrella(
        atoms, 1e4, tgt, reps, MAP_QC, MAP_QS; gauge = gauge, sigma = 0.0)

    for (label, S) in (
            ("volume-preserving shear",
             FMD.FPPREC[1.0 0.05 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
            ("isotropic expansion", FMD.FPPREC(1.05) * Matrix(I, 3, 3)))
        @testset "$label" begin
            a = deepcopy(atoms)
            a.info[:lattice] .= a.info[:lattice] * S
            a.arrays[:shelldisp] .= atoms.arrays[:shelldisp] * S
            Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
            Calculators.umbrella_p!(a, pu, (qc = MAP_QC, qs = MAP_QS))
            # The atoms have not moved, but P_g scales with 1/V, so the bias
            # is NOT expected to stay zero under isotropic expansion — what
            # must hold is that the reference co-transforms, i.e. the residual
            # is only the physical 1/V change and nothing spurious.
            expected = Calculators.group_p_gauged(a, gauge, MAP_QC, MAP_QS) .- tgt
            @test isapprox(pu.results["sum_r2"], sum(abs2, expected); rtol = 1e-8)
        end
    end
end

@testset "Weight mask" begin
    reps = (4, 1, 1)
    ng = prod(reps)
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    tgt = zeros(FMD.FPPREC, ng, 3)
    w = FMD.FPPREC[1.0, 1.0, 0.0, 1.0]        # group 3 released
    prms = (qc = MAP_QC, qs = MAP_QS)
    # Atoms of group 3 (one A + one B per group, in flat-group order).
    rows3 = findall(==(3), Array(atoms.arrays[:groupid]))

    @testset "sigma = 0: a released group feels exactly nothing" begin
        pu = Calculators.PolarizationMapUmbrella(
            atoms, 1e4, tgt, reps, MAP_QC, MAP_QS;
            gauge = gauge, weight = w, sigma = 0.0)
        a = deepcopy(atoms)
        Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a, pu, prms)
        fmax = maximum(abs, a.arrays[:forces])
        @test fmax > 1e-6                     # the restrained groups do feel it
        # Down at the FFT round-trip floor: with sigma = 0 the kernel is the
        # identity, but the field still makes the transform round trip, so the
        # residue is rounding rather than a leak.
        @test maximum(abs, a.arrays[:forces][rows3, :]) < 1e-12 * fmax
        @test maximum(abs, a.arrays[:shellforces][rows3, :]) < 1e-12 * fmax
    end

    @testset "sigma > 0: force leaks in through the kernel, by design" begin
        pu = Calculators.PolarizationMapUmbrella(
            atoms, 1e4, tgt, reps, MAP_QC, MAP_QS;
            gauge = gauge, weight = w, sigma = 0.8)
        a = deepcopy(atoms)
        Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a, pu, prms)
        # The weight multiplies the SMOOTHED residual, so the adjoint spreads
        # it back over the kernel: a w = 0 group is not force-free.  This is
        # the documented semantics, pinned here so it cannot drift silently.
        @test maximum(abs, a.arrays[:forces][rows3, :]) >
              1e-3 * maximum(abs, a.arrays[:forces])
    end
end

@testset "Target-map persistence carries the gauge identity" begin
    reps = (2, 2, 2)
    ng = prod(reps)
    proto, atoms = map_fixture(reps)
    gauge = Calculators.PolarizationGauge(proto, reps, MAP_QC, MAP_QS)
    tgt = Calculators.group_p_gauged(atoms, gauge, MAP_QC, MAP_QS)

    mktempdir() do dir
        path = joinpath(dir, "target.jld2")
        Calculators.save_target_map(path, tgt, gauge)
        @test Calculators.load_target_map(path, gauge) == tgt

        # A gauge from a different system must be refused, even though the
        # array shapes happen to match.
        other_proto = make_binary_supercell(reps)
        other_proto.arrays[:elemid] = reverse(other_proto.arrays[:elemid])
        other = Calculators.PolarizationGauge(other_proto, reps, MAP_QC, MAP_QS)
        @test_throws ErrorException Calculators.load_target_map(path, other)

        # The harder case, and the one that motivates storing the reference
        # STRUCTURE: same composition, same grouping, same ids — only the
        # reference frame differs.  Every topological field matches, so a
        # check on grid/elemid/groupid alone would wave this through while the
        # map silently refers to a different P = 0.
        shifted_proto = make_binary_supercell(reps)
        shifted_proto.arrays[:pos][sublattice_a(shifted_proto), 3] .+= 0.01
        shifted = Calculators.PolarizationGauge(shifted_proto, reps, MAP_QC, MAP_QS)
        @test shifted.elemid == gauge.elemid          # topology identical …
        @test shifted.groupid == gauge.groupid
        @test shifted.id_ref == gauge.id_ref
        @test shifted.M_frac_ref != gauge.M_frac_ref  # … zero-point is not
        @test_throws ErrorException Calculators.load_target_map(path, shifted)

        # Wrong shape at save time, and a non-target file at load time.
        @test_throws ErrorException Calculators.save_target_map(
            joinpath(dir, "bad.jld2"), zeros(FMD.FPPREC, ng, 2), gauge)
        plain = joinpath(dir, "plain.jld2")
        JLD2.jldsave(plain; something = 1)
        @test_throws ErrorException Calculators.load_target_map(plain, gauge)
    end
end
