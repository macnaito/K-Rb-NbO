# test/calculators/test_surface_term.jl
#
# SurfaceTerm (Issue #179): depolarization tensor of an ellipsoid in an
# anisotropic external medium, and the surface-term energy/forces/virial
# built on it.  Reuses make_binary_supercell / sublattice_a from
# test_umbrella.jl (runtests.jl includes that file first).
#
# The depolarization tensor is pinned by closed forms that are independent
# of the quadrature implementation: the isotropic sphere, the uniaxial
# sphere (both branches of the analytic L_∥), the ellipsoid in vacuum
# (classical spheroid depolarization factors), the exact trace identity
# Tr(ε_out·L) = 1, and rotation covariance.

using Test
using LinearAlgebra
using Random

# Deterministic rotation matrix (Rodrigues), no RNG involvement.
function _st_rotation(axis::Vector{Float64}, θ::Float64)
    n = axis ./ norm(axis)
    K = [0.0 -n[3] n[2]; n[3] 0.0 -n[1]; -n[2] n[1] 0.0]
    return Matrix{Float64}(I, 3, 3) + sin(θ) .* K + (1 - cos(θ)) .* (K * K)
end

# Analytic L_∥ of a SPHERE in a uniaxial medium ε = diag(εperp, εperp, εpar)
# (research note "異方誘電率媒質中の表面項" §3.3; both branches).
function _st_L_par_uniaxial(εperp::Float64, εpar::Float64)
    if εpar > εperp
        Δ = εpar - εperp
        return (1 - sqrt(εperp / Δ) * atan(sqrt(Δ / εperp))) / Δ
    else
        δ = εperp - εpar
        return (sqrt(εperp / δ) * atanh(sqrt(δ / εperp)) - 1) / δ
    end
end

@testset "depolarization_tensor: analytic limits" begin
    @testset "isotropic sphere: L = I/(3ε)" begin
        for ε in (1.0, 2.5, 80.0)
            L = Calculators.depolarization_tensor(ε)
            @test isapprox(L, Matrix{Float64}(I, 3, 3) ./ (3ε);
                           rtol = 1e-6, atol = 1e-9)
        end
    end

    @testset "input forms agree (scalar / diagonal / matrix; shape scale cancels)" begin
        L0 = Calculators.depolarization_tensor(2.5)
        @test Calculators.depolarization_tensor([2.5, 2.5, 2.5]) == L0
        @test Calculators.depolarization_tensor(2.5 .* Matrix{Float64}(I, 3, 3)) == L0
        # A scaled sphere is the same shape — the overall scale of A cancels
        # (analytically; in FP the 1/a factors round per-operation, so this
        # is an isapprox, not a bitwise identity).
        @test isapprox(Calculators.depolarization_tensor(2.5; shape = [3.7, 3.7, 3.7]),
                       L0; rtol = 1e-12)
        @test isapprox(Calculators.depolarization_tensor(2.5;
                           shape = 0.2 .* Matrix{Float64}(I, 3, 3)),
                       L0; rtol = 1e-12)
    end

    @testset "uniaxial sphere: closed-form Lpar, Lperp (both branches)" begin
        for (εperp, εpar) in ((2.0, 5.0), (5.0, 2.0), (1.0, 12.0), (7.0, 1.5))
            L = Calculators.depolarization_tensor([εperp, εperp, εpar])
            Lpar = _st_L_par_uniaxial(εperp, εpar)
            Lperp = (1 - εpar * Lpar) / (2εperp)
            @test isapprox(L[3, 3], Lpar; rtol = 1e-6)
            @test isapprox(L[1, 1], Lperp; rtol = 1e-6)
            @test isapprox(L[2, 2], Lperp; rtol = 1e-6)
            @test maximum(abs, L - Diagonal([Lperp, Lperp, Lpar])) < 1e-7
        end
    end

    @testset "ellipsoid in vacuum: classical spheroid factors" begin
        # Prolate spheroid (a, a, c), c > a, in ε_out = 1: L = diag(Nperp, Nperp, Npar)
        # with the textbook Npar = ((1−e²)/e²)·[ln((1+e)/(1−e))/(2e) − 1].
        a, c = 1.0, 3.0
        e = sqrt(1 - a^2 / c^2)
        Npar = ((1 - e^2) / e^2) * (log((1 + e) / (1 - e)) / (2e) - 1)
        Nperp = (1 - Npar) / 2
        L = Calculators.depolarization_tensor(1.0; shape = [a, a, c])
        @test isapprox(L[3, 3], Npar; rtol = 1e-6)
        @test isapprox(L[1, 1], Nperp; rtol = 1e-6)
        @test isapprox(L[2, 2], Nperp; rtol = 1e-6)
    end

    @testset "trace identity + symmetry for a general (ε, A)" begin
        R1 = _st_rotation([1.0, 2.0, 3.0], 0.7)
        R2 = _st_rotation([-2.0, 1.0, 0.5], 1.1)
        ε = R1 * Diagonal([1.5, 3.0, 6.0]) * R1'
        A = R2 * Diagonal([1.0, 1.6, 0.7]) * R2'
        L = Calculators.depolarization_tensor(ε; shape = A)
        # Tr(ε·L) = 1 holds identically for the direction-average form
        # (each direction contributes (v·εv)/(v·εv)), so this pins the
        # NORMALISATION contract only — it cannot fail on an unconverged
        # quadrature.  Accuracy is pinned by the closed-form testsets above.
        @test isapprox(tr(ε * L), 1.0; atol = 1e-7)
        @test maximum(abs, L - L') == 0          # symmetrised by construction
        @test eigmin(Symmetric(L)) > 0
    end

    @testset "rotation covariance: L(RεRᵀ, RARᵀ) = R·L(ε, A)·Rᵀ" begin
        R = _st_rotation([0.3, -1.0, 2.0], 0.9)
        ε = Diagonal([2.0, 3.5, 9.0])
        A = Diagonal([1.0, 1.4, 0.6])
        L_lab = Calculators.depolarization_tensor(Matrix(ε); shape = Matrix(A))
        L_rot = Calculators.depolarization_tensor(R * ε * R'; shape = R * A * R')
        @test isapprox(L_rot, R * L_lab * R'; rtol = 1e-6, atol = 1e-8)
    end

    @testset "invalid inputs error" begin
        @test_throws ErrorException Calculators.depolarization_tensor(-1.0)
        @test_throws ErrorException Calculators.depolarization_tensor([1.0, 2.0])
        @test_throws ErrorException Calculators.depolarization_tensor(
            [1.0 0.5 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0])          # asymmetric
        @test_throws ErrorException Calculators.depolarization_tensor(
            [1.0 0.0 0.0; 0.0 -2.0 0.0; 0.0 0.0 1.0])         # not SPD
        @test_throws ErrorException Calculators.depolarization_tensor(
            2.0; shape = :cube)
        @test_throws ErrorException Calculators.depolarization_tensor(
            2.0; shape = [1.0, 0.0, 1.0])                     # zero semiaxis
        # An unreachable tolerance must refuse, not hand back a guess.
        @test_throws ErrorException Calculators.depolarization_tensor(
            2.0; rtol = 1e-16, n0 = 1000, nmax = 4000)
    end
end

# ---------------------------------------------------------------------------
# SurfaceTerm construction: T identities and gauge/neutrality validation.
# ---------------------------------------------------------------------------

const ST_QC = FMD.FPPREC[1.2, -0.8]
const ST_QS = FMD.FPPREC[0.8, -1.2]     # per element qc+qs = ±2 → neutral

@testset "SurfaceTerm construction" begin
    proto = make_binary_supercell((3, 3, 3))

    @testset "isotropic sphere: T = I/(2ε_out + 1)" begin
        for ε in (1.0, 3.0, 40.0)
            st = Calculators.SurfaceTerm(proto, ST_QC, ST_QS;
                                         eps_out = ε, gauge = proto)
            @test isapprox(st.T, Matrix{Float64}(I, 3, 3) ./ (2ε + 1);
                           rtol = 1e-6, atol = 1e-9)
            @test isapprox(st.L, Matrix{Float64}(I, 3, 3) ./ (3ε);
                           rtol = 1e-6, atol = 1e-9)
        end
    end

    @testset "conductor limit: T → 0 as ε_out → ∞" begin
        st = Calculators.SurfaceTerm(proto, ST_QC, ST_QS;
                                     eps_out = 1e6, gauge = proto)
        @test maximum(abs, st.T) < 1e-6      # tin-foil is "no term at all"
    end

    @testset "M_frac_ref is the gauge frame's total fractional dipole" begin
        st = Calculators.SurfaceTerm(proto, ST_QC, ST_QS;
                                     eps_out = 3.0, gauge = proto)
        mref = Calculators.total_mfrac_ref(proto, (; qc = ST_QC, qs = ST_QS))
        @test st.M_frac_ref == mref
        # ... and a PolarizationGauge built from the same frame agrees.
        gauge = Calculators.PolarizationGauge(proto, (3, 3, 3), ST_QC, ST_QS)
        st2 = Calculators.SurfaceTerm(proto, ST_QC, ST_QS;
                                      eps_out = 3.0, gauge = gauge)
        @test isapprox(st2.M_frac_ref, st.M_frac_ref; rtol = 1e-12)
    end

    @testset "charged system is rejected" begin
        @test_throws "net charge" Calculators.SurfaceTerm(
            proto, FMD.FPPREC[1.0, 1.0], FMD.FPPREC[0.0, 0.0];
            eps_out = 3.0, gauge = proto)
    end

    @testset "gauge frame validation (Atoms path)" begin
        # Different atom count.
        @test_throws "different system" Calculators.SurfaceTerm(
            proto, ST_QC, ST_QS; eps_out = 3.0,
            gauge = make_binary_supercell((2, 2, 2)))
        # Different elemid (compared per :id).
        swapped = deepcopy(proto)
        swapped.arrays[:elemid][1], swapped.arrays[:elemid][2] =
            swapped.arrays[:elemid][2], swapped.arrays[:elemid][1]
        @test_throws ":elemid differs" Calculators.SurfaceTerm(
            proto, ST_QC, ST_QS; eps_out = 3.0, gauge = swapped)
        # Ambiguous branch: a frame more than half a period away.
        far = deepcopy(proto)
        far.arrays[:pos][1, 2] += 0.5
        @test_throws "periodic image is ambiguous" Calculators.SurfaceTerm(
            far, ST_QC, ST_QS; eps_out = 3.0, gauge = proto)
        # A row-permuted frame is the same system (paired by :id).
        perm = [2:length(proto); 1]
        st_perm = Calculators.SurfaceTerm(deepcopy(proto)[perm], ST_QC, ST_QS;
                                          eps_out = 3.0, gauge = proto)
        st_ref = Calculators.SurfaceTerm(proto, ST_QC, ST_QS;
                                         eps_out = 3.0, gauge = proto)
        @test isapprox(st_perm.M_frac_ref, st_ref.M_frac_ref; atol = 1e-12)
    end

    @testset "gauge validation (PolarizationGauge path)" begin
        gauge = Calculators.PolarizationGauge(proto, (3, 3, 3), ST_QC, ST_QS)
        # validate_gauge rejects a different system through the same door
        # the umbrellas use.
        @test_throws ErrorException Calculators.SurfaceTerm(
            make_binary_supercell((2, 2, 2)), ST_QC, ST_QS;
            eps_out = 3.0, gauge = gauge)
    end

    @testset "gauge is required (no default)" begin
        @test_throws UndefKeywordError Calculators.SurfaceTerm(
            proto, ST_QC, ST_QS; eps_out = 3.0)
    end
end

# ---------------------------------------------------------------------------
# Energy value: vacuum sphere reduces to the textbook Ewald surface term.
# ---------------------------------------------------------------------------

@testset "vacuum sphere: U = 2π/(3V)·|δM|²" begin
    proto = make_binary_supercell((3, 3, 3))
    atoms = deepcopy(proto)
    Δfrac = FMD.FPPREC(0.04)
    rows = sublattice_a(atoms)
    atoms.arrays[:pos][rows, 1] .+= Δfrac

    st = Calculators.SurfaceTerm(atoms, ST_QC, ST_QS;
                                 eps_out = 1.0, gauge = proto)
    prms = (qc = ST_QC, qs = ST_QS)
    reset!(atoms)
    Calculators.surface_term!(atoms, st, prms)

    # Sublattice A (elemid 1, qc+qs = +2) slid by Δfrac·L_x against B:
    # δM_x = n_A·(qc₁+qs₁)·Δfrac·L_x, δM_y = δM_z = 0.
    L_x = atoms.info[:lattice][1, 1]
    V = volume(atoms)
    dM_expected = length(rows) * (ST_QC[1] + ST_QS[1]) * Δfrac * L_x
    U_expected = 2π / (3V) * dM_expected^2
    @test isapprox(atoms.info[:potential_energy][1], U_expected; rtol = 1e-6)
    @test isapprox(st.results["bias"], U_expected; rtol = 1e-6)
    @test isapprox(st.results["dM"], [dM_expected, 0, 0];
                   rtol = 1e-10, atol = 1e-10)

    # The polarisation branch is deliberately continuous (:pos unwrapped):
    # shifting ONE atom by a full lattice vector moves M by a polarisation
    # quantum and the surface energy MUST see it — pin the convention.
    shifted = deepcopy(atoms)
    shifted.arrays[:pos][rows[1], 1] += 1
    reset!(shifted)
    Calculators.surface_term!(shifted, st, prms)
    dM_shift = dM_expected + (ST_QC[1] + ST_QS[1]) * L_x
    @test isapprox(shifted.info[:potential_energy][1],
                   2π / (3V) * dM_shift^2; rtol = 1e-6)
end

# ---------------------------------------------------------------------------
# Physical consistency: FD forces / shell forces / virial, invariances.
# Fully anisotropic ε_out and a rotated ellipsoid so no component vanishes
# by symmetry.
# ---------------------------------------------------------------------------

@testset "SurfaceTerm physical consistency (FD, anisotropic ε + ellipsoid)" begin
    R1 = _st_rotation([1.0, 2.0, 3.0], 0.7)
    R2 = _st_rotation([-2.0, 1.0, 0.5], 1.1)
    ε = R1 * Diagonal([1.5, 3.0, 6.0]) * R1'
    A = R2 * Diagonal([1.0, 1.6, 0.7]) * R2'

    proto = make_binary_supercell((3, 3, 3))
    rng = Random.MersenneTwister(179)
    atoms = deepcopy(proto)
    atoms.arrays[:pos] .+= FMD.FPPREC(0.01) .*
        randn(rng, FMD.FPPREC, size(atoms.arrays[:pos]))
    atoms.arrays[:shelldisp] .= FMD.FPPREC(0.05) .*
        randn(rng, FMD.FPPREC, size(atoms.arrays[:shelldisp]))

    st = Calculators.SurfaceTerm(atoms, ST_QC, ST_QS;
                                 eps_out = ε, shape = A, gauge = proto)
    prms = (qc = ST_QC, qs = ST_QS)
    calc! = a -> Calculators.surface_term!(a, st, prms)

    # Non-vacuity: the rattle produced a real δM, energy, forces and a
    # virial with live shear components.
    probe = deepcopy_atoms(atoms)
    reset!(probe)
    calc!(probe)
    @test probe.info[:potential_energy][1] > 1e-6
    @test maximum(abs, probe.arrays[:forces]) > 1e-6
    @test maximum(abs, probe.arrays[:shellforces]) > 1e-6
    @test maximum(abs, Array(probe.info[:virial])[1:3]) > 1e-6
    @test maximum(abs, Array(probe.info[:virial])[4:6]) > 1e-6

    # Force FD against the TOTAL per-atom gradient: moving `pos` at fixed
    # `shelldisp` displaces core AND shell together (x_s = x_c + sd, the
    # HVP.jl variable convention), so −∂E/∂r_i = F_core,i + F_shell,i.
    # The shared helper compares against the core force alone, which is
    # complete only for qs = 0 — with the split charges here both halves
    # are live and their SUM is the position gradient.
    @testset "SurfaceTerm force = -∇E (FD, core+shell)" begin
        Ftot = probe.arrays[:forces] .+ probe.arrays[:shellforces]
        ilat = inv(Array(atoms.info[:lattice]))
        h = 1e-4
        for i in 1:4, axis in 1:3
            δ = zeros(FMD.FPPREC, 3); δ[axis] = h
            δfrac = vec(δ' * ilat)
            ap = deepcopy_atoms(atoms); ap.arrays[:pos][i, :] .+= δfrac
            Ep, _ = evaluate(calc!, ap)
            am = deepcopy_atoms(atoms); am.arrays[:pos][i, :] .-= δfrac
            Em, _ = evaluate(calc!, am)
            num = -(Ep - Em) / (2h)
            @test isapprox(Ftot[i, axis], num; rtol = 1e-4, atol = 1e-8)
        end
    end
    assert_shellforce_matches_finite_diff(calc!, atoms;
        h = 1e-5, rtol = 1e-5, atol = 1e-10,
        label = "SurfaceTerm shellforce = -∂E/∂sd (FD)")
    # symmetrize: like the umbrella, the term couples M to lab-frame
    # tensors, so ∂U/∂ε has an antisymmetric part the Voigt-6 virial does
    # not carry (see the FD helper's header).
    assert_virial_matches_finite_diff(calc!, atoms;
        h = 1e-5, rtol = 1e-4, atol = 1e-8, symmetrize = true,
        label = "SurfaceTerm virial = -∂E/∂ε (strain FD)")
    # Uniform per-charge force on a NEUTRAL system: the core+shell force
    # sum vanishes (Σ core forces alone is −g·Σqc ≠ 0 with split charges).
    @test maximum(abs, vec(sum(probe.arrays[:forces] .+
                               probe.arrays[:shellforces], dims = 1))) < 1e-10
    assert_translation_invariant(calc!, atoms;
        rtol = 1e-8, atol = 1e-10,
        label = "SurfaceTerm translation invariance")
end

# ---------------------------------------------------------------------------
# Gauge covariance: pure cell deformation (atoms static in fractional
# space, shell displacements co-transformed) keeps δM = 0 — no energy, no
# forces, no virial.  Triclinic cell so a lattice-convention mixup between
# M and M_ref cannot cancel (cf. test_umbrella.jl).
# ---------------------------------------------------------------------------

@testset "SurfaceTerm gauge covariance under cell deformation" begin
    L_tri = FMD.FPPREC[10.0 0.0 0.0; 1.7 11.0 0.0; 0.9 0.6 12.0]
    rng = Random.MersenneTwister(42)
    n = 16
    pos = rand(rng, FMD.FPPREC, n, 3)
    elemid = [isodd(i) ? 1 : 2 for i in 1:n]
    ref = make_atoms(L_tri, pos, elemid;
                     shelldisp = FMD.FPPREC(0.3) .* randn(rng, FMD.FPPREC, n, 3))

    st = Calculators.SurfaceTerm(ref, ST_QC, ST_QS;
                                 eps_out = [2.0, 3.0, 5.0], gauge = ref)
    prms = (qc = ST_QC, qs = ST_QS)

    # Guard: the raw dipole this cancellation hides is actually large.
    @test maximum(abs, Calculators.calcdipole!(deepcopy_atoms(ref), prms)) > 1e-1

    for (label, S) in (("volume-preserving shear",
                        FMD.FPPREC[1.0 0.05 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]),
                       ("isotropic expansion",
                        FMD.FPPREC(1.05) .* Matrix{FMD.FPPREC}(I, 3, 3)))
        @testset "$label" begin
            a = deepcopy_atoms(ref)
            a.info[:lattice] .= ref.info[:lattice] * S
            a.arrays[:shelldisp] .= ref.arrays[:shelldisp] * S
            reset!(a)
            Calculators.surface_term!(a, st, prms)
            @test abs(a.info[:potential_energy][1]) < 1e-18
            @test maximum(abs, a.arrays[:forces]) < 1e-10
            @test maximum(abs, a.arrays[:shellforces]) < 1e-10
            @test maximum(abs, Array(a.info[:virial])) < 1e-10
        end
    end

    # A genuine displacement on top of the deformation IS seen.
    a = deepcopy_atoms(ref)
    a.info[:lattice] .= ref.info[:lattice] .* FMD.FPPREC(1.02)
    a.arrays[:shelldisp] .= ref.arrays[:shelldisp] .* FMD.FPPREC(1.02)
    a.arrays[:pos][elemid .== 1, 1] .+= FMD.FPPREC(0.02)
    reset!(a)
    Calculators.surface_term!(a, st, prms)
    @test a.info[:potential_energy][1] > 1e-8
end

# ---------------------------------------------------------------------------
# Screening physics: with the surface term the shell spring is renormalised
# to k2·I + (4π/V)·qs²·T.  One core-shell pair under an external field: the
# shell force must vanish exactly at the analytically renormalised
# equilibrium.  This ties Spring + ExternalField + SurfaceTerm together
# through the same force assembly npt! uses.
# ---------------------------------------------------------------------------

@testset "SurfaceTerm renormalises the shell spring (screening)" begin
    # Small box, soft spring, |qs| = 2 and a strongly uniaxial ε_out so the
    # depolarization stiffness (4π/V)·qs²·T is a large (~50%) and clearly
    # ANISOTROPIC correction to k2 (T⊥/T∥ ≈ 1.17 here; a mild ε_out keeps
    # T nearly isotropic because ε_in = I dominates the mixing).
    lat = FMD.FPPREC(10.0) .* Matrix{FMD.FPPREC}(I, 3, 3)
    one_atom = make_atoms(lat, FMD.FPPREC[0.5 0.5 0.5], [1])
    qc = FMD.FPPREC[2.0]
    qs = FMD.FPPREC[-2.0]     # neutral atom: qc + qs = 0
    k2 = FMD.FPPREC[0.005]
    k4 = FMD.FPPREC[0.0]
    efield = FMD.FPPREC[3e-3, 0.0, 1.5e-3]

    st = Calculators.SurfaceTerm(one_atom, qc, qs;
                                 eps_out = [1.0, 1.0, 30.0], gauge = one_atom)
    prms = (qc = qc, qs = qs, k2 = k2, k4 = k4, efield = efield)
    calc! = a -> begin
        Calculators.calculate!(Calculators.Spring(), a, prms)
        Calculators.calculate!(Calculators.ExternalField(), a, prms)
        Calculators.surface_term!(a, st, prms)
    end

    V = volume(one_atom)
    Keff = k2[1] .* Matrix{Float64}(I, 3, 3) .+ (4π / V) * qs[1]^2 .* st.T
    u_star = Keff \ (qs[1] .* Vector{Float64}(efield))

    a = deepcopy_atoms(one_atom)
    a.arrays[:shelldisp] .= reshape(FMD.FPPREC.(u_star), 1, 3)
    reset!(a)
    calc!(a)
    @test maximum(abs, a.arrays[:shellforces]) < 1e-12

    # Non-vacuity 1: the bare-spring equilibrium (no depolarization
    # screening) does NOT balance once the surface term is present.
    u_bare = (qs[1] / k2[1]) .* Vector{Float64}(efield)
    b = deepcopy_atoms(one_atom)
    b.arrays[:shelldisp] .= reshape(FMD.FPPREC.(u_bare), 1, 3)
    reset!(b)
    calc!(b)
    @test maximum(abs, b.arrays[:shellforces]) > 1e-6
    # Non-vacuity 2: the renormalisation is anisotropic — the uniaxial
    # ε_out makes the x and z responses genuinely different.
    @test abs(u_star[1] / efield[1] - u_star[3] / efield[3]) >
          0.05 * abs(u_star[1] / efield[1])
end
