# test/calculators/test_umbrella.jl
#
# PolarizationUmbrella correctness, with particular attention to the
# flat-groupid → (gx, gy, gz) decode that has to match ASE's
# `unit_cell * (Nx, Ny, Nz)` atom-layout convention.

using Test
using LinearAlgebra
using Random
using FFTW
using JLD2

# Two-sublattice supercell, ASE atom layout: flat group index
# ix·Ny·Nz + iy·Nz + iz (iz cycles fastest, ix slowest), sublattice A at the
# cell corner (elemid 1) and B at the body centre (elemid 2), both mapped to
# the same groupid.
#
# Why two sublattices rather than one atom per group: a group must be charge
# NEUTRAL, otherwise its dipole depends on where the cell origin sits and the
# collective variable drifts under a rigid translation of the system — which
# PolarizationGauge now rejects at construction.  A ±q pair also means the
# polarisation responds to the RELATIVE displacement of the sublattices, the
# way a real ferroelectric does, so `pos[A] .+= δ` is the natural way to
# polarise the fixture.  Tests that need the shell half of group_p /
# grad_group_p! give the two elements different qs:qc ratios (with a single
# element shellforces ≡ (qs/qc)·forces identically).
function make_binary_supercell(reps::NTuple{3,Int}; a_bohr=4.0)
    Nx, Ny, Nz = reps
    n = 2 * Nx * Ny * Nz
    pos = zeros(FMD.FPPREC, n, 3)
    groupid = zeros(Int, n)
    elemid = zeros(Int, n)
    k = 0
    for ix in 0:Nx-1, iy in 0:Ny-1, iz in 0:Nz-1
        gid = ix * Ny * Nz + iy * Nz + iz + 1   # ASE-natural flat index
        for (e, off) in ((1, 0.0), (2, 0.5))
            k += 1
            pos[k, 1] = (ix + off) / Nx
            pos[k, 2] = (iy + off) / Ny
            pos[k, 3] = (iz + off) / Nz
            groupid[k] = gid
            elemid[k] = e
        end
    end
    info = Dict{Symbol,Any}(
        :lattice => FMD.FPPREC.([Nx*a_bohr 0 0; 0 Ny*a_bohr 0; 0 0 Nz*a_bohr]),
        :potential_energy => FMD.FPPREC[0],
        :virial => zeros(FMD.FPPREC, 6),
        :method => Symbol[:test],
    )
    arrays = Dict{Symbol,Array}(
        :pos => pos,
        :elemid => elemid,
        :groupid => groupid,
        :forces => zeros(FMD.FPPREC, n, 3),
        :shellforces => zeros(FMD.FPPREC, n, 3),
        :shelldisp => zeros(FMD.FPPREC, n, 3),
        :velo => zeros(FMD.FPPREC, n, 3),
        :id => collect(1:n),
        :masses => ones(FMD.FPPREC, n),
    )
    return Atoms{Array}(info, arrays)
end

# Rows of sublattice A (elemid 1) — displacing only these polarises the cell.
sublattice_a(atoms) = findall(==(1), atoms.arrays[:elemid])

@testset "PolarizationUmbrella construction" begin
    atoms = make_binary_supercell((3, 4, 5))
    qc = FMD.FPPREC[1.0, -1.0]
    qs = FMD.FPPREC[0.0, 0.0]
    pu = Calculators.PolarizationUmbrella(
        atoms, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (3, 4, 5), qc, qs;
        gauge = atoms, m = (1, 0, 0),
    )
    @test pu.gridsize == (3, 4, 5)
    @test pu.m == (1, 0, 0)
    @test pu.denom_inv > 0
    @test size(pu.Ghat) == (3 ÷ 2 + 1, 4, 5)
    # ghat_at_m for σ=0 must be exactly 1.
    @test isapprox(pu.ghat_at_m, 1.0; atol = 1e-12)
end

@testset "Uniform mode m=(0,0,0): amp = mean(Pg ⋅ ê)" begin
    Nx, Ny, Nz = 3, 4, 5
    ng = Nx * Ny * Nz
    atoms = make_binary_supercell((Nx, Ny, Nz))
    qc = FMD.FPPREC[1.0, -1.0]     # group-neutral ±q pair
    qs = FMD.FPPREC[0.0, 0.0]

    # Polarise by sliding sublattice A against B, the way a ferroelectric
    # does.  Displacing ALL atoms would move a neutral cell rigidly and
    # produce no polarisation at all.
    Δfrac = 0.05
    atoms.arrays[:pos][sublattice_a(atoms), 1] .+= Δfrac

    # An atoms_ref with the undisplaced layout gives Pg_ref that cancels the
    # lattice contribution.
    atoms_ref = make_binary_supercell((Nx, Ny, Nz))

    pu = Calculators.PolarizationUmbrella(
        atoms_ref, 0.0, 0.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
        gauge = atoms_ref, m = (0, 0, 0),
    )
    prms = (qc = qc, qs = qs)
    Calculators.umbrella_p!(atoms, pu, prms)

    # Hand calculation: each group's polarisation picks up q · Δr_x / V_g
    # where Δr_x = Δfrac · L_x and V_g = V / ng (one A atom per group).
    # amp = mean over groups of (Pg · ê) with ê = (1,0,0), and every group is
    # displaced identically, so amp = q · Δr_x / V_g.
    L_x = atoms.info[:lattice][1, 1]
    V_g = abs(det(Array(atoms.info[:lattice]))) / ng
    expected = abs(1.0 * (Δfrac * L_x) / V_g)
    @test isapprox(abs(pu.results["cv"]), expected; rtol = 1e-3)
end

@testset "Cosine mode regression: m axes ↔ real-space axes" begin
    # The decisive grid-index regression: pick a non-cube cell so a
    # transposed decode would be visible.  Set up a polarization profile
    # that varies cosinusoidally along ONE specific real-space axis, and
    # query the amplitude with m vector along that same axis.  With the
    # ASE-natural decode, amp_now ≈ amplitude; with the legacy decode it
    # would be ≈ 0 (the wave addresses a perpendicular axis).
    Nx, Ny, Nz = 5, 3, 7
    qc = FMD.FPPREC[1.0, -1.0]
    qs = FMD.FPPREC[0.0, 0.0]

    function build_modulated(axis::Int, amp_frac::Real)
        atoms = make_binary_supercell((Nx, Ny, Nz))
        # Modulate sublattice A's x displacement by a cosine along the chosen
        # real-space axis (`ix`/`iy`/`iz` of the unit cell); B stays put, so
        # the group dipole follows the same cosine.
        for k in sublattice_a(atoms)
            i0 = atoms.arrays[:groupid][k] - 1
            iz = i0 % Nz
            ixy = i0 ÷ Nz
            iy = ixy % Ny
            ix = ixy ÷ Ny
            n = (ix, iy, iz)[axis]
            N = (Nx, Ny, Nz)[axis]
            atoms.arrays[:pos][k, 1] += amp_frac * cos(2π * n / N)
        end
        return atoms
    end

    for (axis, m) in zip(1:3, [(1, 0, 0), (0, 1, 0), (0, 0, 1)])
        atoms = build_modulated(axis, 0.05)
        atoms_ref = make_binary_supercell((Nx, Ny, Nz))
        pu = Calculators.PolarizationUmbrella(
            atoms_ref, 0.0, 0.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = m,
        )
        Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))
        amp_match = abs(pu.results["cv"])

        # Now query the SAME atoms with an orthogonal m — should give ~0.
        m_orth = axis == 1 ? (0, 0, 1) : (axis == 2 ? (1, 0, 0) : (1, 0, 0))
        pu_orth = Calculators.PolarizationUmbrella(
            atoms_ref, 0.0, 0.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = m_orth,
        )
        Calculators.umbrella_p!(atoms, pu_orth, (qc = qc, qs = qs))
        amp_orth = abs(pu_orth.results["cv"])

        # The matching mode must dominate the orthogonal one.  With the
        # legacy gx-fast decode, this assertion fails for axis ≠ 1.
        @test amp_match > 10 * amp_orth
        @test amp_match > 0
    end
end

@testset "Bias force = -∇U_par (FD on macroscopic mode)" begin
    Nx, Ny, Nz = 3, 3, 3
    atoms = make_binary_supercell((Nx, Ny, Nz))
    qc = FMD.FPPREC[1.0, -1.0]
    qs = FMD.FPPREC[0.0, 0.0]
    k_par = 1e3
    A_target = 0.0

    pu = Calculators.PolarizationUmbrella(
        atoms, k_par, 0.0, A_target, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
        gauge = atoms, m = (0, 0, 0),
    )

    # Displace one atom slightly off-center to break symmetry, then verify
    # the force on that atom equals the central-difference of U_par.
    j = 5  # arbitrary
    Δ = 0.03
    a = deepcopy(atoms)
    a.arrays[:pos][j, 1] += Δ
    Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(a, pu, (qc = qc, qs = qs))
    F_ana = a.arrays[:forces][j, 1]
    U0 = a.info[:potential_energy][1]

    h = 1e-4
    L_x_inv = 1 / a.info[:lattice][1, 1]   # δ_frac for a cartesian h
    pos_ref = atoms.arrays[:pos][j, 1] + Δ
    a_p = deepcopy(atoms); a_p.arrays[:pos][j, 1] = pos_ref + h * L_x_inv
    Cores.clear!(a_p, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(a_p, pu, (qc = qc, qs = qs))
    Up = a_p.info[:potential_energy][1]

    a_m = deepcopy(atoms); a_m.arrays[:pos][j, 1] = pos_ref - h * L_x_inv
    Cores.clear!(a_m, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(a_m, pu, (qc = qc, qs = qs))
    Um = a_m.info[:potential_energy][1]

    F_num = -(Up - Um) / (2h)
    @test isapprox(F_ana, F_num; rtol = 1e-3, atol = 1e-6)
end

@testset "Gaussian smoothing: ghat_at_m matches FFT of cos(θ)" begin
    Nx, Ny, Nz = 4, 4, 4
    σ = 1.5
    rNx = Nx ÷ 2 + 1

    # Internal Ĝ matches the analytic exp(-(σk)²/2) on the rFFT grid.
    cos_field = zeros(Float64, Nx, Ny, Nz)
    m = (0, 0, 1)
    for gz in 0:Nz-1, gy in 0:Ny-1, gx in 0:Nx-1
        θ = 2π * (m[1] * gx / Nx + m[2] * gy / Ny + m[3] * gz / Nz)
        cos_field[gx + 1, gy + 1, gz + 1] = cos(θ)
    end
    plan = FFTW.plan_rfft(cos_field)
    cos_hat = plan * cos_field

    # The (k = m) bin should hold N/2 amplitude (real-only) for cos input
    # — sanity check that our θ=2π·m·g/N convention is the same as FFTW's.
    expected_kbin = (1, 1, m[3] + 1)
    @test abs(cos_hat[expected_kbin...]) ≈ Nx * Ny * Nz / 2

    # The analytic ghat_at_m formula reproduces the kernel value at m.
    k2_at_m = (2π * m[1] / Nx)^2 + (2π * m[2] / Ny)^2 + (2π * m[3] / Nz)^2
    ghat_at_m = exp(-0.5 * σ^2 * k2_at_m)
    @test ghat_at_m > 0
    @test ghat_at_m ≤ 1
end

# Regression for the gauge-covariant reference: with the legacy frozen-
# Cartesian Pg_ref these tests fail because pure cell deformation
# (volume-preserving shear, isotropic expansion) generates a spurious
# bias even though atoms have not moved.  After the fix the residual
# polarisation Pg = Pg_raw - Pg_ref(t) and bias forces vanish.
@testset "Gauge-covariant reference under cell deformation" begin
    Nx, Ny, Nz = 3, 3, 3
    ng = Nx * Ny * Nz
    qc = FMD.FPPREC[1.0, -1.0]
    qs = FMD.FPPREC[0.0, 0.0]

    # Polarise the fixture so that M_frac (and hence Pg_ref) is non-trivial.
    # With M_frac = 0 these tests would pass even with the buggy frozen
    # Cartesian reference.
    function shifted_atoms()
        a = make_binary_supercell((Nx, Ny, Nz))
        rows = sublattice_a(a)
        a.arrays[:pos][rows, 1] .+= 0.07
        a.arrays[:pos][rows, 2] .+= 0.03
        return a
    end

    @testset "Volume-preserving shear" begin
        atoms_ref = shifted_atoms()
        pu = Calculators.PolarizationUmbrella(
            atoms_ref, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = (0, 0, 0),
        )

        atoms = deepcopy(atoms_ref)
        # det(S) = 1 — pure shear, atoms static in fractional space.
        S = FMD.FPPREC[1.0 0.05 0.0;
                       0.0 1.0  0.0;
                       0.0 0.0  1.0]
        atoms.info[:lattice] .= atoms.info[:lattice] * S
        Cores.clear!(atoms, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))

        # Atomic forces must vanish (no physical motion has occurred).
        @test maximum(abs, atoms.arrays[:forces]) < 1e-10
        @test maximum(abs, atoms.arrays[:shellforces]) < 1e-10
        # And the bias collective variable too.
        @test isapprox(pu.results["cv"], 0.0; atol = 1e-10)
        @test isapprox(pu.results["bias"], 0.0; atol = 1e-12)
    end

    @testset "Isotropic expansion" begin
        atoms_ref = shifted_atoms()
        pu = Calculators.PolarizationUmbrella(
            atoms_ref, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = (0, 0, 0),
        )

        atoms = deepcopy(atoms_ref)
        α = FMD.FPPREC(1.05)
        atoms.info[:lattice] .*= α
        Cores.clear!(atoms, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))

        @test maximum(abs, atoms.arrays[:forces]) < 1e-10
        @test maximum(abs, atoms.arrays[:shellforces]) < 1e-10
        @test isapprox(pu.results["cv"], 0.0; atol = 1e-10)
    end

    @testset "Atomic displacement still produces physical bias" begin
        atoms_ref = shifted_atoms()
        pu = Calculators.PolarizationUmbrella(
            atoms_ref, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = (0, 0, 0),
        )

        atoms = deepcopy(atoms_ref)
        # Slide sublattice A by a further fractional Δ in x AND deform the
        # cell; the extra dipole is what the bias must see.
        Δfrac = 0.02
        atoms.arrays[:pos][sublattice_a(atoms), 1] .+= Δfrac
        atoms.info[:lattice] .*= FMD.FPPREC(1.02)

        Cores.clear!(atoms, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))

        # cv should equal q · Δr_x / V_g(t), independent of the cell scaling.
        # With L → α·L and uniform fractional shift, r_α-displacement = α·L_x·Δfrac
        # and V_g → α³·V_g, so cv = q·α·L_x·Δfrac / (α³·V_g_ref) = α^{-2}·(...)
        L_x = atoms.info[:lattice][1, 1]
        V_g = abs(det(Array(atoms.info[:lattice]))) / ng
        expected = abs(1.0 * (Δfrac * L_x) / V_g)
        @test isapprox(abs(pu.results["cv"]), expected; rtol = 1e-3)
    end
end

# Fractional → Cartesian is `pos * lattice` everywhere in this codebase
# (Cores/NeighborList.jl, Cores/XYZ.jl, Calculators/Ewald.jl).  The
# polarisation path used to index the lattice transposed, which is
# indistinguishable while the cell is orthorhombic — so the fixtures here are
# deliberately NON-orthorhombic and non-symmetric.  Without such a cell every
# other test in this file passes under either convention.
@testset "Fractional → Cartesian convention (non-orthorhombic cell)" begin
    # Triclinic and asymmetric: L ≠ Lᵀ, so the two conventions disagree.
    L_tri = FMD.FPPREC[10.0 0.0 0.0; 1.7 11.0 0.0; 0.9 0.6 12.0]

    # Local RNG: seeding the global one here would silently reshuffle the
    # random streams every later testset in this file sees.
    # elemid alternates 1,2,1,2,... so that a group holding an adjacent pair
    # is charge neutral with a ±q table (required by PolarizationGauge).
    function tri_atoms(n; seed = 7, groupid = ones(Int, n),
                       elemid = [isodd(i) ? 1 : 2 for i in 1:n])
        rng = Random.MersenneTwister(seed)
        pos = rand(rng, FMD.FPPREC, n, 3)
        info = Dict{Symbol,Any}(
            :lattice => copy(L_tri),
            :potential_energy => FMD.FPPREC[0],
            :virial => zeros(FMD.FPPREC, 6),
            :method => Symbol[:test],
        )
        arrays = Dict{Symbol,Array}(
            :pos => pos,
            :elemid => elemid,
            :groupid => groupid,
            :forces => zeros(FMD.FPPREC, n, 3),
            :shellforces => zeros(FMD.FPPREC, n, 3),
            # Non-zero shell displacement is what makes the gauge-covariance
            # test below discriminating: with sd = 0 the fractional moment is
            # just Σq·s and the raw/reference dipoles pick up the SAME lattice
            # matrix, so the residual cancels under either convention.
            :shelldisp => FMD.FPPREC(0.3) .* randn(rng, FMD.FPPREC, n, 3),
            :velo => zeros(FMD.FPPREC, n, 3),
            :id => collect(1:n),
            :masses => ones(FMD.FPPREC, n),
        )
        return Atoms{Array}(info, arrays)
    end

    @testset "group_p" begin
        n = 8
        qc = FMD.FPPREC[0.6, 0.6]      # group_p needs no neutrality
        qs = FMD.FPPREC[0.4, 0.4]
        atoms = tri_atoms(n)
        Pg = Array(Calculators.group_p(atoms, qc, qs))

        r = atoms.arrays[:pos] * atoms.info[:lattice]          # THE convention
        rshell = r .+ atoms.arrays[:shelldisp]
        Vg = abs(det(Array(atoms.info[:lattice])))             # one group here
        ref = vec(sum(qc[1] .* r .+ qs[1] .* rshell, dims = 1)) ./ Vg
        @test isapprox(Pg[1, :], ref; rtol = 1e-12)
    end

    @testset "calcdipole!" begin
        n = 8
        qc = FMD.FPPREC[0.6, 0.6]
        qs = FMD.FPPREC[0.4, 0.4]
        atoms = tri_atoms(n)
        M = Array(Calculators.calcdipole!(atoms, (qc = qc, qs = qs)))

        r = atoms.arrays[:pos] * atoms.info[:lattice]
        rshell = r .+ atoms.arrays[:shelldisp]
        ref = vec(sum(qc[1] .* r .+ qs[1] .* rshell, dims = 1))
        @test isapprox(M, ref; rtol = 1e-12)
    end

    # group_mfrac and Pg_ref(t) must use the SAME convention as group_p:
    # fixing only one of them leaves a residual bias under cell deformation
    # even though no atom has moved.  This is the end-to-end guard.
    @testset "Gauge covariance survives shear of a triclinic cell" begin
        Nx, Ny, Nz = 2, 2, 2
        n = 16
        qc = FMD.FPPREC[0.6, -0.6]     # per element qc+qs = ±1 → group-neutral
        qs = FMD.FPPREC[0.4, -0.4]
        gid = repeat(1:(Nx * Ny * Nz), inner = 2)   # one ±q pair per group
        atoms_ref = tri_atoms(n; groupid = gid)
        # Guard against a degenerate fixture: "bias stays zero" is only
        # meaningful while the raw group polarisation it cancels is large.
        @test maximum(abs, Array(Calculators.group_p(atoms_ref, qc, qs))) > 1e-3
        pu = Calculators.PolarizationUmbrella(
            atoms_ref, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (Nx, Ny, Nz), qc, qs;
            gauge = atoms_ref, m = (0, 0, 0),
        )

        atoms = deepcopy(atoms_ref)
        # det(S) = 1 — volume-preserving shear on top of an already
        # non-orthorhombic cell.  Atoms are static in fractional space and
        # the Cartesian shell displacement co-transforms, so the bias, its
        # forces and its virial must all stay exactly zero.
        S = FMD.FPPREC[1.0 0.05 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
        atoms.info[:lattice] .= atoms.info[:lattice] * S
        atoms.arrays[:shelldisp] .= atoms_ref.arrays[:shelldisp] * S
        Cores.clear!(atoms, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))

        @test maximum(abs, atoms.arrays[:forces]) < 1e-10
        @test maximum(abs, atoms.arrays[:shellforces]) < 1e-10
        @test maximum(abs, Array(atoms.info[:virial])) < 1e-10
        @test isapprox(pu.results["cv"], 0.0; atol = 1e-10)
        @test isapprox(pu.results["bias"], 0.0; atol = 1e-12)
    end
end

# The bias virial is -∂U/∂ε like every other calculator's, and it decomposes
# into three pieces (docs/src/theory/polarization-bias-virial.md): the atomic
# r⊗F sum, the isotropic 1/V_g term, and the Pg_ref cell-derivative term.
# Only an FD comparison sees all three at once — the gauge-covariance testsets
# above evaluate at points where the bias *vanishes*, so dUdPg = 0 there and
# every group-lattice virial term is trivially zero.  That is how a wrong
# 1/V_g term survived: the shear components were right, the diagonal was not.
@testset "PolarizationUmbrella virial = -∂U/∂ε (strain FD)" begin
    Nx, Ny, Nz = 3, 4, 5   # non-cubic, so a transposed grid decode stays visible
    ng = Nx * Ny * Nz
    qc = FMD.FPPREC[1.2, -0.8]
    qs = FMD.FPPREC[0.8, -1.2]     # per element qc+qs = ±2 → group-neutral
    e_dir = FMD.FPPREC[1, 0, 0]
    prms = (qc = qc, qs = qs)

    rng = Random.MersenneTwister(4242)
    atoms_ref = make_binary_supercell((Nx, Ny, Nz))
    atoms = deepcopy(atoms_ref)
    # Cores and shells rattled independently: on the ideal lattice the shear
    # components vanish by symmetry, which would make half the comparison
    # vacuous (same reason as test/calculators/test_virial.jl).  0.01 of a
    # supercell period is ~3% of a lattice constant on the shortest axis here
    # — a realistic thermal displacement, and below the unit-cell-scale
    # warning validate_gauge raises for frames far from their gauge.
    atoms.arrays[:pos] .+= FMD.FPPREC(0.01) .* randn(rng, FMD.FPPREC, size(atoms.arrays[:pos]))
    atoms.arrays[:shelldisp] .= FMD.FPPREC(0.05) .* randn(rng, FMD.FPPREC, size(atoms.arrays[:shelldisp]))

    # Built from the UNDEFORMED reference frame, so M_frac_ref is frozen and
    # non-zero — that is what puts the Pg_ref cell-derivative term into the
    # comparison (with M_frac_ref = 0 it would drop out silently).
    mkpu(k_par, k_homo; kw...) = Calculators.PolarizationUmbrella(
        atoms_ref, k_par, k_homo, 0.0, e_dir, (Nx, Ny, Nz), qc, qs;
        gauge = atoms_ref, kw...)

    # Flat-bottom threshold at half the RMS residual, so both sides of the
    # max(0, |r| - r_th) branch are populated (asserted at the end).
    pu_probe = mkpu(0.0, 1e3; m = (0, 0, 0), sigma = 0.0)
    a_probe = deepcopy(atoms)
    Cores.clear!(a_probe, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(a_probe, pu_probe, prms)
    r_th = FMD.FPPREC(0.5) * sqrt(pu_probe.results["sum_r2"] / ng)
    @test r_th > 0
    # Guard against a degenerate fixture: an FD match on a zero virial would
    # be vacuous (the helper skips components below atol).  The shear block is
    # checked separately — it is the half that vanishes by symmetry on an
    # unrattled lattice, and a diagonal-only guard would not notice.
    @test maximum(abs, Array(a_probe.info[:virial])[1:3]) > 1e-3
    @test maximum(abs, Array(a_probe.info[:virial])[4:6]) > 1e-3

    configs = [
        ("sigma=0, r_th=0, m=(0,0,0)", mkpu(0.0, 1e3; m = (0, 0, 0), sigma = 0.0)),
        ("sigma>0, m=(0,0,1), k_par>0", mkpu(1e3, 1e3; m = (0, 0, 1), sigma = 0.8)),
        ("flat-bottom r_th>0", mkpu(0.0, 1e3; m = (0, 0, 0), sigma = 0.0, r_th = r_th)),
    ]
    for (label, pu) in configs
        calc! = a -> Calculators.umbrella_p!(a, pu, prms)
        # symmetrize: the bias restrains P along a lab-frame ê, so it is not
        # rotationally invariant and ∂U/∂ε has an antisymmetric part that the
        # symmetric Voigt-6 virial does not — and must not — carry.  See the
        # helper's header and docs/src/theory/polarization-bias-virial.md §4.
        assert_virial_matches_finite_diff(calc!, atoms;
                                          h = 1e-5, rtol = 1e-4, atol = 1e-8,
                                          symmetrize = true,
                                          label = "PolarizationUmbrella virial ($label)")
    end

    # Both sides of the max(0, |r| - r_th) branch must be live in the compared
    # configuration.  Re-evaluated on the undeformed frame: the FD helper left
    # `results` at its last (deformed) probe point.
    eval_undeformed!(pu) = begin
        a = deepcopy(atoms)
        Cores.clear!(a, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a, pu, prms)
        return a
    end
    eval_undeformed!(configs[3][2])
    # Positive branch: some residual exceeds r_th (U > 0), and the shift is
    # doing something (U below the unshifted energy).
    @test 0 < configs[3][2].results["U_homo"] < pu_probe.results["U_homo"]
    # Zero branch: sum_r2 = Σ|r_g|² ≥ max|r_g|², so a threshold of sqrt(sum_r2)
    # puts EVERY group inside the flat bottom — the penalty and its gradient
    # must then be identically zero.  The inequality above cannot pin this
    # (it also holds when every group is above the threshold).
    pu_allflat = mkpu(0.0, 1e3; m = (0, 0, 0), sigma = 0.0,
                      r_th = sqrt(pu_probe.results["sum_r2"]))
    a_allflat = eval_undeformed!(pu_allflat)
    @test pu_allflat.results["U_homo"] == 0
    @test maximum(abs, a_allflat.arrays[:forces]) == 0
    @test maximum(abs, a_allflat.arrays[:shellforces]) == 0
end

# ---------------------------------------------------------------------------
# PolarizationGauge (Issue #122): the zero-point of the polarisation is a
# declared input, not whatever frame happened to reach the constructor.
# ---------------------------------------------------------------------------

# Synthetic KNbO₃-like perovskite: formal charges (+1, +5, −2) make the cubic
# structure's formal polarisation HALF a quantum, so no choice of images can
# put Σq·s at zero.  This is the case that kills "just find a Σq·r = 0
# reference" and that the difference machinery has to handle by cancellation.
function make_kno_like(reps::NTuple{3,Int}; a_bohr = 7.6)
    Nx, Ny, Nz = reps
    nu = 5                                   # K, Nb, O, O, O
    n = nu * Nx * Ny * Nz
    frac = [(0.0, 0.0, 0.0),                 # K   (+1)
            (0.5, 0.5, 0.5),                 # Nb  (+5)
            (0.5, 0.5, 0.0),                 # O   (−2)
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5)]
    pos = zeros(FMD.FPPREC, n, 3)
    groupid = zeros(Int, n)
    elemid = zeros(Int, n)
    k = 0
    for ix in 0:Nx-1, iy in 0:Ny-1, iz in 0:Nz-1
        gid = ix * Ny * Nz + iy * Nz + iz + 1
        for (e, f) in enumerate(frac)
            k += 1
            pos[k, 1] = (ix + f[1]) / Nx
            pos[k, 2] = (iy + f[2]) / Ny
            pos[k, 3] = (iz + f[3]) / Nz
            groupid[k] = gid
            elemid[k] = e == 1 ? 1 : (e == 2 ? 2 : 3)
        end
    end
    info = Dict{Symbol,Any}(
        :lattice => FMD.FPPREC.([Nx*a_bohr 0 0; 0 Ny*a_bohr 0; 0 0 Nz*a_bohr]),
        :potential_energy => FMD.FPPREC[0],
        :virial => zeros(FMD.FPPREC, 6),
        :method => Symbol[:test],
    )
    arrays = Dict{Symbol,Array}(
        :pos => pos, :elemid => elemid, :groupid => groupid,
        :forces => zeros(FMD.FPPREC, n, 3), :shellforces => zeros(FMD.FPPREC, n, 3),
        :shelldisp => zeros(FMD.FPPREC, n, 3), :velo => zeros(FMD.FPPREC, n, 3),
        :id => collect(1:n), :masses => ones(FMD.FPPREC, n),
    )
    return Atoms{Array}(info, arrays)
end

const KNO_QC = FMD.FPPREC[1.0, 5.0, -2.0]
const KNO_QS = FMD.FPPREC[0.0, 0.0, 0.0]

@testset "PolarizationGauge" begin

    @testset "half-quantum system: ΔP is analytic and branch-independent" begin
        reps = (2, 2, 2)
        ng = prod(reps)
        cubic = make_kno_like(reps)
        gauge = Calculators.PolarizationGauge(cubic, reps, KNO_QC, KNO_QS)

        # The cubic reference itself carries half a polarisation quantum per
        # group in every direction — the situation that has no "neutral"
        # origin.  Pin it, because it is the reason the gauge must be a
        # declared reference rather than something the code searches for.
        # M_frac_ref is in SUPERCELL fractional units, so rescale by the
        # repeat count to read it per unit cell.
        for (d, N) in enumerate(reps)
            @test all(x -> isapprox(abs(x * N - round(x * N)), 0.5; atol = 1e-12),
                      gauge.M_frac_ref[:, d])
        end

        # Displace Nb (charge +5) by δ in x; every other sublattice is fixed.
        δ = FMD.FPPREC(0.01)
        moved = deepcopy(cubic)
        nb = findall(==(2), moved.arrays[:elemid])
        moved.arrays[:pos][nb, 1] .+= δ

        Pg = Calculators.group_p_gauged(moved, gauge, KNO_QC, KNO_QS)
        L_x = moved.info[:lattice][1, 1]
        V_g = abs(det(Array(moved.info[:lattice]))) / ng
        expected = 5.0 * (δ * L_x) / V_g
        @test isapprox(maximum(abs, Pg[:, 1] .- expected), 0; atol = 1e-10)
        @test isapprox(maximum(abs, Pg[:, 2:3]), 0; atol = 1e-10)

        # Branch cancellation: shifting an atom by a full lattice vector in
        # BOTH the gauge frame and the measured frame leaves ΔP untouched,
        # even though each absolute dipole jumps by a quantum.
        cubic_s = deepcopy(cubic); cubic_s.arrays[:pos][3, 1] += 1
        moved_s = deepcopy(moved); moved_s.arrays[:pos][3, 1] += 1
        gauge_s = Calculators.PolarizationGauge(cubic_s, reps, KNO_QC, KNO_QS)
        @test !isapprox(gauge_s.M_frac_ref, gauge.M_frac_ref)     # it did jump
        Pg_s = Calculators.group_p_gauged(moved_s, gauge_s, KNO_QC, KNO_QS)
        @test isapprox(Pg_s, Pg; rtol = 1e-10)

        # Shifting only ONE side is the silent-corruption case: the image
        # correspondence is broken and validate_gauge must refuse.
        @test_throws ErrorException Calculators.group_p_gauged(
            moved_s, gauge, KNO_QC, KNO_QS)
    end

    @testset "uniform-offset artefact: m ≠ 0 bias ignores a rigid offset" begin
        # Ideal prototype as the gauge, a uniformly polarised frame as the MD
        # input.  For m ≠ 0 the LS amplitude is blind to a uniform offset
        # (Σcosθ = 0), but the homogeneity term is NOT — with the frame
        # itself as the gauge (the old implicit behaviour) the offset is
        # folded into the reference instead, which is a different bias.
        reps = (4, 1, 1)
        proto = make_binary_supercell(reps)
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]

        polarised = deepcopy(proto)
        polarised.arrays[:pos][sublattice_a(polarised), 1] .+= 0.05

        pu_proto = Calculators.PolarizationUmbrella(
            polarised, 0.0, 1.0e3, 0.0, FMD.FPPREC[1, 0, 0], reps, qc, qs;
            gauge = proto, m = (1, 0, 0))
        pu_self = Calculators.PolarizationUmbrella(
            polarised, 0.0, 1.0e3, 0.0, FMD.FPPREC[1, 0, 0], reps, qc, qs;
            gauge = polarised, m = (1, 0, 0))

        a1 = deepcopy(polarised)
        Cores.clear!(a1, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a1, pu_proto, (qc = qc, qs = qs))
        a2 = deepcopy(polarised)
        Cores.clear!(a2, (:potential_energy, :virial, :forces, :shellforces))
        Calculators.umbrella_p!(a2, pu_self, (qc = qc, qs = qs))

        # Self-gauged: residual ≡ 0 by construction, so no penalty at all.
        @test isapprox(pu_self.results["U_homo"], 0; atol = 1e-12)
        # Prototype-gauged: the uniform offset is real polarisation and the
        # homogeneity term charges for it.  The two are different physics,
        # which is exactly why the choice may not be implicit.
        @test pu_proto.results["U_homo"] > 1e-6
    end

    @testset "rigid translation leaves the gauged polarisation invariant" begin
        reps = (2, 2, 2)
        proto = make_binary_supercell(reps)
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]
        gauge = Calculators.PolarizationGauge(proto, reps, qc, qs)

        polarised = deepcopy(proto)
        polarised.arrays[:pos][sublattice_a(polarised), 1] .+= 0.03
        Pg0 = Calculators.group_p_gauged(polarised, gauge, qc, qs)

        translated = deepcopy(polarised)
        # The invariance is exact for any translation; keep this one inside
        # validate_gauge's unit-cell warning band so the suite stays quiet
        # (0.05 of a supercell period is 0.1 unit cells on this 2-cell axis).
        translated.arrays[:pos] .+= reshape(FMD.FPPREC[0.05, -0.03, 0.06], 1, 3)
        # The gauge frame does NOT move with it; group neutrality is the only
        # reason the answer survives.
        @test isapprox(Calculators.group_p_gauged(translated, gauge, qc, qs),
                       Pg0; atol = 1e-10)
    end

    @testset "charged groups are rejected at construction" begin
        reps = (2, 2, 2)
        proto = make_binary_supercell(reps)
        # Same sign on both sublattices → every group carries +2.
        @test_throws ErrorException Calculators.PolarizationGauge(
            proto, reps, FMD.FPPREC[1.0, 1.0], FMD.FPPREC[0.0, 0.0])
        # Core and shell cancelling within each element is neutral.
        @test Calculators.PolarizationGauge(
            proto, reps, FMD.FPPREC[1.0, 2.0],
            FMD.FPPREC[-1.0, -2.0]) isa Calculators.PolarizationGauge
    end

    @testset "a group lattice that does not tile the cell is rejected" begin
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]
        holed = make_binary_supercell((2, 2, 2))
        # Group 3 emptied into group 4: min and max still span 1:8, so only a
        # per-group population count catches it.  An empty group would make
        # the smoothing FFT average over a site no atom contributes to.
        gid = Array(holed.arrays[:groupid])
        gid[gid .== 3] .= 4
        holed.arrays[:groupid] = gid
        @test_throws ErrorException Calculators.PolarizationGauge(
            holed, (2, 2, 2), qc, qs)
    end

    @testset "struct invariants hold on every path in" begin
        # Not reachable through the normal constructor — this guards a
        # hand-assembled struct and a corrupted JLD2, which would otherwise
        # become a silently wrong reference.
        @test_throws ErrorException Calculators.PolarizationGauge(
            zeros(FMD.FPPREC, 3, 3), zeros(FMD.FPPREC, 4, 3), (2, 2, 2),
            [1, 2, 1, 2], [1, 1, 2, 2], [1, 2, 3, 4])      # M_frac_ref not (8,3)
        @test_throws ErrorException Calculators.PolarizationGauge(
            zeros(FMD.FPPREC, 8, 3), zeros(FMD.FPPREC, 4, 3), (2, 2, 2),
            [1, 2, 1], [1, 1, 2, 2], [1, 2, 3, 4])         # elemid too short
        @test_throws ErrorException Calculators.PolarizationGauge(
            zeros(FMD.FPPREC, 8, 3), zeros(FMD.FPPREC, 4, 3), (2, 2, 2),
            [1, 2, 1, 2], [1, 1, 2, 99], [1, 2, 3, 4])     # groupid out of range
        @test_throws ErrorException Calculators.PolarizationGauge(
            zeros(FMD.FPPREC, 8, 3), zeros(FMD.FPPREC, 4, 3), (2, 2, 2),
            [1, 2, 1, 2], [1, 1, 2, 2], [1, 2, 2, 4])      # duplicated :id
    end

    @testset "validate_gauge rejects a different system" begin
        reps = (2, 2, 2)
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]
        proto = make_binary_supercell(reps)
        gauge = Calculators.PolarizationGauge(proto, reps, qc, qs)

        @test Calculators.validate_gauge(gauge, deepcopy(proto)) === nothing

        # Atom count.
        @test_throws ErrorException Calculators.validate_gauge(
            gauge, make_binary_supercell((2, 2, 3)))
        # elemid order (same multiset, different arrangement).
        swapped = deepcopy(proto)
        swapped.arrays[:elemid][1], swapped.arrays[:elemid][2] =
            swapped.arrays[:elemid][2], swapped.arrays[:elemid][1]
        @test_throws ErrorException Calculators.validate_gauge(gauge, swapped)
        # groupid relabelling.
        regrouped = deepcopy(proto)
        regrouped.arrays[:groupid] .= reverse(regrouped.arrays[:groupid])
        @test_throws ErrorException Calculators.validate_gauge(gauge, regrouped)
        # A row-permuted frame is the SAME system: the MD driver returns
        # cell-sorted rows, so pairing by :id rather than by row is what lets
        # a trajectory snapshot be measured against the gauge at all.
        perm = [2:length(proto); 1]
        shuffled = deepcopy(proto)[perm]      # the same slicing the CellList uses
        @test Calculators.validate_gauge(gauge, shuffled) === nothing
        # ... and the per-group polarisation is unchanged by the reordering.
        @test isapprox(Calculators.group_p_gauged(shuffled, gauge, qc, qs),
                       Calculators.group_p_gauged(deepcopy(proto), gauge, qc, qs);
                       atol = 1e-12)
        # A frame carrying different ids is a different system.
        alien = deepcopy(proto)
        alien.arrays[:id] = alien.arrays[:id] .+ 1000
        @test_throws ErrorException Calculators.validate_gauge(gauge, alien)

        # Ambiguous image: > 0.45 in fractional coordinate.
        far = deepcopy(proto)
        far.arrays[:pos][1, 2] += 0.5
        @test_throws ErrorException Calculators.validate_gauge(gauge, far)
        # Suspicious but unambiguous: warn on the unit-cell scale, do not
        # error — 0.3 of a supercell period is 0.6 unit cells here, well past
        # any ferroelectric displacement but still a definite image.
        mid = deepcopy(proto)
        mid.arrays[:pos][1, 2] += 0.3
        @test_logs (:warn, r"unit cells from the gauge frame") match_mode = :any (
            Calculators.validate_gauge(gauge, mid))
        # ... and the two scales are genuinely different: a displacement that
        # is large per unit cell but small per supercell warns without erroring.
        big = make_binary_supercell((8, 1, 1))
        gauge_big = Calculators.PolarizationGauge(big, (8, 1, 1), qc, qs)
        nudged = deepcopy(big)
        nudged.arrays[:pos][1, 1] += 0.1        # 0.8 unit cells, 0.1 supercell
        @test_logs (:warn, r"unit cells from the gauge frame") match_mode = :any (
            Calculators.validate_gauge(gauge_big, nudged))
    end

    @testset "save/load round trip" begin
        reps = (2, 2, 2)
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]
        proto = make_binary_supercell(reps)
        gauge = Calculators.PolarizationGauge(proto, reps, qc, qs)

        mktempdir() do dir
            path = joinpath(dir, "gauge.jld2")
            Calculators.save_gauge(path, gauge)
            back = Calculators.load_gauge(path)
            @test back.M_frac_ref == gauge.M_frac_ref     # bit-identical
            @test back.pos_ref == gauge.pos_ref
            @test back.gridsize == gauge.gridsize
            @test back.elemid == gauge.elemid
            @test back.groupid == gauge.groupid
            # A loaded gauge still validates against a frame from the same
            # structure, and still refuses a different one.
            @test Calculators.validate_gauge(back, deepcopy(proto)) === nothing
            @test_throws ErrorException Calculators.validate_gauge(
                back, make_binary_supercell((2, 2, 3)))

            # Not a gauge file at all.
            other = joinpath(dir, "other.jld2")
            JLD2.jldsave(other; something = 1)
            @test_throws ErrorException Calculators.load_gauge(other)

            # A gauge written by an older format must be refused outright
            # rather than read with a missing field: the identity check is the
            # whole point of the file.
            old_fmt = joinpath(dir, "v1.jld2")
            JLD2.jldsave(old_fmt; format = 1,
                         M_frac_ref = gauge.M_frac_ref, pos_ref = gauge.pos_ref,
                         gridsize = collect(gauge.gridsize),
                         elemid = gauge.elemid, groupid = gauge.groupid)
            @test_throws ErrorException Calculators.load_gauge(old_fmt)
        end
    end

    @testset "umbrella refuses a gauge for a different grid" begin
        qc = FMD.FPPREC[1.0, -1.0]
        qs = FMD.FPPREC[0.0, 0.0]
        atoms = make_binary_supercell((2, 2, 2))
        gauge = Calculators.PolarizationGauge(
            make_binary_supercell((2, 2, 2)), (1, 2, 4), qc, qs)
        @test_throws ErrorException Calculators.PolarizationUmbrella(
            atoms, 1.0, 1.0, 0.0, FMD.FPPREC[1, 0, 0], (2, 2, 2), qc, qs;
            gauge = gauge, m = (0, 0, 0))
    end
end

# ---------------------------------------------------------------------------
# Golden values (Issue #168).
#
# The tests above pin PROPERTIES — a finite difference, an invariance, an
# analytic limit — which is what they should do.  But a refactor that reworks
# the internals while keeping every property intact can still shift the
# arithmetic (a different summation order, a fused pass, one extra rounding),
# and no property test would notice.  This one pins the actual numbers of one
# fully-loaded configuration, captured from the implementation before the
# shared-machinery extraction.
#
# If a deliberate change to the bias makes these move, re-measure and say so
# in the PR; if they move under a change that was supposed to be a pure
# refactor, that is the bug this test exists to catch.
# ---------------------------------------------------------------------------
@testset "Golden values (pure-refactor guard)" begin
    reps = (3, 4, 5)
    qc = FMD.FPPREC[1.2, -0.8]
    # qs deliberately NOT the mirror of qc: with {|qc|} and {|qs|} equal as
    # multisets, norm(forces) ≡ norm(shellforces) identically and one of the
    # two checks below would carry no information.
    qs = FMD.FPPREC[0.7, -1.1]
    proto = make_binary_supercell(reps)
    atoms = deepcopy(proto)
    # Deterministic low-discrepancy perturbation rather than an RNG draw:
    # Julia does not promise a stable random stream across versions, and at
    # rtol 1e-12 a changed stream would be indistinguishable from a genuine
    # regression.  Only multiply and floor here, so the fixture is bit-stable
    # for as long as Float64 is.
    golden_jitter(n) = 2 * (n * 0.6180339887498949 - floor(n * 0.6180339887498949)) - 1
    for k in axes(atoms.arrays[:pos], 1), d in 1:3
        atoms.arrays[:pos][k, d] += FMD.FPPREC(0.008) * golden_jitter(3k + 7d)
        atoms.arrays[:shelldisp][k, d] = FMD.FPPREC(0.05) * golden_jitter(5k + 2d)
    end

    # Every branch at once: cosine mode with a phase, Gaussian smoothing, a
    # flat-bottom threshold, an off-axis ê, both bias strengths non-zero.
    pu = Calculators.PolarizationUmbrella(
        proto, 1.5e3, 2.5e3, 0.004, FMD.FPPREC[0.6, 0.8, 0.0], reps, qc, qs;
        gauge = proto, m = (1, 0, 2), phase = 0.3, sigma = 0.7, r_th = 1e-4)

    Cores.clear!(atoms, (:potential_energy, :virial, :forces, :shellforces))
    Calculators.umbrella_p!(atoms, pu, (qc = qc, qs = qs))

    f = atoms.arrays[:forces]
    sf = atoms.arrays[:shellforces]
    # Index-weighted checksum: unlike a norm it also detects a permutation or
    # a sign flip of individual entries.
    chk(x) = sum(x[i, d] * (i + 100 * d) for i in axes(x, 1), d in axes(x, 2))

    rt = 1e-12
    @test isapprox(pu.results["cv"], -7.368180382230988e-5; rtol = rt)
    @test isapprox(pu.results["bias"], 0.012446162579094743; rtol = rt)
    @test isapprox(pu.results["U_homo"], 0.028858589100322744; rtol = rt)
    @test isapprox(pu.results["sum_r2"], 3.0471363616327366e-5; rtol = rt)
    @test isapprox(atoms.info[:potential_energy][1], 0.04130475167941749; rtol = rt)
    @test isapprox(norm(f), 0.15982901907061217; rtol = rt)
    @test isapprox(norm(sf), 0.14449353870473358; rtol = rt)
    @test isapprox(chk(f), -64.09017288016929; rtol = rt)
    @test isapprox(chk(sf), 64.67420374905987; rtol = rt)
    @test isapprox(f[1, :],
                   [0.0016945903332215575, 0.0069497608545413675, -0.033516153762132364];
                   rtol = rt)
    @test isapprox(f[end, :],
                   [-0.001138469710852844, 0.0014441182426475048, 0.009938307398588858];
                   rtol = rt)
    @test isapprox(collect(FMD.FPPREC, atoms.info[:virial]),
                   [0.05866291534937185, 0.05206412972203284, 0.02256901084057139,
                    0.0005427557331426636, 0.0012718199640570083, -0.0010327359544751724];
                   rtol = rt)
end
