# test/calculators/test_hvp.jl
#
# Analytic HVP building blocks (Issue #147).
#
# T1 — scalar second derivatives. d2ϕC / d2ϕB must be the exact
# r-derivatives of the *implemented* dϕC / dϕB (dϕC mixes erfc_approx
# values with the analytic erfc′ — d2ϕC differentiates that expression as
# written, see CoulombBuckingham.jl). The check is a central difference of
# the first-derivative kernels with a δ sweep: truncation is O(δ²) and
# roundoff O(eps/δ), so at least one δ in the sweep sits near the optimum
# (~1e-10 relative in Float64) — far below the asserted rtol.
using Test
using Random
using KernelAbstractions

# Central FD of f at r, accepted if ANY δ in the sweep matches — a single
# fixed δ can sit badly for a given curvature scale.
function _fd_matches(f, df, r; rtol, atol, deltas = (1e-4, 1e-5, 1e-6))
    return any(deltas) do δ
        fd = (f(r + δ) - f(r - δ)) / (2δ)
        isapprox(df(r), fd; rtol = rtol, atol = atol)
    end
end

# ---------------------------------------------------------------------------
# T2 (real-space subset) — hvp! with coulomb = nothing covers the
# short-range pair Hessian + Spring + intra closed forms. FD oracle: the
# same three terms of the force stack (full CoulombBuckingham — its
# core-core term is u-independent and cancels in the central difference —
# plus Spring plus the intra kernel), differentiated along normalized
# random directions with a δ sweep.
#
# Ordering matters: the analytic product is taken FIRST on a fresh
# workspace — the FD evaluator re-syncs the ghost arrays at perturbed
# shell displacements, which would silently violate the workspace's
# frozen-snapshot contract at O(δ).
# ---------------------------------------------------------------------------

function _hvp_sr_forces!(atoms, nl, prms)
    reset!(atoms)
    backend = KernelAbstractions.get_backend(atoms.arrays[:pos])
    Calculators.calculate!(Calculators.CoulombBuckingham(), atoms, nl, prms)
    Calculators.calculate!(Calculators.Spring(), atoms, prms)
    N = length(atoms)
    pe = zeros(FMD.FPPREC, N)
    vir = zeros(FMD.FPPREC, N, 6)
    arrays = (shelldisp = atoms.arrays[:shelldisp], elemid = atoms.arrays[:elemid],
              forces = atoms.arrays[:forces], shellforces = atoms.arrays[:shellforces])
    Calculators.kernel_subtractintraatom!(backend)(arrays, pe, vir,
        prms[:qc], prms[:qs], prms[:alpha]; ndrange = N)
    return copy(atoms.arrays[:shellforces]), copy(atoms.arrays[:forces])
end

function _hvp_sr_fd(atoms, nl, prms, v, δ)
    u = atoms.arrays[:shelldisp]
    @. u += δ * v
    fsp, fcp = _hvp_sr_forces!(atoms, nl, prms)
    @. u -= 2δ * v
    fsm, fcm = _hvp_sr_forces!(atoms, nl, prms)
    @. u += δ * v
    return (fsp .- fsm) ./ (2δ), (fcp .- fcm) ./ (2δ)
end

function _check_hvp_sr(atoms, prms; nvec = 4, rtol = 1e-6, atol = 1e-8)
    nl = neighborlist(atoms, prms[:rc], prms[:skin])
    N = length(atoms)
    for _ in 1:nvec
        v = randn(FMD.FPPREC, N, 3)
        v ./= sqrt(sum(abs2, v))

        ws = Calculators.hvp_workspace(atoms, nl, nothing, prms)
        Jv = zeros(FMD.FPPREC, N, 3)
        Kv = zeros(FMD.FPPREC, N, 3)
        # Purity: hvp! must not touch the atoms arrays.
        snap = (copy(atoms.arrays[:shelldisp]), copy(atoms.arrays[:shellforces]),
                copy(atoms.arrays[:forces]))
        Calculators.hvp!(Jv, Kv, ws.Wscratch, v, atoms, ws, prms)
        @test atoms.arrays[:shelldisp] == snap[1]
        @test atoms.arrays[:shellforces] == snap[2]
        @test atoms.arrays[:forces] == snap[3]
        # The Kv-skipping fast path returns the identical J block.
        Jv2 = zeros(FMD.FPPREC, N, 3)
        Calculators.hvp!(Jv2, nothing, ws.Wscratch, v, atoms, ws, prms)
        @test Jv2 == Jv

        @test any((1e-4, 1e-5, 1e-6)) do δ
            Jfd, Kfd = _hvp_sr_fd(atoms, nl, prms, v, δ)
            isapprox(Jv, Jfd; rtol = rtol, atol = atol) &&
                isapprox(Kv, Kfd; rtol = rtol, atol = atol)
        end
    end
end

@testset "HVP scalar second derivatives" begin
    @testset "derfc_approx = d/dx erfc_approx" begin
        for x in (0.1, 0.5, 1.0, 2.0, 4.0)
            @test _fd_matches(Calculators.erfc_approx, Calculators.derfc_approx,
                              FMD.FPPREC(x); rtol = 1e-8, atol = 1e-12)
        end
    end

    @testset "d2ϕC = d/dr dϕC (α and β branches)" begin
        alpha = FMD.FPPREC(0.3)
        # η pairs: strongly smeared (β ≪ α regime) and tight (β ≫ α) to
        # exercise both erfc branches with distinct magnitudes.
        for (ηi, ηj) in ((0.5, 0.5), (0.5, 1.5), (2.0, 2.0)),
            r in (0.8, 2.0, 5.0, 9.0)

            f = rr -> Calculators.dϕC(rr, alpha, FMD.FPPREC(ηi), FMD.FPPREC(ηj))
            df = rr -> Calculators.d2ϕC(rr, alpha, FMD.FPPREC(ηi), FMD.FPPREC(ηj))
            @test _fd_matches(f, df, FMD.FPPREC(r); rtol = 1e-8, atol = 1e-12)
        end
    end

    @testset "complex-step cross-check (independent oracle)" begin
        # Complex-step differentiation Im f(x+ih)/h is exact to machine
        # precision for these analytic scalar kernels and shares nothing
        # with the central-difference sweep — the implementation-
        # independent verification required by the theory-note house
        # rules (docs/src/theory/index.md).
        h = 1e-20
        for x in (0.1, 1.0, 3.0)
            cs = imag(Calculators.erfc_approx(complex(FMD.FPPREC(x), h))) / h
            @test isapprox(cs, Calculators.derfc_approx(FMD.FPPREC(x)); rtol = 1e-13)
        end
        alpha = FMD.FPPREC(0.3)
        for r in (0.9, 2.7, 6.3)
            cs = imag(Calculators.dϕC(complex(FMD.FPPREC(r), h), alpha,
                                      FMD.FPPREC(0.5), FMD.FPPREC(1.5))) / h
            @test isapprox(cs, Calculators.d2ϕC(FMD.FPPREC(r), alpha,
                                                FMD.FPPREC(0.5), FMD.FPPREC(1.5));
                           rtol = 1e-12)
        end
        for r in (1.5, 4.0)
            args = FMD.FPPREC.((50.0, 0.3, 0.6, 20.0, 1e-3))
            cs = imag(Calculators.dϕB(complex(FMD.FPPREC(r), h), args...)) / h
            @test isapprox(cs, Calculators.d2ϕB(FMD.FPPREC(r), args...); rtol = 1e-12)
        end
    end

    @testset "B-spline partition sum rules (M, dM, d2M)" begin
        # Σ_l M_l = 1, Σ_l M′_l = 0, Σ_l M″_l = 0 over the BSPORDER nodes
        # for any fractional offset — the property that makes a uniform
        # shell translation exact on the PME grid (analytic, not FD).
        for u in (0.0, 0.13, 0.5, 0.77, 0.999)
            u = FMD.FPPREC(u)
            @test isapprox(sum(Calculators.spline(u, -4 + l) for l in 1:4), 1;
                           atol = 1e-14)
            @test abs(sum(Calculators.dspline(u, -4 + l) for l in 1:4)) < 1e-13
            @test abs(sum(Calculators.d2spline(u, -4 + l) for l in 1:4)) < 1e-13
        end
    end

    @testset "d2ϕB = d/dr dϕB" begin
        # (a, b, rho, c, e): pure exponential, B ≠ 0 tail, dispersion with
        # the smooth cap e (never zero in production params — ρ=1 rows keep
        # a=0 so the values are inert there).
        for (a, b, rho, c, e) in ((100.0, 0.0, 0.7, 0.0, 1e-3),
                                  (100.0, 0.3, 0.7, 0.0, 1e-3),
                                  (50.0, 0.0, 0.35, 30.0, 1e-3),
                                  (0.0, 0.0, 1.0, 50.0, 1e-2)),
            r in (1.5, 3.0, 6.0)

            f = rr -> Calculators.dϕB(rr, FMD.FPPREC.((a, b, rho, c, e))...)
            df = rr -> Calculators.d2ϕB(rr, FMD.FPPREC.((a, b, rho, c, e))...)
            @test _fd_matches(f, df, FMD.FPPREC(r); rtol = 1e-8, atol = 1e-12)
        end
    end
end

@testset "hvp! real-space + spring + intra vs FD" begin
    @testset "NaCl-8, Coulomb + spring + intra" begin
        Random.seed!(11)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        prms = nacl_params(atoms; alpha = 0.30, with_shell = true)
        _check_hvp_sr(atoms, prms)
    end

    @testset "NaCl-8, + Buckingham (shell-shell)" begin
        Random.seed!(12)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = nacl_params(atoms; alpha = 0.30, with_shell = true)
        prms = (; base...,
                a = FMD.FPPREC[0.0 5.0; 5.0 8.0],
                b = FMD.FPPREC[0.0 0.2; 0.2 0.0],
                rho = FMD.FPPREC[1.0 0.9; 0.9 0.8],
                c = FMD.FPPREC[0.0 10.0; 10.0 20.0])
        _check_hvp_sr(atoms, prms)
    end

    @testset "perovskite-5, k4 ≠ 0 (quartic spring branch)" begin
        Random.seed!(13)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.08 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = perovskite_params(atoms; with_shell = true)
        prms = (; base..., k4 = FMD.FPPREC[0.0, 0.0, 600.0])
        _check_hvp_sr(atoms, prms)
    end

    @testset "guards: umbrella and stale workspaces are explicit errors" begin
        atoms = make_nacl_8()
        prms = nacl_params(atoms; with_shell = true)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        @test_throws ErrorException Calculators.hvp_workspace(atoms, nl, nothing,
            (; prms..., polarization_umbrella = :dummy))

        # Frozen-snapshot guard: moving the shells after construction must
        # error, not silently answer on the stale cache.
        ws = Calculators.hvp_workspace(atoms, nl, nothing, prms)
        atoms.arrays[:shelldisp][1, 1] += FMD.FPPREC(1e-3)
        N = length(atoms)
        out = zeros(FMD.FPPREC, N, 3)
        @test_throws ErrorException Calculators.hvp!(out, nothing, ws.Wscratch,
            zeros(FMD.FPPREC, N, 3), atoms, ws, prms)
    end
end

# ---------------------------------------------------------------------------
# T5 (Ewald path) — full-stack hvp! (short-range + spring + intra + Ewald
# reciprocal) against the FD oracles fd_jvp! / fd_dFdu_dirderiv! (the
# pre-#147 implementations, moved to test/helpers.jl), which differentiate
# the complete shell_forces! / core_forces! stacks. The external field is
# linear in u and cancels in the central difference, so the stacks match
# term for term. T3 — Hessian symmetry wᵀ(Jv) = vᵀ(Jw): J = -H_ss is
# symmetric by construction; this is the FD-independent check (most
# sensitive to mismatched pair bookkeeping).
# ---------------------------------------------------------------------------
@testset "hvp! full stack vs FD + symmetry (Ewald path)" begin
    function _check_full(atoms, prms; nvec = 3, rtol = 1e-6, atol = 1e-8)
        coulomb = Calculators.EwaldInfo(atoms.info[:lattice], prms)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        N = length(atoms)
        for _ in 1:nvec
            v = randn(FMD.FPPREC, N, 3); v ./= sqrt(sum(abs2, v))
            w = randn(FMD.FPPREC, N, 3); w ./= sqrt(sum(abs2, w))

            # Analytic products first: the FD oracle re-syncs ghosts at
            # perturbed displacements (frozen-snapshot contract).
            ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
            Jv = zeros(FMD.FPPREC, N, 3); Kv = zeros(FMD.FPPREC, N, 3)
            Jw = zeros(FMD.FPPREC, N, 3)
            Calculators.hvp!(Jv, Kv, ws.Wscratch, v, atoms, ws, prms)
            Calculators.hvp!(Jw, nothing, ws.Wscratch, w, atoms, ws, prms)

            # T3: symmetry of J (no FD involved).
            @test isapprox(sum(w .* Jv), sum(v .* Jw); rtol = 1e-10, atol = 1e-12)

            # T5: FD over the full force stacks, δ sweep.
            Jfd = zeros(FMD.FPPREC, N, 3); Kfd = zeros(FMD.FPPREC, N, 3)
            @test any((1e-4, 1e-5, 1e-6)) do δ
                fd_jvp!(Jfd, v, atoms, nl, coulomb, prms; delta = δ)
                fd_dFdu_dirderiv!(Kfd, v, atoms, nl, coulomb, prms; delta = δ)
                isapprox(Jv, Jfd; rtol = rtol, atol = atol) &&
                    isapprox(Kv, Kfd; rtol = rtol, atol = atol)
            end
        end
    end

    @testset "NaCl-8 (Coulomb + spring + intra + reciprocal)" begin
        Random.seed!(31)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        _check_full(atoms, nacl_params(atoms; alpha = 0.30, with_shell = true))
    end

    @testset "perovskite-5 (+ Buckingham, k4 ≠ 0)" begin
        Random.seed!(32)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.08 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = perovskite_params(atoms; with_shell = true)
        prms = (; base...,
                k4 = FMD.FPPREC[0.0, 0.0, 600.0],
                a = FMD.FPPREC[0.0 0.0 4.0; 0.0 0.0 6.0; 4.0 6.0 8.0],
                rho = FMD.FPPREC[1.0 1.0 0.9; 1.0 1.0 0.8; 0.9 0.8 0.85],
                c = FMD.FPPREC[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 15.0])
        _check_full(atoms, prms)
    end
end

# ---------------------------------------------------------------------------
# T5 (PME path) — full-stack hvp! with PMEInfo against the same FD
# oracles. Both sides differentiate the SAME discrete PME model
# (B-spline discretization included), so agreement is at the FD
# truncation level regardless of mesh — this is the
# derive-on-constrained-space property, not a discretization-accuracy
# statement. The sheared cell exercises the icell chain of the dipole
# spread / gathers (a cubic cell cancels transpose mistakes). T4 — the
# discretization statement: PME HVP vs the analytic Ewald HVP on a fine
# mesh, normalized by the reciprocal-only magnitude.
# ---------------------------------------------------------------------------
@testset "hvp! full stack vs FD (PME path)" begin
    function _check_full_pme(atoms, prms; meshsize = 0.5,
                             nvec = 3, rtol = 1e-6, atol = 1e-8)
        prms = (; prms..., pme_meshsize = FMD.FPPREC(meshsize))
        coulomb = Calculators.PMEInfo(atoms.info[:lattice], prms)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        N = length(atoms)
        for _ in 1:nvec
            v = randn(FMD.FPPREC, N, 3); v ./= sqrt(sum(abs2, v))
            w = randn(FMD.FPPREC, N, 3); w ./= sqrt(sum(abs2, w))

            ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
            Jv = zeros(FMD.FPPREC, N, 3); Kv = zeros(FMD.FPPREC, N, 3)
            Jw = zeros(FMD.FPPREC, N, 3)
            Calculators.hvp!(Jv, Kv, ws.Wscratch, v, atoms, ws, prms)
            Calculators.hvp!(Jw, nothing, ws.Wscratch, w, atoms, ws, prms)
            @test isapprox(sum(w .* Jv), sum(v .* Jw); rtol = 1e-10, atol = 1e-12)

            Jfd = zeros(FMD.FPPREC, N, 3); Kfd = zeros(FMD.FPPREC, N, 3)
            @test any((1e-4, 1e-5, 1e-6)) do δ
                fd_jvp!(Jfd, v, atoms, nl, coulomb, prms; delta = δ)
                fd_dFdu_dirderiv!(Kfd, v, atoms, nl, coulomb, prms; delta = δ)
                isapprox(Jv, Jfd; rtol = rtol, atol = atol) &&
                    isapprox(Kv, Kfd; rtol = rtol, atol = atol)
            end
        end
    end

    @testset "NaCl-8, cubic cell" begin
        Random.seed!(41)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        _check_full_pme(atoms, nacl_params(atoms; alpha = 0.30, with_shell = true))
    end

    @testset "NaCl-8, sheared cell (icell chain)" begin
        Random.seed!(42)
        atoms = make_nacl_8()
        lat = Matrix(atoms.info[:lattice])
        lat[2, 1] += 0.08 * lat[1, 1]   # b gains a component along a
        lat[3, 2] += 0.05 * lat[2, 2]
        atoms.info[:lattice] = FMD.FPPREC.(lat)
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        # rc from the (reduced) inscribed sphere of the sheared cell.
        _check_full_pme(atoms, nacl_params(atoms; alpha = 0.30,
                                           rc_factor = 0.40, with_shell = true))
    end
end

@testset "Ewald ↔ PME HVP equivalence (reciprocal discretization)" begin
    # Reciprocal-only comparison via the coulomb-cache terms. Measured on
    # this fixture (NaCl-8, α = 0.30): the J block converges one order
    # slower than the force interpolation — its local term reads the
    # piecewise-linear d2weights (order-4 B-spline second derivative) —
    # mesh 0.5 → 0.25 gives relJ 3.3e-3 → 4.6e-4 while relK goes
    # 1.2e-4 → 1.1e-5 (K has no local term). Pinned here: absolute
    # ceilings at mesh 0.25 (measured × ~4 margin) AND the halving of both
    # errors from mesh 0.5 → 0.25 (the convergence property — a broken
    # d2-gather typically stalls it). Production-parameter accuracy is the
    # bench script's job (bench/check_hvp_bsporder.jl, theory note §5).
    Random.seed!(43)
    atoms = make_nacl_8()
    atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
    base = nacl_params(atoms; alpha = 0.30, with_shell = true)
    N = length(atoms)
    nl = neighborlist(atoms, base[:rc], base[:skin])

    ew = Calculators.EwaldInfo(atoms.info[:lattice], base)
    cew = Calculators._hvp_coulomb_cache(ew, atoms, nl, base)
    v = randn(FMD.FPPREC, N, 3); v ./= sqrt(sum(abs2, v))
    Jew = zeros(FMD.FPPREC, N, 3); Kew = zeros(FMD.FPPREC, N, 3)
    Calculators.hvp_coulomb!(Jew, Kew, v, atoms, cew, base)
    sJ = maximum(abs, Jew); sK = maximum(abs, Kew)

    relerr = Dict{Float64,NTuple{2,Float64}}()
    for mesh in (0.5, 0.25)
        prms = (; base..., pme_meshsize = FMD.FPPREC(mesh))
        pm = Calculators.PMEInfo(atoms.info[:lattice], prms)
        cpm = Calculators._hvp_coulomb_cache(pm, atoms, nl, prms)
        Jpm = zeros(FMD.FPPREC, N, 3); Kpm = zeros(FMD.FPPREC, N, 3)
        Calculators.hvp_coulomb!(Jpm, Kpm, v, atoms, cpm, prms)
        relerr[mesh] = (maximum(abs, Jpm .- Jew)/sJ, maximum(abs, Kpm .- Kew)/sK)
    end
    @test relerr[0.25][1] < 2e-3
    @test relerr[0.25][2] < 5e-5
    @test relerr[0.25][1] < relerr[0.5][1] / 2
    @test relerr[0.25][2] < relerr[0.5][2] / 2
end

# ---------------------------------------------------------------------------
# T6 (Issue #151) — the virial output of hvp!. `Wv` (per-atom, Voigt-6)
# plus the returned reciprocal remainder must be the directional
# derivative of `info[:virial]` along `v`, i.e. the cell-strain
# counterpart of the Jv/Kv blocks. The FD oracle differentiates the SAME
# quantity the force path reports, so agreement is at the FD truncation
# level on both Coulomb paths.
#
# Two statements need no finite differences at all and are the ones that
# catch a mis-signed or double-counted pair block:
#   * linearity — D_{αv+βw} W = α·D_v W + β·D_w W;
#   * branch agreement — the Kv-skipping path must produce the IDENTICAL
#     virial, since the core-shell pair block moves with u whether or not
#     the mixed Hessian block was asked for.
# ---------------------------------------------------------------------------
@testset "hvp! virial derivative vs FD (Issue #151)" begin
    function _virial_fd(atoms, nl, coulomb, prms, v, δ)
        u = atoms.arrays[:shelldisp]
        @. u += δ * v
        Calculators.core_forces!(atoms, nl, coulomb, prms)
        Wp = copy(Array(atoms.info[:virial]))
        @. u -= 2δ * v
        Calculators.core_forces!(atoms, nl, coulomb, prms)
        Wm = copy(Array(atoms.info[:virial]))
        @. u += δ * v
        return (Wp .- Wm) ./ (2δ)
    end

    function _check_virial(atoms, prms, coulomb; nvec = 2, rtol = 1e-6)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        N = length(atoms)
        for _ in 1:nvec
            v = randn(FMD.FPPREC, N, 3); v ./= sqrt(sum(abs2, v))
            w = randn(FMD.FPPREC, N, 3); w ./= sqrt(sum(abs2, w))

            # Analytic first — the FD oracle moves the shells and would
            # break the workspace's frozen-snapshot contract at O(δ).
            ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
            Jv = zeros(FMD.FPPREC, N, 3); Kv = zeros(FMD.FPPREC, N, 3)
            Wv = zeros(FMD.FPPREC, N, 6)
            snap = copy(atoms.arrays[:shelldisp])
            Wrec = Calculators.hvp!(Jv, Kv, Wv, v, atoms, ws, prms)
            @test atoms.arrays[:shelldisp] == snap        # still pure
            Wtot = vec(sum(Wv, dims = 1)) .+ Wrec
            @test maximum(abs, Wtot) > 1e-8               # non-vacuous

            # Branch agreement: no Kv, identical virial.
            Wv2 = zeros(FMD.FPPREC, N, 6)
            Wrec2 = Calculators.hvp!(Jv, nothing, Wv2, v, atoms, ws, prms)
            @test Wv2 == Wv
            @test Wrec2 == Wrec

            # Linearity in the direction (no FD).
            Ww = zeros(FMD.FPPREC, N, 6)
            Wsum = zeros(FMD.FPPREC, N, 6)
            Wrecw = Calculators.hvp!(Jv, nothing, Ww, w, atoms, ws, prms)
            Wrecs = Calculators.hvp!(Jv, nothing, Wsum,
                                     FMD.FPPREC(2.0) .* v .- w, atoms, ws, prms)
            lhs = vec(sum(Wsum, dims = 1)) .+ Wrecs
            rhs = 2 .* Wtot .- (vec(sum(Ww, dims = 1)) .+ Wrecw)
            @test isapprox(lhs, rhs; rtol = 1e-9,
                           atol = 1e-12 * max(1.0, maximum(abs, rhs)))

            @test any((1e-4, 1e-5, 1e-6)) do δ
                isapprox(Wtot, _virial_fd(atoms, nl, coulomb, prms, v, δ);
                         rtol = rtol, atol = 1e-8 * maximum(abs, Wtot))
            end
        end
    end

    @testset "NaCl-8, $(name)" for (name, build) in
            (("Ewald", (a, p) -> Calculators.EwaldInfo(a.info[:lattice], p)),
             ("PME", (a, p) -> Calculators.PMEInfo(a.info[:lattice], p)))
        Random.seed!(71)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = nacl_params(atoms; alpha = 0.30, with_shell = true)
        prms = (; base..., pme_meshsize = FMD.FPPREC(0.5),
                a = FMD.FPPREC[0.0 5.0; 5.0 8.0],
                rho = FMD.FPPREC[1.0 0.9; 0.9 0.8],
                c = FMD.FPPREC[0.0 2.0; 2.0 4.0],
                k4 = FMD.FPPREC[300.0, 500.0])   # quartic spring branch live
        _check_virial(atoms, prms, build(atoms, prms))
    end

    # The intra-atom correction branches at |u| = 0.1/α and BOTH fixtures
    # above sit entirely inside the Taylor side (max|u| ≈ 0.11 and 0.22
    # against thresholds 0.33 and 0.50), so without this case the
    # non-Taylor virial term would never execute in the suite. The
    # displacement is large on purpose; the assertion below fails if a
    # future change to the fixture quietly pulls it back inside.
    @testset "NaCl-8, large shells (intra non-Taylor branch)" begin
        Random.seed!(73)
        atoms = make_nacl_8()
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.2 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = nacl_params(atoms; alpha = 0.30, with_shell = true)
        prms = (; base..., pme_meshsize = FMD.FPPREC(0.5),
                a = FMD.FPPREC[0.0 5.0; 5.0 8.0],
                rho = FMD.FPPREC[1.0 0.9; 0.9 0.8],
                c = FMD.FPPREC[0.0 2.0; 2.0 4.0],
                k4 = FMD.FPPREC[300.0, 500.0])
        @test maximum(sqrt.(sum(abs2, atoms.arrays[:shelldisp], dims = 2))) >
              FMD.FPPREC(1e-1) / prms[:alpha]      # the branch is really taken
        _check_virial(atoms, prms, Calculators.PMEInfo(atoms.info[:lattice], prms))
    end

    # k4 = 0 takes the other spring branch and the harmonic intra path.
    @testset "perovskite-5, k4 = 0 (PME)" begin
        Random.seed!(72)
        atoms = make_perovskite_5(; a_ang = 4.0)
        atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.08 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = perovskite_params(atoms; with_shell = true)
        prms = (; base..., pme_meshsize = FMD.FPPREC(0.5),
                a = FMD.FPPREC[0.0 0.0 4.0; 0.0 0.0 6.0; 4.0 6.0 8.0],
                rho = FMD.FPPREC[1.0 1.0 0.9; 1.0 1.0 0.8; 0.9 0.8 0.85],
                c = FMD.FPPREC[0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 15.0])
        _check_virial(atoms, prms, Calculators.PMEInfo(atoms.info[:lattice], prms))
    end
end

# ---------------------------------------------------------------------------
# T7 (Issue #181) — the surface term's rank-3 block. Three independent
# statements, only one of which involves finite differences:
#
#  (a) the closed form. `U_surf` depends on the positions through the
#      whole-system dipole alone, so the block it adds to hvp! is
#      `-q_i·(4π/V)T·(Σ_j q^s_j v_j)` on cores and shells — written out
#      here from the theory note (docs/src/theory/surface-term.md §3.5),
#      sharing no code with the implementation, and compared against the
#      DIFFERENCE between the products with and without the term;
#  (b) the null space. A direction with `Σ_j q^s_j v_j = 0` moves no
#      dipole, so the surface term must contribute exactly nothing —
#      bit-identical products, no tolerance involved;
#  (c) the FD oracles, which pick up the term for free (fd_jvp! /
#      fd_dFdu_dirderiv! differentiate the full shell_forces! /
#      core_forces! stacks, and those include :surface_term).
#
# The gauge frame is the UNRATTLED prototype on purpose: with
# `gauge = atoms` the run frame is its own reference, `δM = 0`, and both
# `g` and the whole virial derivative vanish identically — the Hessian
# blocks would still be exercised but the `D_v W` half of the term would
# be silently vacuous.
#
# The shell charges are enlarged to |q_s| = 1 (the fixture's ±0.2 makes
# the block ~25× smaller, since it scales as q_s²) so that (a) and the
# virial statement sit well above the FD noise floor.
# ---------------------------------------------------------------------------
@testset "hvp! surface term (Issue #181)" begin
    function _surf_fixture(; seed = 81)
        Random.seed!(seed)
        proto = make_nacl_8()
        atoms = deepcopy_atoms(proto)
        atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
        atoms.arrays[:shelldisp] .= 0.05 .*
            randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
        base = nacl_params(atoms; alpha = 0.30, with_shell = true)
        # qc + qs stays ±0.8 per element (the fixture's neutral split), so
        # the system is neutral and SurfaceTerm accepts it.
        base = (; base..., qc = FMD.FPPREC[1.8, -1.8],
                qs = FMD.FPPREC[-1.0, 1.0], pme_meshsize = FMD.FPPREC(0.5))
        # Anisotropic ε_out and a rotated ellipsoid: T is neither a
        # multiple of I nor diagonal, so a transposed or symmetrised T
        # cannot pass unnoticed.
        R = [0.8 -0.6 0.0; 0.6 0.8 0.0; 0.0 0.0 1.0]
        st = Calculators.SurfaceTerm(atoms, base[:qc], base[:qs];
                                     eps_out = R * Diagonal([1.5, 3.0, 8.0]) * R',
                                     shape = [1.0, 1.4, 0.7], gauge = proto)
        return atoms, base, (; base..., surface_term = st), st
    end

    # Products with and without the term, on one shared snapshot.
    function _pair(atoms, base, prms, coulomb, nl, v)
        N = length(atoms)
        out = map((prms, base)) do p
            ws = Calculators.hvp_workspace(atoms, nl, coulomb, p)
            J = zeros(FMD.FPPREC, N, 3); K = zeros(FMD.FPPREC, N, 3)
            W = zeros(FMD.FPPREC, N, 6)
            Wrec = Calculators.hvp!(J, K, W, v, atoms, ws, p)
            (J = J, K = K, W = W, Wrec = Wrec)
        end
        return out[1], out[2]        # (with surface, without)
    end

    # The block, written from the note rather than from the code.
    function _closed_form(atoms, prms, st, v)
        N = length(atoms)
        qs = [prms[:qs][atoms.arrays[:elemid][i]] for i in 1:N]
        qc = [prms[:qc][atoms.arrays[:elemid][i]] for i in 1:N]
        V = volume(atoms)
        S = vec(sum(qs .* v, dims = 1))
        lat = Matrix{FMD.FPPREC}(Array(atoms.info[:lattice]))
        M = vec(sum(qc .* (atoms.arrays[:pos] * lat) .+
                    qs .* (atoms.arrays[:pos] * lat .+ atoms.arrays[:shelldisp]),
                    dims = 1))
        dM = M .- transpose(lat) * st.M_frac_ref
        g = (4π / V) * st.T * dM
        c = (4π / V) * st.T * S
        dU = dot(g, S)
        sym(a, b) = [a[1]*b[1], a[2]*b[2], a[3]*b[3],
                     (a[2]*b[3] + a[3]*b[2])/2, (a[1]*b[3] + a[3]*b[1])/2,
                     (a[1]*b[2] + a[2]*b[1])/2]
        DW = [dU, dU, dU, 0, 0, 0] .- sym(c, dM) .- sym(g, S)
        return (J = -qs * c', K = -qc * c', DW = DW, S = S)
    end

    @testset "closed form, null space and FD [$label]" for (label, build) in
            (("Ewald", (a, p) -> Calculators.EwaldInfo(a.info[:lattice], p)),
             ("PME", (a, p) -> Calculators.PMEInfo(a.info[:lattice], p)))
        atoms, base, prms, st = _surf_fixture()
        N = length(atoms)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        coulomb = build(atoms, prms)

        v = randn(FMD.FPPREC, N, 3); v ./= sqrt(sum(abs2, v))
        w = randn(FMD.FPPREC, N, 3); w ./= sqrt(sum(abs2, w))

        with, without = _pair(atoms, base, prms, coulomb, nl, v)
        ref = _closed_form(atoms, prms, st, v)

        # (a) exact rank-3 closed form. Non-vacuous first: the block has
        # to be a real contribution, not a rounding artefact.
        jscale = max(1.0, maximum(abs, without.J))
        @test maximum(abs, ref.J) > 1e-4 * jscale
        @test maximum(abs, ref.DW) > 1e-4
        @test maximum(abs, (with.J .- without.J) .- ref.J) <
              1e-12 * max(1.0, maximum(abs, ref.J))
        @test maximum(abs, (with.K .- without.K) .- ref.K) <
              1e-12 * max(1.0, maximum(abs, ref.K))
        # The virial derivative is whole-system: it rides Wrec, and the
        # per-atom rows must be untouched (the peratomstress columns omit
        # it, exactly as they omit the reciprocal part).
        @test with.W == without.W
        @test maximum(abs, (with.Wrec .- without.Wrec) .- ref.DW) <
              1e-12 * max(1.0, maximum(abs, ref.DW))

        # (b) null space: a direction that moves no dipole is invisible to
        # the term. Two opposite-charge rows carrying the SAME vector make
        # `S = Σ q^s_j v_j` exactly zero in floating point — every partial
        # sum of {0, …, 0, -w, +w} is representable, whatever order the
        # reduction takes — so the products may be compared bitwise
        # instead of against a tolerance. (Projecting a random direction
        # would leave S at ~1e-17 and make the comparison depend on that
        # landing below half an ulp of Jv.)
        qs = [prms[:qs][atoms.arrays[:elemid][i]] for i in 1:N]
        ipos = findfirst(>(0), qs)
        ineg = findfirst(<(0), qs)
        # The construction below needs both signs and an exact cancellation
        # — assert the fixture still provides them, rather than failing
        # later as an indexing error on `nothing`.
        @test ipos !== nothing && ineg !== nothing
        @test qs[ipos] == -qs[ineg]      # the fixture's ±1 split
        v0 = zeros(FMD.FPPREC, N, 3)
        v0[ipos, :] .= v[ipos, :]
        v0[ineg, :] .= v[ipos, :]
        @test dot(qs, v0[:, 1]) == 0
        with0, without0 = _pair(atoms, base, prms, coulomb, nl, v0)
        @test maximum(abs, without0.J) > 1e-6     # not comparing zeros
        @test with0.J == without0.J
        @test with0.K == without0.K
        @test with0.Wrec == without0.Wrec

        # T3 with the term in place: J = -H_ss is still symmetric.
        ws = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
        Jv = zeros(FMD.FPPREC, N, 3); Jw = zeros(FMD.FPPREC, N, 3)
        Kv = zeros(FMD.FPPREC, N, 3)
        Wv = zeros(FMD.FPPREC, N, 6)
        Wrec = Calculators.hvp!(Jv, Kv, Wv, v, atoms, ws, prms)
        Calculators.hvp!(Jw, nothing, ws.Wscratch, w, atoms, ws, prms)
        @test isapprox(sum(w .* Jv), sum(v .* Jw); rtol = 1e-10, atol = 1e-12)

        # (c) FD over the full stacks, which include the surface term.
        Jfd = zeros(FMD.FPPREC, N, 3); Kfd = zeros(FMD.FPPREC, N, 3)
        @test any((1e-4, 1e-5, 1e-6)) do δ
            fd_jvp!(Jfd, v, atoms, nl, coulomb, prms; delta = δ)
            fd_dFdu_dirderiv!(Kfd, v, atoms, nl, coulomb, prms; delta = δ)
            isapprox(Jv, Jfd; rtol = 1e-6, atol = 1e-8) &&
                isapprox(Kv, Kfd; rtol = 1e-6, atol = 1e-8)
        end

        # Virial derivative against a central difference of info[:virial]
        # (the analytic product was taken above, before the FD moved the
        # shells). The surface part is 3 orders above the FD residual, so
        # this sees it rather than merely tolerating it.
        Wtot = vec(sum(Wv, dims = 1)) .+ Wrec
        wsnap = copy(atoms.arrays[:shelldisp])
        @test any((1e-4, 1e-5, 1e-6)) do δ
            u = atoms.arrays[:shelldisp]
            @. u += δ * v
            Calculators.core_forces!(atoms, nl, coulomb, prms)
            Wp = copy(Array(atoms.info[:virial]))
            @. u -= 2δ * v
            Calculators.core_forces!(atoms, nl, coulomb, prms)
            Wm = copy(Array(atoms.info[:virial]))
            @. u += δ * v
            isapprox(Wtot, (Wp .- Wm) ./ (2δ); rtol = 1e-6,
                     atol = 1e-8 * maximum(abs, Wtot))
        end
        @test atoms.arrays[:shelldisp] ≈ wsnap
    end

    # A workspace freezes δM and 1/V, so a prms that disagrees with it
    # would silently drop (or double) the term. Refuse instead.
    @testset "workspace / prms mismatch is an error" begin
        atoms, base, prms, st = _surf_fixture()
        N = length(atoms)
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        coulomb = Calculators.PMEInfo(atoms.info[:lattice], prms)
        v = zeros(FMD.FPPREC, N, 3)
        Jv = zeros(FMD.FPPREC, N, 3); Wv = zeros(FMD.FPPREC, N, 6)

        ws_with = Calculators.hvp_workspace(atoms, nl, coulomb, prms)
        ws_without = Calculators.hvp_workspace(atoms, nl, coulomb, base)
        @test_throws ErrorException Calculators.hvp!(Jv, nothing, Wv, v, atoms,
                                                     ws_with, base)
        @test_throws ErrorException Calculators.hvp!(Jv, nothing, Wv, v, atoms,
                                                     ws_without, prms)
        # Same gauge and tensors, different object: still refused — the
        # check is identity, because a SurfaceTerm carries a frozen
        # reference the cache was built from.
        st2 = Calculators.SurfaceTerm(atoms, base[:qc], base[:qs];
                                      eps_out = Matrix(st.eps_out),
                                      shape = Matrix(st.shape), gauge = atoms)
        @test_throws ErrorException Calculators.hvp!(Jv, nothing, Wv, v, atoms,
            ws_with, (; base..., surface_term = st2))
    end

    # The conductor-boundary dielectric tensor is not what a screened
    # response measures, so linear_response! refuses rather than relabel.
    @testset "linear_response! refuses the surface term" begin
        atoms, base, prms, _ = _surf_fixture()
        nl = neighborlist(atoms, prms[:rc], prms[:skin])
        ew = Calculators.EwaldInfo(atoms.info[:lattice], prms)
        @test_throws ErrorException Calculators.linear_response!(atoms, nl, ew,
                                                                 prms)
        # Without it the same call is fine (the refusal is about the term,
        # not about this fixture). :becex/y/z are the BEC slabs the
        # routine writes into and does not allocate itself.
        N = length(atoms)
        for k in (:becex, :becey, :becez)
            atoms.arrays[k] = zeros(FMD.FPPREC, N, 3)
        end
        @test Calculators.linear_response!(atoms, nl, ew, base) isa Integer
    end
end
