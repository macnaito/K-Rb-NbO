# test/apps/test_xlbomd.jl
#
# Shadow XL-BOMD (Issue #149). The load-bearing assertions are the two
# that do not depend on the implementation agreeing with itself:
#
#   T1  the shadow force is the EXACT gradient of the shadow energy at a
#       frozen auxiliary field (central FD over atomic positions) — the
#       property the whole scheme's energy conservation rests on;
#   T3  with the auxiliary field sitting at the converged SCF solution the
#       shadow minimizer coincides with it, the Hessian correction
#       vanishes, and the shadow force must reproduce the ordinary relaxed
#       force — no finite differences involved.
#
# The rest pin the closed form, the driver integration, the fallback, the
# guards, and the invariance of the default (:scf) path.

using Test
using Random
using LinearAlgebra
using FMD.Apps: npt!

const XL = FMD.Apps

# Rattled shells and cores: on the ideal rocksalt the shell forces vanish
# by symmetry and every comparison below would be vacuous.
function _xl_fixture(; seed = 5, quartic = false)
    Random.seed!(seed)
    atoms = make_nacl_8()
    atoms.arrays[:pos] .+= 0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
    atoms.arrays[:shelldisp] .= 0.05 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
    prms = nacl_params(atoms; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    # Non-zero Buckingham: with a ≡ 0 the fixture is bare Coulomb + springs
    # and collapses, which makes any conserved-quantity assertion a test of
    # the collapse rather than of the integrator.
    prms = (; prms..., pme_meshsize = FMD.FPPREC(0.4 / UNITS[:Å]),
            skin = FMD.FPPREC(0.0),
            a = FMD.FPPREC[0.0 5.0; 5.0 8.0],
            rho = FMD.FPPREC[1.0 0.9; 0.9 0.8],
            c = FMD.FPPREC[0.0 2.0; 2.0 4.0],
            k4 = quartic ? FMD.FPPREC[300.0, 500.0] : prms[:k4])
    return atoms, prms
end

# calc!(atoms) closure that evaluates the shadow energy/force at a FIXED
# auxiliary field. The neighbour list and the Coulomb solver are rebuilt
# per call because the FD displaces the atoms.
_xl_calc(prms, shat, make_coulomb) = function (a)
    nl = neighborlist(a, prms[:rc], prms[:skin])
    XL.xl_shadow_forces!(a, nl, make_coulomb(a), prms, shat)
end

@testset "shadow XL-BOMD" begin

    @testset "T1: shadow force = -∇(shadow energy) at frozen ŝ" begin
        for quartic in (false, true), usepme in (false, true)
            atoms, prms = _xl_fixture(; quartic = quartic)
            mk = usepme ? (a -> Calculators.PMEInfo(a.info[:lattice], prms)) :
                          (a -> Calculators.EwaldInfo(a.info[:lattice], prms))
            # ŝ deliberately away from the relaxed shells so the linearized
            # part of the shadow functional is actually exercised.
            Random.seed!(77)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
            label = "shadow force = -∇E" * (quartic ? " [k4≠0]" : "") *
                    (usepme ? " [PME]" : " [Ewald]")
            assert_force_matches_finite_diff(_xl_calc(prms, shat, mk), atoms;
                                             h = 1e-5, rtol = 1e-5, atol = 1e-8,
                                             label = label)
        end
    end

    @testset "T2: shadow minimizer closed form" begin
        for quartic in (false, true)
            atoms, prms = _xl_fixture(; quartic = quartic)
            a = deepcopy_atoms(atoms)
            nl = neighborlist(a, prms[:rc], prms[:skin])
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            shat = copy(a.arrays[:shelldisp])
            XL.xl_shadow_forces!(a, nl, ew, prms, shat)
            s1 = copy(a.arrays[:shelldisp])

            # F_pair at ŝ, recomputed independently of the kernel.
            b = deepcopy_atoms(atoms)
            b.arrays[:shelldisp] .= shat
            nlb = neighborlist(b, prms[:rc], prms[:skin])
            Calculators.core_forces!(b, nlb, Calculators.EwaldInfo(b.info[:lattice], prms), prms)
            N = length(b)
            @test maximum(abs, s1) > 1e-4          # non-vacuous
            for i in 1:N
                e = b.arrays[:elemid][i]
                k2 = prms[:k2][e]; k4 = prms[:k4][e]
                d2 = sum(abs2, view(shat, i, :))
                fpair = view(b.arrays[:shellforces], i, :) .+ (k2 + k4*d2/6) .* view(shat, i, :)
                m2 = sum(abs2, view(s1, i, :))
                resid = (k2 + k4*m2/6) .* view(s1, i, :) .- fpair
                @test maximum(abs, resid) < 1e-10 * max(1.0, maximum(abs, fpair))
            end
        end
    end

    @testset "T3: ŝ = converged SCF solution reproduces the relaxed force" begin
        atoms, base = _xl_fixture()
        prms = (; base..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 500)

        ref = deepcopy_atoms(atoms)
        nlr = neighborlist(ref, prms[:rc], prms[:skin])
        ewr = Calculators.EwaldInfo(ref.info[:lattice], prms)
        Calculators.calculate!(Calculators.CoreShell(), ref, nlr, ewr, prms)
        sstar = copy(ref.arrays[:shelldisp])
        Fref = copy(ref.arrays[:forces])
        Eref = ref.info[:potential_energy][1]

        a = deepcopy_atoms(atoms)
        nl = neighborlist(a, prms[:rc], prms[:skin])
        ew = Calculators.EwaldInfo(a.info[:lattice], prms)
        E, maxdev = XL.xl_shadow_forces!(a, nl, ew, prms, sstar)

        @test maximum(abs, Fref) > 1e-4                       # non-vacuous
        # The shadow minimizer is the SCF solution, so the correction term
        # is zero and both the force and the energy fall back to the
        # relaxed ones (up to the residual shell force at cgtol).
        @test maxdev < 1e-8
        @test maximum(abs, a.arrays[:forces] .- Fref) < 1e-6 * maximum(abs, Fref)
        @test isapprox(E, Eref; rtol = 1e-10)
    end

    @testset "T4/T5: driver integration, ncg column, fallback" begin
        atoms, base = _xl_fixture()
        dt = FMD.FPPREC(1.0 / UNITS[:fs])
        common = (; base...,
                  backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  skin = FMD.FPPREC(0.1),
                  cgtol = FMD.FPPREC(1e-8), cgmaxiter = 200,
                  nsteps = 8, timestep = dt,
                  thermostat = :none, barostat = :none,
                  initial_temperature = 0.0, target_temperature = 0.0,
                  output_interval = 100)

        # conserved column (18) over the post-warmup window.
        _cons(dir) = [parse(Float64, split(l)[18])
                      for l in eachline(joinpath(dir, "log.txt"))]
        _ncg(dir) = [parse(Int, last(split(l)))
                     for l in eachline(joinpath(dir, "log.txt"))]

        @testset "XL steps cost no CG iterations, drift stays SCF-grade" begin
            # Reference: the ordinary path at a tight tolerance, i.e. the
            # best conservation the relaxed surface offers on this fixture.
            drift_ref = mktempdir() do dir
                npt!(deepcopy_atoms(atoms), (; common..., output_dir = dir))
                c = _cons(dir)
                maximum(abs, c[4:end] .- c[4])
            end
            mktempdir() do dir
                p = (; common..., shell_dynamics = :xlbomd, xl_warmup = 3,
                     output_dir = dir)
                out = npt!(deepcopy_atoms(atoms), p)
                @test all(isfinite, out.arrays[:pos])
                ncg = _ncg(dir)
                @test length(ncg) == 8
                @test all(>(0), ncg[1:3])       # warmup relaxes
                @test all(==(0), ncg[4:end])    # then pure shadow steps
                # shell_rms (column 19) is a residual only where a CG ran:
                # finite on the warmup steps, NaN on the shadow steps —
                # never the stale value of the last warmup.
                rms = [parse(Float64, split(l)[19])
                       for l in eachline(joinpath(dir, "log.txt"))]
                @test all(isfinite, rms[1:3])
                @test all(isnan, rms[4:end])
                # Shadow dynamics conserves its own invariant at least as
                # well as the tight SCF path conserves the physical one
                # (relative comparison — the absolute scale is the
                # integrator's, not the scheme's).
                c = _cons(dir)
                @test maximum(abs, c[4:end] .- c[4]) < 5 * drift_ref + 1e-12
            end
        end

        @testset "fallback fires and reseeds when the deviation is capped" begin
            mktempdir() do dir
                p = (; common..., shell_dynamics = :xlbomd, xl_warmup = 3,
                     xl_max_deviation = FMD.FPPREC(1e-14), output_dir = dir)
                logs, out = Test.collect_test_logs() do
                    npt!(deepcopy_atoms(atoms), p)
                end
                @test all(isfinite, out.arrays[:pos])
                @test any(l -> occursin("exceeded the threshold", string(l.message)), logs)
                # The end-of-run summary is the authoritative fallback
                # count. The ncg column is not: a reseed whose shells are
                # already inside cgtol costs zero CG iterations, so
                # ncg == 0 does not imply a free shadow step (it does the
                # other way round — ncg > 0 always means a relaxation ran).
                summary = filter(l -> occursin("fallback relaxation", string(l.message)), logs)
                @test length(summary) == 1
                nfb = parse(Int, match(r"(\d+) fallback", string(summary[1].message))[1])
                @test nfb >= 1
            end
        end
    end

    @testset "T6: guards" begin
        atoms, base = _xl_fixture()
        dt = FMD.FPPREC(1.0 / UNITS[:fs])
        p = (; base..., backend = :cpu, elements = Dict("Na" => 1, "Cl" => 2),
             skin = FMD.FPPREC(0.1), nsteps = 1, timestep = dt,
             tau_t = 20dt, tau_p = 500dt,
             initial_temperature = 0.0, target_temperature = 10.0,
             target_stress = zeros(FMD.FPPREC, 6), output_interval = 10)

        # A barostat is allowed since the shadow virial landed (#151); the
        # cell must actually move and stay finite. (The pressure itself is
        # pinned by T9 and by the bench comparison against :scf.)
        mktempdir() do dir
            out = npt!(deepcopy_atoms(atoms),
                (; p..., thermostat = :none, barostat = :mttk, nsteps = 6,
                 shell_dynamics = :xlbomd, xl_warmup = 2, output_dir = dir))
            @test all(isfinite, out.arrays[:pos])
            @test all(isfinite, Array(out.info[:lattice]))
            @test Array(out.info[:lattice]) != Array(atoms.info[:lattice])
        end
        # Validator: the umbrella rejection, plus the "only read by" ones.
        @test_throws ArgumentError FMD.Apps.validate_params(:NPTMD,
            (; p..., thermostat = :none, barostat = :none,
             input = "x.xyz", output_dir = "o", pme_meshsize = FMD.FPPREC(0.4),
             seed = 0, shell_dynamics = :bogus))
        @test_throws ArgumentError FMD.Apps.validate_params(:NPTMD,
            (; p..., thermostat = :none, barostat = :none,
             input = "x.xyz", output_dir = "o", pme_meshsize = FMD.FPPREC(0.4),
             seed = 0, xl_kappa = 0.5))       # :scf must reject :xl_*

        # k2 = 0 with a live quartic: the closed-form minimizer cannot
        # reach the purely quartic root, and s₁ = 0 would silently
        # disagree with the :scf path — refuse instead.
        mktempdir() do dir
            @test_throws ErrorException npt!(deepcopy_atoms(atoms),
                (; p..., thermostat = :none, barostat = :none,
                 k2 = FMD.FPPREC[0.0, 20.0], k4 = FMD.FPPREC[300.0, 0.0],
                 shell_dynamics = :xlbomd, output_dir = dir))
        end
        # k2 = 0 with k4 = 0 is the ordinary shell-less species: allowed.
        mktempdir() do dir
            out = npt!(deepcopy_atoms(atoms),
                (; p..., thermostat = :none, barostat = :none, nsteps = 4,
                 k2 = FMD.FPPREC[0.0, 20.0], k4 = FMD.FPPREC[0.0, 0.0],
                 shell_dynamics = :xlbomd, xl_warmup = 2, output_dir = dir))
            @test all(isfinite, out.arrays[:pos])
        end
    end

    # T8 (Issue #152). The kernel only decides how ŝ moves; the load-bearing
    # assertion is therefore not about forces but about the direction — that
    # the streaming Horner loop really evaluates the declared polynomial of
    # A⁻¹H_ss, checked against a DENSE operator assembled from hvp! and
    # dense matrix powers, which share no code with the implementation.
    @testset "T8: Chebyshev kernel preconditioning" begin

        @testset "cheb_kernel_coefs: minimax identity and one-sided safety" begin
            for (order, a, b) in ((2, 0.4, 3.5), (3, 0.4, 6.0), (4, 0.3, 5.0))
                c = (a + b) / 2; d = (b - a) / 2
                pc = XL.cheb_kernel_coefs(order, a, b)
                # p has degree order-1, i.e. `order` coefficients; the
                # extra Hessian-vector products per step are length(pc) - 1.
                @test length(pc) == order
                # T_order by its own recurrence, evaluated pointwise.
                function tcheb(t)
                    tm1, tk = 1.0, t
                    for _ in 2:order
                        tm1, tk = tk, 2t*tk - tm1
                    end
                    return tk
                end
                σ = tcheb(c / d)
                for λ in range(a, b; length = 51)
                    @test isapprox(1 - λ * evalpoly(λ, pc), tcheb((c - λ)/d) / σ;
                                   rtol = 1e-9, atol = 1e-12)
                end
                # On [a, b] the preconditioned spectrum is inside 1 ± ρ, so
                # κ·λp < 2κ: κ = 1.8 is stable for ANY declared interval.
                ρ = 1 / abs(σ)
                @test ρ < 1
                for λ in range(a, b; length = 201)
                    @test 1 - ρ - 1e-9 <= λ * evalpoly(λ, pc) <= 1 + ρ + 1e-9
                end
                # Below a: weaker but never negative (safe to under-declare).
                for λ in range(1e-6, a; length = 201)
                    @test λ * evalpoly(λ, pc) >= -1e-12
                end
                # Above b the band is left and κ = 1.8 stops being stable
                # (sign change for even order, growth past 4/κ for odd) —
                # exactly why b must bound the true λmax.
                @test any(!(0 < 1.8 * λ * evalpoly(λ, pc) < 4)
                          for λ in range(b, 3b; length = 200))
            end
            @test_throws ErrorException XL.cheb_kernel_coefs(1, 0.4, 3.5)
            @test_throws ErrorException XL.cheb_kernel_coefs(3, 3.5, 0.4)
            # An absurdly wide interval is not an error: it preconditions
            # almost nothing but stays inside the stable band.
            pw = XL.cheb_kernel_coefs(2, 1e-6, 1e6)
            @test all(0 <= λ * evalpoly(λ, pw) <= 2
                      for λ in range(1e-6, 1e6; length = 1000))
        end

        @testset "the direction is p(A⁻¹H_ss)·A⁻¹F_s(ŝ)" begin
            atoms, prms = _xl_fixture(; quartic = true)
            N = length(atoms)
            Random.seed!(91)
            shat = atoms.arrays[:shelldisp] .+
                   0.02 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))

            # Dense H_ss = -J at ŝ, column by column, and the Jacobi base.
            ref = deepcopy_atoms(atoms)
            ref.arrays[:shelldisp] .= shat
            nlr = neighborlist(ref, prms[:rc], prms[:skin])
            ewr = Calculators.EwaldInfo(ref.info[:lattice], prms)
            Calculators.core_forces!(ref, nlr, ewr, prms)
            Fs = copy(ref.arrays[:shellforces])
            ws = Calculators.hvp_workspace(ref, nlr, ewr, prms)
            H = zeros(FMD.FPPREC, 3N, 3N)
            ek = zeros(FMD.FPPREC, N, 3)
            Jv = similar(ek)
            for k in 1:3N
                fill!(ek, 0); ek[k] = 1
                Calculators.hvp!(Jv, nothing, ws.Wscratch, ek, ref, ws, prms)
                H[:, k] .= vec(-Jv)
            end
            kv3 = repeat(Array(ws.kvals), 3)       # vec() stacks columns
            M = Diagonal(1 ./ kv3) * H
            y = vec(Fs) ./ kv3
            @test maximum(abs, y) > 1e-6                      # non-vacuous

            for (order, a, b) in ((2, 0.4, 6.0), (3, 0.4, 6.0), (4, 0.3, 5.0))
                pc = XL.cheb_kernel_coefs(order, a, b)
                # Dense Horner-free reference: explicit matrix powers.
                xd = zeros(FMD.FPPREC, 3N)
                Mp = Matrix{FMD.FPPREC}(I, 3N, 3N)
                for j in eachindex(pc)
                    xd .+= pc[j] .* (Mp * y)
                    j < length(pc) && (Mp = M * Mp)
                end
                a2 = deepcopy_atoms(atoms)
                a2.arrays[:xl_dir] = zero(a2.arrays[:shelldisp])
                nl = neighborlist(a2, prms[:rc], prms[:skin])
                ew = Calculators.EwaldInfo(a2.info[:lattice], prms)
                XL.xl_shadow_forces!(a2, nl, ew, prms, copy(shat); kernel_coefs = pc)
                @test maximum(abs, vec(a2.arrays[:xl_dir]) .- xd) <
                      1e-9 * maximum(abs, xd)
            end

            # The spring kernel is the 0th-order member of the same family.
            a3 = deepcopy_atoms(atoms)
            a3.arrays[:xl_dir] = zero(a3.arrays[:shelldisp])
            nl = neighborlist(a3, prms[:rc], prms[:skin])
            ew = Calculators.EwaldInfo(a3.info[:lattice], prms)
            XL.xl_shadow_forces!(a3, nl, ew, prms, copy(shat))
            @test a3.arrays[:xl_dir] == a3.arrays[:shelldisp] .- shat
        end

        @testset "the kernel is inert at the relaxed shells" begin
            # ŝ = s* ⇒ F_s(ŝ) = 0 ⇒ zero driver for either kernel, so T3
            # (shadow force = relaxed force) is kernel-independent.
            atoms, base = _xl_fixture()
            prms = (; base..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 500)
            ref = deepcopy_atoms(atoms)
            nlr = neighborlist(ref, prms[:rc], prms[:skin])
            ewr = Calculators.EwaldInfo(ref.info[:lattice], prms)
            Calculators.calculate!(Calculators.CoreShell(), ref, nlr, ewr, prms)
            sstar = copy(ref.arrays[:shelldisp])
            Fref = copy(ref.arrays[:forces])

            for kc in (nothing, XL.cheb_kernel_coefs(3, 0.4, 6.0))
                a = deepcopy_atoms(atoms)
                a.arrays[:xl_dir] = zero(a.arrays[:shelldisp])
                nl = neighborlist(a, prms[:rc], prms[:skin])
                ew = Calculators.EwaldInfo(a.info[:lattice], prms)
                XL.xl_shadow_forces!(a, nl, ew, prms, sstar; kernel_coefs = kc)
                @test maximum(abs, a.arrays[:xl_dir]) < 1e-8
                @test maximum(abs, a.arrays[:forces] .- Fref) < 1e-6 * maximum(abs, Fref)
            end
        end

        @testset "shell-less species keep the spring direction" begin
            # k2 = 0 marks a species with no shell degree of freedom: s₁ ≡ 0,
            # so ŝ must be driven back to zero. The polynomial cannot know
            # that (the Jacobi base substitutes a unit spring), hence the mask.
            atoms, base = _xl_fixture()
            prms = (; base..., k2 = FMD.FPPREC[0.0, 20.0], k4 = FMD.FPPREC[0.0, 0.0])
            Random.seed!(31)
            shat = 0.03 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
            a = deepcopy_atoms(atoms)
            a.arrays[:xl_dir] = zero(a.arrays[:shelldisp])
            nl = neighborlist(a, prms[:rc], prms[:skin])
            ew = Calculators.EwaldInfo(a.info[:lattice], prms)
            XL.xl_shadow_forces!(a, nl, ew, prms, copy(shat);
                                 kernel_coefs = XL.cheb_kernel_coefs(3, 0.4, 6.0))
            dead = findall(i -> prms[:k2][a.arrays[:elemid][i]] <= 0, 1:length(a))
            live = setdiff(1:length(a), dead)
            @test !isempty(dead) && !isempty(live)
            @test a.arrays[:xl_dir][dead, :] == -shat[dead, :]
            @test maximum(abs, a.arrays[:xl_dir][live, :]) > 1e-6   # non-vacuous

            # ...and the shell-less rows must not leak INTO the live ones.
            # `hvp_workspace` substitutes a unit spring for k2 = 0 in its
            # Jacobi diagonal, so without projecting the polynomial's input
            # (and every image of it) those rows would carry a residual of
            # their own through hvp! into the rows that do have shells —
            # preconditioning an operator with a fictitious spring spliced
            # in, whose spectrum has no reason to lie in the declared
            # interval. Zeroing the dead rows' shell force must therefore
            # leave the live directions bit-identical.
            b = deepcopy_atoms(atoms)
            b.arrays[:xl_dir] = zero(b.arrays[:shelldisp])
            b.arrays[:shelldisp] .= shat
            nlb = neighborlist(b, prms[:rc], prms[:skin])
            ewb = Calculators.EwaldInfo(b.info[:lattice], prms)
            Calculators.core_forces!(b, nlb, ewb, prms)
            @test maximum(abs, b.arrays[:shellforces][dead, :]) > 1e-6  # non-vacuous
            b.arrays[:shellforces][dead, :] .= 0
            ws = Calculators.hvp_workspace(b, nlb, ewb, prms)
            vspring = a.arrays[:shelldisp] .- shat
            XL._xl_cheb_dir!(b.arrays[:xl_dir], XL.cheb_kernel_coefs(3, 0.4, 6.0),
                             b, ws, prms, vspring)
            @test b.arrays[:xl_dir][live, :] == a.arrays[:xl_dir][live, :]
        end

        @testset "κ default follows the kernel" begin
            atoms, base = _xl_fixture()
            xl(p...) = XL.xl_init!(deepcopy_atoms(atoms),
                                   (; base..., shell_dynamics = :xlbomd, p...))
            st = xl()
            @test st.kernel === :spring && st.kappa == 0.5 && isempty(st.kcoef)
            # With the spectrum flattened the literature value applies —
            # the whole point of the kernel (theory note §6.1).
            st = xl(:xl_kernel => :chebyshev)
            @test st.kernel === :chebyshev && st.kappa == 1.8 && length(st.kcoef) == 3
            st = xl(:xl_kernel => :chebyshev, :xl_kappa => 0.7,
                    :xl_kernel_order => 4)
            @test st.kappa == 0.7 && length(st.kcoef) == 4
            @test_throws ErrorException xl(:xl_kernel => :bogus)
            @test_throws ErrorException xl(:xl_kernel => :chebyshev,
                                           :xl_kernel_order => 1)
            @test_throws ErrorException xl(:xl_kernel => :chebyshev,
                                           :xl_kernel_spectrum => (3.0, 1.0))
        end

        @testset "driver: kernel steps cost no CG iterations and stay bounded" begin
            atoms, base = _xl_fixture()
            dt = FMD.FPPREC(1.0 / UNITS[:fs])
            common = (; base..., backend = :cpu,
                      elements = Dict("Na" => 1, "Cl" => 2),
                      skin = FMD.FPPREC(0.1),
                      cgtol = FMD.FPPREC(1e-8), cgmaxiter = 200,
                      nsteps = 8, timestep = dt,
                      thermostat = :none, barostat = :none,
                      initial_temperature = 0.0, target_temperature = 0.0,
                      output_interval = 100)
            drift_ref = mktempdir() do dir
                npt!(deepcopy_atoms(atoms), (; common..., output_dir = dir))
                c = [parse(Float64, split(l)[18]) for l in eachline(joinpath(dir, "log.txt"))]
                maximum(abs, c[4:end] .- c[4])
            end
            mktempdir() do dir
                out = npt!(deepcopy_atoms(atoms),
                           (; common..., shell_dynamics = :xlbomd, xl_warmup = 3,
                            xl_kernel = :chebyshev, output_dir = dir))
                @test all(isfinite, out.arrays[:pos])
                rows = [split(l) for l in eachline(joinpath(dir, "log.txt"))]
                ncg = [parse(Int, last(r)) for r in rows]
                @test all(>(0), ncg[1:3])
                @test all(==(0), ncg[4:end])       # no fallback, no relaxation
                c = [parse(Float64, r[18]) for r in rows]
                @test maximum(abs, c[4:end] .- c[4]) < 5 * drift_ref + 1e-12
            end
        end

        @testset "validator: kernel keys are conditional" begin
            atoms, base = _xl_fixture()
            dt = FMD.FPPREC(1.0 / UNITS[:fs])
            mktempdir() do dir
                inp = joinpath(dir, "x.xyz")
                touch(inp)                    # validate_params stats the input
                base_p = (; base..., backend = :cpu, nsteps = 1, timestep = dt,
                          thermostat = :none, barostat = :none,
                          initial_temperature = 0.0, target_temperature = 0.0,
                          output_interval = 10, input = inp, output_dir = dir,
                          pme_meshsize = FMD.FPPREC(0.4), seed = 0,
                          elements = Dict("Na" => 1, "Cl" => 2))
                v(p...) = FMD.Apps.validate_params(:NPTMD, (; base_p..., p...))
                # The accepted combination, so the rejections below are not
                # vacuously satisfied by an unrelated validation failure.
                @test v(:shell_dynamics => :xlbomd, :xl_kernel => :chebyshev,
                        :xl_kernel_order => 3,
                        :xl_kernel_spectrum => (0.4, 6.0)) === nothing
                @test_throws ArgumentError v(:xl_kernel => :chebyshev)     # :scf
                @test_throws ArgumentError v(:shell_dynamics => :xlbomd,
                                             :xl_kernel => :bogus)
                @test_throws ArgumentError v(:shell_dynamics => :xlbomd,
                                             :xl_kernel_order => 3)        # :spring
                @test_throws ArgumentError v(:shell_dynamics => :xlbomd,
                                             :xl_kernel => :chebyshev,
                                             :xl_kernel_order => 1)
                @test_throws ArgumentError v(:shell_dynamics => :xlbomd,
                                             :xl_kernel => :chebyshev,
                                             :xl_kernel_spectrum => (6.0, 0.4))
            end
        end

        @testset "the spring kernel is bit-identical to the default" begin
            atoms, base = _xl_fixture()
            dt = FMD.FPPREC(1.0 / UNITS[:fs])
            common = (; base..., backend = :cpu,
                      elements = Dict("Na" => 1, "Cl" => 2),
                      skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(1e-8),
                      cgmaxiter = 200, nsteps = 6, timestep = dt,
                      thermostat = :none, barostat = :none,
                      initial_temperature = 0.0, target_temperature = 0.0,
                      output_interval = 100,
                      shell_dynamics = :xlbomd, xl_warmup = 3)
            a = mktempdir() do dir
                npt!(deepcopy_atoms(atoms), (; common..., output_dir = dir))
            end
            b = mktempdir() do dir
                npt!(deepcopy_atoms(atoms),
                     (; common..., xl_kernel = :spring, output_dir = dir))
            end
            @test a.arrays[:pos] == b.arrays[:pos]
            @test a.arrays[:shelldisp] == b.arrays[:shelldisp]
        end
    end

    # -----------------------------------------------------------------
    # T9 (Issue #151) — the shadow virial. Same shape as T1/T3: the
    # load-bearing check is a central difference of the shadow ENERGY,
    # here with respect to a cell strain at frozen Cartesian ŝ, and the
    # rest are exact statements that involve no finite differences.
    # -----------------------------------------------------------------
    @testset "T9: shadow virial" begin
        VOIGT = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
        function _strain(k, δ)
            E = Matrix{FMD.FPPREC}(I, 3, 3)
            (α, β) = VOIGT[k]
            E[α, β] += δ
            E[β, α] = E[α, β]
            return E
        end
        _symouter(A, B) = [sum((A[:, α] .* B[:, β] .+ A[:, β] .* B[:, α]) ./ 2)
                           for (α, β) in VOIGT]

        # (a) W_shadow = -∂E_shadow/∂ε with ŝ held FIXED in Cartesian —
        # the convention the auxiliary Verlet implements (the barostat
        # never rescales ŝ). PME only: the grid size is frozen by
        # `cellupdate!`, so the strained energies belong to one smooth
        # function of the cell, whereas an Ewald k-set can gain or lose a
        # vector under the strain and break the difference. The Ewald
        # assembly is covered without FD by (b) below, and its reciprocal
        # virial derivative by test_hvp.jl's T6.
        @testset "= -∂E_shadow/∂ε at frozen ŝ [k4≠0: $quartic]" for quartic in
                (false, true)
            atoms, prms = _xl_fixture(; quartic = quartic)
            L0 = Matrix{FMD.FPPREC}(Array(atoms.info[:lattice]))
            Random.seed!(78)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
            pme = Calculators.PMEInfo(atoms.info[:lattice], prms)

            # No `cellupdate!` here on purpose: the strained energies are
            # only one smooth function of the cell if xl_shadow_forces!
            # refreshes the reciprocal factors itself, which is what a
            # barostat run depends on. Leaving it out makes this test the
            # check for that too.
            function _run(L)
                a = deepcopy_atoms(atoms)
                a.info[:lattice] .= L
                nl = neighborlist(a, prms[:rc], prms[:skin])
                E, _ = XL.xl_shadow_forces!(a, nl, pme, prms, copy(shat))
                return E, copy(Array(a.info[:virial]))
            end

            _, W = _run(L0)
            @test maximum(abs, W) > 1e-6                     # non-vacuous
            @test any((1e-4, 1e-5, 1e-6)) do δ
                all(1:6) do k
                    dE = (_run(L0 * _strain(k, FMD.FPPREC(δ)))[1] -
                          _run(L0 * _strain(k, -FMD.FPPREC(δ)))[1]) / (2δ)
                    # ε_αβ = ε_βα = δ for the shears, so dE/dδ = -2·W_αβ.
                    isapprox(-dE / (k <= 3 ? 1 : 2), W[k];
                             rtol = 1e-5, atol = 1e-6 * maximum(abs, W))
                end
            end
        end

        # (b) ŝ at the converged SCF solution: the correction vanishes and
        # the shadow virial must be the ordinary relaxed one (no FD).
        @testset "ŝ = s* reproduces the relaxed virial [$label]" for
                (label, mk) in
                (("Ewald", (a, p) -> Calculators.EwaldInfo(a.info[:lattice], p)),
                 ("PME", (a, p) -> Calculators.PMEInfo(a.info[:lattice], p)))
            atoms, base = _xl_fixture()
            prms = (; base..., cgtol = FMD.FPPREC(1e-10), cgmaxiter = 500)
            ref = deepcopy_atoms(atoms)
            nlr = neighborlist(ref, prms[:rc], prms[:skin])
            Calculators.calculate!(Calculators.CoreShell(), ref, nlr,
                                   mk(ref, prms), prms)
            sstar = copy(ref.arrays[:shelldisp])
            Wref = copy(Array(ref.info[:virial]))

            a = deepcopy_atoms(atoms)
            nl = neighborlist(a, prms[:rc], prms[:skin])
            XL.xl_shadow_forces!(a, nl, mk(a, prms), prms, sstar)
            @test maximum(abs, Wref) > 1e-6
            @test maximum(abs, Array(a.info[:virial]) .- Wref) <
                  1e-6 * maximum(abs, Wref)
        end

        # (c) The u-only cancellation, exactly. With every pair term off
        # the shadow energy is E_spring(s₁) alone, which depends on a
        # Cartesian displacement and therefore has NO strain derivative —
        # so the spring virial `core_forces!` reports at ŝ must be
        # cancelled to roundoff by the three correction terms. This is the
        # strain counterpart of the force-side cancellation lemma, and it
        # fails loudly if any of the four pieces drops or double-counts
        # its spring/intra part.
        @testset "spring-only cell ⇒ exactly zero [k4≠0: $quartic]" for quartic in
                (false, true)
            # The fixture's own k2/k4 are overwritten below (the point is a
            # cell with nothing but a spring), so it is built unrattled.
            atoms, base = _xl_fixture()
            prms = (; base...,
                    qc = zeros(FMD.FPPREC, 2), qs = zeros(FMD.FPPREC, 2),
                    a = zeros(FMD.FPPREC, 2, 2), c = zeros(FMD.FPPREC, 2, 2),
                    k2 = FMD.FPPREC[10.0, 20.0],
                    k4 = quartic ? FMD.FPPREC[300.0, 500.0] : zeros(FMD.FPPREC, 2))
            a = deepcopy_atoms(atoms)
            nl = neighborlist(a, prms[:rc], prms[:skin])
            pme = Calculators.PMEInfo(a.info[:lattice], prms)
            shat = copy(atoms.arrays[:shelldisp])
            XL.xl_shadow_forces!(a, nl, pme, prms, shat)

            # The spring virial that has to be cancelled, measured on its own.
            b = deepcopy_atoms(atoms)
            b.arrays[:shelldisp] .= shat
            Cores.clear!(b, (:potential_energy, :virial, :shellforces, :forces))
            Calculators.calculate!(Calculators.Spring(), b, prms)
            scale = maximum(abs, Array(b.info[:virial]))
            @test scale > 1e-4                                # non-vacuous
            @test maximum(abs, Array(a.info[:virial])) < 1e-12 * scale
        end

        # (d) External field. `ExternalField.calculate!` adds to the shell
        # force but nothing to `info[:virial]`, so the assembly must not
        # convert that shell force into a strain derivative either — the
        # reported virial has to match a hand assembly built from
        # F_s − q_s·E, and must NOT match the one built from the raw F_s.
        @testset "external field is excluded from the assembly" begin
            atoms, base = _xl_fixture()
            prms = (; base..., efield = FMD.FPPREC[2e-3, -1e-3, 5e-4])
            Random.seed!(79)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))

            a = deepcopy_atoms(atoms)
            nl = neighborlist(a, prms[:rc], prms[:skin])
            pme = Calculators.PMEInfo(a.info[:lattice], prms)
            XL.xl_shadow_forces!(a, nl, pme, prms, copy(shat))
            Wrep = copy(Array(a.info[:virial]))

            # Independent assembly: W(ŝ) and D_v W by finite difference in
            # λ, F_s / s₁ / Jv from a separate evaluation.
            b = deepcopy_atoms(atoms)
            b.arrays[:shelldisp] .= shat
            nlb = neighborlist(b, prms[:rc], prms[:skin])
            pmeb = Calculators.PMEInfo(b.info[:lattice], prms)
            Calculators.core_forces!(b, nlb, pmeb, prms)
            W0 = copy(Array(b.info[:virial]))
            Fs = copy(b.arrays[:shellforces])
            s1 = copy(a.arrays[:shelldisp])
            v = s1 .- shat
            ws = Calculators.hvp_workspace(b, nlb, pmeb, prms)
            Jv = similar(v)
            Calculators.hvp!(Jv, nothing, ws.Wscratch, v, b, ws, prms)
            function _W(s)
                b.arrays[:shelldisp] .= s
                Calculators.core_forces!(b, nlb, pmeb, prms)
                copy(Array(b.info[:virial]))
            end
            h = FMD.FPPREC(1e-4)
            DvW = (_W(shat .+ h .* v) .- _W(shat .- h .* v)) ./ (2h)

            qsv = FMD.FPPREC[prms[:qs][e] for e in atoms.arrays[:elemid]]
            Fs_nofield = Fs .- qsv .* permutedims(Array(prms[:efield]))
            Wok = W0 .+ DvW .- _symouter(Fs_nofield, s1) .- _symouter(Jv, shat)
            Wraw = W0 .+ DvW .- _symouter(Fs, s1) .- _symouter(Jv, shat)

            scale = maximum(abs, Wrep)
            resid = maximum(abs, Wrep .- Wok)
            @test resid < 1e-6 * scale
            # …and the field term is material: the assembly that keeps it
            # sits orders of magnitude further away than the agreement
            # floor above, so this is a real branch and not rounding.
            @test maximum(abs, Wrep .- Wraw) > 1e3 * max(resid, eps(FMD.FPPREC))
        end
    end

    # T10 (Issue #181). The surface term used to be refused here because
    # its depolarization field had no analytic HVP; now that it has one,
    # the two statements that carried the scheme in T1/T9 have to hold
    # with the term switched on — and one new one: the surface shell force
    # must be treated as the OPPOSITE of the external field in the virial
    # assembly (it does contribute to `info[:virial]`, so its conversion
    # term belongs there).
    #
    # The gauge is the unrattled prototype: with the run frame as its own
    # gauge δM = 0, and the entire term — energy, force and virial —
    # vanishes identically, so every check below would pass vacuously.
    @testset "T10: surface term (Issue #181)" begin
        function _xl_surf_fixture(; seed = 5, quartic = false)
            atoms, base = _xl_fixture(; seed = seed, quartic = quartic)
            proto = make_nacl_8()
            # |q_s| = 1 instead of the fixture's ±0.2: the block scales as
            # q_s², and at ±0.2 it sits 25× lower, close enough to the FD
            # floor to make the comparisons weak. qc + qs stays ±0.8, so
            # the frame is still neutral (SurfaceTerm requires it).
            prms = (; base..., qc = FMD.FPPREC[1.8, -1.8],
                    qs = FMD.FPPREC[-1.0, 1.0])
            R = [0.8 -0.6 0.0; 0.6 0.8 0.0; 0.0 0.0 1.0]
            st = Calculators.SurfaceTerm(atoms, prms[:qc], prms[:qs];
                eps_out = R * Diagonal([1.5, 3.0, 8.0]) * R',
                shape = [1.0, 1.4, 0.7], gauge = proto)
            return atoms, (; prms..., surface_term = st), st
        end

        # (a) T1 with the term: the shadow force is still the exact
        # gradient of the shadow energy at frozen ŝ.
        @testset "shadow force = -∇E_shadow [$label]" for (label, mkc) in
                (("Ewald", (p, a) -> Calculators.EwaldInfo(a.info[:lattice], p)),
                 ("PME", (p, a) -> Calculators.PMEInfo(a.info[:lattice], p)))
            atoms, prms, _ = _xl_surf_fixture()
            Random.seed!(77)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
            assert_force_matches_finite_diff(
                _xl_calc(prms, shat, a -> mkc(prms, a)), atoms;
                h = 1e-5, rtol = 1e-5, atol = 1e-8,
                label = "shadow force = -∇E [surface, $label]")
        end

        # (b) T9(a) with the term: the reported virial is still the exact
        # cell-strain derivative of E_shadow at frozen Cartesian ŝ. The
        # strain is symmetric by construction, which is what makes the
        # comparison legitimate for a term whose ∂U/∂ε has an
        # antisymmetric part (surface-term.md §4).
        @testset "shadow virial = -∂E_shadow/∂ε" begin
            VOIGT = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
            function _strain(k, δ)
                E = Matrix{FMD.FPPREC}(I, 3, 3)
                (α, β) = VOIGT[k]
                E[α, β] += δ
                E[β, α] = E[α, β]
                return E
            end
            atoms, prms, _ = _xl_surf_fixture()
            L0 = Matrix{FMD.FPPREC}(Array(atoms.info[:lattice]))
            Random.seed!(78)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
            pme = Calculators.PMEInfo(atoms.info[:lattice], prms)
            function _run(L)
                a = deepcopy_atoms(atoms)
                a.info[:lattice] .= L
                nl = neighborlist(a, prms[:rc], prms[:skin])
                E, _ = XL.xl_shadow_forces!(a, nl, pme, prms, copy(shat))
                return E, copy(Array(a.info[:virial]))
            end
            _, W = _run(L0)
            @test maximum(abs, W) > 1e-6
            @test any((1e-4, 1e-5, 1e-6)) do δ
                all(1:6) do k
                    dE = (_run(L0 * _strain(k, FMD.FPPREC(δ)))[1] -
                          _run(L0 * _strain(k, -FMD.FPPREC(δ)))[1]) / (2δ)
                    isapprox(-dE / (k <= 3 ? 1 : 2), W[k];
                             rtol = 1e-5, atol = 1e-6 * maximum(abs, W))
                end
            end
        end

        # (c) The mirror of T9(d): unlike the external field, the surface
        # term DOES contribute to `info[:virial]`, so the assembly must
        # convert its shell force too. Hand-assembled both ways — the one
        # that keeps the surface force must match, the one that strips it
        # (the ExternalField treatment) must not.
        @testset "the surface shell force stays in the assembly" begin
            VOIGT = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))
            _symouter(A, B) = [sum((A[:, α] .* B[:, β] .+ A[:, β] .* B[:, α]) ./ 2)
                               for (α, β) in VOIGT]
            atoms, prms, st = _xl_surf_fixture()
            Random.seed!(79)
            shat = atoms.arrays[:shelldisp] .+
                   0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))

            a = deepcopy_atoms(atoms)
            nl = neighborlist(a, prms[:rc], prms[:skin])
            pme = Calculators.PMEInfo(a.info[:lattice], prms)
            XL.xl_shadow_forces!(a, nl, pme, prms, copy(shat))
            Wrep = copy(Array(a.info[:virial]))

            b = deepcopy_atoms(atoms)
            b.arrays[:shelldisp] .= shat
            nlb = neighborlist(b, prms[:rc], prms[:skin])
            pmeb = Calculators.PMEInfo(b.info[:lattice], prms)
            Calculators.core_forces!(b, nlb, pmeb, prms)
            W0 = copy(Array(b.info[:virial]))
            Fs = copy(b.arrays[:shellforces])
            # The surface part of F_s, from the closed form: -q_s·g.
            gsurf = Vector{FMD.FPPREC}(st.results["dM"])
            gsurf = (4π / volume(b)) .* (st.T * gsurf)
            qsv = FMD.FPPREC[prms[:qs][e] for e in atoms.arrays[:elemid]]
            Fs_nosurf = Fs .+ qsv .* permutedims(gsurf)

            s1 = copy(a.arrays[:shelldisp])
            v = s1 .- shat
            ws = Calculators.hvp_workspace(b, nlb, pmeb, prms)
            Jv = similar(v)
            Calculators.hvp!(Jv, nothing, ws.Wscratch, v, b, ws, prms)
            function _W(s)
                b.arrays[:shelldisp] .= s
                Calculators.core_forces!(b, nlb, pmeb, prms)
                copy(Array(b.info[:virial]))
            end
            h = FMD.FPPREC(1e-4)
            DvW = (_W(shat .+ h .* v) .- _W(shat .- h .* v)) ./ (2h)

            Wok = W0 .+ DvW .- _symouter(Fs, s1) .- _symouter(Jv, shat)
            Wstripped = W0 .+ DvW .- _symouter(Fs_nosurf, s1) .-
                        _symouter(Jv, shat)
            scale = maximum(abs, Wrep)
            resid = maximum(abs, Wrep .- Wok)
            @test resid < 1e-6 * scale
            @test maximum(abs, Wrep .- Wstripped) >
                  1e3 * max(resid, eps(FMD.FPPREC))
        end

        # (d) End-to-end: the combination the validator and npt! used to
        # refuse now runs, moves the cell under a barostat, and stays
        # finite.
        @testset "npt! accepts :xlbomd + :surface_term" begin
            atoms, prms, _ = _xl_surf_fixture()
            dt = FMD.FPPREC(1.0 / UNITS[:fs])
            p = (; prms..., backend = :cpu,
                 elements = Dict("Na" => 1, "Cl" => 2),
                 skin = FMD.FPPREC(0.1), nsteps = 6, timestep = dt,
                 tau_t = 20dt, tau_p = 500dt,
                 initial_temperature = 0.0, target_temperature = 10.0,
                 target_stress = zeros(FMD.FPPREC, 6), output_interval = 10,
                 thermostat = :none, shell_dynamics = :xlbomd, xl_warmup = 2)
            mktempdir() do dir
                out = npt!(deepcopy_atoms(atoms),
                           (; p..., barostat = :mttk, output_dir = dir))
                @test all(isfinite, out.arrays[:pos])
                @test all(isfinite, Array(out.info[:lattice]))
                @test Array(out.info[:lattice]) != Array(atoms.info[:lattice])
            end
            # The validator agrees with npt! (it used to reject this).
            mktempdir() do dir
                inp = joinpath(dir, "in.xyz")
                touch(inp)
                @test FMD.Apps.validate_params(:NPTMD,
                    (; p..., barostat = :none, input = inp,
                     output_dir = dir, seed = 0)) === nothing
            end
        end

        # (e) The spectral necessary conditions (theory note §6). The
        # depolarization block raises λmax(spring⁻¹H_ss) by a computable
        # amount, and Weyl makes that a LOWER bound on the true λmax — so
        # a declared interval or κ below it is provably wrong. Built on a
        # deliberately soft single-atom cell where the bump dominates.
        @testset "xl_init! refuses a provably wrong spectrum / kappa" begin
            lat = FMD.FPPREC(10.0) .* Matrix{FMD.FPPREC}(I, 3, 3)
            one_atom = make_atoms(lat, FMD.FPPREC[0.5 0.5 0.5], [1])
            qc = FMD.FPPREC[2.0]
            qs = FMD.FPPREC[-2.0]
            st = Calculators.SurfaceTerm(one_atom, qc, qs; eps_out = 1.0,
                                         gauge = one_atom)
            base = (; qc = qc, qs = qs, k4 = FMD.FPPREC[0.0],
                    shell_dynamics = :xlbomd, surface_term = st)
            # Δλ = (4π/V)·(q_s²/k2)·λmax(T); T = I/3 for a vacuum sphere.
            Δλ(k2) = 4π / volume(one_atom) * (qs[1]^2 / k2) / 3

            soft = FMD.FPPREC(1e-3)                 # Δλ ≈ 16.8
            stiff = FMD.FPPREC(1.0)                 # Δλ ≈ 1.7e-2
            @test Δλ(soft) > 8 && Δλ(stiff) < 0.1   # the fixture is the claim

            # κ·Δλ ≥ 4 on the surface block alone: provably unstable —
            # but only for the :spring kernel, whose Verlet coupling IS λ.
            @test_throws ErrorException XL.xl_init!(deepcopy_atoms(one_atom),
                (; base..., k2 = FMD.FPPREC[soft], xl_kappa = 0.5))
            # The default Chebyshev interval cannot contain it either.
            @test_throws ErrorException XL.xl_init!(deepcopy_atoms(one_atom),
                (; base..., k2 = FMD.FPPREC[soft], xl_kernel = :chebyshev,
                 xl_kappa = 0.1))
            # …but a Chebyshev interval that DOES contain it is accepted at
            # the kernel's default κ = 1.8, even though κ·Δλ ≈ 30 ≫ 4: the
            # preconditioned coupling is λ·p(λ) ∈ [1-ρ, 1+ρ] ⊂ (0, 2), not
            # λ, so the :spring bound does not apply (theory note §6.1).
            # Refusing here would send a user who hit the κ guard — whose
            # documented escape is exactly this kernel — into a dead end.
            @test XL.xl_init!(deepcopy_atoms(one_atom),
                (; base..., k2 = FMD.FPPREC[soft], xl_kernel = :chebyshev,
                 xl_kernel_spectrum = (0.4, 40.0))) isa XL.XLState
            # Stiff spring: the bump is negligible and both pass.
            @test XL.xl_init!(deepcopy_atoms(one_atom),
                (; base..., k2 = FMD.FPPREC[stiff], xl_kappa = 0.5)) isa
                  XL.XLState
            @test XL.xl_init!(deepcopy_atoms(one_atom),
                (; base..., k2 = FMD.FPPREC[stiff],
                 xl_kernel = :chebyshev)) isa XL.XLState
            # …and without a surface term the soft spring is accepted:
            # the guard is about the depolarization block, not about k2.
            @test XL.xl_init!(deepcopy_atoms(one_atom),
                (; qc = qc, qs = qs, k4 = FMD.FPPREC[0.0],
                 shell_dynamics = :xlbomd, k2 = FMD.FPPREC[soft],
                 xl_kappa = 0.5)) isa XL.XLState
        end
    end

    @testset "T7: the default (:scf) path is unchanged" begin
        atoms, base = _xl_fixture()
        dt = FMD.FPPREC(1.0 / UNITS[:fs])
        common = (; base..., backend = :cpu,
                  elements = Dict("Na" => 1, "Cl" => 2),
                  skin = FMD.FPPREC(0.1), cgtol = FMD.FPPREC(2e-5),
                  nsteps = 5, timestep = dt,
                  thermostat = :none, barostat = :none,
                  initial_temperature = 0.0, target_temperature = 0.0,
                  output_interval = 100)
        a = mktempdir() do dir
            npt!(deepcopy_atoms(atoms), (; common..., output_dir = dir))
        end
        b = mktempdir() do dir
            npt!(deepcopy_atoms(atoms),
                 (; common..., shell_dynamics = :scf, output_dir = dir))
        end
        # Bitwise: the new branch must be inert on the default path.
        @test a.arrays[:pos] == b.arrays[:pos]
        @test a.arrays[:velo] == b.arrays[:velo]
        @test a.arrays[:shelldisp] == b.arrays[:shelldisp]
    end
end
