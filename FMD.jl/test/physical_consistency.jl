# test/physical_consistency.jl
#
# Reusable physical-invariance assertions, parameterized by a closure
# `calc!(atoms)` that performs `clear!(atoms)` followed by one or more
# `Calculators.calculate!(...)` calls.  All quantities are in atomic units.

using Test
using LinearAlgebra
using Random

# --------------------------------------------------------------------
# 1. Translation invariance: E and F unchanged when every atom shifts
# by the same Cartesian vector.  We add the shift in fractional space.
# --------------------------------------------------------------------

function assert_translation_invariant(calc!, atoms::Atoms;
                                      shift_frac = (0.137, -0.211, 0.073),
                                      rtol = 1e-8, atol = 1e-10,
                                      label = "translation invariance")
    a1 = deepcopy_atoms(atoms)
    E1, F1 = evaluate(calc!, a1)

    a2 = deepcopy_atoms(atoms)
    a2.arrays[:pos] .+= reshape(collect(FMD.FPPREC.(shift_frac)), 1, 3)
    E2, F2 = evaluate(calc!, a2)

    @testset "$label" begin
        @test isapprox(E1, E2; rtol = rtol, atol = atol)
        @test isapprox(F1, F2; rtol = rtol, atol = atol)
    end
end

# --------------------------------------------------------------------
# 2. PBC integer-shift invariance: shifting one atom's fractional
# coordinate by exactly 1 in any axis must leave E, F unchanged.
# --------------------------------------------------------------------

function assert_pbc_image_invariant(calc!, atoms::Atoms;
                                    rtol = 1e-10, atol = 1e-12,
                                    label = "PBC integer shift invariance")
    a1 = deepcopy_atoms(atoms)
    E1, F1 = evaluate(calc!, a1)

    a2 = deepcopy_atoms(atoms)
    # Shift atom 1 by (+1, 0, 0) in fractional coordinates.
    a2.arrays[:pos][1, 1] += 1
    E2, F2 = evaluate(calc!, a2)

    @testset "$label" begin
        @test isapprox(E1, E2; rtol = rtol, atol = atol)
        @test isapprox(F1, F2; rtol = rtol, atol = atol)
    end
end

# --------------------------------------------------------------------
# 3. Newton's third law (action-reaction): for any
# translation-invariant Hamiltonian, sum(forces, dims=1) = 0.
# --------------------------------------------------------------------

function assert_force_sum_zero(calc!, atoms::Atoms;
                               atol = 1e-8,
                               label = "sum(forces) ≈ 0 (Newton III)")
    a = deepcopy_atoms(atoms)
    _, F = evaluate(calc!, a)
    s = vec(sum(F, dims = 1))
    @testset "$label" begin
        for k in 1:3
            @test isapprox(s[k], 0; atol = atol)
        end
    end
end

# --------------------------------------------------------------------
# 4. Force = -∇E (central finite difference).
#
# Move atom `i` by ±h Bohr along Cartesian axis `axis` (converted to
# fractional via lattice⁻¹), evaluate E± and compare to the analytic
# force component F[i, axis].  Skip atoms whose force component is
# numerically zero — central differences would also give 0 there.
# --------------------------------------------------------------------

function assert_force_matches_finite_diff(calc!, atoms::Atoms;
                                          h::Real = 1e-4,
                                          atom_indices = 1:min(length(atoms), 4),
                                          axes = (1, 2, 3),
                                          rtol = 1e-4, atol = 1e-6,
                                          label = "force = -∇E (FD)")
    a = deepcopy_atoms(atoms)
    _, F = evaluate(calc!, a)
    ilattice = inv(Array(atoms.info[:lattice]))

    @testset "$label" begin
        for i in atom_indices, axis in axes
            δ = zeros(FMD.FPPREC, 3); δ[axis] = h
            δfrac = δ' * ilattice  # row vector

            ap = deepcopy_atoms(atoms)
            ap.arrays[:pos][i, :] .+= vec(δfrac)
            Ep, _ = evaluate(calc!, ap)

            am = deepcopy_atoms(atoms)
            am.arrays[:pos][i, :] .-= vec(δfrac)
            Em, _ = evaluate(calc!, am)

            num = -(Ep - Em) / (2h)
            ana = F[i, axis]

            # Skip components that are numerically zero by symmetry.
            if abs(ana) < atol && abs(num) < atol
                continue
            end
            @test isapprox(ana, num; rtol = rtol, atol = atol)
        end
    end
end

# --------------------------------------------------------------------
# 4a. Shellforce = -∂E/∂(shell displacement) (central finite difference).
#
# Mirror of (4) for the core–shell degree of freedom.  `:shelldisp` is
# stored in Cartesian Bohr (see Ewald/CoreShell `spos_r = pos·lattice +
# shelldisp`), so we perturb it directly without the lattice transform.
# Compares to the analytic `:shellforces` left in `atoms` by `calc!`.
# --------------------------------------------------------------------

function assert_shellforce_matches_finite_diff(calc!, atoms::Atoms;
                                               h::Real = 1e-5,
                                               atom_indices = 1:min(length(atoms), 4),
                                               axes = (1, 2, 3),
                                               rtol = 1e-6, atol = 1e-8,
                                               label = "shellforce = -∂E/∂shelldisp (FD)")
    a = deepcopy_atoms(atoms)
    reset!(a)
    calc!(a)
    SF = copy(a.arrays[:shellforces])

    @testset "$label" begin
        for i in atom_indices, axis in axes
            ap = deepcopy_atoms(atoms)
            ap.arrays[:shelldisp][i, axis] += h
            Ep, _ = evaluate(calc!, ap)

            am = deepcopy_atoms(atoms)
            am.arrays[:shelldisp][i, axis] -= h
            Em, _ = evaluate(calc!, am)

            num = -(Ep - Em) / (2h)
            ana = SF[i, axis]

            if abs(ana) < atol && abs(num) < atol
                continue
            end
            @test isapprox(ana, num; rtol = rtol, atol = atol)
        end
    end
end

# --------------------------------------------------------------------
# 4b. Inversion symmetry through `center_frac`.  For a calculator whose
# energy is built from scalar invariants (distances, angles), inverting
# every atom through a chosen center should leave E unchanged and flip F.
# Use a fixture whose lattice + species labelling is fixed under the
# inversion (e.g. `make_perovskite_5` with its B-site at (½,½,½)).
# --------------------------------------------------------------------

function assert_inversion_invariant(calc!, atoms::Atoms;
                                    center_frac = (0.5, 0.5, 0.5),
                                    rtol = 1e-10, atol = 1e-12,
                                    label = "inversion symmetry")
    a1 = deepcopy_atoms(atoms)
    E1, F1 = evaluate(calc!, a1)

    a2 = deepcopy_atoms(atoms)
    cf = reshape(collect(FMD.FPPREC.(center_frac)), 1, 3)
    a2.arrays[:pos] .= 2 .* cf .- a2.arrays[:pos]
    E2, F2 = evaluate(calc!, a2)

    @testset "$label" begin
        @test isapprox(E1, E2; rtol = rtol, atol = atol)
        @test isapprox(F1, -F2; rtol = rtol, atol = atol)
    end
end

# --------------------------------------------------------------------
# 4c. Virial = -∂E/∂ε (affine-strain central finite difference).
#
# The stored virial (Voigt [xx, yy, zz, yz, zx, xy]) is
#   W_αβ = Σ F·r = -∂E/∂ε_αβ
# under an affine deformation gradient D = I + δ·(e_α ⊗ e_β) applied to BOTH
# the lattice (rows = lattice vectors → lattice·Dᵀ) and the Cartesian shell
# displacement (→ shelldisp·Dᵀ), with fractional coordinates held fixed.
# A single (α, β) entry is perturbed at a time so there is no factor-of-two
# ambiguity that a symmetric ε would introduce; rotational invariance kills
# the antisymmetric (rotation) part to first order.
#
# `symmetrize = true` perturbs (α, β) and (β, α) together by δ/2 each, i.e.
# it measures the SYMMETRIC derivative ½(∂E/∂ε_αβ + ∂E/∂ε_βα).  Needed for
# energy terms that are NOT rotationally invariant — the polarisation
# umbrella restrains P along a lab-frame direction, so its ∂E/∂ε has a
# genuine antisymmetric part.  That part is a torque on the atoms (it lives
# in the forces, which have their own FD check) and does no work on any cell
# degree of freedom, since the barostats evolve a symmetric 6-DOF cell
# (docs/src/theory/mttk-cell-dof.md); the stored Voigt-6 virial is therefore
# the symmetric part by construction.  For a rotationally invariant
# calculator both modes give the same value.
#
# `calc!` MUST rebuild any cell-dependent state (neighbour list, EwaldInfo,
# PMEInfo) internally, since the lattice changes between evaluations.
# Components that are zero by symmetry (both stored and FD below atol) are
# skipped, exactly like the force FD check — so an asymmetric fixture is
# needed to exercise the off-diagonal (shear) components.
#
# CPU-only: `deepcopy_atoms` returns `Atoms{Array}`.  GPU calculators (PME)
# are checked with an inline equivalent in test/gpu/test_pme_gpu.jl.
# --------------------------------------------------------------------

const _VOIGT_IDX = ((1, 1), (2, 2), (3, 3), (2, 3), (3, 1), (1, 2))

function assert_virial_matches_finite_diff(calc!, atoms::Atoms;
                                           h::Real = 1e-5,
                                           rtol = 1e-4, atol = 1e-8,
                                           symmetrize::Bool = false,
                                           label = "virial = -∂E/∂ε (FD)")
    a = deepcopy_atoms(atoms)
    _, _ = evaluate(calc!, a)
    W = copy(Array(a.info[:virial]))
    lat0 = copy(Array(atoms.info[:lattice]))
    sd0  = copy(Array(atoms.arrays[:shelldisp]))

    deform(δ, α, β) = begin
        D = Matrix{FMD.FPPREC}(I, 3, 3)
        if symmetrize
            # For α == β the two half-steps land on the same entry and sum
            # to δ, so the diagonal perturbation is the same in both modes.
            D[α, β] += δ / 2
            D[β, α] += δ / 2
        else
            D[α, β] += δ
        end
        ad = deepcopy_atoms(atoms)
        ad.info[:lattice]     .= lat0 * D'
        ad.arrays[:shelldisp] .= sd0  * D'
        first(evaluate(calc!, ad))
    end

    @testset "$label" begin
        for (k, (α, β)) in enumerate(_VOIGT_IDX)
            Ep = deform(+h, α, β)
            Em = deform(-h, α, β)
            num = -(Ep - Em) / (2h)   # == W_αβ iff virial == -∂E/∂ε
            ana = W[k]
            # Skip components that are numerically zero by symmetry.
            if abs(ana) < atol && abs(num) < atol
                continue
            end
            @test isapprox(ana, num; rtol = rtol, atol = atol)
        end
    end
end

# --------------------------------------------------------------------
# 5. Stress / virial symmetry.
# Voigt order in this codebase: [xx, yy, zz, yz, xz, xy].
# Symmetry of Cauchy stress means σ_yz = σ_zy, σ_xz = σ_zx, σ_xy = σ_yx,
# which is automatically satisfied by Voigt packing — what we check is
# that the stored virial is a Voigt 6-vector (length 6, not exploded).
# --------------------------------------------------------------------

function assert_virial_voigt(atoms::Atoms;
                             label = "virial is 6-component Voigt")
    @testset "$label" begin
        @test length(atoms.info[:virial]) == 6
        # σ_yz, σ_xz, σ_xy must be finite (not NaN/Inf) when the
        # calculator has been invoked.
        for k in 4:6
            @test isfinite(atoms.info[:virial][k])
        end
    end
end
