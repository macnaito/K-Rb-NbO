# bench/check_hvp_bsporder.jl — BSPORDER 4→6 decision protocol
# (Issue #147 / #117 P1; results recorded in
# docs/src/theory/coreshell-analytic-hvp.md §5).
#
# Measures the PME reciprocal HVP discretization error against the
# analytic Ewald HVP on a production-parameter BaTiO₃ fixture
# (example/ShellModel-MD potential, α = ewaldcoeff(rc = 11 Å, tol = 1e-6))
# across mesh spacings, alongside the force interpolation error at the
# same meshes. System size is irrelevant for the per-atom discretization
# error, so a 2×2×2 supercell (40 atoms) suffices; the neighbour list is
# a dummy — the reciprocal caches never read it.
#
# Run:  JULIA_LOAD_PATH=... julia --project bench/check_hvp_bsporder.jl
using Random
using LinearAlgebra
using Printf
using FMD
using FMD: UNITS
using FMD.Cores
using FMD.Calculators

include(joinpath(@__DIR__, "..", "test", "helpers.jl"))
include(joinpath(@__DIR__, "..", "example", "ShellModel-MD", "potential.jl"))

Random.seed!(1)
atoms = replicate_atoms(make_perovskite_5(; a_ang = 4.00, A = "Ba", B = "Ti"),
                        (2, 2, 2))
atoms.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(atoms.arrays[:pos]))
atoms.arrays[:shelldisp] .= 0.10 .* randn(FMD.FPPREC, size(atoms.arrays[:shelldisp]))
N = length(atoms)

nelem = 3
base = (; POT...,
        ewald_tolerance = FMD.FPPREC(1e-6),
        eta_c = fill(FMD.FPPREC(1e-5/UNITS[:Å]), nelem),
        eta_s = fill(FMD.FPPREC(1e-5/UNITS[:Å]), nelem),
        efield = zeros(FMD.FPPREC, 3),
        skin = FMD.FPPREC(0.0))
base = (; base..., alpha = Calculators.ewaldcoeff(base))
@printf("BaTiO₃ 2×2×2 (%d atoms), α = %.4f a.u. (rc = %.1f Å, tol = %.0e)\n",
        N, base[:alpha], base[:rc]*UNITS[:Å], base[:ewald_tolerance])

nl = neighborlist(atoms, FMD.FPPREC(6.0), base[:skin])   # dummy (unused by caches)

ew = Calculators.EwaldInfo(atoms.info[:lattice], base)
cew = Calculators._hvp_coulomb_cache(ew, atoms, nl, base)

reset!(atoms); Calculators.calculate!(ew, atoms, base)
Few = copy(atoms.arrays[:shellforces])
sF = maximum(abs, Few)

vs = [begin
          v = randn(FMD.FPPREC, N, 3)
          v ./ sqrt(sum(abs2, v))
      end for _ in 1:4]
Jews = Matrix{FMD.FPPREC}[]; Kews = Matrix{FMD.FPPREC}[]
for v in vs
    Jew = zeros(FMD.FPPREC, N, 3); Kew = zeros(FMD.FPPREC, N, 3)
    Calculators.hvp_coulomb!(Jew, Kew, v, atoms, cew, base)
    push!(Jews, Jew); push!(Kews, Kew)
end
sJ = maximum(maximum.(abs, Jews)); sK = maximum(maximum.(abs, Kews))

@printf("%-8s %-14s %10s %10s %10s   (max over %d directions, max-norm rel.)\n",
        "mesh", "grid", "relJ", "relK", "relF", length(vs))
for mesh in (2.0, 1.0, 0.5)
    prms = (; base..., pme_meshsize = FMD.FPPREC(mesh))
    pm = Calculators.PMEInfo(atoms.info[:lattice], prms)
    cpm = Calculators._hvp_coulomb_cache(pm, atoms, nl, prms)
    relJ = 0.0; relK = 0.0
    for (v, Jew, Kew) in zip(vs, Jews, Kews)
        Jpm = zeros(FMD.FPPREC, N, 3); Kpm = zeros(FMD.FPPREC, N, 3)
        Calculators.hvp_coulomb!(Jpm, Kpm, v, atoms, cpm, prms)
        relJ = max(relJ, maximum(abs, Jpm .- Jew)/sJ)
        relK = max(relK, maximum(abs, Kpm .- Kew)/sK)
    end
    reset!(atoms); Calculators.calculate!(pm, atoms, prms)
    relF = maximum(abs, atoms.arrays[:shellforces] .- Few)/sF
    @printf("%-8.3g %-14s %10.3g %10.3g %10.3g\n",
            mesh, string(pm.fftsize), relJ, relK, relF)
end
println("""
Decision guide (theory note §5): the meaningful criterion is the RATIO
relJ/relF at the production mesh — at a coarse mesh the generic
interpolation error dominates forces and HVP alike, so an absolute relJ
target below relF is unreachable for any BSPORDER. Keep BSPORDER = 4 if
relJ/relF ≲ 10 at the production mesh (measured 2026-08: 1.6 at
mesh 2.0) and the downstream ε/BEC shift of linear_response!(PME) vs
(Ewald) stays well under the fitting residual (~1e-3; the fitting path
itself runs Ewald and is exact regardless). The d2-order loss only opens
the J/F gap at finer meshes where both errors are already small
(mesh 0.5: relJ 1.4e-4 vs relF 3.8e-5). For HVP-heavy workloads needing
more reciprocal accuracy, shrink pme_meshsize before touching BSPORDER
(a 4→6 change rewrites the hardcoded spline kernels and costs ~3.4× in
spread/interpolation).""")
