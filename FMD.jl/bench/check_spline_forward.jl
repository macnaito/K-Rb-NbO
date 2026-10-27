using FMD
using FMD.Cores
using FMD.Calculators
using Random
using Statistics
using LinearAlgebra
Random.seed!(0)

# Local finite-difference force check.  The in-`src` `Calculators.checkforce`
# was removed (verification now lives in the test suite); this small inline
# copy keeps the benchmark scripts self-contained.
function checkforce(arg, splp)
    h = 1e-4
    # applyA re-syncs the ghost positions from arg.atoms at every call, so
    # in-place perturbation of the fractional coordinates is followed (the
    # ghost-format arg has no standalone pos/lattice fields).
    pos = arg.atoms.arrays[:pos]
    δf = vec([h 0.0 0.0] * inv(Array(arg.atoms.info[:lattice])))
    _, F, _ = Calculators.applyA(splp, arg)
    pos[1, :] .+= δf
    EP, _, _ = Calculators.applyA(splp, arg)
    pos[1, :] .-= 2 .* δf
    EM, _, _ = Calculators.applyA(splp, arg)
    pos[1, :] .+= δf
    analytical = F[1, 1]; numerical = -(EP - EM) / (2h)
    @show analytical numerical
    @assert isapprox(analytical, numerical; rtol = 1e-3, atol = 1e-5) "checkforce failed: |Δ| = $(abs(analytical - numerical))"
end

println("=== Loading dataset ===")
atoms = readxyz(open("example/dataset/KRNO_PBEsol_MACE.xyz", "r"))[1]
setelemid!(atoms, Dict("K" => 1, "Rb" => 2, "Nb" => 3, "O" => 4))
println("N atoms = ", length(atoms))

println("=== Constructing SplineParam (small for fast test) ===")
splp = SplineParam(6.0/FMD.UNITS[:Å], 5.0/FMD.UNITS[:Å], 4, 6, 5)
@show size(splp.c1) size(splp.c2) size(splp.c3)

println("=== Filling with random coefficients ===")
splp.c1 .= randn(size(splp.c1)) .* 1e-3
splp.c2 .= randn(size(splp.c2)) .* 1e-3
splp.c3 .= randn(size(splp.c3)) .* 1e-3

println("=== Building neighbor-list arg ===")
arg = Calculators.makearg_spline(atoms, splp.rmax2, splp.rmax3)

println("=== applyA call (forward only) ===")
E, F, V = Calculators.applyA(splp, arg)
@show E
@show extrema(F)
@show V

println("=== checkforce: finite-diff on atom 1 ===")
checkforce(arg, splp)
println("OK: forward checkforce passed.")

println()
println("=== Variant: c2 = 0, only 3-body ===")
splp2 = deepcopy(splp); splp2.c1 .= 0; splp2.c2 .= 0
arg2 = Calculators.makearg_spline(atoms, splp2.rmax2, splp2.rmax3)
checkforce(arg2, splp2)
println("OK: 3-body only checkforce passed.")

println()
println("=== Variant: c3 = 0, only 1+2-body ===")
splp3 = deepcopy(splp); splp3.c3 .= 0
arg3 = Calculators.makearg_spline(atoms, splp3.rmax2, splp3.rmax3)
checkforce(arg3, splp3)
println("OK: 1+2-body only checkforce passed.")
