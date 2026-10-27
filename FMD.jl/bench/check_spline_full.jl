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

println()
println("=== Building SplineParam ===")
splp = SplineParam(6.0/FMD.UNITS[:Å], 5.0/FMD.UNITS[:Å], 4, 6, 5)
splp.c1 .= randn(size(splp.c1)) .* 1e-3
splp.c2 .= randn(size(splp.c2)) .* 1e-3
splp.c3 .= randn(size(splp.c3)) .* 1e-3
@show size(splp.c1) size(splp.c2) size(splp.c3)
@show length(vec(splp))

arg = Calculators.makearg_spline(atoms, splp.rmax2, splp.rmax3)

println()
println("=== checkforce: full (1+2+3) ===")
checkforce(arg, splp)

println()
println("=== checkforce: 3-body only ===")
splp_3 = deepcopy(splp); splp_3.c1 .= 0; splp_3.c2 .= 0
checkforce(arg, splp_3)

println()
println("=== checkforce: 1+2-body only ===")
splp_12 = deepcopy(splp); splp_12.c3 .= 0
checkforce(arg, splp_12)

println()
println("=== Bilinear duality test: <Av, u> = <v, A^T u> ===")
function duality_test(arg, splp; tag = "full")
    Random.seed!(1)
    v = randn(length(vec(splp)))
    splp_v = deepcopy(splp); Calculators.setval!(splp_v, v)
    E, F, V = Calculators.applyA(splp_v, arg)

    uE = randn()
    uF = randn(size(F))
    uV = randn(6)

    g = deepcopy(splp); Calculators.setval!(g, 0.0)
    Calculators.applyAT!(g, arg, uE, uF, uV)
    g_vec = vec(g)

    lhs = uE * E + sum(uF .* F) + sum(uV .* V)   # <Av, u>
    rhs = sum(v .* g_vec)                         # <v, A^T u>
    rel = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    println("[$tag] lhs = $lhs, rhs = $rhs, rel.err = $rel")
    @assert rel < 1e-10 "duality failed for $tag, rel.err = $rel"
end

duality_test(arg, splp; tag = "full 1+2+3")
duality_test(arg, splp_3; tag = "3-body only")
duality_test(arg, splp_12; tag = "1+2-body only")

println()
println("=== ALL CHECKS PASSED ===")
