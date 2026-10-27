using FMD
using FMD.Cores
using FMD.Calculators
using BenchmarkTools
using Random
Random.seed!(0)

println("=== Loading dataset ===")
atoms = readxyz(open("example/dataset/KRNO_PBEsol_MACE.xyz", "r"))[1]
setelemid!(atoms, Dict("K" => 1, "Rb" => 2, "Nb" => 3, "O" => 4))
println("N atoms = ", length(atoms))

# Production-size SplineParam: nb=(12, 6) like example/FitSpline/input.jl
splp = SplineParam(6.0/FMD.UNITS[:Å], 5.0/FMD.UNITS[:Å], 4, 12, 6)
splp.c1 .= randn(size(splp.c1)) .* 1e-3
splp.c2 .= randn(size(splp.c2)) .* 1e-3
splp.c3 .= randn(size(splp.c3)) .* 1e-3
@show size(splp.c2) size(splp.c3) length(vec(splp))

arg = Calculators.makearg_spline(atoms, splp.rmax2, splp.rmax3)

# Warmup
E, F, V = Calculators.applyA(splp, arg)
@show E

# MD-style: applyA only
println()
println("=== applyA (MD-like) ===")
b = @benchmark Calculators.applyA($splp, $arg)
display(b)
println("median: ", BenchmarkTools.median(b).time / 1e6, " ms")

# FIT-style: applyA + applyAT! per CG step
println()
println("=== applyA + applyAT! (FIT-like) ===")
g = deepcopy(splp); Calculators.setval!(g, 0.0)

function fit_step!(g, splp, arg, F, V)
    Calculators.setval!(g, 0.0)
    E2, F2, V2 = Calculators.applyA(splp, arg)
    Calculators.applyAT!(g, arg, 1.0, F2, V2)
    return E2
end

b2 = @benchmark fit_step!($g, $splp, $arg, $F, $V)
display(b2)
println("median: ", BenchmarkTools.median(b2).time / 1e6, " ms")
