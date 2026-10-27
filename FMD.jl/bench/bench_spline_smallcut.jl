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

for (rmax3, label) in [(5.0, "rmax3=5.0Å"), (4.0, "rmax3=4.0Å"), (3.5, "rmax3=3.5Å")]
    println()
    println("================ $label ================")
    splp = SplineParam(6.0/FMD.UNITS[:Å], rmax3/FMD.UNITS[:Å], 4, 12, 6)
    splp.c1 .= randn(size(splp.c1)) .* 1e-3
    splp.c2 .= randn(size(splp.c2)) .* 1e-3
    splp.c3 .= randn(size(splp.c3)) .* 1e-3

    arg = Calculators.makearg_spline(atoms, splp.rmax2, splp.rmax3)

    Calculators.applyA(splp, arg)  # warmup

    b = @benchmark Calculators.applyA($splp, $arg)
    println("MD applyA median: ", BenchmarkTools.median(b).time / 1e6, " ms")

    g = deepcopy(splp); Calculators.setval!(g, 0.0)
    E2, F2, V2 = Calculators.applyA(splp, arg)
    b2 = @benchmark begin
        Calculators.setval!($g, 0.0)
        E_, F_, V_ = Calculators.applyA($splp, $arg)
        Calculators.applyAT!($g, $arg, 1.0, F_, V_)
    end
    println("FIT applyA+applyAT! median: ", BenchmarkTools.median(b2).time / 1e6, " ms")
end
