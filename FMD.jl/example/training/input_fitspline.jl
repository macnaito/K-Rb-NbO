# Stage 3' — Spline MLIP の CG フィット（ACE の代わりに 2/3 体 B-spline を使う場合）
#
# input_fitace.jl と同じ train.xyz・同じベースライン減算で，MLIP の表現だけ
# B-spline（2 体 + 3 体）に替えたもの．化学・データに合わせて使い分ける．
#
# 実行（このディレクトリで，train.xyz 生成後）:
#     mpiexec -n 4 julia --heap-size-hint=1G --project=../.. input_fitspline.jl
#
# 出力: SplineP.jld2, parity_{force,energy,virial}.csv

using FMD
using FMD: UNITS

include("../ShellModel-MD/potential.jl")   # -> POT: Azuma 2025, r²SCAN-D4 (Ewald) BaTiO₃ set

params = (;
     # 文献セット完全形．SplineP.jld2 にも Buckingham は保存されないので，
     # MD 側では potential.jl の Buckingham を重ねること（input_fitace.jl の注意
     # と input_md.jl を参照）．
     POT...,

     dataset = "train.xyz",
     dataset_idx = :,
     traindata_ratio = 0.9,
     nsteps = 2000,

     weight_force = 10.0,
     weight_stress = 1.0,
     method_weight = (; MACE_R2SCAN = 1.0,),

     cgtol = 1e-8,
     cgmaxiter = 100,
     ewald_tolerance = 1e-8,

     spline_rmax2 = 6.0/(UNITS[:Å]),
     spline_rmax3 = 3.5/(UNITS[:Å]),
     spline_nbasis2 = 25,
     spline_nbasis3 = 25,
     rigid = (c1 = 1e-8, c2 = 1e-4, c3 = 1e-4,),
     rigid_curv = (
          c2       = 1e-2,
          c3_r     = 1e-2,
          c3_θ     = 1e-3,
          c2_pin   = 1e-2,   # 2-body ϕ(rmax2) → 0 (taper relief)
          c3_r_pin = 1e-2,   # 3-body ϕ(rmax3) → 0 along rij and rik
     ),

     # NLH 短距離斥力ベースライン（input_fitace.jl の同項と同じ；省略可）．
     nlh = FMD.Calculators.NLHParam(Dict("Ba" => 1, "Ti" => 2, "O" => 3),
                                    5.0/(UNITS[:Å])),
)

params = (; FMD.default_params(Val(:FitSpline), params)..., params...)

FMD.app[:FitSpline](params)
