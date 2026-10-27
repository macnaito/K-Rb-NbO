# [optional] Full path — shell モデル自体を CMA-ES でフィットし直す場合
#
# Quick path（README 参照）は文献 shell パラメータをそのまま使うのでこの
# ステップは不要．shell を作り替えたい場合（組成・参照理論を変える等）だけ
# dfpt.xyz（BEC + 誘電テンソル，gen_dfpt_macefield.py か VASP DFPT で生成）を
# 用意してこれを回す．
#
# 実行（このディレクトリで，dfpt.xyz 生成後）:
#     mpiexec -n 16 julia --project=../.. input_fitsm.jl
#
# MPI ランク配置: popsize × data_workers = 16 × 1．HPC では popsize = 72 程度
# （dataset/dfpt/kno の本番設定）に上げると探索が安定する．
#
# 出力: ShellParam.jld2（loadpotential で読める 8-field shell）
#       これを使う場合は input_fitace.jl の `POT...` を
#       `FMD.loadpotential("ShellParam.jld2")...` + Buckingham 項に差し替える．

using MPI
using FMD
using FMD: UNITS

MPI.Init()

params = (
     dataset = "dfpt.xyz",
     dataset_idx = :,

     opt_weights = (
          bec = 10.0,
          eps = 1.0,
     ),

     elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3),
     formal_charges = [2.0, 4.0, -2.0],
     rc = 11.0/(UNITS[:Å]),

     # 初期値: cation は formal charge + 1，anion は + 1
     qref = 1.0,
     qc = [3.0, 5.0, 1.0],
     k2 = fill(1.0, 3),
     eta_c = fill(1.0, 3),
     eta_s = fill(1.0, 3),

     lb = (
          qref  = 0.5,
          qc    = [0.1, 0.1, 0.1],
          k2    = [0.8, 0.8, 0.8],
          eta_c = fill(0.02/(UNITS[:Å]), 3),
          eta_s = fill(0.02/(UNITS[:Å]), 3),
     ),
     ub = (
          qref  = 1.5,
          qc    = [10.0, 10.0, 10.0],
          k2    = [5.0, 5.0, 5.0],
          eta_c = fill(4.0/(UNITS[:Å]), 3),
          eta_s = fill(4.0/(UNITS[:Å]), 3),
     ),

     nsteps = 1000,
     sigma0 = 0.2,
     popsize = 16,
     data_workers = 1,

     cgtol = 1e-6,
     cgmaxiter = 50,
)

params = (; FMD.default_params(Val(:FitShellModel), params)..., params...)

FMD.app[:FitShellModel](params)

MPI.Finalize()
