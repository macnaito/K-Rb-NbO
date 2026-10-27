# Stage 3 — 残差の純 ACE フィット（null shell）
#
# shell / NLH は**渡さない**：FitACE のデフォルトは電荷・spring・Buckingham
# すべてゼロの null shell で，ベースライン減算が厳密にゼロになる．残差は
# make_residual.jl で既にベースモデル全体を引いてあるので，これで二重減算が
# 構造的に起きない．shell 電荷が別ロジックで最適化済みの場合でも，delta の
# フィットはこの形のまま成立する．
#
# ★ この出力（ACEP.jld2）は null shell を含む中間生成物．loadpotential の
#   バリデーションを通ってしまうが，**単体で MD に使ってはいけない**
#   （静電なしのポテンシャルとして黙って動く）．必ず merge_acep.jl で
#   ベースと合成した ACEP_merged.jld2 を使うこと．
#
# 実行（このディレクトリで，residual.xyz 生成後）:
#     mpiexec -n 4 julia --heap-size-hint=1G --project=../.. input_fitdelta.jl
#
# 出力: ACEP.jld2（delta 係数，中間生成物）, parity_{force,energy,virial}.csv
#       （この parity は「残差をどれだけ説明できたか」を示す）

using FMD
using FMD: UNITS

# 基底の互換条件（rmin / rmax / elements 順）はベースから取って強制する．
# ここを手打ちにすると，ずれても係数マージまで気付けない．
_base = FMD.loadpotential("../training/ACEP.jld2")

params = (;
     elements = _base.elements,
     dataset = "residual.xyz",
     dataset_idx = :,
     traindata_ratio = 0.9,

     weight_force = 10.0,
     weight_stress = 1.0,
     method_weight = (; MACE_MPA0 = 1.0,),   # label_mace.py --method-tag と一致させる

     cgtol = 1e-8,
     cgmaxiter = 100,
     ewald_tolerance = 1e-8,

     # radial 基底はベースと同一ドメイン（必須），基底数はベースより小さくてよい
     # （マージは各レベル max の union 基底に埋め込む）．残差は小さく滑らかなので
     # 小さな基底で足りる．
     ace_rmin = _base.acep.rmin,
     ace_rmax = _base.acep.rmax,
     ace_nk = (5, 4, 3, 3),
     nsteps = 2000,
     cg_preconditioner = :blockchol_si,
     rigid = (
          cp = 1e-8,
          c0 = 1e0, c00 = 1e0,
          c11 = 1e0, c22 = 1e0, c33 = 1e0,
          c011 = 1e0, c022 = 1e0, c033 = 1e0,
     ),
)

params = (; FMD.default_params(Val(:FitACE), params)..., params...)

FMD.app[:FitACE](params)
