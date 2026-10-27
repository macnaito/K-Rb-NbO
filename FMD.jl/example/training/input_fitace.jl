# Stage 3 — ACE MLIP の CG フィット（shell モデルをベースラインとする Δ 学習）
#
# fitting! は各フレームから shell モデル（Coulomb + Buckingham + spring）の
# 寄与を自動で減算し，残差だけを ACE に学習させる．予測・MD 側は同じ
# ベースラインを加算して返すため，往復は構成上厳密に一致する．
#
# 実行（このディレクトリで，train.xyz 生成後）:
#     mpiexec -n 4 julia --heap-size-hint=1G --project=../.. input_fitace.jl
# MPI が無ければ `julia --project=../.. input_fitace.jl`（1 ランク）でも動く．
# ランクはデータ並列（フレーム分割）に使われる．RAM 16 GB 級では 4 ランク +
# heap-size-hint が安全（1 ランクあたり実測 ~1.5 GB）．
#
# 出力: ACEP.jld2（shell + ACE 係数 + provenance; loadpotential でそのまま
#       MD に読める）, parity_{force,energy,virial}.csv

using FMD
using FMD: UNITS

include("../ShellModel-MD/potential.jl")   # -> POT: Azuma 2025, r²SCAN-D4 (Ewald) BaTiO₃ set

params = (;
     # ベースライン shell モデル（文献セット完全形: 電荷 / spring / Buckingham / rc）．
     # この文献セットは点電荷 shell モデルで，Buckingham 斥力が shell の分極崩壊を
     # 止める設計になっている — 電荷 + spring だけを baseline にすると shell 緩和が
     # 破綻し（実測で baseline 力 ~19 eV/Å），残差フィットが成立しない．
     #
     # ★ 注意: FitACE が保存する ACEP.jld2 には Buckingham（a/rho/c）が入らない．
     #   MD 側では loadpotential の結果に potential.jl の Buckingham を必ず重ねる
     #   こと（input_md.jl がその完全な例）．忘れると黙って別のポテンシャルになる．
     #
     # eta_c / eta_s / qref は指定せず FitACE デフォルト（点電荷極限 1e-5/Å, qref = 1）
     # に任せる — NPTMD 側も同じデフォルトなので fit と MD のスタックが一致する．
     POT...,

     dataset = "train.xyz",
     dataset_idx = :,
     traindata_ratio = 0.9,

     weight_force = 10.0,
     weight_stress = 1.0,
     method_weight = (; MACE_R2SCAN = 1.0,),   # label_mace.py --method-tag と一致させる

     cgtol = 1e-8,
     cgmaxiter = 100,
     ewald_tolerance = 1e-8,

     # ── コンシューマ PC で完走する規模設定 ──────────────────────────
     # ace_nk はチェビシェフ次数 (n0, n1, n2, n3)．係数数は次数の 2 乗で増え，
     # CG 反復コストはデータ量 × 係数数に比例する．ここでは 8 コアの
     # ワークステーションで数分に収まる最小規模にしてある．
     # 精度を上げる場合は HPC で ace_nk = (12, 6, 4, 4)（デフォルト）等に．
     ace_rmin = 1.0/(UNITS[:Å]),
     ace_rmax = 6.0/(UNITS[:Å]),
     ace_nk = (6, 5, 4, 3),
     nsteps = 2000,
     cg_preconditioner = :blockchol_si,   # radial 基底の共線形性を除去し CG を加速
     rigid = (
          cp = 1e-8,
          c0 = 1e0, c00 = 1e0,
          c11 = 1e0, c22 = 1e0, c33 = 1e0,
          c011 = 1e0, c022 = 1e0, c033 = 1e0,
     ),

     # NLH 短距離斥力ベースライン（Δ学習）: 学習データが届かない短距離で
     # ポテンシャルが発散するよう，遮蔽核斥力を fit 時に減算し予測時に加算する．
     # 係数は元素対ごとの公表値（data/nlh/nlh_coeffs.dat, CC-BY 4.0）．
     # cutoff は shifted-force で打ち切るので E も F も連続．rc 以下であること．
     # 減算した係数は ACEP.jld2 に焼き込まれ，MD 側は自動で同じものを加算する．
     # 省略すればベースライン無し．
     nlh = FMD.Calculators.NLHParam(Dict("Ba" => 1, "Ti" => 2, "O" => 3),
                                    5.0/(UNITS[:Å])),
)

params = (; FMD.default_params(Val(:FitACE), params)..., params...)

FMD.app[:FitACE](params)
