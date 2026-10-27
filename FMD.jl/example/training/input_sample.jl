# Stage 1 — 学習配置サンプリング: 純 shell モデルの NPT-MD（MLIP なし）
#
# Δ 学習の学習データは「ベースライン（shell モデル）の PES からサンプルし，
# 参照理論（MACE 単点評価，Stage 2）でラベル付けする」方針で作る．
# ベースラインが既に妥当な領域を中心に配置が集まるため残差が小さく，
# 小さな ACE 基底でもフィットが利く．uMLIP による MD サンプリング
# （dataset/dft/ 参照，本番向け）と違い GPU も torch も不要．
#
# 温度 3 点（強誘電相 200 K / 相転移近傍 400 K / 高温 700 K）で位相空間の
# 多様性を，静水圧 ±2 GPa で体積の多様性を確保する．
#
# 実行（このディレクトリで）:
#     python3 ../ShellModel-MD/makecell.py -n 3 -o input.xyz
#     julia --project=../.. input_sample.jl
#
# 出力: out_<条件>/step_<n>.xyz（各 60 スナップショット）
#       Stage 2 の label_mace.py が --burnin で熱平衡化前を捨てて回収する．

using FMD
using FMD: UNITS

include("../ShellModel-MD/potential.jl")   # -> POT: Azuma 2025, r²SCAN-D4 (Ewald) BaTiO₃ set

dt = 1.0 / UNITS[:fs]     # MD timestep: 1 fs in a.u.

# (タグ, 温度 [K], 静水圧 [GPa], 乱数シード)
conditions = [
    ("T200_P0",  200.0,  0.0, 1),
    ("T400_P0",  400.0,  0.0, 2),
    ("T700_P0",  700.0,  0.0, 3),
    ("T400_Pm2", 400.0, -2.0, 4),
    ("T400_Pp2", 400.0, +2.0, 5),
]

for (tag, T, P, seed) in conditions
    params = (;
        # ポテンシャル（物理形状を決めるものすべて）．
        POT...,

        # Job I/O.
        input = "input.xyz",           # makecell.py -n 3 の 3×3×3 セル（135 原子）
        output_dir = "out_$(tag)",

        # 数値ノブ．
        pme_meshsize = 2.0,
        efield = [0.0, 0.0, 0.0],
        cgtol = 2e-5,                  # per-step adiabatic shell relaxation
        cgmaxiter = 50,
        skin = 0.2,

        # 時間制御: 3 ps，50 step ごとにスナップショット．
        timestep = dt,
        nsteps = 3000,
        output_interval = 50,

        # NPT(T, σ = −P)．CSVR + SCR は一次緩和でリンギングが無い．
        # target_stress は応力（引張が正）なので圧縮圧力 P は符号反転．
        thermostat = :csvr,
        tau_t = 100dt,
        barostat = :scr,
        tau_p = 500dt,
        compressibility = UNITS[:GPa] / 160.0,   # κ_T from B ≈ 160 GPa
        target_stress = [-P, -P, -P, 0.0, 0.0, 0.0] ./ UNITS[:GPa],
        initial_temperature = T,
        target_temperature = T,
        seed = seed,

        # 135 原子は GPU では小さすぎるため CPU 固定（CUDA 不要で完走する）．
        backend = :cpu,
    )
    params = (; FMD.default_params(Val(:NPTMD), params)..., params...)
    FMD.app[:NPTMD](params)
end
