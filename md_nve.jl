# 純 KNbO₃ (2,560 原子) の NVE-MD．150 K・0 GPa で平衡化した NPT (CSVR + MTTK, 20 ps) の
# step_39000.xyz (19.5 ps; セルが 12–20 ps 平均に最も近いフレーム) から，速度とシェル変位を引き継いで始める．
#   julia -t auto --project=FMD.jl md_nve.jl   (KNbO フォルダで; Mac ローカル実行: CPU バックエンド)
# 基準の 1 本: dt・cgtol・shell_extrap_order は平衡化と同じ．
using FMD
using FMD: UNITS

pot = FMD.loadpotential("./ACEP.jld2")

fs = 1.0/UNITS[:fs]

params = (;
     pot...,
     output_dir = "out_nve_1fs",
     input = "output_result/out-150K_0Gpa_relax/step_39000.xyz",
     backend = :cpu,
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = 1e-6,
     cgmaxiter = 200,
     shell_extrap_order = 2,
     skin = 0.2,
     timestep = 1.0fs,
     thermostat = :none,          # NVE: 熱浴なし
     barostat = :none,            # NVE: セル固定
     output_interval = 1000,
     nsteps = 20_000,             # 10 ps
     initial_temperature = 0.0,   # 0 = 入力の速度をそのまま使う (乱数で引き直さない)
     target_temperature = 150.0,  # NVE では使われない (必須キーなので残す)
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
