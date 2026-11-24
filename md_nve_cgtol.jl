# 純 KNbO₃ (2,560 原子) の NVE-MD．150 K・0 GPa で平衡化した NPT (CSVR + MTTK, 20 ps) の
# step_39000.xyz (19.5 ps; セルが 12–20 ps 平均に最も近いフレーム) から，速度とシェル変位を引き継いで始める．
#   julia -t auto --project=FMD.jl md_nve.jl   (KNbO フォルダで; Mac ローカル実行: CPU バックエンド)
# cgtol スキャン: dt=0.5 fs, 4 ps, 3点外挿 (shell_extrap_order=2) で cgtol のみ変える．
using FMD
using FMD: UNITS

pot = FMD.loadpotential("./ACEP.jld2")

fs = 1.0/UNITS[:fs]
const CGTOL_STR = ARGS[1]          # julia ... md_nve_cgtol.jl <cgtol> [k]  (例: 4e-7 3)
const CGTOL = parse(Float64, CGTOL_STR)
const EXTRAP = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2   # 第2引数: 外挿次数 k (省略時 2 = 3点外挿)
const OUTDIR = EXTRAP == 2 ? "output_result/out_nve_0.5fs_cgtol$(CGTOL_STR)" :
                             "output_result/out_nve_0.5fs_k$(EXTRAP)_cgtol$(CGTOL_STR)"

params = (;
     pot...,
     output_dir = OUTDIR,
     input = "output_result/out-150K_0Gpa_relax/step_39000.xyz",
     backend = :cpu,
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = CGTOL,
     cgmaxiter = 200,
     shell_extrap_order = EXTRAP,  # k+1 点外挿 (2 = 3点, 3 = 4点)
     skin = 0.2,
     timestep = 0.5fs,
     thermostat = :none,          # NVE: 熱浴なし
     barostat = :none,            # NVE: セル固定
     output_interval = 1000,
     nsteps = 8000,             # 4 ps
     initial_temperature = 0.0,   # 0 = 入力の速度をそのまま使う (乱数で引き直さない)
     target_temperature = 150.0,  # NVE では使われない (必須キーなので残す)
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
