# dt × cgtol スキャンの 1 パターン分 (メモ calculation cost.rtf)．
#   julia -t auto --project=FMD.jl md_scan.jl <dt[fs]> <cgtol>
#   例: julia -t auto --project=FMD.jl md_scan.jl 0.5 1e-6
# 条件: 8×8×8 (2,560 原子)，150 K，0 GPa，CSVR + MTTK，shell_extrap_order = 3，3 ps (dt = 0.2〜1.0 fs)．
# 出力: scan/cgtol=<cgtol>_dt=<dt>fs/  (log.txt の wall 列 = 1 ステップの計算時間)
using FMD
using FMD: UNITS

length(ARGS) == 2 || error("usage: md_scan.jl <dt[fs]> <cgtol>")
dt_fs = parse(Float64, ARGS[1])
cgtol = parse(Float64, ARGS[2])
dt_fs > 0 && cgtol > 0 || error("dt と cgtol は正の値 (got dt = $dt_fs, cgtol = $cgtol)")

total_fs = 3000.0                          # 3 ps
# 割り切れない dt (0.7, 0.9 fs) は切り上げて 3 ps 以上にする (メモ: 4286, 3334 step)
nsteps = ceil(Int, total_fs / dt_fs - 1e-9)

fs = 1.0/UNITS[:fs]
pot = FMD.loadpotential("./ACEP.jld2")

params = (;
     pot...,
     output_dir = "scan/cgtol=$(ARGS[2])_dt=$(ARGS[1])fs",
     input = "input.xyz",
     backend = :cpu,
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = cgtol,
     cgmaxiter = 200,
     shell_extrap_order = 3,
     skin = 0.2,
     timestep = dt_fs*fs,
     thermostat = :csvr,
     tau_t = 100fs,                         # 時定数は fs で固定 (dt に依らない)
     barostat = :mttk,
     tau_p = 500fs,
     output_interval = 1000,
     nsteps = nsteps,
     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     target_temperature = 150.0,
     initial_temperature = 150.0,
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
