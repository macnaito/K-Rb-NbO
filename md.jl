# (K,Rb)NbO₃ の NPT-MD (殻模型 + ACE)．熱浴 Nosé–Hoover，圧力浴 MTTK (Parrinello–Rahman 型の可変セル)．
#   julia -t auto --project=FMD.jl md.jl   (Mac ローカル実行: CPU バックエンド，-t でスレッド並列)
# input.xyz: 理想立方の 8×8×8 セル (2,560 原子)，Rb 0% (純 KNbO₃; python3 make_input.py --rb 0)
using FMD
using FMD: UNITS

pot = FMD.loadpotential("./ACEP.jld2")

dt = 1.0/UNITS[:fs]   # 1 fs

params = (;
     pot...,
     output_dir = "out",
     input = "input.xyz",
     backend = :cpu,              # Mac (GPU 無し) で実行．GPU クラスタでは :cuda
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = 1e-6,
     cgmaxiter = 200,
     shell_extrap_order = 2,      # shell CG 初期値の外挿次数 k ∈ 0:4 (直近 k+1 点; 2 = 従来の3点外挿)
     skin = 0.2,
     timestep = 0.5dt,
     thermostat = :csvr,   #csvr or nhc or nh
     tau_t = 100dt,
     barostat = :mttk,     #mttk or scr
     tau_p = 500dt,
     output_interval = 1000,
     nsteps = 40_000,
     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     target_temperature = 150.0,
     initial_temperature = 150.0,
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
