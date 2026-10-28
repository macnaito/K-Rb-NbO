# (K,Rb)NbO₃ の NPT-MD (殻模型 + ACE)．熱浴 Nosé–Hoover，圧力浴 MTTK (Parrinello–Rahman 型の可変セル)．
#   julia --project=<FMD の入った環境> md.jl
# input.xyz: 理想立方の 8×8×8 セル (2,560 原子)，A サイトの 51/512 (x ≈ 0.10) を乱択で Rb にしたもの
using FMD
using FMD: UNITS

pot = FMD.loadpotential("./ACEP.jld2")

dt = 1.0/UNITS[:fs]   # 1 fs

params = (;
     pot...,
     output_dir = "out",
     input = "input.xyz",
     backend = :cpu,             # GPU が無ければ :cpu
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = 1e-6,
     cgmaxiter = 200,
     skin = 0.2,
     timestep = 0.5dt,
     thermostat = :csvr,
     tau_t = 100dt,
     barostat = :mttk,
     tau_p = 500dt,
     output_interval = 100,
     nsteps = 20_000,
     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     target_temperature = 150.0,
     initial_temperature = 150.0,
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
