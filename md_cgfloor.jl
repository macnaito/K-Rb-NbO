# cgtol をどこまで締められるか (CG の精度限界) を見る短い NPT-MD．
#   julia -t auto --project=FMD.jl md_cgfloor.jl <cgtol>
# dt = 0.5 fs，10 ステップ，cgmaxiter = 400．限界より下の cgtol では ncg が cgmaxiter に張り付く．
using FMD
using FMD: UNITS
cgtol = parse(Float64, ARGS[1])
fs = 1.0/UNITS[:fs]
pot = FMD.loadpotential("./ACEP.jld2")
params = (; pot...,
     output_dir = "scan_floor/cgtol=$(ARGS[1])",
     input = "input.xyz", backend = :cpu, pme_meshsize = 1.0, efield = [0.0, 0.0, 0.0],
     cgtol = cgtol, cgmaxiter = 400, shell_extrap_order = 3, skin = 0.2,
     timestep = 0.5fs, thermostat = :csvr, tau_t = 100fs, barostat = :mttk, tau_p = 500fs,
     output_interval = 1000, nsteps = 10,
     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     target_temperature = 150.0, initial_temperature = 150.0)
params = (; FMD.default_params(Val(:NPTMD), params)..., params...)
FMD.app[:NPTMD](params)
