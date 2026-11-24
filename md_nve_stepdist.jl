# 1 ステップあたりのコア・シェルの移動距離と，予測子 / CG による移動距離を測る計測用 NVE．
#   julia -t auto --project=FMD.jl md_nve_stepdist.jl <k> [nsteps]
# md_nve_extrap.jl と同じ条件 (dt=0.5 fs, cgtol=1e-6, 150 K の step_39000 から)．
# FMD._rotate_shist! を差し替えて，CG 直後・履歴回転前の時点で
#   Δr      = velo·dt                 (コアの 1 ステップ移動; drift ステップそのもの)
#   Δd      = d_{n+1} − d_n           (コア–シェル相対変位の変化)
#   Δs      = Δr + Δd                 (シェルの絶対位置の移動)
#   pred    = d_pred − d_n            (予測子 (外挿) が動かした量)
#   cg      = d_{n+1} − d_pred        (CG が動かした量)
# の原子あたりのノルムを元素ごとに平均して CSV に出す (単位 Å)．
using FMD
using FMD: UNITS
using Printf

const EXTRAP = parse(Int, ARGS[1])
const NSTEPS = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 400
const SKIP = 10   # 外挿のウォームアップ (shist が埋まるまで) を除く
const BOHR_A = UNITS[:Å]

pot = FMD.loadpotential("./ACEP.jld2")
fs = 1.0/UNITS[:fs]
outdir = "output_result/stepdist_extrap$(EXTRAP)"
mkpath(outdir)

const STATE = Dict{Symbol,Any}(:step => 0, :dt => 0.5fs, :rows => Vector{String}())
pred_of(a, k) = k == 0 ? a[:shist1] :
                k == 1 ? 2a[:shist1] .- a[:shist2] :
                k == 2 ? 3a[:shist1] .- 3a[:shist2] .+ a[:shist3] :
                k == 3 ? 4a[:shist1] .- 6a[:shist2] .+ 4a[:shist3] .- a[:shist4] :
                         5a[:shist1] .- 10a[:shist2] .+ 10a[:shist3] .- 5a[:shist4] .+ a[:shist5]
rownorm(x) = sqrt.(sum(abs2, Array(x), dims = 2))[:, 1]

@eval FMD.Apps function _rotate_shist!(atoms::Atoms, shist_count::Int)
    a = atoms.arrays
    S = Main.STATE
    S[:step] += 1
    # 予測子が実際に使われたステップ (shist が order+1 点そろってから) だけ集計
    if shist_count >= Main.EXTRAP + 1 && S[:step] > Main.SKIP
        dn   = Array(a[:shist1])
        d1   = Array(a[:shelldisp])
        dp   = Array(Main.pred_of(a, Main.EXTRAP))
        dr   = Array(a[:velo]) .* S[:dt]
        m    = round.(Int, Array(a[:masses])[:, 1] ./ (Main.UNITS[:dalton]))
        cols = (Main.rownorm(dr), Main.rownorm(d1 .- dn), Main.rownorm(dr .+ d1 .- dn),
                Main.rownorm(dp .- dn), Main.rownorm(d1 .- dp), Main.rownorm(d1))
        for (name, mm) in (("all", 0), ("K", 39), ("Nb", 93), ("O", 16))
            sel = mm == 0 ? trues(length(m)) : m .== mm
            any(sel) || continue
            push!(S[:rows], join((S[:step], name,
                (Main.@sprintf("%.6e", sum(c[sel]) / count(sel) * Main.BOHR_A) for c in cols)...), ","))
        end
    end
    for j in _shist_nslots(atoms):-1:2
        a[_shist_key(j)] .= a[_shist_key(j - 1)]
    end
    a[:shist1] .= a[:shelldisp]
    return shist_count + 1
end

params = (;
     pot...,
     output_dir = outdir,
     input = "output_result/out-150K_0Gpa_relax/step_39000.xyz",
     backend = :cpu,
     pme_meshsize = 1.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = 1e-6,
     cgmaxiter = 200,
     shell_extrap_order = EXTRAP,
     skin = 0.2,
     timestep = 0.5fs,
     thermostat = :none,
     barostat = :none,
     output_interval = 100_000,   # xyz は書かない
     nsteps = NSTEPS,
     initial_temperature = 0.0,
     target_temperature = 150.0,
)
params = (; FMD.default_params(Val(:NPTMD), params)..., params...)
FMD.app[:NPTMD](params)

open(joinpath(outdir, "stepdist.csv"), "w") do io
    println(io, "step,elem,core_dr,shell_dd,shell_ds_abs,pred_move,cg_move,shelldisp")
    foreach(r -> println(io, r), STATE[:rows])
end
println("wrote ", joinpath(outdir, "stepdist.csv"), " rows=", length(STATE[:rows]))
