# Stage 5 — 検証: マージ恒等式と，新参照に対する base / merged のパリティ比較
#
# (1) マージ恒等式: 任意配置で E/F/V(merged) = E/F/V(base ACE) + E/F/V(delta ACE)
#     が機械精度で成り立つこと（係数線形性の直接検証）
# (2) パリティ: 新参照ラベル（newref.xyz）に対する全モデル予測
#     （shell + Buckingham + NLH + ACE）の RMSE を base / merged で比較する．
#     フレームは delta フィットの **test split（out-of-sample）** を再現して使う
#     — FitACE は seed 0 で shuffle してから末尾 10% を test にするので，
#     同じ置換を再現すれば delta が学習に使っていないフレームだけで評価できる．
#
# 実行（このディレクトリで，ACEP_merged.jld2 生成後）:
#     julia --project=../.. compare.jl

using Random
using Statistics
using FMD
using FMD.Cores
using FMD.Calculators
using FMD: UNITS

include("../ShellModel-MD/potential.jl")

base   = FMD.loadpotential("../training/ACEP.jld2")
delta  = FMD.loadpotential("./ACEP.jld2")
merged = FMD.loadpotential("./ACEP_merged.jld2")
@assert base.elements == POT.elements

dataset = readxyz(open("newref.xyz", "r"))
for atoms in dataset
    setelemid!(atoms, base.elements)
end

# delta フィット（input_fitdelta.jl，traindata_ratio = 0.9）と同じ split を再現．
# residual.xyz と newref.xyz はフレーム順が同一なので index の置換で対応が付く．
idx = collect(1:length(dataset))
Random.seed!(0)
Random.shuffle!(idx)
ntrain = floor(Int, length(dataset) * 0.9)
test_frames = dataset[idx[(ntrain+1):end]]
println("out-of-sample frames (delta の test split): $(length(test_frames))")

# ---- (1) マージ恒等式（純 ACE 評価，機械精度） ---------------------------
function ace_efv(acep, atoms)
    a = deepcopy(atoms)
    clear!(a)
    calculate!(acep, a)
    return a.info[:potential_energy][1], copy(a.arrays[:forces]), copy(a.info[:virial])
end

for atoms in test_frames[1:min(5, end)]
    Ebase, Fbase, Vbase = ace_efv(base.acep, atoms)
    Edel,  Fdel,  Vdel  = ace_efv(delta.acep, atoms)
    Emrg,  Fmrg,  Vmrg  = ace_efv(merged.acep, atoms)
    # 成分最大値ベースの厳しい恒等式チェック（ノルム比較は相殺を見逃す）
    @assert abs(Emrg - (Ebase + Edel)) <=
            1e-12 * max(abs(Ebase), abs(Edel), abs(Emrg)) + 1e-15 "energy identity"
    @assert maximum(abs, Fmrg .- (Fbase .+ Fdel)) <=
            1e-12 * maximum(abs, Fmrg) + 1e-14 "force identity"
    @assert maximum(abs, Vmrg .- (Vbase .+ Vdel)) <=
            1e-12 * maximum(abs, Vmrg) + 1e-14 "virial identity"
end
println("(1) merge identity: E/F/V(merged) == base + delta at machine precision ✓")

# ---- (2) 新参照に対する全モデル予測のパリティ（out-of-sample） -----------
function predict(pot, atoms)
    prms = (; pot..., a = POT.a, rho = POT.rho, c = POT.c)
    # default_params は数値ノブ（cgtol / ewald_tolerance = 1e-8 等）を補うために
    # 呼ぶ．dataset 以下のキーは validate の形式を満たすためのダミーで，予測経路
    # では読まれない．
    prms = (; FMD.default_params(Val(:FitACE), (; prms...,
                dataset = "-", traindata_ratio = 1.0, method_weight = (;),
                weight_force = 1.0, weight_stress = 1.0, rigid = (;),
                ace_rmax = pot.acep.rmax))..., prms...)
    pred = deepcopy(atoms)
    clear!(pred)
    add_baseline!(pred, prms)
    calculate!(pot.acep, pred)
    return pred
end

function rmse_report(pot, label)
    eV, Å = UNITS[:eV], UNITS[:Å]
    dE2, dF2, nE, nF = 0.0, 0.0, 0, 0
    for atoms in test_frames
        pred = predict(pot, atoms)
        N = length(atoms)
        dE2 += ((pred.info[:potential_energy][1] - atoms.info[:potential_energy][1])/N)^2
        nE += 1
        dF2 += sum(abs2, pred.arrays[:forces] .- atoms.arrays[:forces])
        nF += 3N
    end
    rE, rF = sqrt(dE2/nE)*eV*1000, sqrt(dF2/nF)*eV/Å*1000
    @info label energy_rmse_meV_atom = rE force_rmse_meV_A = rF
    return rE, rF
end

println("(2) parity vs newref, out-of-sample ($(length(test_frames)) frames):")
rE_base, rF_base = rmse_report(base,   "base   (r²SCAN 系のまま)")
rE_mrg,  rF_mrg  = rmse_report(merged, "merged (retarget 後)")
println()
println("energy RMSE: $(round(rE_base, digits = 3)) → $(round(rE_mrg, digits = 3)) meV/atom")
println("force  RMSE: $(round(rF_base, digits = 2)) → $(round(rF_mrg, digits = 2)) meV/Å")
rE_mrg < rE_base && rF_mrg < rF_base ||
    @warn "merged が base を改善していません — 手順を確認してください"
