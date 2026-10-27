# Stage 2 — 残差データセットの生成: 残差 = 新参照ラベル − ベースモデル全体の予測
#
# ベースモデル全体 = 文献 shell（電荷 / spring / Buckingham）+ NLH + ベース ACE．
# この「全体」を引くのが重要で，こうすると Stage 3 の delta フィットは
# shell を一切知らない純 ACE フィット（ベースライン厳密ゼロ）にでき，
# 二重減算が構造的に起きない．
#
# 実行（このディレクトリで，newref.xyz 生成後）:
#     julia --project=../.. make_residual.jl
#
# 入力: newref.xyz（新参照ラベル．label_mace.py で生成）
#       ../training/ACEP.jld2（ベースポテンシャル）
# 出力: residual.xyz（残差ラベル: energy / forces / virial，method= は継承）

using Printf
using FMD
using FMD.Cores
using FMD.Calculators
using FMD: UNITS

include("../ShellModel-MD/potential.jl")   # -> POT（Buckingham の単一ソース）

pot = FMD.loadpotential("../training/ACEP.jld2")
@assert pot.elements == POT.elements "ACEP.jld2 と potential.jl の elements が不一致"

# ベースモデルのフィット時と同じスタック・同じ数値ノブ（cgtol 等は
# FitACE デフォルト = 1e-8 系）で予測する．残差の定義の一部なので緩めない．
# default_params に渡す dataset 以下のキーは validate の形式を満たすための
# ダミーで，予測経路では読まれない．
base = (; pot..., a = POT.a, rho = POT.rho, c = POT.c)
params = (; FMD.default_params(Val(:FitACE), (; base...,
              dataset = "newref.xyz", traindata_ratio = 1.0,
              method_weight = (;), weight_force = 1.0, weight_stress = 1.0,
              rigid = (;), ace_rmax = pot.acep.rmax))..., base...)

dataset = readxyz(open("newref.xyz", "r"))
println("loaded newref.xyz: $(length(dataset)) frames")

for atoms in dataset
    setelemid!(atoms, params[:elements])
end

# ---- ベースモデル全体の予測を引く ---------------------------------------
for (i, atoms) in enumerate(dataset)
    pred = deepcopy(atoms)
    clear!(pred)
    add_baseline!(pred, params)            # shell + Buckingham + NLH（shell 緩和込み）
    calculate!(pot.acep, pred)             # + ベース ACE
    atoms.arrays[:forces]         .-= pred.arrays[:forces]
    atoms.info[:virial]           .-= pred.info[:virial]
    atoms.info[:potential_energy] .-= pred.info[:potential_energy]
    i % 20 == 0 && println("  $i/$(length(dataset))")
end

# ---- 残差の extxyz 書き出し ----------------------------------------------
# FMD の writexyz は学習ラベル（virial= / method=）を出力しないため，
# readxyz の規約（pos Å / forces eV/Å / potential_energy eV / virial eV Voigt-6）
# をここで明示的にミラーする．直後の読み戻し検証が単位・書式の誤りを捕まえる．
elements = collect(keys(params[:elements]))[sortperm(collect(values(params[:elements])))]
eV, Å = UNITS[:eV], UNITS[:Å]
open("residual.xyz", "w") do io
    for atoms in dataset
        lat = atoms.info[:lattice] .* Å
        latstr = join((@sprintf("%.17g", lat[i, j]) for i in 1:3 for j in 1:3), " ")
        vir = atoms.info[:virial] .* eV
        virstr = join((@sprintf("%.17g", v) for v in vir), " ")
        E = atoms.info[:potential_energy][1] * eV
        method = atoms.info[:method][1]
        println(io, length(atoms))
        println(io, "Lattice=\"$latstr\" Properties=species:S:1:pos:R:3:forces:R:3 " *
                    "potential_energy=$(@sprintf("%.17g", E)) virial=\"$virstr\" " *
                    "method=$method pbc=\"T T T\"")
        cart = atoms.arrays[:pos] * atoms.info[:lattice]
        F = atoms.arrays[:forces]
        for n in 1:length(atoms)
            @printf(io, "%-2s %.17g %.17g %.17g %.17g %.17g %.17g\n",
                    elements[atoms.arrays[:elemid][n]],
                    cart[n, 1]*Å, cart[n, 2]*Å, cart[n, 3]*Å,
                    F[n, 1]*eV/Å, F[n, 2]*eV/Å, F[n, 3]*eV/Å)
        end
    end
end

# ---- 読み戻し検証（書式・単位の往復が機械精度で閉じること） --------------
check = readxyz(open("residual.xyz", "r"))
@assert length(check) == length(dataset)
for (a, b) in zip(dataset, check)
    setelemid!(b, params[:elements])
    @assert isapprox(a.info[:potential_energy][1], b.info[:potential_energy][1];
                     rtol = 1e-12) "energy round-trip mismatch"
    @assert isapprox(a.arrays[:forces], b.arrays[:forces]; rtol = 1e-10, atol = 1e-14) "forces round-trip mismatch"
    @assert isapprox(a.info[:virial], b.info[:virial]; rtol = 1e-10, atol = 1e-14) "virial round-trip mismatch"
    @assert a.info[:method][1] == b.info[:method][1] "method tag mismatch"
    # 幾何（格子・デカルト座標）の往復 — ラベルが合っていても分率⇄デカルト変換
    # や Lattice= の行列順を取り違えると壊れるのはこちら
    @assert isapprox(a.info[:lattice], b.info[:lattice]; rtol = 1e-12) "lattice round-trip mismatch"
    @assert isapprox(a.arrays[:pos] * a.info[:lattice], b.arrays[:pos] * b.info[:lattice];
                     rtol = 1e-9, atol = 1e-12) "cartesian position round-trip mismatch"
end
println("wrote residual.xyz: $(length(dataset)) frames (round-trip verified)")
