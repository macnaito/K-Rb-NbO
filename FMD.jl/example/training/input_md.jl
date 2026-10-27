# Stage 4 — 学習済みポテンシャルで NPT-MD（動作確認を兼ねた読み込み側の完全な例）
#
# ★ この example の ACEP.jld2 は単体では完結しない：fit のベースラインに文献
#   shell セットの Buckingham 項を使っているが，FitACE はそれを ACEP.jld2 に
#   保存しない．MD 側では下のように loadpotential の結果へ potential.jl の
#   Buckingham（a / rho / c）を必ず重ねること．忘れても即座にはエラーに
#   ならず，黙って別のポテンシャル（Buckingham = 0）で走ってしまう．
#
# 実行（このディレクトリで，ACEP.jld2 生成後）:
#     julia --project=../.. input_md.jl
#
# 出力: out_md/log.txt, step_<n>.xyz（1 ps の NPT @ 300 K）

using FMD
using FMD: UNITS

include("../ShellModel-MD/potential.jl")   # -> POT（Buckingham の単一ソース）

pot = FMD.loadpotential("./ACEP.jld2")     # shell + ACE 係数 + NLH + rc

# Buckingham 行列は種 ID 添字なので，elements の対応が食い違うと黙って
# 別ポテンシャルになる．マージ前に一致を確認する．
@assert pot.elements == POT.elements "ACEP.jld2 と potential.jl の elements が不一致"

dt = 1.0 / UNITS[:fs]

params = (;
    pot...,
    a = POT.a, rho = POT.rho, c = POT.c,   # ← fit ベースラインと同じ Buckingham
    input = "input.xyz",                   # makecell.py -n 3 の 3×3×3 セル
    output_dir = "out_md",

    # MD 用の緩い数値ノブ（fit 時は cgtol = ewald_tolerance = 1e-8 の高精度
    # Ewald）．ベースラインの shell 緩和が学習時よりわずかに粗くなるが，MD の
    # 常用精度としては十分．Ewald α は (rc, pme_meshsize) から自動決定され，
    # 推定 RMS 力誤差がログに出る（Issue #216）
    pme_meshsize = 2.0,
    efield = [0.0, 0.0, 0.0],
    cgtol = 2e-5,
    cgmaxiter = 50,
    skin = 0.2,

    timestep = dt,
    nsteps = 1000,
    output_interval = 100,

    thermostat = :csvr,
    tau_t = 100dt,
    barostat = :scr,
    tau_p = 500dt,
    compressibility = UNITS[:GPa] / 160.0,
    target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    initial_temperature = 300.0,
    target_temperature = 300.0,
    seed = 7,

    backend = :cpu,                        # 135 原子は CPU が速い
)
params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
