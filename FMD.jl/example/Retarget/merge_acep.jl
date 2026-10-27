# Stage 4 — 係数マージ: ベース ACE + delta ACE → 拡張モデル
#
# ACE はエネルギーが係数について線形なので，同一 radial ドメインなら
# 「係数の和 = ポテンシャルの和」が厳密に成り立つ（基底数が違っても
# `+(::ACEParam, ::ACEParam)` が union 基底へ埋め込んで合成する．
# rmin/rmax・種数の不一致はそこでエラーになる）．
#
# 出力の ACEP_merged.jld2 はベースの shell / elements / rc / NLH を継承した
# loadpotential 互換ファイル．MD ではベースと同様，potential.jl の Buckingham
# を手動で重ねること（../training/input_md.jl と同じ理由・同じ形）．
#
# 実行（このディレクトリで，delta の ACEP.jld2 生成後）:
#     julia --project=../.. merge_acep.jl

using Dates
using JLD2
using FMD
using FMD.Calculators: nlh_saved

base  = FMD.loadpotential("../training/ACEP.jld2")
delta = FMD.loadpotential("./ACEP.jld2")

# `+` は rmin/rmax と種数しか検証できない．二重カウントの余地を assert で塞ぐ：
# (1) elements の並びが同一（種数が同じでも並び違いは検出不能）
@assert delta.elements == base.elements "elements の並びが base と delta で不一致"
# (2) delta が null shell（電荷・spring 全ゼロ）で学習されたこと — shell を渡して
#     フィットした delta をマージすると shell 寄与が二重カウントになる
@assert all(iszero, delta.qc) && all(iszero, delta.qs) &&
        all(iszero, delta.k2) && all(iszero, delta.k4) "delta が null shell で学習されていない"
# (3) delta 側に NLH が無いこと — あると delta が減算した NLH が merged に載らず失われる
@assert !haskey(delta, :nlh) "delta 側に NLH ベースラインがある（merged に継承されない）"

merged = base.acep + delta.acep    # 不一致（rmin/rmax/種数）はここでエラー

jldsave("ACEP_merged.jld2";
    acep = merged,
    shell = (
        qref  = base.qref,
        qc    = base.qc,
        qs    = base.qs,
        k2    = base.k2,
        k4    = base.k4,
        eta_c = base.eta_c,
        eta_s = base.eta_s,
    ),
    elements = base.elements,
    rc = base.rc,
    (haskey(base, :nlh) ? (; nlh = nlh_saved(base.nlh)) : (;))...,
    provenance = (
        merged_from = ("../training/ACEP.jld2", "./ACEP.jld2"),
        merge_timestamp = string(now()),
        note = "base + Δ (residual refit); shell/elements/rc/nlh inherited from base",
    ),
)
nk = (size(merged.c0, 1), size(merged.c11, 1), size(merged.c22, 1),
      size(merged.c33, 1)) .÷ length(merged.cp)
println("wrote ACEP_merged.jld2 (ace_nk = $nk)")
