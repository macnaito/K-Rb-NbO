# Retarget — 差分学習によるベースポテンシャルの参照乗り換え

既存のポテンシャルを**作り直さずに**別の参照理論へ適応させる：
「新しい学習データ − ベースモデルの予測」の残差だけを小さな ACE でフィットし，
係数をマージして拡張モデルを得る．学習が軽く（残差は小さく滑らか），
ベースとの結果比較が同一土俵でできるのが利点．

この手順がそのまま使える用途：

- **汎関数の乗り換え**（この example のデモ: r²SCAN 系 → PBE 系）
- **電子ドープなど摂動系への適応**（ドープ濃度ごとに 1 つの delta を作る）
- **類似材料・組成への微調整**

原理: ACE のエネルギーは係数について厳密に線形なので，同一 radial 基底
ドメインなら「係数の和 = ポテンシャルの和」が機械精度で成り立つ．基底数が
違っても Chebyshev 基底は入れ子なので union 基底への埋め込みで合成できる
（`+(::ACEParam, ::ACEParam)`．非互換な基底はエラーで止まる）．

前提: `example/training` を実行済みで `../training/ACEP.jld2` と
配置データ（`../training/out_T*/step_*.xyz`）があること．

## Files

| file | role | 段階 |
|---|---|---|
| （`../training/label_mace.py`） | 同じ配置を新参照で再ラベル → `newref.xyz` | Stage 1 |
| `make_residual.jl` | 残差 = 新 ref − ベースモデル全体の予測 → `residual.xyz` | Stage 2 |
| `input_fitdelta.jl` | 残差の純 ACE フィット（null shell）→ delta `ACEP.jld2` | Stage 3 |
| `merge_acep.jl` | ベース + delta → `ACEP_merged.jld2` | Stage 4 |
| `compare.jl` | マージ恒等式 + base vs merged のパリティ | Stage 5 |

## Quickstart

```bash
cd example/Retarget

# 1. 同じ配置を新参照（PBE 系 mpa-0）で再ラベル
# glob は out_T*（サンプリング出力）に限定する — out_* だと input_md.jl の
# 出力 out_md/ まで拾ってしまう
python3 ../training/label_mace.py '../training/out_T*/step_*.xyz' \
    -o newref.xyz --burnin 20 --model medium-mpa-0 --method-tag MACE_MPA0

# 2. 残差生成（ベースモデル全体の予測を引く）
julia --project=../.. make_residual.jl

# 3. 残差を小さな ACE でフィット（null shell）
mpiexec -n 4 julia --heap-size-hint=1G --project=../.. input_fitdelta.jl

# 4. 係数マージ
julia --project=../.. merge_acep.jl

# 5. 検証（マージ恒等式 + パリティ改善）
julia --project=../.. compare.jl
```

`ACEP_merged.jld2` はベースの shell / elements / rc / NLH を継承した
`loadpotential` 互換ファイル．MD での使い方はベースと同じ
（`../training/input_md.jl` の `ACEP.jld2` を `ACEP_merged.jld2` に
読み替える．Buckingham の手動マージが必要な点も同じ）．

## 罠（この構成が守っていること）

1. **残差は「ベースモデル全体」の予測で作る**（shell + Buckingham + NLH + ACE）．
   そのうえで delta フィットには shell / NLH を**渡さない**（FitACE のデフォルト =
   null shell でベースライン減算が厳密ゼロ）．残差に shell を渡すと二重減算になる
2. **delta の `ACEP.jld2`（中間生成物）を MD に使わない**．null shell を含み
   `loadpotential` のバリデーションを通ってしまうが，静電なしのポテンシャルとして
   黙って動く．MD に使ってよいのは `ACEP_merged.jld2` だけ
3. **radial ドメイン（rmin/rmax）と elements 順はベースから機械的に継承**
   （`input_fitdelta.jl` が `../training/ACEP.jld2` から読む）．手打ちでずらすと
   係数マージが無効になる — 不一致は `+` がエラーで捕捉するが，elements の
   **並び順**だけは種数が同じ限り検出できないので継承で守る
4. **残差生成の数値ノブは fit 相当の厳密さ**（cgtol = ewald_tolerance = 1e-8）．
   残差ラベルの定義の一部なので MD 用の緩い値を使わない
5. shell 電荷そのものを作り直した場合はベースラインが変わるため本手順の適用外
   （その時点で別モデルとして扱う）

## 期待される結果と所要時間（8 コア CPU 実測）

| 段階 | 時間 |
|---|---|
| Stage 1 relabel（200 フレーム，CPU 倍精度） | 約 6 分 |
| Stage 2 residual（shell 緩和込み全予測 + 往復検証） | 約 6 分 |
| Stage 3 delta フィット（`ace_nk = (5,4,3,3)`，4 ランク） | 約 5 分 |
| Stage 4–5 merge + compare | 約 3 分 |

`compare.jl` の実測（新参照 = `medium-mpa-0`，delta の test split =
**out-of-sample** 20 フレーム）：

- マージ恒等式は機械精度で成立
- 新参照に対する RMSE（base → merged）:
  - energy: 7146.6 → **0.26** meV/atom（base の値はフレーバー間の
    エネルギー原点オフセットが支配的．オフセットは delta の per-element
    定数 `cp` が吸収する）
  - force: 115.6 → **34.0** meV/Å

残差の構造も参考まで: r²SCAN 系 → PBE 系の乗り換えでは
「巨大な定数オフセット（+7.15 eV/atom）+ 小さな変動（sd 6.3 meV/atom，
力の残差 max 1.3 eV/Å）」となり，「残差は小さく滑らか」という
差分学習の前提が成り立つ．

delta の基底はベースより小さくてよい（残差は小さく滑らか）．本番では
新参照の配置を追加サンプリングして混ぜるのも有効（`label_mace.py` は
任意の xyz を受け付ける）．
