# BaTiO₃ Shell Model + MLIP Training

BaTiO₃ の shell モデルをベースラインに，ACE（または Spline）MLIP を
Δ 学習でフィットするワークフロー．**VASP 不要・コンシューマ PC
（8 コア，GPU なし）で完走する**規模に調整してある．

構成は「ベースライン PES からサンプル → 参照理論でラベル → 残差を MLIP に
学習させる」の 3 段階．shell パラメータは文献値
（H. Azuma et al., Acta Mater. 296, 121216 (2025),
doi:10.1016/j.actamat.2025.121216 の r²SCAN-D4 セット，
`../ShellModel-MD/potential.jl` を単一ソースとして include）を使い，
ラベルは同じ r²SCAN 系の MACE foundation model（MATPES r²SCAN，分散補正
なし）で付ける．参照汎関数がベースラインとほぼ揃うため残差が小さい
（D4 分散の分だけ厳密には異なり，その寄与は残差として MLIP が学習する）．

## Files

| file | role | 段階 |
|---|---|---|
| `input_sample.jl` | shell モデル NPT-MD で学習配置をサンプル | Stage 1 |
| `label_mace.py` | MACE 単点評価でラベル付け → `train.xyz` | Stage 2 |
| `input_fitace.jl` | CG フィット（ACE）→ `ACEP.jld2` | Stage 3 |
| `input_fitspline.jl` | CG フィット（2/3 体 Spline）→ `SplineP.jld2` | Stage 3' |
| `input_md.jl` | 学習済みポテンシャルで NPT-MD（読み込み側の完全な例） | Stage 4 |
| `train.xyz` | 生成済み学習データ（同梱; Stage 1–2 を飛ばせる） | — |
| `gen_dfpt_macefield.py` | [optional] MACE-Field で BEC/ε → `dfpt.xyz` | Full path |
| `input_fitsm.jl` | [optional] shell モデル自体の CMA-ES フィット | Full path |

## Quickstart（同梱の train.xyz を使う最短経路）

```bash
cd example/training
mpiexec -n 4 julia --heap-size-hint=1G --project=../.. input_fitace.jl   # → ACEP.jld2
```

（1 ランクあたり実測 ~1.5 GB 使う．RAM 16 GB 級では `--heap-size-hint` を
付けて 4 ランク程度に抑えるのが安全．RAM に余裕があればランクを増やしてよい）

MPI が無ければ `julia --project=../.. input_fitace.jl`（1 ランク）でも動く．
ランクはフレーム分割のデータ並列に使われる．

でき上がった `ACEP.jld2` で MD を回す：

```bash
python3 ../ShellModel-MD/makecell.py -n 3 -o input.xyz   # 未生成なら
julia --project=../.. input_md.jl                        # 1 ps NPT @ 300 K
```

**注意**: この example の `ACEP.jld2` は単体では完結しない．fit のベースライン
に文献セットの Buckingham 項を使うが，`ACEP.jld2` には Buckingham が保存され
ないため，MD 側では `loadpotential` の結果に `potential.jl` の `a / rho / c`
を必ず重ねる（`input_md.jl` がその完全な例．忘れてもエラーにならず**黙って
別のポテンシャルになる**）．

## データ生成から回す場合（Stage 1–2）

```bash
cd example/training
python3 ../ShellModel-MD/makecell.py -n 3 -o input.xyz   # 3×3×3 = 135 原子
julia --project=../.. input_sample.jl                    # → out_T*/step_*.xyz
pip install ase mace-torch                               # 初回のみ
# glob は out_T*（サンプリング出力）に限定する — out_* だと input_md.jl の
# 出力 out_md/ まで拾ってしまう
python3 label_mace.py 'out_T*/step_*.xyz' -o train.xyz --burnin 20
```

- サンプリングは 5 条件（200/400/700 K × 0 GPa，400 K × ±2 GPa）×
  3 ps．熱平衡化前の 20 スナップショット/条件は `--burnin` で捨て，
  計 200 フレームになる
- ラベルは倍精度・CPU で評価する（`--device cuda` で GPU も可）．
  モデルは初回実行時に自動ダウンロードされる．MATPES 系 MACE モデルは
  **ASL（Academic Software License）**での配布なので利用条件に注意

## 規模と精度について

example は「手元で完走する」ことを優先した最小規模：

- セル 135 原子・200 フレーム・`ace_nk = (6, 5, 4, 3)`・`nsteps = 2000`
- 本番の目安は `ace_nk = (12, 6, 4, 4)`（デフォルト）・数千フレーム・
  HPC の MPI 並列（`docs/src/parameters.md`）．配置サンプリングも本番では
  uMLIP MD（`dataset/dft/kno` の VASP + MACE fine-tune パイプラインが雛形）
  が望ましい
- NLH 短距離斥力ベースライン（`data/nlh`，遮蔽核斥力）は本番と同様に有効．
  学習データが届かない短距離での発散を保証する（`ACEP.jld2` に焼き込まれ，
  MD 側は自動で加算する）
- fit のベースラインは**文献 shell セット完全形（静電 + spring + Buckingham）
  + NLH**．この文献セットは点電荷 shell モデルで Buckingham が shell の分極
  崩壊を止めているため，Buckingham を外した baseline では shell 緩和が破綻し
  残差フィットが成立しない（実測: baseline 力 ~19 eV/Å）．代償として
  `ACEP.jld2` 単体では完結しない（上の注意と `input_md.jl` を参照）

所要時間の目安（8 コア CPU 実測）：

| 段階 | 時間 |
|---|---|
| Stage 1 sampling（5 条件 × 3 ps，逐次） | 約 16 分（約 3 分/条件） |
| Stage 2 labeling（200 フレーム，CPU 倍精度） | 約 7 分（0.5 frames/s） |
| Stage 3 FitACE（4 ランク） | 約 6 分 |

## Full path — shell モデルも自分でフィットする場合

Quick path は文献 shell パラメータを使うので不要だが，組成や参照理論を
変えて shell から作り直す場合は BEC + 誘電テンソルのデータが要る：

```bash
# mace-field（github.com/mdi-group/mace-field）を pip install し，学習済み
# モデル（例: MACEField-MH-0-omat-dielectric.model）をプロジェクトの配布
# （リポジトリの案内 / 論文 arXiv:2508.17870）から取得したうえで
python3 gen_dfpt_macefield.py 'out_T400_P0/step_*.xyz' \
    --model MACEField-MH-0-omat-dielectric.model --nsamples 20 -o dfpt.xyz
mpiexec -n 16 julia --project=../.. input_fitsm.jl        # → ShellParam.jld2
```

- **精度の注意**: BaTiO₃ では MACE-Field の Z*/ε は DFPT 比 ±10% 程度．
  ワークフローの学習・検証には十分だが，論文品質のフィットには VASP DFPT
  を使う（`dataset/dfpt/kno` が生成パイプラインの雛形．KNbO₃ など
  d⁰ 遷移金属の一部は MACE-Field の外挿が破綻するので材料ごとに要検証）
- 新しい `ShellParam.jld2` を使う場合，`input_fitace.jl` の `POT...` を
  `FMD.loadpotential("ShellParam.jld2")...` に差し替え，Buckingham 項
  （a / rho / c）は別途与えること（shell フィットは静電応答だけを見る）

## dataset/ との役割分担

- `example/training` — 手元で再現できる学習ワークフローの教材（MACE ラベル）
- `dataset/{dft,dfpt}/kno` — 論文品質のデータ生成パイプライン（VASP，HPC 前提）

## Dataset format

`train.xyz` — FMD extxyz．per-frame: `potential_energy=`（eV），
`virial=`（Voigt-6，eV，`= −stress·V`），`method=`（`method_weight` の
キーに対応），per-atom: `forces`．

`dfpt.xyz` — FMD extxyz．per-atom `becex/becey/becez`（Born 有効電荷，
`becex[i,α] = ∂P_x/∂u_{i,α}`），per-frame `dielectric_tensor`（Voigt-6）．
