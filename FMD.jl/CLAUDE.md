# CLAUDE.md

## 原則：結果の信頼性が第一

数値シミュレーション結果の誤りは，コードひいては論文の信頼性を揺るがす．誤った結果を出して動くくらいなら，動かない方が良い．

**プレースホルダは置かない**：やむをえずプレースホルダ（未実装部分）を残すとしても，その箇所への到達時にはエラーを出し，未処理のまま動作を続行しない．
**誤った結果を与える簡略化をしない**：特定の引数・条件下でのみ正しい結果を与える処理は，引数・条件チェックを行い，正しい結果を与えないならエラーを出す．
**テストを妥協しない**：コードが動作することだけでなく，理論的に期待される出力を与えるか，等価になるべきものが正しく等価になるかなど，徹底的にチェックする．
**ユーザーの期待を裏切らない**：ユーザーがインターフェースを見て期待するものと異なる動作をすると，ユーザーは結果の解釈を誤る．
**コードを簡潔に保つ**：プロジェクトが複雑になるほど管理が難しくなり，バグも混入しやすくなる．小規模プロジェクト（開発は一人，使用は研究室内の数人）なので，後方互換はそれほど意識しなくて良い．

## 開発フロー

main直pushは禁止．すべての変更は **Issue → branch → PR** を経由する．

1. Issueを立てる（or 既存をpick）
2. `git switch main && git pull`
3. `gh issue develop <N> --checkout` でIssue連携branchを作成
4. 修正 + テスト追加・実行（後述「テスト」節）
5. `git push -u origin HEAD`
6. `gh pr create --title "..." --body "...\n\nCloses #<N>"`
7. セルフレビュー（後述「セルフレビュー」節）
8. `gh pr merge --squash --delete-branch`（squash merge + ローカル/リモートbranch削除 + mainへ切替が一括で行われる．ローカルbranchは `-D` 相当で強制削除される点に注意）
9. `git pull`（mainを最新化．gh が fast-forward 済みなら no-op）

## テスト

環境依存（GPU / MPI）や処理時間を理由にテストを省略・削除しない．
実行環境の制約はプロファイル分割で吸収する：

```bash
julia --project test/runtests.jl                 # default: core + calculators (CPU) + apps smoke
FMD_TEST_GPU=1 julia --project test/runtests.jl  # + gpu/ + Ewald==PME等価性 + npt_short（要 CUDA.functional()）
FMD_TEST_MPI=1 julia --project test/runtests.jl  # + mpi/（subprocess driver，要 mpiexec）
FMD_TEST_GPU=1 FMD_TEST_MPI=1 julia --project test/runtests.jl  # + GPU+MPI tier（GPU DD，要 CUDA-aware MPI）
```

- プロファイルは包含関係にある：両フラグ実行は default + GPU + MPI + GPU+MPI の**全 tier を実行する**
  （runtests.jl の MPI tier gate は GPU フラグと独立．MPI 子スクリプトは `FMD_TEST_*` を一切読まず，
  GPU 経路は ARGS の backend 引数で明示指定されるため，親の環境変数は子の挙動を変えない．
  この不変条件は runtests.jl の「profile inclusion invariant」テストで担保）（Issue #109）
- したがって GPU + CUDA-aware MPI のある機（開発機）でのフルゲートは **両フラグ 1 実行のみ**．
  default / GPU / MPI 単独はその完全な部分集合なので**重ねて実行しない**（重複テストは時間の無駄）．
  単独プロファイルを使うのは，CUDA や mpiexec の無い環境で回す場合と，runtests.jl の gating 分岐
  自体を変更した場合（単独でも緑になることの確認）だけ．
  新規テストの単体ランナー（scratchpad の include スクリプト）も両フラグ実行と重複する：早期
  フィードバックが要る時だけ使い，通ったらゲートは両フラグ 1 回で締める
- `Pkg.test()` は使わない：test 用 sandbox 環境が作られ，`LocalPreferences.toml`（CUDA_Runtime_jll の
  local 設定など）が反映されず `CUDA.functional()` が false になる（`script/test_gpu_mpi.sh` 冒頭の注意書き参照）
- HPC では GPU / MPI プロファイルは `pjsub script/test_gpu_mpi.sh` で回す
- GPU / MPI コードに触れる変更では，該当プロファイルを必ず実行してからPRを出す（両フラグの併用可）．
  `FMD_TEST_GPU=1` で `CUDA.functional() == false` の場合は error で即失敗する（Issue #18）ので，
  `FMD_TEST_GPU=1` を付けた実行が緑なら GPU テストは実際に走っている
- テストは `FMD.FPPREC === Float64` を前提とする（`LocalPreferences.toml` の `[FMD] precision = "Float64"`）．
  Float32 のままでは `runtests.jl` 冒頭の assert で即エラーになる（有限差分比較が rtol < 1e-4 に
  届かないため）．したがって Float32 ビルドは現状テストされていない

## PR

例: `feat: add Langevin thermostat`

PR本文に `Closes #<N>` を必ず入れる（mergeでIssueが自動close）．

## セルフレビュー

PRを出した後，`fmd-reviewer` サブエージェント（`.claude/agents/fmd-reviewer.md`）を
**context isolated で**起動して一次レビューを行う．

- 起動: Agent ツールで `subagent_type: fmd-reviewer`，**必ず `isolation: "worktree"` を付ける**
  （レビュアーは読み取り専用が原則だが，万一逸脱しても本体の作業ツリーが汚れない保険．
  Issue #29 参照）
- 観点: 運用ルール準拠 / コードレベル / 変更内容に応じた物理的整合性
- 出力: Blocker / Should-fix / Nit の3段階で総評

レビュアーは実行による検証を行わない（読み取り専用）．「未検証」として返された
検証コマンドは，**親会話が実行して結果を確認する**こと — 実行しないまま
指摘を放置しない．
レビュー観点（特に物理的整合性の例）は `.claude/agents/fmd-reviewer.md` で更新する．

## マージ方針

- **Squash merge のみ**
- 1 Issue = 1 PR = 1 機能
  - ただし同一変更で自然に解決される場合は複数 Closes 可
- 無関係な変更を混ぜない

## NG

- mainへの直接push
- `--force` push（必要時は --force-with-lease）
- テストなしの新機能追加
