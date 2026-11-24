#!/bin/zsh
# cgtol スキャン NVE (dt=0.5fs, 4ps, 4点外挿 k=3) を順番に実行 (各回全スレッド使用, 計算時間比較のため直列)
cd /Users/hirokinaito/Desktop/KNbO
for tol in 5e-6 1e-6 7e-7 4e-7 1e-7; do
  d=output_result/out_nve_0.5fs_k3_cgtol$tol
  mkdir -p $d
  echo "=== k3 cgtol $tol start $(date)"
  /usr/bin/time -p julia -t auto --project=FMD.jl md_nve_cgtol.jl $tol 3 > $d/stdout.txt 2>&1
  echo "=== k3 cgtol $tol end $(date) exit=$?"
done
