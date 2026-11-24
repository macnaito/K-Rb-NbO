#!/bin/zsh
# 外挿次数 0..4 の NVE (dt=0.5fs, 4ps, cgtol=1e-6) を順番に実行 (各回全スレッド使用, 計算時間比較のため直列)
cd /Users/hirokinaito/Desktop/KNbO
for k in 0 1 2 3 4; do
  mkdir -p output_result/out_nve_0.5fs_extrap$k
  echo "=== order $k start $(date)"
  /usr/bin/time -p julia -t auto --project=FMD.jl md_nve_extrap.jl $k > output_result/out_nve_0.5fs_extrap$k/stdout.txt 2>&1
  echo "=== order $k end $(date) exit=$?"
done
