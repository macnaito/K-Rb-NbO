#!/bin/zsh
# dt × cgtol の各パターンを 1 本ずつ順に実行する (計算時間を公平に比べるため並列にしない)．
# 途中で止めても，もう一度実行すれば終わったパターン (log.txt の行数 = nsteps) は飛ばす．
# 1 本が失敗しても (大きい dt で発散など) 記録して次へ進む．
#   ./run_scan.sh                        (KNbO フォルダで; dt = 0.5 0.4 0.3 0.2)
#   ./run_scan.sh 0.6 0.7 0.8 0.9 1.0    (dt を指定)
cd "${0:A:h}"
mkdir -p scan
dts=("$@"); (( ${#dts} )) || dts=(0.5 0.4 0.3 0.2)
for cgtol in 1e-6 7e-7 4e-7 1e-7; do
  for dt in "${dts[@]}"; do
    dir="scan/cgtol=${cgtol}_dt=${dt}fs"
    nsteps=$(python3 -c "import math; print(math.ceil(3000/$dt - 1e-9))")
    if [[ -f $dir/log.txt ]] && (( $(wc -l < $dir/log.txt) == nsteps )); then
      echo "$(date '+%F %T') skip  $dir (done)"; continue
    fi
    echo "$(date '+%F %T') start $dir ($nsteps steps)"
    mkdir -p $dir
    julia -t auto --project=FMD.jl md_scan.jl $dt $cgtol > $dir/stdout.txt 2>&1 \
      || { echo "$(date '+%F %T') FAILED $dir (see $dir/stdout.txt)"; continue; }
    echo "$(date '+%F %T') done  $dir"
  done
done
echo "$(date '+%F %T') all done"
