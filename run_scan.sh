#!/bin/zsh
# dt × cgtol の 16 パターンを 1 本ずつ順に実行する (計算時間を公平に比べるため並列にしない)．
# 途中で止めても，もう一度実行すれば終わったパターン (log.txt の行数 = nsteps) は飛ばす．
#   ./run_scan.sh            (KNbO フォルダで)
cd "${0:A:h}"
mkdir -p scan
for cgtol in 1e-6 7e-7 4e-7 1e-7; do
  for dt in 0.5 0.4 0.3 0.2; do
    dir="scan/cgtol=${cgtol}_dt=${dt}fs"
    nsteps=$(python3 -c "print(round(3000/$dt))")
    if [[ -f $dir/log.txt ]] && (( $(wc -l < $dir/log.txt) == nsteps )); then
      echo "$(date '+%F %T') skip  $dir (done)"; continue
    fi
    echo "$(date '+%F %T') start $dir ($nsteps steps)"
    mkdir -p $dir
    julia -t auto --project=FMD.jl md_scan.jl $dt $cgtol > $dir/stdout.txt 2>&1 \
      || { echo "$(date '+%F %T') FAILED $dir (see $dir/stdout.txt)"; exit 1; }
    echo "$(date '+%F %T') done  $dir"
  done
done
echo "$(date '+%F %T') all done"
