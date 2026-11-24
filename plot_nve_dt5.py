#!/usr/bin/env python3
"""NVE の保存量ドリフトを dt = 1.0 / 0.5 / 0.4 / 0.3 / 0.2 fs の 5 本で比べる図を作る．

    python3 plot_nve_dt5.py   →  output_result/nve_compare_dt5.html

各 out_nve_<dt>fs/log.txt の先頭 4 ps を使う（列 18 = conserved [Hartree]，列 1 = 温度）．
"""
import json
import statistics as st
from pathlib import Path

HA_MEV = 27211.386245988
NATOM = 2560
T_END = 4.0  # ps
DTS = [1.0, 0.5, 0.4, 0.3, 0.2]
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
ROOT = Path(__file__).parent / "output_result"


def linfit(t, y):
    mt, my = st.fmean(t), st.fmean(y)
    b = sum((a - mt) * (c - my) for a, c in zip(t, y)) / sum((a - mt) ** 2 for a in t)
    res = [c - (my + b * (a - mt)) for a, c in zip(t, y)]
    return b, st.pstdev(res)


series, stats = [], []
for dt in DTS:
    nmax = round(T_END / (dt * 1e-3))
    E, T, ncg, wall = [], [], [], []
    with open(ROOT / f"out_nve_{dt}fs" / "log.txt") as f:
        for i, line in enumerate(f):
            if i >= nmax:
                break
            v = line.split()
            E.append(float(v[17]))
            T.append(float(v[0]))
            ncg.append(int(v[-1]))
            wall.append(float(v[19]))
    t = [(i + 1) * dt * 1e-3 for i in range(len(E))]
    dE = [(e - E[0]) * HA_MEV / NATOM for e in E]
    slope, sd = linfit(t, dE)
    stride = max(1, round(2.0 / dt))  # 約 2 fs ごとに間引く（描画用のみ）
    pts = [[round(t[i], 4), round(dE[i], 6)] for i in range(0, len(t), stride)]
    if pts[-1][0] != round(t[-1], 4):
        pts.append([round(t[-1], 4), round(dE[-1], 6)])
    series.append({"dt": f"{dt:.1f}", "pts": pts})
    stats.append({"dt": f"{dt:.1f}", "dtv": dt, "nsteps": len(E), "end": dE[-1], "slope": slope, "sd": sd,
                  "ncg": st.fmean(ncg), "wall": st.fmean(wall),
                  "wall_ps": st.fmean(wall) / (dt * 1e-3) / 3600})
    print(f"dt={dt}: steps={len(E)} ΔE(4ps)={dE[-1]:+.4f} slope={slope:+.4f} sd={sd:.4f} "
          f"<ncg>={st.fmean(ncg):.1f} wall/step={st.fmean(wall):.3f}s")

html = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVE dt 比較</title>
<style>
:root{--bg:#fbfbfa;--card:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e6e5e1;--grid:rgba(0,0,0,.06);
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#141413;--card:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--grid:rgba(255,255,255,.08);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}}
:root[data-theme="dark"]{--bg:#141413;--card:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3a38;--grid:rgba(255,255,255,.08);
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,BlinkMacSystemFont,"Hiragino Sans","Segoe UI",sans-serif}
main{max-width:1000px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:20px;margin:0 0 4px;font-weight:600}
.sub{color:var(--muted);margin:0 0 16px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin:0 0 16px}
.card h2{font-size:15px;margin:0 0 2px;font-weight:600}
.card p{margin:0 0 10px;color:var(--muted);font-size:13px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;margin:0 0 8px}
.legend button{all:unset;cursor:pointer;display:inline-flex;align-items:center;gap:6px;color:var(--fg)}
.legend button[aria-pressed="false"]{opacity:.35}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.wrap{position:relative;height:420px}
.wrap.sm{height:300px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:500}
.tw{overflow-x:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
</style>
</head>
<body>
<main>
<h1>NVE 保存量の dt 依存性</h1>
<p class="sub">KNbO₃ 2,560 原子・150 K・同一初期配置（NPT step_39000）から 4 ps．縦軸は保存量の初期値からの変化 ΔH（meV/atom）．</p>

<div class="card">
<h2>ΔH(t)</h2>
<p>凡例をクリックで表示を切り替え（縦軸は表示中の系列に合わせて伸縮）．</p>
<div class="legend" id="lg"></div>
<div class="wrap"><canvas id="c1" aria-label="保存量の時間変化"></canvas></div>
</div>

<div class="card">
<h2>ドリフトと揺らぎ（4 ps の線形フィット）</h2>
<p>傾き = 系統的ドリフト，揺らぎ = フィット残差の標準偏差（Verlet の dt² 誤差が主）．</p>
<div class="wrap sm"><canvas id="c2" aria-label="dt ごとのドリフトと揺らぎ"></canvas></div>
<div class="tw" style="margin-top:12px"><table id="tb"><thead><tr>
<th>dt (fs)</th><th>ステップ数</th><th>ΔH(4 ps) meV/atom</th><th>傾き meV/atom/ps</th><th>揺らぎ σ meV/atom</th><th>平均 CG 反復</th><th>1 ステップ (s)</th><th>1 ps あたり (h)</th>
</tr></thead><tbody></tbody></table></div>
</div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const SERIES = __SERIES__;
const STATS = __STATS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const col = i => css('--s' + (i + 1));
const fmt = (x, d) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);

function base() {
  return {
    responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
    interaction: {mode: 'nearest', axis: 'x', intersect: false},
    plugins: {legend: {display: false}, decimation: {enabled: false}},
  };
}
function axis(title) {
  return {title: {display: true, text: title, color: css('--muted')},
          ticks: {color: css('--muted')}, grid: {color: css('--grid')}, border: {color: css('--line')}};
}

let c1, c2;
function draw() {
  c1 && c1.destroy(); c2 && c2.destroy();
  const o1 = base();
  o1.plugins.tooltip = {callbacks: {
    title: it => `t = ${it[0].parsed.x.toFixed(3)} ps`,
    label: c => `dt=${SERIES[c.datasetIndex].dt} fs: ${fmt(c.parsed.y, 4)} meV/atom`}};
  o1.scales = {x: {...axis('時間 (ps)'), type: 'linear', min: 0, max: 4},
               y: axis('ΔH (meV/atom)')};
  c1 = new Chart(document.getElementById('c1'), {type: 'line', options: o1,
    data: {datasets: SERIES.map((s, i) => ({label: `dt=${s.dt} fs`,
      data: s.pts.map(p => ({x: p[0], y: p[1]})), borderColor: col(i), backgroundColor: col(i),
      borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, hidden: !vis[i]}))}});

  const o2 = base();
  o2.parsing = true;
  o2.interaction = {mode: 'index', intersect: false};
  o2.plugins.legend = {display: true, labels: {color: css('--fg'), boxWidth: 14, boxHeight: 3}};
  o2.plugins.tooltip = {callbacks: {title: it => `dt = ${it[0].label} fs`,
    label: c => c.datasetIndex === 0 ? `傾き: ${fmt(c.parsed.y, 4)} meV/atom/ps`
                                     : `揺らぎ σ: ${c.parsed.y.toFixed(4)} meV/atom`}};
  o2.scales = {x: {...axis('dt (fs)'), type: 'category'}, y: axis('meV/atom（/ps）')};
  const asc = [...STATS].sort((a, b) => a.dtv - b.dtv);
  c2 = new Chart(document.getElementById('c2'), {type: 'line', options: o2,
    data: {labels: asc.map(s => s.dt),
      datasets: [
        {label: 'ドリフト傾き (meV/atom/ps)', data: asc.map(s => s.slope), borderColor: css('--fg'),
         backgroundColor: css('--fg'), borderWidth: 2, pointRadius: 4},
        {label: '揺らぎ σ (meV/atom)', data: asc.map(s => s.sd), borderColor: css('--muted'),
         backgroundColor: css('--muted'), borderWidth: 2, borderDash: [5, 4], pointRadius: 4}]}});
}

const vis = SERIES.map(() => true);
const lg = document.getElementById('lg');
SERIES.forEach((s, i) => {
  const b = document.createElement('button');
  b.setAttribute('aria-pressed', 'true');
  b.innerHTML = `<i style="background:${col(i)}"></i>dt=${s.dt} fs`;
  b.onclick = () => { vis[i] = !vis[i]; b.setAttribute('aria-pressed', vis[i]);
    c1.setDatasetVisibility(i, vis[i]); c1.update(); };
  lg.appendChild(b);
});

const tb = document.querySelector('#tb tbody');
STATS.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="sw" style="background:${col(i)}"></span>${s.dt}</td><td>${s.nsteps}</td>
    <td>${fmt(s.end, 4)}</td><td>${fmt(s.slope, 4)}</td><td>${s.sd.toFixed(4)}</td>
    <td>${s.ncg.toFixed(1)}</td><td>${s.wall.toFixed(3)}</td><td>${s.wall_ps.toFixed(2)}</td>`;
  tb.appendChild(tr);
});

draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  draw();
  lg.querySelectorAll('i').forEach((el, i) => el.style.background = col(i));
  tb.querySelectorAll('.sw').forEach((el, i) => el.style.background = col(i));
});
</script>
</body>
</html>
"""
html = html.replace("__SERIES__", json.dumps(series, separators=(",", ":")))
html = html.replace("__STATS__", json.dumps(stats, separators=(",", ":")))
out = ROOT / "nve_compare_dt5.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e3:.0f} kB")
