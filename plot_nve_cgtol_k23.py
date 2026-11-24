#!/usr/bin/env python3
"""cgtol スキャン (dt=0.5 fs, 4 ps) を 3点外挿 (k=2) と 4点外挿 (k=3) で比べる図を作る．

    python3 plot_nve_cgtol_k23.py   →  output_result/nve_compare_cgtol_k23.html

列 1 = T, 18 = conserved, 19 = CG 終了残差, 20 = 1 ステップの時間 (s), 21 = conserved+W_d,
22 = shell_cos, 末尾 = ncg．
"""
import json
import statistics as st
from pathlib import Path

HA_MEV = 27211.386245988
NATOM = 2560
KB3 = 3 * 0.08617333262  # meV/K (調和近似の 1 原子あたり熱容量)
TOLS = ["5e-6", "1e-6", "7e-7", "4e-7", "1e-7"]
RUNS = {2: "out_nve_0.5fs_cgtol{}", 3: "out_nve_0.5fs_k3_cgtol{}"}
DT = 0.5  # fs
ROOT = Path(__file__).parent / "output_result"


def slope(t, y):
    mt, my = st.fmean(t), st.fmean(y)
    return sum((a - mt) * (c - my) for a, c in zip(t, y)) / sum((a - mt) ** 2 for a in t)


stats, series = [], []
for k, pat in RUNS.items():
    for tol in TOLS:
        E, Ew, res, wall, cos, ncg = [], [], [], [], [], []
        with open(ROOT / pat.format(tol) / "log.txt") as f:
            for line in f:
                v = line.split()
                E.append(float(v[17])); res.append(float(v[18])); wall.append(float(v[19]))
                Ew.append(float(v[20])); cos.append(float(v[21])); ncg.append(int(v[-1]))
        t = [(i + 1) * DT * 1e-3 for i in range(len(E))]
        dE = [(e - E[0]) * HA_MEV / NATOM for e in E]
        s = slope(t, dE)
        sw = slope(t, [(e - Ew[0]) * HA_MEV / NATOM for e in Ew])
        stats.append({"k": k, "tol": tol, "slope": s, "slw": sw, "dT": s / KB3 * 1000,  # K/ns
                      "end": dE[-1], "cos": st.fmean(cos), "res": st.fmean(res),
                      "ncg": st.fmean(ncg), "wall": st.fmean(wall),
                      "wall_ps": st.fmean(wall) / (DT * 1e-3) / 3600})
        idx = list(range(0, len(t), 8)) + [len(t) - 1]
        series.append({"k": k, "tol": tol, "pts": [[round(t[i], 4), round(dE[i], 6)] for i in idx]})
        print(f"k={k} cgtol={tol}: slope={s:+.5f} ncg={st.fmean(ncg):.1f} wall={st.fmean(wall):.3f}")

html = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVE cgtol 3点・4点外挿比較</title>
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
.grid3{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}
.wrap{position:relative;height:280px}
.wrap.lg{height:380px}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;margin:0 0 8px}
.legend span{display:inline-flex;align-items:center;gap:6px}
.legend i{width:14px;height:3px;border-radius:2px;display:inline-block}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:0 0 8px}
.seg button{all:unset;cursor:pointer;padding:3px 12px;font-size:13px;color:var(--muted)}
.seg button[aria-pressed="true"]{background:var(--line);color:var(--fg)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
th{color:var(--muted);font-weight:500}
.tw{overflow-x:auto}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
</style>
</head>
<body>
<main>
<h1>NVE 保存量の cgtol 依存性：3点外挿 (k=2) と 4点外挿 (k=3)</h1>
<p class="sub">KNbO₃ 2,560 原子・150 K・同一初期配置（NPT step_39000）から dt=0.5 fs で 4 ps．傾きは保存量 ΔH の 4 ps 線形フィット．</p>

<div class="card">
<h2>cgtol を変えたときのドリフト・CG 反復・計算時間</h2>
<p>k=2 はドリフトが残差に比例して残り，締めるほど CG 反復と時間が増える．k=3 はどの cgtol でもドリフトがほぼ 0 で，締めるほど CG 反復と時間がむしろ減る．</p>
<div class="legend"><span><i style="background:var(--s1)"></i>k=2（3点外挿）</span><span><i style="background:var(--s2)"></i>k=3（4点外挿）</span></div>
<div class="grid3">
<div><div class="wrap"><canvas id="cA" aria-label="cgtol ごとのドリフト"></canvas></div></div>
<div><div class="wrap"><canvas id="cB" aria-label="cgtol ごとの平均 CG 反復"></canvas></div></div>
<div><div class="wrap"><canvas id="cC" aria-label="cgtol ごとの 1 ステップ時間"></canvas></div></div>
</div>
</div>

<div class="card">
<h2>ΔH(t)</h2>
<p>同じ縦軸で k=2 と k=3 を切り替えて表示．</p>
<div class="seg" id="seg"><button aria-pressed="true" data-k="2">k=2</button><button aria-pressed="false" data-k="3">k=3</button></div>
<div class="legend" id="lg"></div>
<div class="wrap lg"><canvas id="cD" aria-label="保存量の時間変化"></canvas></div>
</div>

<div class="card">
<h2>表</h2>
<p>加熱速度 = 傾き / 3k<sub>B</sub>（調和近似，エネルギーが運動と位置に等分される場合）．⟨cos⟩ = 残差 F<sub>s</sub> と Δd のなす角の平均（− で加熱）．</p>
<div class="tw"><table id="tb"><thead><tr>
<th>k</th><th>cgtol</th><th>傾き meV/atom/ps</th><th>傾き (+W_d)</th><th>加熱 K/ns</th><th>⟨cos⟩</th><th>⟨CG 残差⟩</th><th>平均 CG 反復</th><th>1 ステップ (s)</th><th>1 ps あたり (h)</th>
</tr></thead><tbody></tbody></table></div>
</div>

<div class="card">
<h2>k=3 で cgtol を締めるほど CG が速くなる理由</h2>
<p>k=3 の予測子は直近 4 点の収束解から外挿する（係数 4, −6, 4, −1）．各点には CG を打ち切ったときの誤差（大きさ ∝ cgtol）が残っていて，外挿でこれが √(16+36+16+1) ≈ 8.3 倍に増幅される．cgtol=1e-6 ではこの雑音が予測誤差の大半を占め，シェル運動の打ち切り誤差（θ<sup>4</sup> 項）より大きい．cgtol を締めると過去の解がきれいになり，予測が良くなるので，次のステップで CG が必要とする反復が減る．k=2 では予測誤差が打ち切り誤差（θ<sup>3</sup> 項）で決まり雑音の寄与が小さいため，この効果は見えず，締めた分だけ反復が増える．</p>
<p>またこの雑音はランダムな向きなので，k=3 の残差は Δd と相関しない（⟨cos⟩ ≈ 0）．このためドリフトがほぼ 0 になる．</p>
</div>
</main>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<script>
const STATS = __STATS__;
const SERIES = __SERIES__;
const TOLS = __TOLS__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const kcol = k => css(k === 2 ? '--s1' : '--s2');
const tcol = i => css('--s' + (i + 1));
const fmt = (x, d) => (x >= 0 ? '+' : '−') + Math.abs(x).toFixed(d);
const axis = t => ({title: {display: true, text: t, color: css('--muted')}, ticks: {color: css('--muted')},
                    grid: {color: css('--grid')}, border: {color: css('--line')}});
const base = () => ({responsive: true, maintainAspectRatio: false, animation: false,
                     interaction: {mode: 'index', intersect: false}, plugins: {legend: {display: false}}});

let charts = [], curK = 2;
function small(id, key, ylab, dec, min0) {
  const o = base();
  o.scales = {x: {...axis('cgtol'), type: 'category'}, y: {...axis(ylab), ...(min0 ? {min: 0} : {})}};
  o.plugins.tooltip = {callbacks: {title: it => `cgtol = ${it[0].label}`,
    label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(dec)}`}};
  return new Chart(document.getElementById(id), {type: 'line', options: o, data: {labels: TOLS,
    datasets: [2, 3].map(k => ({label: `k=${k}`, data: TOLS.map(t => STATS.find(s => s.k === k && s.tol === t)[key]),
      borderColor: kcol(k), backgroundColor: kcol(k), borderWidth: 2, pointRadius: 4, pointHoverRadius: 6}))}});
}
function draw() {
  charts.forEach(c => c.destroy());
  const o = base();
  o.parsing = false;
  o.interaction = {mode: 'nearest', axis: 'x', intersect: false};
  o.scales = {x: {...axis('時間 (ps)'), type: 'linear', min: 0, max: 4},
              y: {...axis('ΔH (meV/atom)'), min: -0.01, max: 0.06}};
  o.plugins.tooltip = {callbacks: {title: it => `t = ${it[0].parsed.x.toFixed(3)} ps`,
    label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 4)} meV/atom`}};
  const cD = new Chart(document.getElementById('cD'), {type: 'line', options: o,
    data: {datasets: SERIES.filter(s => s.k === curK).map((s, i) => ({label: `cgtol=${s.tol}`,
      data: s.pts.map(p => ({x: p[0], y: p[1]})), borderColor: tcol(i), backgroundColor: tcol(i),
      borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4}))}});
  charts = [small('cA', 'slope', 'ドリフト (meV/atom/ps)', 5, true),
            small('cB', 'ncg', '平均 CG 反復', 1, true),
            small('cC', 'wall', '1 ステップ (s)', 3, true), cD];
  document.getElementById('lg').innerHTML = TOLS.map((t, i) => `<span><i style="background:${tcol(i)}"></i>cgtol=${t}</span>`).join('');
  document.querySelectorAll('#tb .sw').forEach(el => el.style.background = kcol(+el.dataset.k));
}
document.querySelectorAll('#seg button').forEach(b => b.onclick = () => {
  curK = +b.dataset.k;
  document.querySelectorAll('#seg button').forEach(x => x.setAttribute('aria-pressed', x === b));
  draw();
});
const tb = document.querySelector('#tb tbody');
STATS.forEach(s => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="sw" data-k="${s.k}"></span>${s.k}</td><td>${s.tol}</td><td>${fmt(s.slope, 5)}</td><td>${fmt(s.slw, 5)}</td>
    <td>${fmt(s.dT, 1)}</td><td>${fmt(s.cos, 3)}</td><td>${s.res.toExponential(2)}</td><td>${s.ncg.toFixed(1)}</td>
    <td>${s.wall.toFixed(3)}</td><td>${s.wall_ps.toFixed(2)}</td>`;
  tb.appendChild(tr);
});
draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
</script>
</body>
</html>
"""
html = html.replace("__STATS__", json.dumps(stats, separators=(",", ":")))
html = html.replace("__SERIES__", json.dumps(series, separators=(",", ":")))
html = html.replace("__TOLS__", json.dumps(TOLS))
out = ROOT / "nve_compare_cgtol_k23.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e3:.0f} kB")
