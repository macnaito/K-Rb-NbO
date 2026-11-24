#!/usr/bin/env python3
"""cgtol = 5e-6..1e-7 の NVE 保存量を比べる図を作る (dt=0.5 fs, 4 ps, 3点外挿 k=2)．

    python3 plot_nve_cgtol.py   →  output_result/nve_compare_cgtol.html

列 18 = conserved, 19 = CG 終了残差, 21 = conserved+W_d, 22 = shell_cos, 末尾 = ncg．
"""
import json
import statistics as st
from pathlib import Path

HA_MEV = 27211.386245988
NATOM = 2560
ORDERS = ["5e-6", "1e-6", "7e-7", "4e-7", "1e-7"]
DT = 0.5  # fs
LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"]
ROOT = Path(__file__).parent / "output_result"


def linfit(t, y):
    mt, my = st.fmean(t), st.fmean(y)
    b = sum((a - mt) * (c - my) for a, c in zip(t, y)) / sum((a - mt) ** 2 for a in t)
    res = [c - (my + b * (a - mt)) for a, c in zip(t, y)]
    return b, st.pstdev(res)


series, stats = [], []
for k in ORDERS:
    E, Ew, T, ncg, wall, cos, res = [], [], [], [], [], [], []
    with open(ROOT / f"out_nve_0.5fs_cgtol{k}" / "log.txt") as f:
        for line in f:
            v = line.split()
            E.append(float(v[17])); T.append(float(v[0])); wall.append(float(v[19]))
            Ew.append(float(v[20])); cos.append(float(v[21])); ncg.append(int(v[-1])); res.append(float(v[18]))
    t = [(i + 1) * DT * 1e-3 for i in range(len(E))]
    byn = {}
    for c, n in zip(cos, ncg):
        byn.setdefault(n, []).append(c)
    cosn = [[n, round(st.fmean(byn[n]), 3), len(byn[n])] for n in sorted(byn) if len(byn[n]) > 100]
    dE = [(e - E[0]) * HA_MEV / NATOM for e in E]
    dEw = [(e - Ew[0]) * HA_MEV / NATOM for e in Ew]
    slope, sd = linfit(t, dE)
    slw, _ = linfit(t, dEw)
    idx = list(range(0, len(t), 4)) + ([len(t) - 1] if (len(t) - 1) % 4 else [])
    series.append({"dt": str(k), "pts": [[round(t[i], 4), round(dE[i], 6)] for i in idx],
                   "ptsw": [[round(t[i], 4), round(dEw[i], 6)] for i in idx]})
    stats.append({"dt": str(k), "dtv": -float(k), "res": st.fmean(res), "rc": st.fmean([a * b for a, b in zip(res, cos)]),
                  "cosn": cosn, "nsteps": len(E), "end": dE[-1], "slope": slope, "sd": sd,
                  "slw": slw, "cos": st.fmean(cos), "ncg": st.fmean(ncg), "wall": st.fmean(wall),
                  "wall_ps": st.fmean(wall) / (DT * 1e-3) / 3600})
    print(f"cgtol={k}: res={st.fmean(res):.2e} ncg={st.fmean(ncg):.1f} wall={st.fmean(wall):.3f} steps={len(E)} slope={slope:+.4f} slope(+Wd)={slw:+.4f} cos={st.fmean(cos):+.3f}")

html = """<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NVE cgtol 比較</title>
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
<h1>NVE 保存量の cgtol 依存性</h1>
<p class="sub">KNbO₃ 2,560 原子・150 K・同一初期配置（NPT step_39000）から dt=0.5 fs・3点外挿 (k=2) で 4 ps．縦軸は保存量の初期値からの変化 ΔH（meV/atom）．</p>

<div class="card">
<h2>ΔH(t)</h2>
<p>凡例をクリックで表示を切り替え．「W_d を足す」をオンにすると，シェル残差がした仕事 W_d = Σ F_s·Δd を足し戻した保存量を表示（ドリフトが消えれば，漏れが残差の仕事で説明しきれている）．</p>\n<label style="font-size:13px;color:var(--muted);display:inline-flex;gap:6px;align-items:center;margin:0 0 8px;cursor:pointer"><input type="checkbox" id="wd">W_d を足す</label>
<div class="legend" id="lg"></div>
<div class="wrap"><canvas id="c1" aria-label="保存量の時間変化"></canvas></div>
</div>

<div class="card">
<h2>ドリフトとシェル残差の向き</h2>
<p>傾き = 4 ps の線形フィット．⟨cos⟩ = 残差 F_s とシェル変位の変化 Δd のなす角の平均（+ で冷却，− で加熱）．</p>
<div class="wrap sm"><canvas id="c2" aria-label="dt ごとのドリフトと揺らぎ"></canvas></div>
<div class="tw" style="margin-top:12px"><table id="tb"><thead><tr>
<th>cgtol</th><th>ΔH(4 ps) meV/atom</th><th>傾き meV/atom/ps</th><th>傾き (+W_d)</th><th>⟨cos⟩</th><th>⟨CG 残差⟩</th><th>平均 CG 反復</th><th>1 ステップ (s)</th><th>1 ps あたり (h)</th>
</tr></thead><tbody></tbody></table></div>
</div>
<div class="card">
<h2>ドリフトは ⟨res·cos⟩ に比例する</h2>
<p>1 ステップで漏れるエネルギーは −F<sub>s</sub>·Δd = −|F<sub>s</sub>||Δd|·cos．F<sub>s</sub> は CG を打ち切ったときにシェルに残る残差力，Δd はそのステップのシェル変位．dt が同じなら |Δd| はどの cgtol でも同じなので，ドリフトは ⟨res·cos⟩ に比例するはず．右端の比がほぼ一定なのがその確認．</p>
<div class="tw"><table id="tb2"><thead><tr>
<th>cgtol</th><th>⟨res⟩</th><th>⟨cos⟩</th><th>⟨res·cos⟩</th><th>傾き meV/atom/ps</th><th>傾き / ⟨res·cos⟩</th>
</tr></thead><tbody></tbody></table></div>
<p style="margin-top:12px">7e-7 と 4e-7 の傾きが同じなのは偶然．4e-7 は残差が 1/1.72 に小さいが，|cos| が 1.66 倍大きく（−0.40 対 −0.24），⟨res·cos⟩ がほぼ同じ値になる．cos は 1 ps ごとに区切っても一定で，ノイズではなく系統的な値．</p>
<p>cos が cgtol に対して単調に変わらないのは，CG を何反復目で打ち切るかで決まるため．CG の残差は反復ごとにそれまでと直交するので，残差の向き（Δd とのなす角）は反復ごとに変わる．下の表は打ち切り反復数 ncg ごとの ⟨cos⟩（100 ステップ以上ある ncg のみ，括弧内はステップ数）．7e-7 はおもに ncg 9–11（cos ≈ −0.2），4e-7 は ncg 13–14（cos ≈ −0.42）で止まっている．</p>
<div class="tw"><table id="tb3"><thead><tr><th>cgtol</th><th style="text-align:left">ncg: ⟨cos⟩ (ステップ数)</th></tr></thead><tbody></tbody></table></div>
<p style="margin-top:12px">残差は cgtol とともに単調に小さくなるので，ドリフトも全体としては下がる．ただ cos が上下するぶん cgtol 依存はギザギザになり，隣の設定と値が並ぶこともある．k=2 のまま cgtol でエネルギー保存を合わせ込むより，cos ≈ 0 になる外挿次数（k=3, 4）で偏りそのものを消すほうが確実．</p>
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
    label: c => `cgtol=${SERIES[c.datasetIndex].dt}: ${fmt(c.parsed.y, 4)} meV/atom`}};
  o1.scales = {x: {...axis('時間 (ps)'), type: 'linear', min: 0, max: 4},
               y: axis('ΔH (meV/atom)')};
  c1 = new Chart(document.getElementById('c1'), {type: 'line', options: o1,
    data: {datasets: SERIES.map((s, i) => ({label: `cgtol=${s.dt}`,
      data: (useWd ? s.ptsw : s.pts).map(p => ({x: p[0], y: p[1]})), borderColor: col(i), backgroundColor: col(i),
      borderWidth: 1.5, pointRadius: 0, pointHoverRadius: 4, hidden: !vis[i]}))}});

  const o2 = base();
  o2.parsing = true;
  o2.interaction = {mode: 'index', intersect: false};
  o2.plugins.legend = {display: true, labels: {color: css('--fg'), boxWidth: 14, boxHeight: 3}};
  o2.plugins.tooltip = {callbacks: {title: it => `cgtol = ${it[0].label}`,
    label: c => `${c.dataset.label}: ${fmt(c.parsed.y, 4)}`}};
  o2.scales = {x: {...axis('cgtol'), type: 'category'}, y: axis('ドリフト (meV/atom/ps)')};
  const asc = [...STATS].sort((a, b) => a.dtv - b.dtv);
  c2 = new Chart(document.getElementById('c2'), {type: 'line', options: o2,
    data: {labels: asc.map(s => s.dt),
      datasets: [
        {label: 'ΔH の傾き', data: asc.map(s => s.slope), borderColor: css('--fg'),
         backgroundColor: css('--fg'), borderWidth: 2, pointRadius: 4},
        {label: '+W_d', data: asc.map(s => s.slw), borderColor: css('--muted'),
         backgroundColor: css('--muted'), borderWidth: 2, borderDash: [5, 4], pointRadius: 4}]}});
}

const vis = SERIES.map(() => true);
let useWd = false;
document.getElementById('wd').onchange = e => { useWd = e.target.checked; draw(); };
const lg = document.getElementById('lg');
SERIES.forEach((s, i) => {
  const b = document.createElement('button');
  b.setAttribute('aria-pressed', 'true');
  b.innerHTML = `<i style="background:${col(i)}"></i>cgtol=${s.dt}`;
  b.onclick = () => { vis[i] = !vis[i]; b.setAttribute('aria-pressed', vis[i]);
    c1.setDatasetVisibility(i, vis[i]); c1.update(); };
  lg.appendChild(b);
});

const tb = document.querySelector('#tb tbody');
STATS.forEach((s, i) => {
  const tr = document.createElement('tr');
  tr.innerHTML = `<td><span class="sw" style="background:${col(i)}"></span>${s.dt}</td><td>${fmt(s.end, 4)}</td><td>${fmt(s.slope, 4)}</td><td>${fmt(s.slw, 4)}</td><td>${fmt(s.cos, 3)}</td><td>${s.res.toExponential(2)}</td>
    <td>${s.ncg.toFixed(1)}</td><td>${s.wall.toFixed(3)}</td><td>${s.wall_ps.toFixed(2)}</td>`;
  tb.appendChild(tr);
});

const tb2 = document.querySelector('#tb2 tbody');
const tb3 = document.querySelector('#tb3 tbody');
STATS.forEach((s, i) => {
  const hl = (s.dt === '7e-7' || s.dt === '4e-7') ? ' style="font-weight:600"' : '';
  const tr = document.createElement('tr');
  tr.innerHTML = `<td${hl}><span class="sw" style="background:${col(i)}"></span>${s.dt}</td><td${hl}>${s.res.toExponential(2)}</td><td${hl}>${fmt(s.cos, 3)}</td>
    <td${hl}>${s.rc.toExponential(2).replace(/^-/, "−")}</td><td${hl}>${fmt(s.slope, 4)}</td><td${hl}>${(s.slope / -s.rc).toExponential(2)}</td>`;
  tb2.appendChild(tr);
  const tr3 = document.createElement('tr');
  tr3.innerHTML = `<td><span class="sw" style="background:${col(i)}"></span>${s.dt}</td><td style="text-align:left;white-space:normal">${s.cosn.map(c => `${c[0]}: ${fmt(c[1], 2)} (${c[2]})`).join('　')}</td>`;
  tb3.appendChild(tr3);
});

draw();
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
  draw();
  lg.querySelectorAll('i').forEach((el, i) => el.style.background = col(i));
  [tb, tb2, tb3].forEach(t => t.querySelectorAll('.sw').forEach((el, i) => el.style.background = col(i)));
});
</script>
</body>
</html>
"""
html = html.replace("__SERIES__", json.dumps(series, separators=(",", ":")))
html = html.replace("__STATS__", json.dumps(stats, separators=(",", ":")))
out = ROOT / "nve_compare_cgtol.html"
out.write_text(html, encoding="utf-8")
print("wrote", out, f"{out.stat().st_size/1e3:.0f} kB")
