#!/usr/bin/env python3
"""複数の out*/log.txt の保存量の変化 ΔH′ を 1 枚のグラフに重ねる（標準ライブラリだけで動く）．

使い方（KNO フォルダで）:
    python3 plot_conserved.py                         # out-*_150K_dt=* を全部 → conserved_compare.html
    python3 plot_conserved.py outA outB ...           # フォルダを指定

時間刻みが違う計算を比べるので，横軸はステップ数ではなく時間（ps）．
刻み幅はフォルダ名の "dt=<数値>fs" から読む．
"""
import json
import math
import re
import sys
from pathlib import Path

from plot_log import HARTREE_EV, natoms, read_log

MAXPTS = 2000


def dt_fs(folder):
    m = re.search(r"dt=([0-9.]+)fs", folder.name)
    if not m:
        sys.exit(f"{folder}: フォルダ名から dt が読めません（'dt=<数値>fs' を含めてください）")
    return float(m.group(1))


def main():
    folders = [Path(a) for a in sys.argv[1:]] or sorted(Path(".").glob("out-*_150K_dt=*fs"))
    folders.sort(key=dt_fs, reverse=True)
    series, stats = [], []
    for d in folders:
        rows = read_log(d / "log.txt")
        snaps = sorted(d.glob("step_*.xyz"))
        n = natoms(snaps[0]) if snaps else None
        if not n:
            sys.exit(f"{d}: step_*.xyz から原子数が読めません")
        dt = dt_fs(d)
        h0 = rows[0][17]
        scale = HARTREE_EV * 1000 / n
        dh = [(r[17] - h0) * scale for r in rows]
        t = [(k + 1) * dt / 1000 for k in range(len(rows))]  # ps
        # ドリフト: 直線フィットの傾き（meV/原子/ps），ゆらぎ: フィットからの RMS
        tm, ym = sum(t) / len(t), sum(dh) / len(dh)
        slope = sum((a - tm) * (b - ym) for a, b in zip(t, dh)) / sum((a - tm) ** 2 for a in t)
        rms = math.sqrt(sum((b - ym - slope * (a - tm)) ** 2 for a, b in zip(t, dh)) / len(t))
        stats.append(dict(name=f"dt = {dt:g} fs", nsteps=len(rows), final=dh[-1], slope=slope, rms=rms,
                          maxabs=max(abs(v) for v in dh)))
        stride = max(1, math.ceil(len(rows) / MAXPTS))
        keep = sorted(set([0] + list(range(stride - 1, len(rows), stride))))
        series.append(dict(name=f"dt = {dt:g} fs", x=[float(f"{t[k]:.7g}") for k in keep],
                           y=[float(f"{dh[k]:.7g}") for k in keep]))
        print(f"{d}: {len(rows)} steps, dt={dt:g} fs, {n} atoms")
    out = Path("conserved_compare.html")
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(dict(series=series, stats=stats))), encoding="utf-8")
    print(f"wrote {out}")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>保存量の比較</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3ef; --surface-1: #fcfcfb; --grid: #e6e5df; --axis: #b9b8b0;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #8a5cd1;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --grid: #2c2c2a; --axis: #4a4a46;
    --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #9b72dc;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --grid: #2c2c2a; --axis: #4a4a46;
  --text-primary: #ffffff; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #9b72dc;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif;
}
main { max-width: 980px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.card { background: var(--surface-1); border-radius: 12px; padding: 14px 14px 8px; position: relative; }
.card h2 { font-size: 14px; margin: 0; font-weight: 600; }
.card .unit { color: var(--text-muted); font-weight: 400; margin-left: 6px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; margin: 4px 0 6px; color: var(--text-secondary); font-size: 12px; }
.legend label { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; user-select: none; }
.legend input { margin: 0; }
.legend i { width: 14px; height: 2px; border-radius: 1px; display: inline-block; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 11px; }
.tip {
  position: absolute; pointer-events: none; background: var(--surface-0); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 8px; padding: 6px 9px; font-size: 12px;
  white-space: nowrap; box-shadow: 0 2px 8px rgb(0 0 0 / .12); display: none; z-index: 2;
}
.tip b { font-weight: 600; }
.tip .row { display: flex; align-items: center; gap: 6px; }
.tip i { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
table { border-collapse: collapse; font-size: 13px; margin-top: 20px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 5px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th:first-child, td:first-child { text-align: left; }
th { color: var(--text-secondary); font-weight: 600; }
.tablewrap { overflow-x: auto; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<main>
  <h1>保存量の変化 ΔH′ の比較</h1>
  <p class="sub">時間刻み dt を変えた NPT-MD（150 K）。ΔH′ = H′(t) − H′(最初のステップ)、1 原子あたり。</p>
  <section class="card">
    <h2>ΔH′<span class="unit">meV/原子 ・ 横軸 時間 (ps)</span></h2>
    <div class="legend" id="legend"></div>
    <div id="plot"></div>
    <div class="tip" id="tip"></div>
  </section>
  <div class="tablewrap"><table id="table"></table></div>
  <p class="note">ドリフト = ΔH′ を時間で直線フィットした傾き。ゆらぎ = その直線からの RMS。</p>
</main>
<script>
const DATA = __DATA__;
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];
const W = 760, H = 360, M = { l: 60, r: 16, t: 12, b: 30 };
const YMAX = 0.4;  // y 軸の上限（meV/原子）
const on = DATA.series.map(() => true);

function niceTicks(lo, hi, n) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo, raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const stepv = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => span / s <= n) || 10 * mag;
  const t0 = Math.floor(lo / stepv) * stepv, ticks = [];
  for (let t = t0; t <= hi + stepv * 1e-9; t += stepv) ticks.push(+t.toPrecision(12));
  if (ticks[0] > lo) ticks.unshift(+(ticks[0] - stepv).toPrecision(12));
  if (ticks[ticks.length - 1] < hi) ticks.push(+(ticks[ticks.length - 1] + stepv).toPrecision(12));
  return { ticks, lo: ticks[0], hi: ticks[ticks.length - 1] };
}
function tickLabel(v, ticks) {
  const d = ticks.length > 1 ? Math.abs(ticks[1] - ticks[0]) : 1;
  const dec = Math.max(0, -Math.floor(Math.log10(d)) + (d / Math.pow(10, Math.floor(Math.log10(d))) % 1 ? 1 : 0));
  return v.toFixed(Math.min(dec, 6));
}
const el = (tag, attrs) => {
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
};
const fmt = v => Math.abs(v) >= 10 ? v.toFixed(2) : v.toPrecision(3);

const legend = document.getElementById("legend");
DATA.series.forEach((s, i) => {
  const lab = document.createElement("label");
  lab.innerHTML = `<input type="checkbox" checked><i style="background:${COLORS[i]}"></i>${s.name}`;
  lab.querySelector("input").addEventListener("change", e => { on[i] = e.target.checked; draw(); });
  legend.appendChild(lab);
});

const plot = document.getElementById("plot"), tip = document.getElementById("tip"), card = plot.parentElement;
function draw() {
  plot.innerHTML = "";
  const vis = DATA.series.map((s, i) => [s, i]).filter(([, i]) => on[i]);
  const xmax = Math.max(...DATA.series.map(s => s.x[s.x.length - 1]));
  const ally = vis.flatMap(([s]) => s.y).concat([0]);
  // y 軸の上限は YMAX に固定（はみ出した線は切る）
  const lo = Math.min(...ally), pad = (YMAX - lo) * 0.06;
  const yt = niceTicks(lo - pad, YMAX, 6), xt = niceTicks(0, xmax, 8);
  yt.ticks = yt.ticks.filter(t => t <= YMAX); yt.hi = YMAX;
  const xs = v => M.l + v / xmax * (W - M.l - M.r);
  const ys = v => M.t + (yt.hi - v) / (yt.hi - yt.lo) * (H - M.t - M.b);
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "dt ごとの保存量の変化" });
  for (const t of yt.ticks) {
    svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: ys(t), y2: ys(t), stroke: t === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }));
    const tx = el("text", { x: M.l - 6, y: ys(t) + 4, "text-anchor": "end" });
    tx.textContent = tickLabel(t, yt.ticks); svg.appendChild(tx);
  }
  for (const t of xt.ticks) {
    if (t < 0 || t > xmax) continue;
    const tx = el("text", { x: xs(t), y: H - 10, "text-anchor": "middle" });
    tx.textContent = tickLabel(t, xt.ticks); svg.appendChild(tx);
  }
  svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1 }));
  const clip = el("clipPath", { id: "plotarea" });
  clip.appendChild(el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }));
  svg.appendChild(clip);
  for (const [s, i] of vis) {
    const d = s.x.map((x, k) => `${k ? "L" : "M"}${xs(x).toFixed(2)},${ys(s.y[k]).toFixed(2)}`).join("");
    svg.appendChild(el("path", { d, fill: "none", stroke: COLORS[i], "stroke-width": 1.6, "stroke-linejoin": "round",
      "clip-path": "url(#plotarea)" }));
  }
  const cross = el("line", { y1: M.t, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  svg.appendChild(cross);
  const dots = vis.map(([, i]) => { const c = el("circle", { r: 4, fill: COLORS[i], stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }); svg.appendChild(c); return c; });
  const hit = el("rect", { x: M.l, y: 0, width: W - M.l - M.r, height: H, fill: "transparent" });
  svg.appendChild(hit);
  plot.appendChild(svg);
  hit.addEventListener("pointermove", evt => {
    const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const tx = Math.max(0, Math.min(xmax, (p.x - M.l) / (W - M.l - M.r) * xmax));
    const x = xs(tx);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    let html = `<b>t = ${tx.toFixed(3)} ps</b>`;
    vis.forEach(([s, i], j) => {
      let k = s.x.findIndex(v => v >= tx); if (k < 0) k = s.x.length - 1;
      dots[j].setAttribute("cx", xs(s.x[k])); dots[j].setAttribute("cy", ys(s.y[k])); dots[j].setAttribute("visibility", "visible");
      html += `<div class="row"><i style="background:${COLORS[i]}"></i>${s.name}: ${fmt(s.y[k])} meV/原子</div>`;
    });
    tip.innerHTML = html; tip.style.display = "block";
    const box = card.getBoundingClientRect(), sb = svg.getBoundingClientRect();
    const px = sb.left - box.left + x / W * sb.width;
    tip.style.left = `${px + 12 + tip.offsetWidth > box.width ? px - 12 - tip.offsetWidth : px + 12}px`;
    tip.style.top = `${sb.top - box.top + 8}px`;
  });
  hit.addEventListener("pointerleave", () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); dots.forEach(c => c.setAttribute("visibility", "hidden")); });
}
draw();

document.getElementById("table").innerHTML =
  "<thead><tr><th>dt</th><th>ステップ数</th><th>最終 ΔH′ (meV/原子)</th><th>最大 |ΔH′| (meV/原子)</th>" +
  "<th>ドリフト (meV/原子/ps)</th><th>ゆらぎ RMS (meV/原子)</th></tr></thead><tbody>" +
  DATA.stats.map((s, i) => `<tr><td><i style="display:inline-block;width:8px;height:8px;border-radius:50%;background:${COLORS[i]};margin-right:6px"></i>${s.name}</td>` +
    `<td>${s.nsteps}</td><td>${fmt(s.final)}</td><td>${fmt(s.maxabs)}</td><td>${fmt(s.slope)}</td><td>${fmt(s.rms)}</td></tr>`).join("") +
  "</tbody>";
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
