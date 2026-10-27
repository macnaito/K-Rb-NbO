#!/usr/bin/env python3
"""FMD の NPT-MD 出力 out/log.txt をグラフにする（標準ライブラリだけで動く）．

使い方（KNO フォルダで）:
    python3 plot_log.py                 # out/log.txt → out/log_plot.html
    python3 plot_log.py out2/log.txt    # 別の出力フォルダ

log.txt の列（FMD.jl/src/Apps/NPT.jl が毎ステップ 1 行書く）:
    temp(K) | lattice 9 成分(Å, 列優先) | dipole 3 (e·Å) | dipole_phys 3 (e·Å)
    | volume(Å³) | conserved(Hartree) | ncg
"""
import json
import math
import sys
from pathlib import Path

HARTREE_EV = 27.211386245988
E_PER_A2_TO_UC_PER_CM2 = 1602.176634  # 1 e/Å² = 1602.18 μC/cm²
ATOMS_PER_CELL = 5  # ペロブスカイト ABO3 の単位格子あたりの原子数


def read_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) != 19:
                continue
            rows.append([float(x) for x in v[:18]] + [int(v[18])])
    if not rows:
        sys.exit(f"{path}: データ行がありません")
    return rows


def natoms(xyz_path):
    try:
        with open(xyz_path) as f:
            return int(f.readline())
    except (OSError, ValueError):
        return None


def ncell_from_natoms(n):
    """原子数から立方スーパーセルの各方向の単位格子数を求める（2560 原子 → 8）．"""
    if not n or n % ATOMS_PER_CELL:
        return None
    k = round((n // ATOMS_PER_CELL) ** (1 / 3))
    return k if k ** 3 * ATOMS_PER_CELL == n else None


def main():
    log = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out/log.txt")
    rows = read_log(log)
    # 原子数は出力スナップショットから取る（保存量を 1 原子あたりにするため）
    snaps = sorted(log.parent.glob("step_*.xyz"))
    n = natoms(snaps[0]) if snaps else None
    # 格子長を単位格子あたりにするための各方向のセル数（求まらなければ割らない）
    ncell = ncell_from_natoms(n)
    if ncell is None:
        print(f"warning: 原子数 {n} から立方スーパーセルのセル数が決まらないので，格子長はセル全体の値のまま",
              file=sys.stderr)

    step = list(range(1, len(rows) + 1))
    temp = [r[0] for r in rows]
    # 格子は列優先で 9 個: L11 L21 L31 L12 L22 L32 L13 L23 L33．
    # FMD の lattice は「行 = 格子ベクトル」なので a1 = (L11, L12, L13)
    def length(r, i):
        return math.sqrt(r[1 + i] ** 2 + r[4 + i] ** 2 + r[7 + i] ** 2) / (ncell or 1)
    a = [length(r, 0) for r in rows]
    b = [length(r, 1) for r in rows]
    c = [length(r, 2) for r in rows]
    vol = [r[16] for r in rows]
    # 分極 P = M_phys / V（ゲージ補正済みの双極子を使う）
    px = [r[13] / r[16] * E_PER_A2_TO_UC_PER_CM2 for r in rows]
    py = [r[14] / r[16] * E_PER_A2_TO_UC_PER_CM2 for r in rows]
    pz = [r[15] / r[16] * E_PER_A2_TO_UC_PER_CM2 for r in rows]
    h0 = rows[0][17]
    scale = HARTREE_EV * 1000 / n if n else HARTREE_EV
    dh = [(r[17] - h0) * scale for r in rows]
    ncg = [r[18] for r in rows]

    charts = [
        dict(id="temp", title="温度", unit="K", series=[dict(name="温度", y=temp)],
             ref=dict(y=None, label="")),
        dict(id="lat", title="格子長（単位格子あたり）" if ncell else "格子長", unit="Å",
             series=[dict(name="a", y=a), dict(name="b", y=b), dict(name="c", y=c)]),
        dict(id="vol", title="体積", unit="Å³", series=[dict(name="体積", y=vol)]),
        dict(id="pol", title="分極（dipole_phys / V）", unit="μC/cm²",
             series=[dict(name="Px", y=px), dict(name="Py", y=py), dict(name="Pz", y=pz)],
             ref=dict(y=0, label="0")),
        dict(id="cons", title="保存量の変化 ΔH′",
             unit="meV/原子" if n else "eV", series=[dict(name="ΔH′", y=dh)],
             ref=dict(y=0, label="0")),
        dict(id="ncg", title="シェル緩和の反復回数", unit="回",
             series=[dict(name="ncg", y=ncg)], integer=True),
    ]
    # 長い計算では HTML が重くなるので，最大 MAXPTS 点に間引く（等間隔）．
    # 有効数字も 7 桁に丸める（グラフと表には十分）
    MAXPTS = 2000
    stride = max(1, math.ceil(len(rows) / MAXPTS))
    keep = list(range(stride - 1, len(rows), stride))
    if keep[0] != 0:
        keep.insert(0, 0)
    step = [step[k] for k in keep]
    for ch in charts:
        for se in ch["series"]:
            se["y"] = [se["y"][k] if isinstance(se["y"][k], int) else float(f"{se['y'][k]:.7g}")
                       for k in keep]
    meta = dict(source=str(log), nsteps=len(rows), natoms=n, ncell=ncell, stride=stride)
    html = TEMPLATE.replace("__DATA__", json.dumps(dict(step=step, charts=charts, meta=meta)))
    out = log.parent / "log_plot.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({len(rows)} steps)")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MD ログ</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3ef;
  --surface-1: #fcfcfb;
  --grid: #e6e5df;
  --axis: #b9b8b0;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #6f6e69;
  --series-1: #2a78d6;
  --series-2: #eb6834;
  --series-3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0: #121211;
    --surface-1: #1a1a19;
    --grid: #2c2c2a;
    --axis: #4a4a46;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #9a998f;
    --series-1: #3987e5;
    --series-2: #d95926;
    --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211;
  --surface-1: #1a1a19;
  --grid: #2c2c2a;
  --axis: #4a4a46;
  --text-primary: #ffffff;
  --text-secondary: #c3c2b7;
  --text-muted: #9a998f;
  --series-1: #3987e5;
  --series-2: #d95926;
  --series-3: #199e70;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 24px 16px 40px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 20px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(340px, 100%), 1fr)); gap: 16px; }
.card { background: var(--surface-1); border-radius: 12px; padding: 14px 14px 8px; position: relative; }
.card h2 { font-size: 14px; margin: 0; font-weight: 600; }
.card .unit { color: var(--text-muted); font-weight: 400; margin-left: 6px; }
.legend { display: flex; gap: 14px; margin: 4px 0 0; color: var(--text-secondary); font-size: 12px; min-height: 18px; }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
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
details { margin-top: 24px; }
summary { cursor: pointer; color: var(--text-secondary); }
table { border-collapse: collapse; font-size: 12px; margin-top: 8px; font-variant-numeric: tabular-nums; }
th, td { padding: 3px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; position: sticky; top: 0; background: var(--surface-0); }
.tablewrap { max-height: 360px; overflow: auto; }
</style>
</head>
<body>
<main>
  <h1>MD ログ</h1>
  <p class="sub" id="sub"></p>
  <div class="grid" id="grid"></div>
  <details>
    <summary>数値の表を表示</summary>
    <div class="tablewrap"><table id="table"></table></div>
  </details>
</main>
<script>
const DATA = __DATA__;
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)"];
const W = 360, H = 200, M = { l: 52, r: 30, t: 12, b: 26 };
const step = DATA.step;
const fmt = (v, integer) => integer ? String(v) :
  (Math.abs(v) >= 1000 ? v.toFixed(1) : Math.abs(v) >= 10 ? v.toFixed(3) : v.toPrecision(4));

document.getElementById("sub").textContent =
  `${DATA.meta.source}・${DATA.meta.nsteps} ステップ` +
  (DATA.meta.natoms ? `・${DATA.meta.natoms} 原子` : "") +
  (DATA.meta.ncell ? `（${DATA.meta.ncell}×${DATA.meta.ncell}×${DATA.meta.ncell} セル）` : "") +
  (DATA.meta.stride > 1 ? `・${DATA.meta.stride} ステップごとに表示` : "");

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

for (const ch of DATA.charts) {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML = `<h2>${ch.title}<span class="unit">${ch.unit}</span></h2>`;
  const legend = document.createElement("div");
  legend.className = "legend";
  if (ch.series.length > 1)
    ch.series.forEach((s, i) => legend.insertAdjacentHTML("beforeend",
      `<span><i style="background:${COLORS[i]}"></i>${s.name}</span>`));
  card.appendChild(legend);

  const all = ch.series.flatMap(s => s.y).concat(ch.ref && ch.ref.y != null ? [ch.ref.y] : []);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = (hi - lo) * 0.06 || Math.abs(hi) * 0.01 || 1;
  const yt = niceTicks(lo - pad, hi + pad, 4);
  const xt = niceTicks(step[0], step[step.length - 1], 5);
  const xs = v => M.l + (v - step[0]) / (step[step.length - 1] - step[0] || 1) * (W - M.l - M.r);
  const ys = v => M.t + (yt.hi - v) / (yt.hi - yt.lo) * (H - M.t - M.b);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": `${ch.title}（${ch.unit}）の時間変化` });
  for (const t of yt.ticks) {
    svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: ys(t), y2: ys(t), stroke: "var(--grid)", "stroke-width": 1 }));
    const tx = el("text", { x: M.l - 6, y: ys(t) + 4, "text-anchor": "end" });
    tx.textContent = tickLabel(t, yt.ticks); svg.appendChild(tx);
  }
  for (const t of xt.ticks) {
    if (t < step[0] || t > step[step.length - 1]) continue;
    const tx = el("text", { x: xs(t), y: H - 8, "text-anchor": "middle" });
    tx.textContent = t; svg.appendChild(tx);
  }
  svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1 }));
  const xl = el("text", { x: W - M.r, y: H - 8, "text-anchor": "end" });
  if (ch.ref && ch.ref.y != null && ch.ref.y >= yt.lo && ch.ref.y <= yt.hi)
    svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: ys(ch.ref.y), y2: ys(ch.ref.y),
      stroke: "var(--axis)", "stroke-width": 1 }));

  ch.series.forEach((s, i) => {
    // 点が多いときは横 1px ごとに最小・最大だけ残して描く（形は変わらない）
    const bins = Math.ceil(W - M.l - M.r) * 2, per = Math.max(1, Math.floor(s.y.length / bins));
    const idx = [];
    for (let k0 = 0; k0 < s.y.length; k0 += per) {
      let lo = k0, hi = k0;
      for (let k = k0; k < Math.min(k0 + per, s.y.length); k++) { if (s.y[k] < s.y[lo]) lo = k; if (s.y[k] > s.y[hi]) hi = k; }
      idx.push(...(lo < hi ? [lo, hi] : lo > hi ? [hi, lo] : [lo]));
    }
    if (idx[idx.length - 1] !== s.y.length - 1) idx.push(s.y.length - 1);
    const d = idx.map((k, j) => `${j ? "L" : "M"}${xs(step[k]).toFixed(2)},${ys(s.y[k]).toFixed(2)}`).join("");
    svg.appendChild(el("path", { d, fill: "none", stroke: COLORS[i], "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = s.y.length - 1;
    svg.appendChild(el("circle", { cx: xs(step[last]), cy: ys(s.y[last]), r: 4, fill: COLORS[i],
      stroke: "var(--surface-1)", "stroke-width": 2 }));
  });
  // 直線フィット（任意）: ch.fits = [{c: 色番号, x0, y0, x1, y1}] を同じ色の破線で描く
  for (const f of ch.fits || [])
    svg.appendChild(el("line", { x1: xs(f.x0), y1: ys(f.y0), x2: xs(f.x1), y2: ys(f.y1),
      stroke: COLORS[f.c], "stroke-width": 1.5, "stroke-dasharray": "6 4" }));
  // 系列が複数あるときは右端に直接ラベル（重なるときはずらす）
  if (ch.series.length > 1) {
    const ends = ch.series.map((s, i) => ({ i, y: ys(s.y[s.y.length - 1]) })).sort((p, q) => p.y - q.y);
    for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 12) ends[k].y = ends[k - 1].y + 12;
    for (const e of ends) {
      const tx = el("text", { x: W - M.r + 8, y: e.y + 4 });
      tx.style.fill = "var(--text-secondary)";
      tx.textContent = ch.series[e.i].name; svg.appendChild(tx);
    }
  }

  // ホバー: 十字線 + ツールチップ
  const cross = el("line", { y1: M.t, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  svg.appendChild(cross);
  const dots = ch.series.map((s, i) => {
    const c = el("circle", { r: 4, fill: COLORS[i], stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" });
    svg.appendChild(c); return c;
  });
  const hit = el("rect", { x: M.l, y: 0, width: W - M.l - M.r, height: H, fill: "transparent" });
  svg.appendChild(hit);
  const tip = document.createElement("div");
  tip.className = "tip";
  card.appendChild(svg); card.appendChild(tip);

  const show = evt => {
    const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
    const p = pt.matrixTransform(svg.getScreenCTM().inverse());
    const frac = (p.x - M.l) / (W - M.l - M.r);
    const k = Math.max(0, Math.min(step.length - 1, Math.round(frac * (step.length - 1))));
    const x = xs(step[k]);
    cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    dots.forEach((c, i) => { c.setAttribute("cx", x); c.setAttribute("cy", ys(ch.series[i].y[k])); c.setAttribute("visibility", "visible"); });
    tip.innerHTML = `<b>ステップ ${step[k]}</b>` + ch.series.map((s, i) =>
      `<div class="row">${ch.series.length > 1 ? `<i style="background:${COLORS[i]}"></i>${s.name} ` : ""}${fmt(s.y[k], ch.integer)} ${ch.unit}</div>`).join("");
    tip.style.display = "block";
    const box = card.getBoundingClientRect(), sb = svg.getBoundingClientRect();
    const px = sb.left - box.left + x / W * sb.width;
    const left = px + 12 + tip.offsetWidth > box.width ? px - 12 - tip.offsetWidth : px + 12;
    tip.style.left = `${left}px`; tip.style.top = `${sb.top - box.top + 8}px`;
  };
  const hide = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); dots.forEach(c => c.setAttribute("visibility", "hidden")); };
  hit.addEventListener("pointermove", show);
  hit.addEventListener("pointerleave", hide);
  document.getElementById("grid").appendChild(card);
}

// 表（行が多いときは間引いて最大 1000 行）
const every = Math.ceil(step.length / 1000);
const rowsk = step.map((_, k) => k).filter(k => k % every === every - 1 || k === 0);
if (every * DATA.meta.stride > 1) document.querySelector("summary").textContent = `数値の表を表示（${every * DATA.meta.stride} ステップごと）`;
const cols = [["ステップ", step, true]];
for (const ch of DATA.charts) for (const s of ch.series)
  cols.push([`${ch.series.length > 1 ? ch.title + " " + s.name : ch.title} (${ch.unit})`, s.y, ch.integer]);
const table = document.getElementById("table");
table.innerHTML = "<thead><tr>" + cols.map(c => `<th>${c[0]}</th>`).join("") + "</tr></thead><tbody>" +
  rowsk.map(k => "<tr>" + cols.map(c => `<td>${fmt(c[1][k], c[2])}</td>`).join("") + "</tr>").join("") + "</tbody>";
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
