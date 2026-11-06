#!/usr/bin/env python3
"""シェル CG 予測子の次数ごとに保存量を並べた HTML を作る．

使い方:
    python3 plot_conserved.py [出力 html] [run ディレクトリ ...]

run ディレクトリを省略すると out_nhc_0.5fs_1e-6_extrap0〜4 のうち存在するものを使う．
グラフは 3 枚: ΔH′，Δ(H′ + W_d)，W_d = Δ(H′ + W_d) − ΔH′（いずれも meV/原子）．
log.txt は 22 列形式（conserved = 18 列目，conserved+W_d = 20 列目，shell_cos = 21 列目，ncg = 22 列目）．
"""
import json
import math
import sys
from pathlib import Path

HARTREE_EV = 27.211386245988
DT_PS = 0.5e-3          # 0.5 fs
FIT_FROM_PS = 0.5       # 傾きは t ≥ 0.5 ps で最小二乗

HERE = Path(__file__).resolve().parent


def read_run(d):
    rows = []
    with open(d / "log.txt") as f:
        for line in f:
            v = line.split()
            if len(v) != 22:
                sys.exit(f"{d}/log.txt: 22 列でない行があります（{len(v)} 列）．conserved+W_d の列が必要です")
            rows.append((float(v[17]), float(v[19]), float(v[20]), int(v[21])))
    snaps = sorted(d.glob("step_*.xyz"))
    if not snaps:
        sys.exit(f"{d}: 原子数を取る step_*.xyz がありません")
    with open(snaps[0]) as f:
        n = int(f.readline())
    return rows, n


def slope(t, y):
    pts = [(a, b) for a, b in zip(t, y) if a >= FIT_FROM_PS]
    m = len(pts)
    mt = sum(a for a, _ in pts) / m
    my = sum(b for _, b in pts) / m
    return sum((a - mt) * (b - my) for a, b in pts) / sum((a - mt) ** 2 for a, _ in pts)


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "conserved_extrap.html"
    dirs = [Path(p) for p in sys.argv[2:]] or [
        HERE / f"out_nhc_0.5fs_1e-6_extrap{k}" for k in range(5)
        if (HERE / f"out_nhc_0.5fs_1e-6_extrap{k}" / "log.txt").exists()]
    if not dirs:
        sys.exit("run ディレクトリが見つかりません")

    runs, nsteps = [], None
    for d in dirs:
        rows, n = read_run(d)
        nsteps = len(rows) if nsteps is None else min(nsteps, len(rows))
        scale = HARTREE_EV * 1000 / n
        h0, hw0 = rows[0][0], rows[0][1]
        dh = [(r[0] - h0) * scale for r in rows]
        dhw = [(r[1] - hw0) * scale for r in rows]
        wd = [b - a for a, b in zip(dh, dhw)]
        cos = [r[2] for r in rows if math.isfinite(r[2])]
        name = d.name.removeprefix("out_nhc_0.5fs_1e-6_")
        label = f"{name[-1]} 次" if name.startswith("extrap") else name
        runs.append(dict(name=label, dir=d.name, natoms=n, dh=dh, dhw=dhw, wd=wd,
                         ncg=sum(r[3] for r in rows) / len(rows),
                         cos=sum(cos) / len(cos) if cos else float("nan")))
    # 長さを揃える（最短の run に合わせる）
    t = [(i + 1) * DT_PS for i in range(nsteps)]
    for r in runs:
        for k in ("dh", "dhw", "wd"):
            r[k] = r[k][:nsteps]
        r["slope_dh"] = slope(t, r["dh"])
        r["slope_dhw"] = slope(t, r["dhw"])

    charts = [
        dict(id="dh", title="保存量の変化 ΔH′", key="dh"),
        dict(id="dhw", title="シェル残差の仕事を足した保存量 Δ(H′ + W_d)", key="dhw"),
        dict(id="wd", title="シェル残差の仕事 W_d", key="wd"),
    ]
    data = dict(t=[round(x, 5) for x in t], charts=charts,
                runs=[{**r, **{k: [round(v, 6) for v in r[k]] for k in ("dh", "dhw", "wd")}}
                      for r in runs],
                natoms=runs[0]["natoms"], fit_from=FIT_FROM_PS)
    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, ensure_ascii=False)), encoding="utf-8")
    print(f"wrote {out}")


TEMPLATE = r"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>外挿次数と保存量</title>
<style>
:root {
  color-scheme: light;
  --surface-0: #f4f3ef; --surface-1: #fcfcfb; --grid: #e6e5df; --axis: #b9b8b0;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #6f6e69;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100; --series-5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    color-scheme: dark;
    --surface-0: #121211; --surface-1: #1a1a19; --grid: #2c2c2a; --axis: #4a4a46;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0: #121211; --surface-1: #1a1a19; --grid: #2c2c2a; --axis: #4a4a46;
  --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #9a998f;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--surface-0); color: var(--text-primary);
  font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Hiragino Sans", "Noto Sans JP", sans-serif; }
main { max-width: 1100px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { font-size: 20px; margin: 0 0 4px; }
.sub { color: var(--text-secondary); margin: 0 0 16px; }
.legend { display: flex; flex-wrap: wrap; gap: 16px; margin: 0 0 16px; color: var(--text-secondary); }
.legend i, .tip i { display: inline-block; width: 12px; height: 3px; border-radius: 2px; vertical-align: middle; margin-right: 6px; }
.card { background: var(--surface-1); border-radius: 10px; padding: 14px 14px 8px; margin-bottom: 16px; position: relative; }
.card h2 { font-size: 15px; margin: 0 0 6px; font-weight: 600; }
.card .unit { color: var(--text-muted); font-weight: 400; margin-left: 6px; }
svg { display: block; width: 100%; height: auto; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
svg text.lbl { fill: var(--text-secondary); font-size: 12px; }
.tip { position: absolute; pointer-events: none; background: var(--surface-0); color: var(--text-primary);
  border: 1px solid var(--grid); border-radius: 8px; padding: 6px 10px; font-size: 12px;
  font-variant-numeric: tabular-nums; white-space: nowrap; display: none; z-index: 2; }
.tip .row { display: flex; justify-content: space-between; gap: 12px; }
.tablewrap { overflow-x: auto; background: var(--surface-1); border-radius: 10px; padding: 8px 14px; }
table { border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; width: 100%; }
th, td { padding: 6px 10px; text-align: right; border-bottom: 1px solid var(--grid); white-space: nowrap; }
th { color: var(--text-secondary); font-weight: 600; }
th:first-child, td:first-child { text-align: left; }
.note { color: var(--text-muted); font-size: 12px; margin-top: 8px; }
</style>
</head>
<body>
<main>
<h1>シェル CG 外挿次数ごとの保存量</h1>
<p class="sub" id="sub"></p>
<div class="legend" id="legend"></div>
<div id="charts"></div>
<h2 style="font-size:15px;margin:24px 0 8px">まとめ</h2>
<div class="tablewrap"><table id="summary"></table></div>
<p class="note" id="note"></p>
</main>
<script>
const DATA = __DATA__;
const COLORS = DATA.runs.map((_, i) => `var(--series-${i + 1})`);
const t = DATA.t;
const NS = "http://www.w3.org/2000/svg";
const el = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };
const fmt = v => (Math.abs(v) < 5e-5 ? 0 : v).toFixed(4);

document.getElementById("sub").textContent =
  `NHC・150 K・0.5 fs・cgtol 1e-6・${DATA.natoms} 原子・${t.length} ステップ（${t[t.length - 1].toFixed(2)} ps）。値は 1 原子あたり、初期値からの変化。`;
document.getElementById("legend").innerHTML = DATA.runs.map((r, i) =>
  `<span><i style="background:${COLORS[i]}"></i>${r.name}</span>`).join("");

function niceTicks(lo, hi, n) {
  const step0 = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0);
  const out = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(10));
  return out;
}

const W = 1000, H = 300, M = { l: 64, r: 70, t: 10, b: 30 };
for (const ch of DATA.charts) {
  const card = document.createElement("div"); card.className = "card";
  card.innerHTML = `<h2>${ch.title}<span class="unit">meV/原子</span></h2>`;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": ch.title });
  card.appendChild(svg);
  const tip = document.createElement("div"); tip.className = "tip"; card.appendChild(tip);
  document.getElementById("charts").appendChild(card);

  let lo = Infinity, hi = -Infinity;
  for (const r of DATA.runs) for (const v of r[ch.key]) { if (v < lo) lo = v; if (v > hi) hi = v; }
  lo = Math.min(lo, 0); hi = Math.max(hi, 0);
  const pad = (hi - lo) * 0.06 || 1e-3; lo -= pad; hi += pad;
  const xs = v => M.l + (v - t[0]) / (t[t.length - 1] - t[0]) * (W - M.l - M.r);
  const ys = v => M.t + (hi - v) / (hi - lo) * (H - M.t - M.b);

  for (const v of niceTicks(lo, hi, 5)) {
    svg.appendChild(el("line", { x1: M.l, x2: W - M.r, y1: ys(v), y2: ys(v),
      stroke: v === 0 ? "var(--axis)" : "var(--grid)", "stroke-width": 1 }));
    const tx = el("text", { x: M.l - 8, y: ys(v) + 4, "text-anchor": "end" }); tx.textContent = v; svg.appendChild(tx);
  }
  for (const v of niceTicks(t[0], t[t.length - 1], 6)) {
    const tx = el("text", { x: xs(v), y: H - 8, "text-anchor": "middle" }); tx.textContent = v + " ps"; svg.appendChild(tx);
  }
  // 線（表示用に間引き：1 px あたり最大・最小を残す）
  const ends = [];
  DATA.runs.forEach((r, i) => {
    const y = r[ch.key]; let d = "";
    const bins = Math.floor(W - M.l - M.r), per = Math.max(1, Math.floor(y.length / bins));
    for (let s = 0; s < y.length; s += per) {
      let mn = s, mx = s;
      for (let k = s; k < Math.min(s + per, y.length); k++) { if (y[k] < y[mn]) mn = k; if (y[k] > y[mx]) mx = k; }
      for (const k of mn < mx ? [mn, mx] : [mx, mn]) d += (d ? "L" : "M") + xs(t[k]).toFixed(1) + "," + ys(y[k]).toFixed(1);
    }
    svg.appendChild(el("path", { d, fill: "none", stroke: COLORS[i], "stroke-width": 1.5, "stroke-linejoin": "round" }));
    ends.push({ i, y: ys(y[y.length - 1]) });
  });
  // 右端の直接ラベル（重なりは縦にずらす）
  ends.sort((a, b) => a.y - b.y);
  for (let k = 1; k < ends.length; k++) if (ends[k].y - ends[k - 1].y < 14) ends[k].y = ends[k - 1].y + 14;
  for (const e of ends) {
    const tx = el("text", { x: W - M.r + 8, y: e.y + 4, class: "lbl" }); tx.textContent = DATA.runs[e.i].name; svg.appendChild(tx);
  }
  // ホバー：縦線 + ツールチップ
  const cross = el("line", { y1: M.t, y2: H - M.b, stroke: "var(--axis)", "stroke-width": 1, visibility: "hidden" });
  svg.appendChild(cross);
  const dots = DATA.runs.map((_, i) => { const c = el("circle", { r: 4, fill: COLORS[i], stroke: "var(--surface-1)", "stroke-width": 2, visibility: "hidden" }); svg.appendChild(c); return c; });
  const hit = el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" });
  svg.appendChild(hit);
  const hide = () => { tip.style.display = "none"; cross.setAttribute("visibility", "hidden"); dots.forEach(c => c.setAttribute("visibility", "hidden")); };
  hit.addEventListener("pointerleave", hide);
  hit.addEventListener("pointermove", ev => {
    const rect = svg.getBoundingClientRect(), sx = (ev.clientX - rect.left) * W / rect.width;
    const frac = (sx - M.l) / (W - M.l - M.r), k = Math.max(0, Math.min(t.length - 1, Math.round(frac * (t.length - 1))));
    const x = xs(t[k]); cross.setAttribute("x1", x); cross.setAttribute("x2", x); cross.setAttribute("visibility", "visible");
    DATA.runs.forEach((r, i) => { dots[i].setAttribute("cx", x); dots[i].setAttribute("cy", ys(r[ch.key][k])); dots[i].setAttribute("visibility", "visible"); });
    tip.innerHTML = `<div style="color:var(--text-secondary)">t = ${t[k].toFixed(3)} ps（${k + 1} step）</div>` +
      DATA.runs.map((r, i) => `<div class="row"><span><i style="background:${COLORS[i]}"></i>${r.name}</span><span>${r[ch.key][k].toFixed(5)}</span></div>`).join("");
    tip.style.display = "block";
    const cr = card.getBoundingClientRect(), px = ev.clientX - cr.left, py = ev.clientY - cr.top;
    tip.style.left = (px + 14 + tip.offsetWidth > cr.width ? px - 14 - tip.offsetWidth : px + 14) + "px";
    tip.style.top = Math.max(0, py - tip.offsetHeight - 10) + "px";
  });
}

const last = a => a[a.length - 1];
document.getElementById("summary").innerHTML =
  "<thead><tr><th>外挿次数</th><th>平均 CG 回数</th><th>平均 cos</th><th>ΔH′ 終値</th><th>Δ(H′+W_d) 終値</th><th>W_d 終値</th>" +
  "<th>ΔH′ 傾き</th><th>Δ(H′+W_d) 傾き</th></tr></thead><tbody>" +
  DATA.runs.map((r, i) => `<tr><td><i style="display:inline-block;width:12px;height:3px;border-radius:2px;background:${COLORS[i]};vertical-align:middle;margin-right:6px"></i>${r.name}</td>` +
    `<td>${r.ncg.toFixed(1)}</td><td>${r.cos >= 0 ? "+" : ""}${r.cos.toFixed(3)}</td>` +
    `<td>${fmt(last(r.dh))}</td><td>${fmt(last(r.dhw))}</td><td>${fmt(last(r.wd))}</td>` +
    `<td>${fmt(r.slope_dh)}</td><td>${fmt(r.slope_dhw)}</td></tr>`).join("") + "</tbody>";
document.getElementById("note").textContent =
  `終値の単位は meV/原子、傾きは meV/原子/ps（t ≥ ${DATA.fit_from} ps の最小二乗）。cos はシェル残差と Δs のなす角の余弦の全ステップ平均。`;
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
