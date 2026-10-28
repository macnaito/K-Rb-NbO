#!/usr/bin/env python3
"""2〜3 個の out*/log.txt を同じステップ範囲で重ねて描く（標準ライブラリだけで動く）．

使い方（出力結果 フォルダで）:
    python3 plot_compare.py out out-20000step_0.0Rb_150K_dt=0.5fs     # → compare_plot.html
    python3 plot_compare.py A B 5000                                   # 先頭 5000 ステップだけ
    python3 plot_compare.py A B --labels 'cgtol=1e-6,cgtol=1e-5' -o cgtol_compare.html
    python3 plot_compare.py A B C --only cons,ncgavg                   # 3 つ，グラフを選んで

両方の dt が同じ前提でステップ数をそろえる（短い方に合わせる）．
グラフの部品は plot_log.py の TEMPLATE をそのまま使う．
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from plot_log import HA_PER_BOHR_TO_MEV_PER_A, HARTREE_EV, TEMPLATE, natoms, ncell_from_natoms, read_log

MAXPTS = 2000


def quantities(rows, n, ncell):
    def length(r, i):
        return math.sqrt(r[1 + i] ** 2 + r[4 + i] ** 2 + r[7 + i] ** 2) / (ncell or 1)
    pol = 1602.176634  # e/Å² → μC/cm²
    h0 = rows[0][17]
    return dict(
        temp=[r[0] for r in rows],
        a=[length(r, 0) for r in rows], b=[length(r, 1) for r in rows], c=[length(r, 2) for r in rows],
        vol=[r[16] for r in rows],
        px=[r[13] / r[16] * pol for r in rows], py=[r[14] / r[16] * pol for r in rows],
        pz=[r[15] / r[16] * pol for r in rows],
        dh=[(r[17] - h0) * HARTREE_EV * 1000 / n for r in rows],
        ncg=[r[18] for r in rows],
        # shell_rms 列の無い古いログ・NaN の入る XL-BOMD では None
        rms=([r[19] * HA_PER_BOHR_TO_MEV_PER_A for r in rows]
             if all(r[19] is not None and math.isfinite(r[19]) for r in rows) else None),
    )


def main():
    ap = argparse.ArgumentParser(description="2〜3 個の log.txt を重ねて描く")
    ap.add_argument("dirs", nargs="+", type=Path, help="比べる出力フォルダ（2〜3 個）")
    ap.add_argument("-n", "--nsteps", type=int, help="先頭から描くステップ数（省略時は一番短いものに合わせる）")
    ap.add_argument("--labels", help="凡例の名前をカンマ区切りで（例: 'cgtol=1e-6,cgtol=1e-5'）")
    ap.add_argument("--only", help="描くグラフをカンマ区切りで（temp,a,b,c,vol,px,py,pz,cons,ncg,ncgavg,rms）")
    ap.add_argument("--window", type=int, default=100, help="ncgavg の移動平均の幅（ステップ，既定 100）")
    ap.add_argument("--fit-from", type=int, default=1000,
                    help="ΔH′ の傾きの直線フィットで除外する先頭ステップ数（既定 1000）")
    ap.add_argument("-o", "--output", default="compare_plot.html", help="出力 HTML（既定 compare_plot.html）")
    args = ap.parse_args()
    dirs = args.dirs
    # 系列の色は plot_log.py の TEMPLATE にある 3 色だけ
    if not 2 <= len(dirs) <= 3:
        sys.exit("比べるフォルダは 2〜3 個にしてください")
    logs = [read_log(d / "log.txt") for d in dirs]
    nsteps = min(len(l) for l in logs)
    if args.nsteps:
        nsteps = min(nsteps, args.nsteps)
    snaps = sorted(dirs[0].glob("step_*.xyz"))
    n = natoms(snaps[0]) if snaps else None
    if not n:
        sys.exit(f"{dirs[0]}: step_*.xyz から原子数が読めません")
    ncell = ncell_from_natoms(n)
    q = [quantities(l[:nsteps], n, ncell) for l in logs]
    # 凡例が長くならないよう，フォルダ名の最後の "_" 以降だけを使う（out-…_dt=0.5fs → dt=0.5fs）
    names = [d.name.rsplit("_", 1)[-1] for d in dirs]
    if len(set(names)) < len(names):
        names = [d.name for d in dirs]
    if args.labels:
        names = args.labels.split(",")
        if len(names) != len(dirs):
            sys.exit(f"--labels はカンマ区切りで {len(dirs)} つ指定してください")
    # ncg は 1 ステップごとの揺れが大きいので，移動平均も用意する
    w = args.window
    for qi in q:
        cs = [0]
        for v in qi["ncg"]:
            cs.append(cs[-1] + v)
        qi["ncgavg"] = [(cs[k + 1] - cs[max(0, k + 1 - w)]) / min(k + 1, w) for k in range(len(qi["ncg"]))]

    def two(key):
        return [dict(name=nm, y=qi[key]) for nm, qi in zip(names, q)]

    charts = [
        dict(id="temp", title="温度", unit="K", series=two("temp")),
        dict(id="a", title="格子長 a", unit="Å", series=two("a")),
        dict(id="b", title="格子長 b", unit="Å", series=two("b")),
        dict(id="c", title="格子長 c", unit="Å", series=two("c")),
        dict(id="vol", title="体積", unit="Å³", series=two("vol")),
        dict(id="px", title="分極 Px", unit="μC/cm²", series=two("px"), ref=dict(y=0, label="0")),
        dict(id="py", title="分極 Py", unit="μC/cm²", series=two("py"), ref=dict(y=0, label="0")),
        dict(id="pz", title="分極 Pz", unit="μC/cm²", series=two("pz"), ref=dict(y=0, label="0")),
        dict(id="cons", title="保存量の変化 ΔH′", unit="meV/原子", series=two("dh"), ref=dict(y=0, label="0")),
        dict(id="ncg", title="シェル緩和の反復回数", unit="回", series=two("ncg"), integer=True),
        dict(id="ncgavg", title=f"シェル緩和の反復回数（{w} ステップ移動平均）", unit="回", series=two("ncgavg")),
    ]
    # 残差の rms は全部のログに shell_rms 列がある（かつ NaN が無い）ときだけ描く
    if all(qi["rms"] is not None for qi in q):
        charts.append(dict(id="rms", title="シェル緩和の残差 rms", unit="meV/Å", series=two("rms")))
    elif any(qi["rms"] is not None for qi in q):
        print("note: shell_rms 列の無い（または NaN を含む）ログがあるので残差のグラフは省略")
    # ΔH′ の傾き: fit_from ステップ以降を最小二乗で直線フィット（傾きは meV/原子/ステップ）
    if not 0 <= args.fit_from < nsteps - 1:
        sys.exit(f"--fit-from は 0〜{nsteps - 2} の範囲で指定してください")
    fits = []
    for i, qi in enumerate(q):
        xsr = range(args.fit_from + 1, nsteps + 1)
        ysr = qi["dh"][args.fit_from:]
        xm, ym = sum(xsr) / len(xsr), sum(ysr) / len(ysr)
        slope = sum((x - xm) * (y - ym) for x, y in zip(xsr, ysr)) / sum((x - xm) ** 2 for x in xsr)
        y0 = ym + slope * (xsr[0] - xm)
        fits.append(dict(c=i, x0=xsr[0], y0=y0, x1=xsr[-1], y1=ym + slope * (xsr[-1] - xm), slope=slope))
    for ch in charts:
        if ch["id"] == "cons":
            ch["fits"] = fits
            ch["title"] += "（破線: 直線フィット）"
    # 時間あたりの傾きにするための dt はフォルダ名の "dt=<数値>fs" から読む（読めなければ /1000 ステップで表示）
    dts = [re.search(r"dt=([0-9.]+)fs", d.name) for d in dirs]
    dts = [float(m.group(1)) if m else None for m in dts]
    rows = []
    for nm, f, dt in zip(names, fits, dts):
        per_ps = f"{f['slope'] / (dt / 1000):.4f}" if dt else "—"
        rows.append(f"<tr><td style=\"text-align:left;white-space:nowrap\">{nm}</td><td>{f['slope'] * 1000:.4f}</td><td>{per_ps}</td></tr>")
    slope_html = (
        '<section class="card" style="margin-bottom:16px"><h2>保存量の変化 ΔH′ の傾き'
        f'<span class="unit">{args.fit_from + 1}〜{nsteps} ステップを直線フィット</span></h2>'
        '<table><thead><tr><th style="text-align:left">計算</th><th>meV/原子/1000 ステップ</th><th>meV/原子/ps</th></tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table></section>')

    if args.only:
        ids = args.only.split(",")
        bad = [i for i in ids if i not in {ch["id"] for ch in charts}]
        if bad:
            sys.exit(f"--only に知らないグラフ名: {bad}")
        charts = [ch for i in ids for ch in charts if ch["id"] == i]
    stride = max(1, math.ceil(nsteps / MAXPTS))
    keep = sorted(set([0] + list(range(stride - 1, nsteps, stride))))
    for ch in charts:
        for se in ch["series"]:
            se["y"] = [se["y"][k] if isinstance(se["y"][k], int) else float(f"{se['y'][k]:.7g}") for k in keep]
    meta = dict(source=" と ".join(d.name for d in dirs), nsteps=nsteps, natoms=n, ncell=ncell, stride=stride)
    html = TEMPLATE.replace("__DATA__", json.dumps(dict(step=[k + 1 for k in keep], charts=charts, meta=meta)))
    html = html.replace("<title>MD ログ</title>", "<title>MD ログ比較</title>").replace(
        "<h1>MD ログ</h1>", "<h1>MD ログの比較</h1>").replace(
        '<div class="grid" id="grid"></div>', slope_html + '<div class="grid" id="grid"></div>')
    out = Path(args.output)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}  ({nsteps} steps)")


if __name__ == "__main__":
    main()
