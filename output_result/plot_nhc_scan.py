#!/usr/bin/env python3
"""NHC の dt スキャン・cgtol スキャンを時間軸（ps）でそろえて PNG に描く（matplotlib が必要）．

使い方（output_result フォルダで）:
    python3 plot_nhc_scan.py              # → fig_nhc/dt_compare.png, cgtol_compare.png, summary.txt
                                          #   + dt_compare.html, cgtol_compare.html（ΔH′ とシェル CG 反復回数）
    python3 plot_nhc_scan.py --avg-from 5 # 平均を取る区間の開始時刻 [ps]（既定 5）

フォルダ名 out_nhc_<dt>fs_<cgtol>_<長さ>ps から dt と cgtol を読む．
dt が違う計算はステップ数ではなく時間 t = step·dt でそろえる．
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from plot_log import HARTREE_EV, TEMPLATE, natoms, ncell_from_natoms, read_log

POL = 1602.176634  # e/Å² → μC/cm²
COLORS = ["#2a6fdb", "#e0663a", "#2a9d5c"]


def load(d):
    m = re.match(r"out_nhc_([0-9.]+)fs_([0-9.e+-]+)_", d.name)
    dt, cgtol = float(m.group(1)), m.group(2)
    rows = np.array([r[:19] for r in read_log(d / "log.txt")], dtype=float)
    n = natoms(sorted(d.glob("step_*.xyz"))[0])
    ncell = ncell_from_natoms(n)
    lat = rows[:, 1:10]
    length = [np.sqrt(lat[:, i] ** 2 + lat[:, 3 + i] ** 2 + lat[:, 6 + i] ** 2) / ncell for i in range(3)]
    vol = rows[:, 16]
    return dict(
        name=d.name, dt=dt, cgtol=cgtol, n=n,
        t=np.arange(1, len(rows) + 1) * dt / 1000,       # ps
        temp=rows[:, 0], a=length[0], b=length[1], c=length[2],
        px=rows[:, 13] / vol * POL, py=rows[:, 14] / vol * POL, pz=rows[:, 15] / vol * POL,
        dh=(rows[:, 17] - rows[0, 17]) * HARTREE_EV * 1000 / n,   # meV/原子
        ncg=rows[:, 18],
    )


def movavg(y, w):
    c = np.cumsum(np.insert(y, 0, 0.0))
    k = np.arange(1, len(y) + 1)
    lo = np.maximum(0, k - w)
    return (c[k] - c[lo]) / (k - lo)


def figure(runs, labels, title, out, window_fs):
    fig, axs = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
    panels = [
        ("temp", "温度 [K]"), ("dh", "保存量の変化 ΔH′ [meV/原子]"), ("ncg", f"シェル CG 反復回数（{window_fs:g} fs 移動平均）"),
        ("a", "格子長 a [Å]"), ("b", "格子長 b [Å]"), ("c", "格子長 c [Å]"),
        ("px", "分極 Px [μC/cm²]"), ("py", "分極 Py [μC/cm²]"), ("pz", "分極 Pz [μC/cm²]"),
    ]
    for ax, (key, ylabel) in zip(axs.flat, panels):
        for r, lab, col in zip(runs, labels, COLORS):
            y = r[key]
            if key == "ncg":
                y = movavg(y, max(1, round(window_fs / r["dt"])))
            ax.plot(r["t"], y, color=col, lw=0.8, label=lab)
        if key in ("dh", "px", "py", "pz"):
            ax.axhline(0, color="0.6", lw=0.6, ls="--")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axs[-1]:
        ax.set_xlabel("時間 [ps]")
    axs[0, 0].legend(loc="best")
    fig.suptitle(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def html(runs, labels, title, out, window_fs, grid_fs=5.0, tmax_ps=None):
    """ΔH′ と CG 反復回数を plot_log.py の HTML 部品で描く．横軸は時間 [ps]．

    dt の違う計算を同じ横軸に載せるため，grid_fs ごとの時刻の値を各計算から拾う
    （grid_fs は全ての dt の倍数であること）．tmax_ps を与えるとそこまでを描き，
    表の数値（傾き・最大 |ΔH′|・CG 回数）もその区間で求める．
    """
    for r in runs:
        if abs(grid_fs / r["dt"] - round(grid_fs / r["dt"])) > 1e-9:
            raise ValueError(f"grid_fs={grid_fs} が dt={r['dt']} の倍数ではありません")
    tmax = min(r["t"][-1] for r in runs) * 1000
    if tmax_ps is not None:
        if tmax_ps * 1000 > tmax + 1e-9:
            raise ValueError(f"tmax_ps={tmax_ps} が一番短い計算（{tmax / 1000:g} ps）より長い")
        tmax = tmax_ps * 1000
    times = np.arange(grid_fs, tmax + 1e-9, grid_fs)               # fs
    def at(r, y):
        return [float(f"{v:.7g}") for v in y[np.rint(times / r["dt"]).astype(int) - 1]]
    fits, rows = [], []
    for i, (r, lab) in enumerate(zip(runs, labels)):
        w = r["t"] <= tmax / 1000 + 1e-9                             # 描く区間
        f = w & (r["t"] >= 0.5)
        slope, icpt = np.polyfit(r["t"][f], r["dh"][f], 1)
        x0, x1 = 0.5, tmax / 1000
        fits.append(dict(c=i, x0=x0, y0=icpt + slope * x0, x1=x1, y1=icpt + slope * x1))
        ncg = r["ncg"][w].mean()
        rows.append(f"<tr><td style=\"text-align:left;white-space:nowrap\">{lab}</td><td>{slope:.4f}</td>"
                    f"<td>{np.abs(r['dh'][w]).max():.4f}</td><td>{ncg:.1f}</td><td>{ncg * 1000 / r['dt']:,.0f}</td></tr>")
    series = lambda key: [dict(name=lab, y=at(r, r[key] if key != "ncgavg" else movavg(r["ncg"], max(1, round(window_fs / r["dt"])))))
                          for r, lab in zip(runs, labels)]
    charts = [
        dict(id="cons", title="保存量の変化 ΔH′（破線: 0.5 ps 以降の直線フィット）", unit="meV/原子",
             series=series("dh"), ref=dict(y=0, label="0"), fits=fits),
        dict(id="ncgavg", title=f"シェル緩和（CG）の反復回数 / ステップ（{window_fs:g} fs 移動平均）", unit="回",
             series=series("ncgavg")),
    ]
    table = ('<section class="card" style="margin-bottom:16px"><h2>まとめ<span class="unit">'
             f'0〜{tmax / 1000:g} ps．ΔH′ の傾きは 0.5 ps 以降の直線フィット．CG 回数/ps は 1 ps あたりの計算コストの目安</span></h2>'
             '<table><thead><tr><th style="text-align:left">計算</th><th>ΔH′ の傾き [meV/原子/ps]</th>'
             '<th>最大 |ΔH′| [meV/原子]</th><th>CG 回数/ステップ</th><th>CG 回数/ps</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></section>')
    meta = dict(source=title, nsteps=len(times), natoms=runs[0]["n"], ncell=ncell_from_natoms(runs[0]["n"]), stride=1)
    page = TEMPLATE.replace("__DATA__", json.dumps(dict(step=[float(f"{t / 1000:.6g}") for t in times],
                                                        charts=charts, meta=meta), ensure_ascii=False))
    page = (page.replace("<title>MD ログ</title>", "<title>NHC スキャン比較</title>")
                .replace("<h1>MD ログ</h1>", "<h1>NHC スキャン: 保存量とシェル緩和回数</h1>")
                .replace("${DATA.meta.nsteps} ステップ", f"{tmax / 1000:g} ps・{grid_fs:g} fs ごとに表示")
                .replace("<b>ステップ ${step[k]}</b>", "<b>${step[k]} ps</b>")
                .replace('[["ステップ", step, true]]', '[["時間 (ps)", step, false]]')
                .replace("ステップごと）`", "点ごと）`")
                .replace('<div class="grid" id="grid"></div>', table + '<div class="grid" id="grid"></div>'))
    out.write_text(page, encoding="utf-8")
    print(f"wrote {out}")


def summary(runs, labels, t0):
    hdr = (f"{'計算':<22}{'T [K]':>9}{'a [Å]':>9}{'b [Å]':>9}{'c [Å]':>9}"
           f"{'|P| [μC/cm²]':>14}{'ΔH′ 傾き [meV/原子/ps]':>24}{'max|ΔH′| [meV/原子]':>22}"
           f"{'ncg/step':>10}{'ncg/ps':>10}")
    lines = [f"平均は t ≥ {t0:g} ps，ΔH′ の傾きは t ≥ 0.5 ps の直線フィット", hdr]
    for r, lab in zip(runs, labels):
        s = r["t"] >= t0
        p = np.sqrt(r["px"] ** 2 + r["py"] ** 2 + r["pz"] ** 2)
        f = r["t"] >= 0.5
        slope = np.polyfit(r["t"][f], r["dh"][f], 1)[0]
        ncg = r["ncg"].mean()
        lines.append(
            f"{lab:<22}{r['temp'][s].mean():9.1f}{r['a'][s].mean():9.4f}{r['b'][s].mean():9.4f}{r['c'][s].mean():9.4f}"
            f"{p[s].mean():14.2f}{slope:24.4f}{np.abs(r['dh']).max():22.4f}"
            f"{ncg:10.1f}{ncg * 1000 / r['dt']:10.0f}")
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--avg-from", type=float, default=5.0, help="平均を取る区間の開始時刻 [ps]（既定 5）")
    ap.add_argument("--window", type=float, default=50.0, help="ncg の移動平均の幅 [fs]（既定 50）")
    ap.add_argument("-o", "--outdir", type=Path, default=Path("fig_nhc"))
    args = ap.parse_args()
    names = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["font.family"] = [f for f in ("Hiragino Sans", "Hiragino Maru Gothic Pro", "Hiragino Sans GB") if f in names] or ["sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    runs = {d.name: load(d) for d in sorted(Path(".").glob("out_nhc_*fs_*_*ps")) if (d / "log.txt").exists()}
    pick = lambda pred: sorted((r for r in runs.values() if pred(r)), key=lambda r: (-r["dt"], -float(r["cgtol"])))
    dt_runs = pick(lambda r: r["cgtol"] == "1e-6")
    cg_runs = pick(lambda r: r["dt"] == 0.5)
    args.outdir.mkdir(exist_ok=True)
    dt_lab = [f"dt = {r['dt']:g} fs" for r in dt_runs]
    cg_lab = [f"cgtol = {r['cgtol']}" for r in cg_runs]
    figure(dt_runs, dt_lab, "NHC (M=3) + MTTK, 150 K: 時間刻み dt の比較（cgtol = 1e-6）",
           args.outdir / "dt_compare.png", args.window)
    figure(cg_runs, cg_lab, "NHC (M=3) + MTTK, 150 K: シェル緩和の収束判定 cgtol の比較（dt = 0.5 fs）",
           args.outdir / "cgtol_compare.png", args.window)
    html(dt_runs, dt_lab, "NHC (M=3) + MTTK, 150 K：dt の比較（cgtol = 1e-6）",
         args.outdir / "dt_compare.html", args.window)
    html(cg_runs, cg_lab, "NHC (M=3) + MTTK, 150 K：cgtol の比較（dt = 0.5 fs）",
         args.outdir / "cgtol_compare.html", args.window, tmax_ps=5.0)
    lines = (["[dt スキャン, cgtol = 1e-6]"] + summary(dt_runs, dt_lab, args.avg_from)
             + ["", "[cgtol スキャン, dt = 0.5 fs]"] + summary(cg_runs, cg_lab, args.avg_from))
    (args.outdir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
