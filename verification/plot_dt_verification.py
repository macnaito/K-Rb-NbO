#!/usr/bin/env python3
"""dt = 0.5〜0.2 fs（NHC + MTTK, 150 K, cgtol 1e-6）の保存量を比べる HTML を描く（標準ライブラリのみ）．

使い方（どこからでも）:
    python3 verification/plot_dt_verification.py
    → verification/dt_conserved.html, verification/dt_conserved_summary.txt

描くもの: ΔH′, Δ(H′+W_d), W_d, shell_cos の移動平均（時間幅 --cos-window-fs）．
dt の違う計算は時間 t = step·dt でそろえ，grid_fs（全 dt の公倍数）ごとの値を拾う．
図の横軸は一番長い計算に合わせ，短い計算はそこで線が終わる．
表の数値は各計算の全長で集計する．
0.3 fs は segfault で 8.1 ps で止まった out_pre_nhc_0.3fs_1e-6_verification を使う．
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "output_result"))
from plot_log import HARTREE_EV, TEMPLATE, natoms, ncell_from_natoms, read_log  # noqa: E402

RUNS = [  # (dt [fs], フォルダ)
    (0.5, "out_nhc_0.5fs_1e-6_verification"),
    (0.4, "out_nhc_0.4fs_1e-6_verification"),
    (0.3, "out_pre_nhc_0.3fs_1e-6_verification"),
    (0.2, "out_nhc_0.2fs_1e-6_verification"),
]
HA2MEV = HARTREE_EV * 1000


def movavg_finite(y, w):
    """後ろ向き w 点の単純移動平均（NaN は除く）．"""
    cs, cn, out = [0.0], [0], []
    for v in y:
        ok = math.isfinite(v)
        cs.append(cs[-1] + (v if ok else 0.0))
        cn.append(cn[-1] + ok)
    for k in range(len(y)):
        lo = max(0, k + 1 - w)
        m = cn[k + 1] - cn[lo]
        out.append((cs[k + 1] - cs[lo]) / m if m else math.nan)
    return out


def linfit(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    a = sxy / sxx
    return a, my - a * mx


def load(dt, folder, cos_window_fs):
    d = ROOT / folder
    rows = read_log(d / "log.txt")
    if rows[0][20] is None:
        sys.exit(f"{folder}: conserved+W_d / shell_cos の列がありません")
    n = natoms(sorted(d.glob("step_*.xyz"))[0])
    h0, hw0 = rows[0][17], rows[0][20]
    cos = [r[21] for r in rows]
    return dict(
        name=folder, dt=dt, n=n,
        t=[(k + 1) * dt / 1000 for k in range(len(rows))],          # ps
        temp=[r[0] for r in rows], ncg=[r[18] for r in rows],
        dh=[(r[17] - h0) * HA2MEV / n for r in rows],                # meV/原子
        dhw=[(r[20] - hw0) * HA2MEV / n for r in rows],
        wd=[(r[20] - r[17]) * HA2MEV / n for r in rows],             # W_d（1 ステップ目は 0）
        cos=cos, cosavg=movavg_finite(cos, max(1, round(cos_window_fs / dt))),
    )


def after(r, key, t0):
    return [v for t, v in zip(r["t"], r[key]) if t >= t0]


def stats(r, t0):
    ts = after(r, "t", t0)
    fit = {k: linfit(ts, after(r, k, t0)) for k in ("dh", "dhw", "wd")}
    cos = [v for v in after(r, "cos", t0) if math.isfinite(v)]
    temp = after(r, "temp", t0)
    return dict(span=r["t"][-1], T=sum(temp) / len(temp), fit=fit,
                dh=r["dh"][-1], dhw=r["dhw"][-1], wd=r["wd"][-1],
                cos=sum(cos) / len(cos), ncg=sum(r["ncg"]) / len(r["ncg"]))


PANELS = [
    ("dh", "保存量の変化 ΔH′", "meV/原子"),
    ("dhw", "W_d を足した保存量の変化 Δ(H′+W_d)", "meV/原子"),
    ("wd", "シェル残差の仕事 W_d", "meV/原子"),
    ("cosavg", "シェル残差の向きの偏り cos", ""),
]


def build(runs, labels, st, cos_window_fs, t0, grid_fs):
    for r in runs:
        if abs(grid_fs / r["dt"] - round(grid_fs / r["dt"])) > 1e-9:
            sys.exit(f"grid_fs={grid_fs} が dt={r['dt']} の倍数ではありません")
    # 全 dt に共通する時刻だけ拾うと，計算の長さが違っても横軸を 1 本にできる．
    # TEMPLATE は全系列が同じ長さの step 配列を前提にするので，
    # 4 本そろう区間 [0, 一番短い計算] で描く．
    tmax = min(r["t"][-1] for r in runs) * 1000
    nt = int(tmax / grid_fs + 1e-9)
    times = [grid_fs * (j + 1) for j in range(nt)]
    def at(r, y):
        return [float(f"{y[round(t / r['dt']) - 1]:.7g}") for t in times]
    charts = []
    for key, title, unit in PANELS:
        ch = dict(id=key, title=title + (f"（{cos_window_fs:g} fs 移動平均）" if key == "cosavg" else
                                         f"（破線: t ≥ {t0:g} ps の直線フィット）"),
                  unit=unit or "（無次元）", series=[dict(name=lab, y=at(r, r[key])) for r, lab in zip(runs, labels)],
                  ref=dict(y=0, label="0"))
        if key != "cosavg":
            ch["fits"] = []
            for i, r in enumerate(runs):
                ts = [t for t in r["t"] if t0 <= t <= tmax / 1000 + 1e-9]
                ys = [v for t, v in zip(r["t"], r[key]) if t0 <= t <= tmax / 1000 + 1e-9]
                a, b = linfit(ts, ys)
                ch["fits"].append(dict(c=i, x0=t0, y0=b + a * t0, x1=tmax / 1000, y1=b + a * tmax / 1000))
        charts.append(ch)
    rows = "".join(
        f"<tr><td style=\"text-align:left;white-space:nowrap\">{lab}</td><td>{s['span']:.2f}</td><td>{s['T']:.1f}</td>"
        f"<td>{s['fit']['dh'][0]:+.5f}</td><td>{s['fit']['wd'][0]:+.5f}</td><td>{s['fit']['dhw'][0]:+.5f}</td>"
        f"<td>{s['dh']:+.4f}</td><td>{s['wd']:+.4f}</td><td>{s['dhw']:+.4f}</td><td>{s['cos']:+.3f}</td>"
        f"<td>{s['ncg']:.1f}</td><td>{s['ncg'] * 1000 / r['dt']:,.0f}</td></tr>"
        for lab, s, r in zip(labels, st, runs))
    table = ('<section class="card" style="margin-bottom:16px"><h2>まとめ<span class="unit">'
             f'各計算の全長で集計（グラフは 4 本そろう 0〜{tmax / 1000:.2f} ps）．傾きは t ≥ {t0:g} ps の直線フィット [meV/原子/ps]，'
             f'終値は最終ステップの値 [meV/原子]，T と cos 平均は t ≥ {t0:g} ps．'
             '0.3 fs は segfault で約 8.1 ps で停止した計算</span></h2>'
             '<table><thead><tr><th style="text-align:left">計算</th><th>長さ [ps]</th><th>T [K]</th>'
             '<th>ΔH′ 傾き</th><th>W_d 傾き</th><th>Δ(H′+W_d) 傾き</th>'
             '<th>ΔH′ 終値</th><th>W_d 終値</th><th>Δ(H′+W_d) 終値</th>'
             '<th>cos 平均</th><th>CG 回数/ステップ</th><th>CG 回数/ps</th></tr></thead>'
             f'<tbody>{rows}</tbody></table></section>')
    meta = dict(source="dt 比較（verification）", nsteps=nt, natoms=runs[0]["n"],
                ncell=ncell_from_natoms(runs[0]["n"]), stride=1)
    page = TEMPLATE.replace("__DATA__", json.dumps(dict(step=[float(f"{t / 1000:.6g}") for t in times],
                                                        charts=charts, meta=meta), ensure_ascii=False))
    # TEMPLATE は 3 色までなので 4 色目（紫）を足す
    page = (page.replace("--series-3: #1baf7a;", "--series-3: #1baf7a;\n  --series-4: #8a5cd0;")
                .replace("--series-3: #199e70;", "--series-3: #199e70;\n    --series-4: #9d74e0;")
                .replace('"var(--series-3)"];', '"var(--series-3)", "var(--series-4)"];'))
    if page.count("--series-4") != 4:
        sys.exit("TEMPLATE の色定義が想定と違います（4 色目を足せません）")
    return (page.replace("<title>MD ログ</title>", "<title>dt と保存量</title>")
                .replace("<h1>MD ログ</h1>", "<h1>dt と保存量（NHC + MTTK, 150 K, cgtol 1e-6）</h1>")
                .replace("${DATA.meta.nsteps} ステップ", f"0〜{tmax / 1000:.2f} ps・{grid_fs:g} fs ごとに表示")
                .replace("<b>ステップ ${step[k]}</b>", "<b>${step[k]} ps</b>")
                .replace('[["ステップ", step, true]]', '[["時間 (ps)", step, false]]')
                .replace("ステップごと）`", "点ごと）`")
                .replace('<div class="grid" id="grid"></div>', table + '<div class="grid" id="grid"></div>'))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cos-window-fs", type=float, default=300.0, help="cos の移動平均の時間幅 [fs]（既定 300）")
    ap.add_argument("--fit-from", type=float, default=0.5, help="傾き・平均を取る開始時刻 [ps]（既定 0.5）")
    ap.add_argument("--grid-fs", type=float, default=6.0, help="グラフに拾う時刻の間隔 [fs]（全 dt の倍数，既定 6）")
    args = ap.parse_args()
    runs = [load(dt, f, args.cos_window_fs) for dt, f in RUNS]
    labels = [f"dt = {r['dt']:g} fs" for r in runs]
    st = [stats(r, args.fit_from) for r in runs]
    outdir = ROOT / "verification"
    (outdir / "dt_conserved.html").write_text(build(runs, labels, st, args.cos_window_fs, args.fit_from, args.grid_fs),
                                              encoding="utf-8")
    print(f"wrote {outdir / 'dt_conserved.html'}")
    lines = [f"傾き [meV/原子/ps] は t ≥ {args.fit_from:g} ps の直線フィット，終値 [meV/原子]，T・cos 平均は t ≥ {args.fit_from:g} ps",
             f"{'計算':<12}{'長さps':>7}{'T[K]':>7}{'ΔH′傾き':>10}{'W_d傾き':>10}{'Δ(H′+W_d)傾き':>14}"
             f"{'ΔH′終値':>9}{'W_d終値':>9}{'Δ(H′+W_d)終値':>14}{'cos平均':>8}{'ncg':>6}"]
    for lab, s in zip(labels, st):
        lines.append(f"{lab:<12}{s['span']:7.2f}{s['T']:7.1f}{s['fit']['dh'][0]:+10.5f}{s['fit']['wd'][0]:+10.5f}"
                     f"{s['fit']['dhw'][0]:+14.5f}{s['dh']:+9.4f}{s['wd']:+9.4f}{s['dhw']:+14.4f}{s['cos']:+8.3f}{s['ncg']:6.1f}")
    (outdir / "dt_conserved_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
