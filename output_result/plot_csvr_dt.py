#!/usr/bin/env python3
"""dt や cgtol の違う計算の保存量の変化 ΔH′ とシェル緩和（CG）の反復回数を，時間軸（ps）で重ねて HTML にする
（標準ライブラリだけで動く）．

使い方（output_result フォルダで）:
    python3 plot_csvr_dt.py                       # out_1fs_1e-6, out_0.5fs_1e-6, out_0.2fs_1e-6 → csvr_dt_compare.html
    python3 plot_csvr_dt.py --cgtol               # out_0.5fs_1e-5, out_0.5fs_1e-6, out_0.5fs_1e-7 → csvr_cgtol_compare.html
    python3 plot_csvr_dt.py A B C -o out.html     # フォルダを指定（2〜3 個）

dt と cgtol はフォルダ名 out_<dt>fs_<cgtol>… から読む．凡例は，dt が全部同じなら cgtol，そうでなければ dt．
長さの違う計算は一番短いものに合わせる．横軸は GRID_FS ごとの時刻（全ての dt の倍数であること）．
"""
import argparse
import json
import re
import sys
from pathlib import Path

from plot_log import HARTREE_EV, TEMPLATE, natoms, ncell_from_natoms, read_log

GRID_FS = 5.0      # 横軸の間隔 [fs]
FIT_FROM_PS = 0.5  # ΔH′ の傾きの直線フィットはこの時刻以降


def load(d):
    m = re.match(r"out_([0-9.]+)fs_([0-9.eE+-]+?)(?:_|$)", d.name)
    if not m:
        sys.exit(f"{d}: フォルダ名から dt と cgtol が読めません（out_<dt>fs_<cgtol>… の形にしてください）")
    rows = read_log(d / "log.txt")
    snaps = sorted(d.glob("step_*.xyz"))
    n = natoms(snaps[0]) if snaps else None
    if not n:
        sys.exit(f"{d}: step_*.xyz から原子数が読めません")
    dt = float(m.group(1))
    h0 = rows[0][17]
    return dict(dt=dt, cgtol=m.group(2), n=n, t=[(k + 1) * dt / 1000 for k in range(len(rows))],
                dh=[(r[17] - h0) * HARTREE_EV * 1000 / n for r in rows], ncg=[r[18] for r in rows])


def movavg(y, w):
    cs = [0]
    for v in y:
        cs.append(cs[-1] + v)
    return [(cs[k + 1] - cs[max(0, k + 1 - w)]) / min(k + 1, w) for k in range(len(y))]


def linfit(x, y):
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    return slope, ym - slope * xm


def main():
    ap = argparse.ArgumentParser(description="dt の違う計算の ΔH′ と CG 反復回数を重ねて描く")
    ap.add_argument("dirs", nargs="*", type=Path)
    ap.add_argument("--cgtol", action="store_true", help="フォルダ省略時に cgtol スキャン（dt = 0.5 fs）を描く")
    ap.add_argument("--title", help="ページの説明行（省略時は条件から作る）")
    ap.add_argument("--window", type=float, default=50.0, help="CG 反復回数の移動平均の幅 [fs]（既定 50）")
    ap.add_argument("-o", "--output", help="出力 HTML（既定 csvr_dt_compare.html / csvr_cgtol_compare.html）")
    args = ap.parse_args()
    if not args.dirs:
        args.dirs = ([Path("out_0.5fs_1e-5"), Path("out_0.5fs_1e-6"), Path("out_0.5fs_1e-7")] if args.cgtol else
                     [Path("out_1fs_1e-6"), Path("out_0.5fs_1e-6"), Path("out_0.2fs_1e-6")])
    if not 2 <= len(args.dirs) <= 3:
        sys.exit("比べるフォルダは 2〜3 個にしてください（グラフの色が 3 色まで）")
    runs = [load(d) for d in args.dirs]
    for r in runs:
        if abs(GRID_FS / r["dt"] - round(GRID_FS / r["dt"])) > 1e-9:
            sys.exit(f"GRID_FS={GRID_FS} が dt={r['dt']} の倍数ではありません")
    tmax_fs = min(r["t"][-1] for r in runs) * 1000
    times = [GRID_FS * k for k in range(1, int(round(tmax_fs / GRID_FS)) + 1)]
    by_cgtol = len({r["dt"] for r in runs}) == 1
    if by_cgtol:
        labels = [f"cgtol = {r['cgtol']}" for r in runs]
        what, fixed = "cgtol", f"dt = {runs[0]['dt']:g} fs"
    else:
        labels = [f"dt = {r['dt']:g} fs" for r in runs]
        what = "dt"
        fixed = f"cgtol = {runs[0]['cgtol']}" if len({r["cgtol"] for r in runs}) == 1 else "cgtol いろいろ"
    title = args.title or f"CSVR + MTTK, 150 K, {fixed}（シェル力の補正前）"
    output = args.output or f"csvr_{what}_compare.html"

    def pick(r, y):
        return [float(f"{y[int(round(t / r['dt'])) - 1]:.7g}") for t in times]

    fits, rows = [], []
    for i, (r, lab) in enumerate(zip(runs, labels)):
        nk = int(round(tmax_fs / r["dt"]))
        f = [k for k in range(nk) if r["t"][k] >= FIT_FROM_PS]
        slope, icpt = linfit([r["t"][k] for k in f], [r["dh"][k] for k in f])
        x0, x1 = FIT_FROM_PS, tmax_fs / 1000
        fits.append(dict(c=i, x0=x0, y0=icpt + slope * x0, x1=x1, y1=icpt + slope * x1))
        ncg = sum(r["ncg"][:nk]) / nk
        rows.append(f"<tr><td style=\"text-align:left;white-space:nowrap\">{lab}</td><td>{slope:.4f}</td>"
                    f"<td>{max(abs(v) for v in r['dh'][:nk]):.4f}</td><td>{ncg:.1f}</td>"
                    f"<td>{ncg * 1000 / r['dt']:,.0f}</td></tr>")
        print(f"{lab:<14} slope {slope:.4f} meV/atom/ps  max|dH| {max(abs(v) for v in r['dh'][:nk]):.4f}  "
              f"ncg {ncg:.1f}/step  {ncg * 1000 / r['dt']:,.0f}/ps")
    charts = [
        dict(id="cons", title=f"保存量の変化 ΔH′（破線: {FIT_FROM_PS:g} ps 以降の直線フィット）", unit="meV/原子",
             series=[dict(name=lab, y=pick(r, r["dh"])) for r, lab in zip(runs, labels)],
             ref=dict(y=0, label="0"), fits=fits),
        dict(id="ncg", title=f"シェル緩和（CG）の反復回数 / ステップ（{args.window:g} fs 移動平均）", unit="回",
             series=[dict(name=lab, y=pick(r, movavg(r["ncg"], max(1, round(args.window / r["dt"])))))
                     for r, lab in zip(runs, labels)]),
    ]
    table = ('<section class="card" style="margin-bottom:16px"><h2>まとめ<span class="unit">'
             f'0〜{tmax_fs / 1000:g} ps．傾きは {FIT_FROM_PS:g} ps 以降の直線フィット．'
             'CG 回数/ps は 1 ps あたりの計算コストの目安</span></h2>'
             '<table><thead><tr><th style="text-align:left">計算</th><th>ΔH′ の傾き [meV/原子/ps]</th>'
             '<th>最大 |ΔH′| [meV/原子]</th><th>CG 回数/ステップ</th><th>CG 回数/ps</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></section>')
    n = runs[0]["n"]
    meta = dict(source=title, nsteps=len(times), natoms=n, ncell=ncell_from_natoms(n), stride=1)
    html = TEMPLATE.replace("__DATA__", json.dumps(dict(step=[float(f"{t / 1000:.6g}") for t in times],
                                                        charts=charts, meta=meta), ensure_ascii=False))
    html = (html.replace("<title>MD ログ</title>", f"<title>CSVR：{what} の比較</title>")
                .replace("<h1>MD ログ</h1>", f"<h1>CSVR：{what} を変えたときの保存量とシェル緩和回数</h1>")
                .replace("${DATA.meta.nsteps} ステップ", f"{tmax_fs / 1000:g} ps・{GRID_FS:g} fs ごとに表示")
                .replace("<b>ステップ ${step[k]}</b>", "<b>${step[k]} ps</b>")
                .replace('[["ステップ", step, true]]', '[["時間 (ps)", step, false]]')
                .replace("ステップごと）`", "点ごと）`")
                .replace('<div class="grid" id="grid"></div>', table + '<div class="grid" id="grid"></div>'))
    Path(output).write_text(html, encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
