#!/usr/bin/env python3
"""シェル力をコアに足す修正の前（out_nhc_*）と後（out_fix_nhc_*）を，dt・cgtol ごとに HTML で比べる
（標準ライブラリだけで動く）．

使い方（output_result フォルダで）:
    python3 plot_fix_compare.py           # → fig_nhc/fix_dt_compare.html, fix_cgtol_compare.html

フォルダ名 out_nhc_<dt>fs_<cgtol>_… / out_fix_nhc_<dt>fs_<cgtol>… から dt・cgtol を読み，
同じ (dt, cgtol) の修正前・修正後を組にする．修正後の計算の方が短いので，
各組は短い方の時間（修正後の長さ）までで比べる．横軸は時間 [ps]（GRID_FS ごとの値）．
"""
import json
import re
import sys
from pathlib import Path

from plot_log import HA_PER_BOHR_TO_MEV_PER_A, HARTREE_EV, TEMPLATE, natoms, ncell_from_natoms, read_log

GRID_FS = 1.0     # 横軸の間隔 [fs]．全ての dt の倍数であること
FIT_FROM_PS = 0.5  # ΔH′ の傾きの直線フィットはこの時刻以降


def load(d):
    m = re.fullmatch(r"out_(fix_)?nhc_([0-9.]+)fs_([0-9.eE+-]+?)(?:_[0-9.]+ps)?", d.name)
    if not m:
        return None
    rows = read_log(d / "log.txt")
    n = natoms(sorted(d.glob("step_*.xyz"))[0])
    dt = float(m.group(2))
    h0 = rows[0][17]
    return dict(fix=bool(m.group(1)), dt=dt, cgtol=m.group(3), n=n, name=d.name,
                t=[(k + 1) * dt / 1000 for k in range(len(rows))],
                dh=[(r[17] - h0) * HARTREE_EV * 1000 / n for r in rows],
                ncg=[r[18] for r in rows],
                rms=[r[19] * HA_PER_BOHR_TO_MEV_PER_A if r[19] is not None else None for r in rows])


def linfit(x, y):
    xm, ym = sum(x) / len(x), sum(y) / len(y)
    slope = sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)
    return slope, ym - slope * xm


def stats(r, nk):
    """先頭 nk ステップでの数値．"""
    t, dh = r["t"][:nk], r["dh"][:nk]
    f = [k for k in range(nk) if t[k] >= FIT_FROM_PS]
    slope, icpt = linfit([t[k] for k in f], [dh[k] for k in f])
    rms = r["rms"][:nk]
    return dict(end=dh[-1], maxabs=max(abs(v) for v in dh), slope=slope, icpt=icpt,
                ncg=sum(r["ncg"][:nk]) / nk,
                rms=sum(rms) / nk if all(v is not None for v in rms) else None)


def page(pairs, key_label, title, out):
    """pairs: [(ラベル, 修正前, 修正後)]（2〜3 組）．"""
    tmax_fs = min(min(b["t"][-1], a["t"][-1]) for _, b, a in pairs) * 1000
    times = [GRID_FS * k for k in range(1, int(round(tmax_fs / GRID_FS)) + 1)]
    for _, b, a in pairs:
        for r in (b, a):
            if abs(GRID_FS / r["dt"] - round(GRID_FS / r["dt"])) > 1e-9:
                sys.exit(f"GRID_FS={GRID_FS} が dt={r['dt']} の倍数ではありません")
    idx = lambda r: [int(round(t / r["dt"])) - 1 for t in times]
    pick = lambda r, y: [float(f"{y[k]:.7g}") for k in idx(r)]
    nk = lambda r: int(round(tmax_fs / r["dt"]))

    rows, charts = [], []
    st = {}
    for lab, b, a in pairs:
        sb, sa = stats(b, nk(b)), stats(a, nk(a))
        st[lab] = (sb, sa)
        ratio = lambda x, y: f"{y / x * 100:.0f} %" if x else "—"
        rms = f"{sa['rms']:.4f}" if sa["rms"] is not None else "—"
        rows.append(
            f"<tr><td style=\"text-align:left;white-space:nowrap\">{lab}</td>"
            f"<td>{sb['end']:.5f}</td><td>{sa['end']:.5f}</td><td>{ratio(sb['end'], sa['end'])}</td>"
            f"<td>{sb['slope']:.5f}</td><td>{sa['slope']:.5f}</td><td>{ratio(sb['slope'], sa['slope'])}</td>"
            f"<td>{sb['ncg']:.2f}</td><td>{sa['ncg']:.2f}</td>"
            f"<td>{rms}</td></tr>")

    def fitline(c, s):
        x0, x1 = FIT_FROM_PS, tmax_fs / 1000
        return dict(c=c, x0=x0, y0=s["icpt"] + s["slope"] * x0, x1=x1, y1=s["icpt"] + s["slope"] * x1)

    # 図は条件ごとに 1 枚：修正前と修正後の ΔH′ を重ねる
    for lab, b, a in pairs:
        charts.append(dict(
            id=f"pair_{lab}", title=f"ΔH′：{lab}（修正前 vs 修正後）", unit="meV/原子", ref=dict(y=0, label="0"),
            series=[dict(name="修正前", y=pick(b, b["dh"])), dict(name="修正後", y=pick(a, a["dh"]))],
            fits=[fitline(0, st[lab][0]), fitline(1, st[lab][1])]))

    table = ('<section class="card" style="margin-bottom:16px"><h2>まとめ<span class="unit">'
             f'0〜{tmax_fs / 1000:g} ps．傾きは {FIT_FROM_PS:g} ps 以降の直線フィット．比 = 修正後 / 修正前</span></h2>'
             '<table><thead><tr><th style="text-align:left">' + key_label + '</th>'
             f'<th>ΔH′({tmax_fs / 1000:g} ps) 修正前</th><th>修正後</th><th>比</th>'
             '<th>傾き [meV/原子/ps] 修正前</th><th>修正後</th><th>比</th>'
             '<th>CG 回数/ステップ 修正前</th><th>修正後</th><th>残差 rms [meV/Å] 修正後</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table>'
             '<p class="sub" style="margin:8px 0 4px">ΔH′ の単位は meV/原子．修正前の計算（10 ps）は先頭だけを使っている．</p></section>')
    # 3 枚の縦軸をそろえる：全グラフの最小・最大を各グラフの範囲に含める
    ys = [v for ch in charts for se in ch["series"] for v in se["y"]]
    for ch in charts:
        ch["yrange"] = [min(ys), max(ys)]
    n = pairs[0][2]["n"]
    meta = dict(source=title, nsteps=len(times), natoms=n, ncell=ncell_from_natoms(n), stride=1)
    html = TEMPLATE.replace("__DATA__", json.dumps(dict(step=[float(f"{t / 1000:.6g}") for t in times],
                                                        charts=charts, meta=meta), ensure_ascii=False))
    html = (html.replace("<title>MD ログ</title>", f"<title>修正前後の比較（{key_label}）</title>")
                .replace("<h1>MD ログ</h1>", f"<h1>シェル力の修正前後の比較：{key_label}</h1>")
                .replace("${DATA.meta.nsteps} ステップ", f"{tmax_fs / 1000:g} ps・{GRID_FS:g} fs ごとに表示")
                .replace("<b>ステップ ${step[k]}</b>", "<b>${step[k]} ps</b>")
                .replace('[["ステップ", step, true]]', '[["時間 (ps)", step, false]]')
                .replace("ステップごと）`", "点ごと）`")
                .replace('<div class="grid" id="grid"></div>', table + '<div class="grid" id="grid"></div>'))
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    for lab, (sb, sa) in st.items():
        print(f"  {lab:<14} ΔH′end {sb['end']:.5f} → {sa['end']:.5f}   slope {sb['slope']:.5f} → {sa['slope']:.5f}"
              f"   ncg {sb['ncg']:.2f} → {sa['ncg']:.2f}")


def main():
    runs = [r for r in (load(d) for d in sorted(Path(".").glob("out_*nhc_*fs_*")) if (d / "log.txt").exists()) if r]
    by = {}
    for r in runs:
        key = (r["dt"], r["cgtol"], r["fix"])
        if key in by:
            sys.exit(f"同じ条件のフォルダが 2 つあります: {by[key]['name']} と {r['name']}")
        by[key] = r
    def pairs(keys, label):
        out = []
        for dt, cg in keys:
            b, a = by.get((dt, cg, False)), by.get((dt, cg, True))
            if not (b and a):
                sys.exit(f"dt={dt} fs, cgtol={cg} の修正前・修正後がそろっていません")
            out.append((label(dt, cg), b, a))
        return out
    outdir = Path("fig_nhc")
    outdir.mkdir(exist_ok=True)
    page(pairs([(1.0, "1e-6"), (0.5, "1e-6"), (0.2, "1e-6")], lambda dt, cg: f"dt = {dt:g} fs"),
         "dt（cgtol = 1e-6）", "NHC (M=3) + MTTK, 150 K", outdir / "fix_dt_compare.html")
    page(pairs([(0.5, "1e-5"), (0.5, "1e-6"), (0.5, "1e-7")], lambda dt, cg: f"cgtol = {cg}"),
         "cgtol（dt = 0.5 fs）", "NHC (M=3) + MTTK, 150 K", outdir / "fix_cgtol_compare.html")


if __name__ == "__main__":
    main()
