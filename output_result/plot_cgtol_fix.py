#!/usr/bin/env python3
"""dt = 0.5 fs，cgtol = 1e-5 / 1e-6 / 1e-7 で，コード修正前後の保存量の変化 ΔH′ を重ねた HTML を作る．

使い方（KNbO フォルダで）:
    python3 output_result/plot_cgtol_fix.py output_result output_result/cgtol_fix_compare.html
"""
import json, math, statistics as st, sys
from pathlib import Path

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
SC = 27.21138625 * 1000 / 2560          # Ha → meV/atom
DT, NSTEP, SKIP_PS, MAXPTS = 0.5, 2000, 0.2, 1200
# (cgtol, 修正前のフォルダ, 修正後のフォルダ)
RUNS = [("1e-5", "out_0.5fs_1e-5", "out_0.5fs_1e-5_fix"),
        ("1e-6", "out_0.5fs_1e-6_diag2", "out_0.5fs_1e-6_fix"),
        ("1e-7", "out_0.5fs_1e-7", "out_0.5fs_1e-7_fix")]


def rows(p):
    return [list(map(float, l.split())) for l in open(p) if l.strip() and not l.startswith("#")]


def fit(x, y):
    xm, ym = st.mean(x), st.mean(y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)


panels = []
for cg, bdir, adir in RUNS:
    lb, la = rows(ROOT / bdir / "log.txt")[:NSTEP], rows(ROOT / adir / "log.txt")[:NSTEP]
    dg = rows(ROOT / adir / "diag_shellwork.txt")[:NSTEP]
    n = min(len(lb), len(la), len(dg))
    if n < NSTEP:
        print(f"{cg}: {n}/{NSTEP} ステップ → 未完のため飛ばす"); continue
    t = [(k + 1) * DT / 1000 for k in range(n)]
    HB = [(r[17] - lb[0][17]) * SC for r in lb[:n]]
    HA = [(r[17] - la[0][17]) * SC for r in la[:n]]
    Wd = [0.0]
    for k in range(1, n):
        Wd.append(Wd[-1] + dg[k][1] * SC)
    k0 = int(round(SKIP_PS / (DT / 1000))); T = t[k0:]
    tail = dg[k0:n]; ok = [r for r in tail if not math.isnan(r[9])]
    ncgB = st.mean(r[-1] for r in lb[k0:n])
    s = dict(B=fit(T, HB[k0:]), A=fit(T, HA[k0:]), AWd=fit(T, Wd[k0:]),
             rmsRel=st.mean(r[4] for r in tail) / float(cg), cosdd=st.mean(r[2] for r in tail),
             cose=st.mean(r[9] for r in ok), e=st.mean(r[6] for r in tail),
             ncg=st.mean(r[10] for r in tail), ncgB=ncgB)
    stride = max(1, math.ceil(n / MAXPTS))
    keep = sorted(set(list(range(0, n, stride)) + [n - 1]))
    f = lambda v: [float(f"{v[k]:.6g}") for k in keep]
    panels.append(dict(label=f"cgtol = {cg}", cg=cg, dt=DT, nsteps=n,
                       t=[float(f"{t[k]:.6g}") for k in keep], B=f(HB), A=f(HA), stats=s))
    print(f"cgtol={cg}: before {s['B']:.4f} after {s['A']:.4f} ({s['A']/s['B']*100:.0f}%)  Wd {s['AWd']:.4f}  "
          f"rms/cgtol {s['rmsRel']:.3f}  cos(F,dd) {s['cosdd']:+.3f}  cos(e,dd) {s['cose']:+.3f}  |e| {s['e']:.2e}  ncg {ncgB:.1f}->{s['ncg']:.1f}")

tpl = (Path(__file__).parent / "cgtol_fix_template.html").read_text(encoding="utf-8")
OUT.write_text(tpl.replace("__DATA__", json.dumps(dict(panels=panels, skip=SKIP_PS))), encoding="utf-8")
print("wrote", OUT)
