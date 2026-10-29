#!/usr/bin/env python3
"""output_result/out_*_1e-6_diag2 の log.txt と diag_shellwork.txt から，
保存量の変化 ΔH′ とシェル残差の仕事 W = W_d + W_r の比較グラフ (HTML) を作る．"""
import json, math, statistics as st, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SC = 27.21138625 * 1000 / 2560          # Ha → meV/atom
RUNS = [(0.2, "out_0.2fs_1e-6_diag2"), (0.5, "out_0.5fs_1e-6_diag2"), (1.0, "out_1fs_1e-6_diag2")]
MAXPTS = 1200
SKIP_PS = 0.2


def fit(x, y):
    xm, ym = st.mean(x), st.mean(y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)


panels = []
for dt, name in RUNS:
    d = ROOT / name
    log = [list(map(float, l.split())) for l in open(d / "log.txt") if l.strip()]
    dg = [list(map(float, l.split())) for l in open(d / "diag_shellwork.txt")
          if l.strip() and not l.startswith("#")]
    n = min(len(log), len(dg))
    t = [(k + 1) * dt / 1000 for k in range(n)]
    H = [(log[k][17] - log[0][17]) * SC for k in range(n)]
    Wd, Wr = [0.0], [0.0]
    for k in range(1, n):               # log row 0 = 基準 (step 1)，W は step 2 から足す
        Wd.append(Wd[-1] + dg[k][1] * SC)
        Wr.append(Wr[-1] + dg[k][11] * SC)
    W = [a + b for a, b in zip(Wd, Wr)]
    R = [h - w for h, w in zip(H, W)]
    k0 = int(round(SKIP_PS / (dt / 1000)))
    T = t[k0:]
    s = {key: fit(T, v[k0:]) for key, v in (("H", H), ("Wd", Wd), ("Wr", Wr), ("W", W), ("R", R))}
    rows = dg[k0:]
    s["cos_dd"] = st.mean(r[2] for r in rows)
    s["cos_dr"] = st.mean(r[12] for r in rows)
    s["rmsF"] = st.mean(r[4] for r in rows)
    s["ncg"] = st.mean(r[10] for r in rows)
    stride = max(1, math.ceil(n / MAXPTS))
    keep = sorted(set(list(range(0, n, stride)) + [n - 1]))
    f = lambda v: [float(f"{v[k]:.6g}") for k in keep]
    panels.append(dict(dt=dt, nsteps=n, t=[float(f"{t[k]:.6g}") for k in keep],
                       H=f(H), W=f(W), R=f(R), Wd=f(Wd), Wr=f(Wr), stats=s))
    print(f"{name}: {n} steps")

tpl = (Path(__file__).parent / "budget_template.html").read_text(encoding="utf-8")
OUT.write_text(tpl.replace("__DATA__", json.dumps(dict(panels=panels, skip=SKIP_PS))), encoding="utf-8")
print("wrote", OUT)
