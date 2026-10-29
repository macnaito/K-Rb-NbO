#!/usr/bin/env python3
"""修正前 (out_*_1e-6_diag2) と修正後 (out_*_1e-6_fix) の保存量の変化 ΔH′ を重ねた HTML を作る．
修正後の予想 = 修正前の ΔH′ − W_r (diag_shellwork.txt の W_r の累積) も点線で重ねる．

使い方（KNbO フォルダで）:
    python3 output_result/plot_fix_compare.py output_result output_result/fix_conserved_compare.html
修正後の log.txt が無い dt は飛ばす．
"""
import json, math, statistics as st, sys
from pathlib import Path

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SC = 27.21138625 * 1000 / 2560          # Ha → meV/atom
RUNS = [(0.2, "0.2fs"), (0.5, "0.5fs"), (1.0, "1fs")]
MAXPTS = 1200
SKIP_PS = 0.2


def fit(x, y):
    xm, ym = st.mean(x), st.mean(y)
    return sum((a - xm) * (b - ym) for a, b in zip(x, y)) / sum((a - xm) ** 2 for a in x)


def load(d):
    log = [list(map(float, l.split())) for l in open(d / "log.txt") if l.strip()]
    dg = [list(map(float, l.split())) for l in open(d / "diag_shellwork.txt")
          if l.strip() and not l.startswith("#")]
    n = min(len(log), len(dg))
    H = [(log[k][17] - log[0][17]) * SC for k in range(n)]
    Wd, Wr = [0.0], [0.0]
    for k in range(1, n):               # log row 0 = 基準 (step 1)，W は step 2 から足す
        Wd.append(Wd[-1] + dg[k][1] * SC)
        Wr.append(Wr[-1] + dg[k][11] * SC)
    return H, Wd, Wr, dg[:n]


panels = []
for dt, tag in RUNS:
    before, after = ROOT / f"out_{tag}_1e-6_diag2", ROOT / f"out_{tag}_1e-6_fix"
    nsteps = round(1.0 / (dt / 1000))            # 1 ps
    if not (after / "log.txt").exists():
        print(f"{after}: なし → 飛ばす"); continue
    HB, _, WrB, dgB = load(before)
    HA, WdA, _, dgA = load(after)
    n = min(len(HB), len(HA))
    if n < nsteps:
        print(f"{after}: {n}/{nsteps} ステップ → 未完のため飛ばす"); continue
    t = [(k + 1) * dt / 1000 for k in range(n)]
    P = [h - w for h, w in zip(HB, WrB)]
    k0 = int(round(SKIP_PS / (dt / 1000)))
    T = t[k0:n]
    s = dict(B=fit(T, HB[k0:n]), A=fit(T, HA[k0:n]), P=fit(T, P[k0:n]), AWd=fit(T, WdA[k0:n]),
             rmsB=st.mean(r[4] for r in dgB[k0:n]), rmsA=st.mean(r[4] for r in dgA[k0:n]),
             ncg=st.mean(r[10] for r in dgA[k0:n]))
    stride = max(1, math.ceil(n / MAXPTS))
    keep = sorted(set(list(range(0, n, stride)) + [n - 1]))
    f = lambda v: [float(f"{v[k]:.6g}") for k in keep]
    panels.append(dict(dt=dt, nsteps=n, t=[float(f"{t[k]:.6g}") for k in keep],
                       B=f(HB), A=f(HA), P=f(P), stats=s))
    print(f"dt={dt} fs: {n} steps  before {s['B']:.4f}  after {s['A']:.4f}  pred {s['P']:.4f} meV/atom/ps")

tpl = (Path(__file__).parent / "fix_template.html").read_text(encoding="utf-8")
OUT.write_text(tpl.replace("__DATA__", json.dumps(dict(panels=panels, skip=SKIP_PS))), encoding="utf-8")
print("wrote", OUT)
