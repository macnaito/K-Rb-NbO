#!/usr/bin/env python3
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Dict, Tuple
import scienceplots

plt.style.use(["science", "bright", "no-latex"])
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# ============================================================
# Temperature dependence:
#   - 2nd: a1(T) = c0 + c1*T + c2*T^2
#   - 4th: a11(T), a12(T) = (d0+d1*T+d2*T^2), (e0+e1*T+e2*T^2)   <-- CHANGED
#   - 6th: linear in T (as before)
#   - higher orders: T-independent
# ============================================================

# -----------------------------
# I/O
# -----------------------------
def load_xy(path: str, symmetrize: bool = True, nbins: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, comments="#")
    P = arr[:, 0].astype(float) # e/Å^2
    G = arr[:, 2].astype(float) # eV
    V = arr[:, 5].astype(float) # Å^3

    m = np.isfinite(P) & np.isfinite(G) & np.isfinite(V)
    P, G, V = P[m], G[m], V[m]
    if P.size == 0:
        return P, G, V

    arg = np.argmin(np.abs(P))
    G = G - G[arg]
    g = G / V[arg]

    if not symmetrize:
        return P, g

    Pm = np.concatenate([P, -P])
    gm = np.concatenate([g,  g])

    if nbins <= 0:
        idx = np.argsort(Pm)
        return Pm[idx], gm[idx]

    A = np.abs(Pm)
    Amax = np.max(A)
    edges = np.linspace(0.0, Amax, nbins + 1)
    ib = np.clip(np.digitize(A, edges) - 1, 0, nbins - 1)

    s = np.zeros(nbins)
    c = np.zeros(nbins)
    np.add.at(s, ib, gm)
    np.add.at(c, ib, 1.0)

    centers = 0.5 * (edges[:-1] + edges[1:])
    good = c > 0
    g_even = s[good] / c[good]

    P_out = np.concatenate([centers[good], -centers[good]])
    g_out = np.concatenate([g_even, g_even])

    idx = np.argsort(P_out)
    return P_out[idx], g_out[idx]

# -----------------------------
# Direction invariants
# -----------------------------
def n_from_hkl(hkl):
    v = np.array(hkl, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def inv4(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    return [nx4 + ny4 + nz4, nx2*ny2 + ny2*nz2 + nz2*nx2]  # [4], [2,2]

def inv6(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    return [
        nx6 + ny6 + nz6,                                        # [6]
        nx4*(ny2+nz2) + ny4*(nx2+nz2) + nz4*(nx2+ny2),          # [4,2]
        nx2*ny2*nz2                                             # [2,2,2]
    ]

def inv8(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    return [
        nx8 + ny8 + nz8,
        nx6*(ny2+nz2) + ny6*(nx2+nz2) + nz6*(nx2+ny2),
        nx4*ny4 + ny4*nz4 + nz4*nx4,
        nx4*ny2*nz2 + ny4*nx2*nz2 + nz4*nx2*ny2
    ]

def inv10(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    return [
        nx10 + ny10 + nz10,
        nx8*(ny2+nz2) + ny8*(nx2+nz2) + nz8*(nx2+ny2),
        nx6*(ny4+nz4) + ny6*(nx4+nz4) + nz6*(nx4+ny4),
        nx6*ny2*nz2 + ny6*nx2*nz2 + nz6*nx2*ny2,
        nx4*ny4*nz2 + ny4*nz4*nx2 + nz4*nx4*ny2
    ]

def inv12(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    nx12, ny12, nz12 = nx6*nx6, ny6*ny6, nz6*nz6
    return [
        nx12 + ny12 + nz12,
        nx10*(ny2+nz2) + ny10*(nx2+nz2) + nz10*(nx2+ny2),
        nx8*(ny4+nz4) + ny8*(nx4+nz4) + nz8*(nx4+ny4),
        nx6*ny6 + ny6*nz6 + nz6*nx6,
        nx8*ny2*nz2 + ny8*nx2*nz2 + nz8*nx2*ny2,
        nx6*(ny4*nz2 + nz4*ny2) + ny6*(nx4*nz2 + nz4*nx2) + nz6*(nx4*ny2 + ny4*nx2),
        nx4*ny4*nz4
    ]

def inv14(hkl):
    n = n_from_hkl(hkl)
    nx2, ny2, nz2 = n**2
    nx4, ny4, nz4 = nx2**2, ny2**2, nz2**2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    nx12, ny12, nz12 = nx6*nx6, ny6*ny6, nz6*nz6
    nx14, ny14, nz14 = nx8*nx6, ny8*ny6, nz8*nz6
    return [
        nx14 + ny14 + nz14,
        nx12*(ny2+nz2) + ny12*(nx2+nz2) + nz12*(nx2+ny2),
        nx10*(ny4+nz4) + ny10*(nx4+nz4) + nz10*(nx4+ny4),
        nx8*(ny6+nz6) + ny8*(nx6+nz6) + nz8*(nx6+ny6),
        nx10*ny2*nz2 + ny10*nx2*nz2 + nz10*nx2*ny2,
        nx8*(ny4*nz2 + nz4*ny2) + ny8*(nx4*nz2 + nz4*nx2) + nz8*(nx4*ny2 + ny4*nx2),
        nx6*ny4*nz4 + ny6*nx4*nz4 + nz6*nx4*ny4,
        nx6*ny6*nz2 + ny6*nz6*nx2 + nz6*nx6*ny2
    ]

def get_invariant_func(order):
    return {4: inv4, 6: inv6, 8: inv8, 10: inv10, 12: inv12, 14: inv14}.get(order, None)

# -----------------------------
# Ridge solver (Tikhonov)
# -----------------------------
def ridge_solve(A: np.ndarray, y: np.ndarray, lam: float, penalty_mask: np.ndarray):
    A = np.asarray(A, float)
    y = np.asarray(y, float)

    col = np.linalg.norm(A, axis=0)
    col[col == 0] = 1.0
    As = A / col

    if lam > 0:
        D = np.diag(penalty_mask.astype(float))
        As_aug = np.vstack([As, np.sqrt(lam) * D])
        y_aug  = np.concatenate([y, np.zeros(As.shape[1])])
    else:
        As_aug, y_aug = As, y

    xs, *_ = np.linalg.lstsq(As_aug, y_aug, rcond=None)
    return xs / col

# -----------------------------
# Model blocks
#   - 2nd: a1(T) quadratic (3 params)
#   - 4th: a11(T),a12(T) quadratic (2 invariants * 3 params = 6)  <-- CHANGED
#   - 6th: linear (unchanged)
# -----------------------------
def build_blocks_map(max_order: int):
    inv_counts = {8: 4, 10: 5, 12: 7, 14: 8}
    orders = list(range(2, max_order + 1, 2))

    blocks_map = {}
    idx = 0

    # 2nd: a1(T)=c0+c1*T+c2*T^2  -> 3 params
    blocks_map[2] = (idx, 3); idx += 3

    # 4th: a11(T),a12(T) each quadratic -> 2 invariants * 3 params = 6  <-- CHANGED
    blocks_map[4] = (idx, 6); idx += 6

    # 6th: a111(T),a112(T),a123(T) each linear -> 3 invariants * 2 params = 6
    blocks_map[6] = (idx, 6); idx += 6

    # higher orders T-independent
    for o in orders:
        if o in (2, 4, 6):
            continue
        blocks_map[o] = (idx, inv_counts[o]); idx += inv_counts[o]

    return blocks_map, orders, idx

def build_row(Pval: float, Tval: float, hkl, blocks_map, orders, nparam: int):
    r = np.zeros(nparam, dtype=float)
    P2 = Pval**2
    P4 = Pval**4
    P6 = Pval**6

    # --- 2nd: a1(T)=c0+c1*T+c2*T^2
    i2, _ = blocks_map[2]
    r[i2 + 0] = 1.0 * P2
    r[i2 + 1] = Tval * P2
    r[i2 + 2] = (Tval**2) * P2

    # --- 4th: a11(T),a12(T) each quadratic  <-- CHANGED
    i4, _ = blocks_map[4]
    I11, I12 = inv4(hkl)

    # a11(T)=d0+d1*T+d2*T^2
    r[i4 + 0] = 1.0        * I11 * P4
    r[i4 + 1] = Tval       * I11 * P4
    r[i4 + 2] = (Tval**2)  * I11 * P4

    # a12(T)=e0+e1*T+e2*T^2
    r[i4 + 3] = 1.0        * I12 * P4
    r[i4 + 4] = Tval       * I12 * P4
    r[i4 + 5] = (Tval**2)  * I12 * P4

    # --- 6th: a111(T),a112(T),a123(T) each linear (unchanged)
    i6, _ = blocks_map[6]
    J1, J2, J3 = inv6(hkl)
    r[i6 + 0] = 1.0  * J1 * P6
    r[i6 + 1] = Tval * J1 * P6
    r[i6 + 2] = 1.0  * J2 * P6
    r[i6 + 3] = Tval * J2 * P6
    r[i6 + 4] = 1.0  * J3 * P6
    r[i6 + 5] = Tval * J3 * P6

    # --- higher orders (T-independent)
    for o in orders:
        if o in (2, 4, 6):
            continue
        invs = get_invariant_func(o)(hkl)
        i0, ln = blocks_map[o]
        r[i0:i0+ln] = (Pval**o) * np.asarray(invs, float)

    return r

def a1_of_T(T, x, blocks_map):
    i2, _ = blocks_map[2]
    c0, c1, c2 = x[i2], x[i2+1], x[i2+2]
    return c0 + c1*T + c2*T*T

def a11_a12_of_T(T, x, blocks_map):
    # 4th block is now 6 params: (d0,d1,d2,e0,e1,e2)  <-- CHANGED
    i4, _ = blocks_map[4]
    d0, d1, d2 = x[i4+0], x[i4+1], x[i4+2]  # a11
    e0, e1, e2 = x[i4+3], x[i4+4], x[i4+5]  # a12
    return (d0 + d1*T + d2*T*T), (e0 + e1*T + e2*T*T)

def a111_a112_a123_of_T(T, x, blocks_map):
    i6, _ = blocks_map[6]
    f0, f1 = x[i6+0], x[i6+1]  # a111
    g0, g1 = x[i6+2], x[i6+3]  # a112
    h0, h1 = x[i6+4], x[i6+5]  # a123
    return (f0 + f1*T), (g0 + g1*T), (h0 + h1*T)

def predict_g(P: np.ndarray, T: float, hkl, x: np.ndarray, blocks_map, orders):
    g = np.zeros_like(P, dtype=float)

    g += a1_of_T(T, x, blocks_map) * (P**2)

    a11, a12 = a11_a12_of_T(T, x, blocks_map)
    I11, I12 = inv4(hkl)
    g += (a11*I11 + a12*I12) * (P**4)

    a111, a112, a123 = a111_a112_a123_of_T(T, x, blocks_map)
    J1, J2, J3 = inv6(hkl)
    g += (a111*J1 + a112*J2 + a123*J3) * (P**6)

    for o in orders:
        if o in (2, 4, 6):
            continue
        i0, ln = blocks_map[o]
        invs = np.asarray(get_invariant_func(o)(hkl), float)
        coef = float(np.dot(x[i0:i0+ln], invs))
        g += coef * (P**o)

    return g

# -----------------------------
# Main
# -----------------------------
def main():
    max_order = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    lam = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-6
    symm = bool(int(sys.argv[3])) if len(sys.argv) > 3 else True
    nbins = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    if max_order % 2 != 0 or max_order < 6 or max_order > 14:
        raise ValueError("max_order must be even and in [6,14]")

    print(f"max_order={max_order}, lambda={lam:.2e}, symmetrize={symm}, nbins={nbins}")
    print("T-dep: a1(T) quadratic; a11(T),a12(T) quadratic; 6th linear; higher orders T-independent.")

    datafiles = {
        ((1,0,0), 200): "./US_200K_100/wham_fes_1d_withV.dat",
        ((1,1,0), 200): "./US_200K_110/wham_fes_1d_withV.dat",
        ((1,1,1), 200): "./US_200K_111/wham_fes_1d_withV.dat",
        ((1,1,2), 200): "./US_200K_112/wham_fes_1d_withV.dat",

        ((1,0,0), 300): "./US_300K_100/wham_fes_1d_withV.dat",
        ((1,1,0), 300): "./US_300K_110/wham_fes_1d_withV.dat",
        ((1,1,1), 300): "./US_300K_111/wham_fes_1d_withV.dat",
        ((1,1,2), 300): "./US_300K_112/wham_fes_1d_withV.dat",

        ((1,0,0), 400): "./US_400K_100/wham_fes_1d_withV.dat",
        ((1,1,0), 400): "./US_400K_110/wham_fes_1d_withV.dat",
        ((1,1,1), 400): "./US_400K_111/wham_fes_1d_withV.dat",
        ((1,1,2), 400): "./US_400K_112/wham_fes_1d_withV.dat",
    }

    raw: Dict[Tuple[Tuple[int,int,int], int], Tuple[np.ndarray, np.ndarray]] = {}
    for (hkl, T), fp in datafiles.items():
        P, g = load_xy(fp, symmetrize=symm, nbins=nbins)
        if P.size == 0:
            continue
        raw[(tuple(hkl), int(T))] = (P, g)
        print(f"Loaded {hkl}@{T}K: N={P.size}")

    if not raw:
        print("No data loaded.")
        return

    blocks_map, orders, nparam = build_blocks_map(max_order)

    rows = []
    y = []
    for (hkl, T), (P, g) in raw.items():
        Tval = float(T)
        for Pi, gi in zip(P, g):
            rows.append(build_row(float(Pi), Tval, hkl, blocks_map, orders, nparam))
            y.append(float(gi))

    A = np.vstack(rows)
    y = np.array(y, dtype=float)

    # penalty mask: example policy (必要なら調整)
    penalty_mask = np.ones(nparam, dtype=float)

    # a1(T): c0はペナルティなし（任意）、c1,c2はペナルティあり、など
    i2, _ = blocks_map[2]
    penalty_mask[i2 + 0] = 0.0  # c0
    # penalty_mask[i2 + 1] = 1.0
    # penalty_mask[i2 + 2] = 1.0

    # 4th: 必要なら定数項だけペナルティなしにする等も可
    # i4, _ = blocks_map[4]
    # penalty_mask[i4 + 0] = 0.0  # a11 d0
    # penalty_mask[i4 + 3] = 0.0  # a12 e0

    x = ridge_solve(A, y, lam=lam, penalty_mask=penalty_mask)

    resid = A @ x - y
    print("\n=== Direct ridge fit (a1,a11,a12 quadratic in T) ===")
    print("A shape:", A.shape, " rank:", np.linalg.matrix_rank(A), "/", A.shape[1])
    print("RMSE:", float(np.sqrt(np.mean(resid**2))))
    print("max|res|:", float(np.max(np.abs(resid))))

    # ---- "Curie point(s)" from a1(T)=0: c0 + c1*T + c2*T^2 = 0
    c0, c1, c2 = x[i2 + 0], x[i2 + 1], x[i2 + 2]
    print(f"\na1(T) = c0 + c1*T + c2*T^2 with c0={c0:.6e}, c1={c1:.6e}, c2={c2:.6e}")

    roots = np.roots([c2, c1, c0])  # c2*T^2 + c1*T + c0 = 0
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
    real_roots.sort()
    if real_roots:
        print("Roots of a1(T)=0 (candidates):", ", ".join(f"{rt:.6f} K" for rt in real_roots))
    else:
        print("a1(T)=0 has no real roots (with fitted coefficients).")

    outdat = "landau.dat"
    np.savetxt(
        outdat,
        np.column_stack([np.arange(nparam), x]),
        header=f"idx value; blocks_map={blocks_map}; a1_roots={real_roots}"
    )
    print("Saved:", outdat)

    temps = sorted({T for (_, T) in raw.keys()})
    hkls_by_T = {T: sorted([hkl for (hkl, TT) in raw.keys() if TT == T]) for T in temps}

    for T in temps:
        plt.figure(figsize=(6.0, 4.0))
        for hkl in hkls_by_T[T]:
            P, gdat = raw[(hkl, T)]
            Pgrid = np.linspace(np.min(P), np.max(P), 500)
            gfit = predict_g(Pgrid, float(T), hkl, x, blocks_map, orders)

            # shift using point closest to P=0 for consistency
            arg0 = np.argmin(np.abs(P))
            gdat0 = gdat - gdat[arg0]
            arg1 = np.argmin(np.abs(Pgrid))
            gfit0 = gfit - gfit[arg1]

            lbl = f"[{hkl[0]}{hkl[1]}{hkl[2]}]"
            plt.plot(P*16.02, gdat0, "o", ms=4, label=f"data {lbl}")
            plt.plot(Pgrid*16.02, gfit0, "-", lw=2, label=f"fit  {lbl}")

        plt.xlabel(r"$P$ (C/m$^2$)")
        plt.ylabel(r"$g(P)$ (GJ/m$^3$)")
        plt.legend()
        plt.tight_layout()
        outpng = f"P_vs_gP_T{T}K.png"
        plt.savefig(outpng, dpi=200)
        plt.close()
        print("Saved plot:", outpng)

if __name__ == "__main__":
    main()
