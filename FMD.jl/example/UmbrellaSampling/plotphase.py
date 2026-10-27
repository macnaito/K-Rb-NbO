#!/usr/bin/env python3
import sys, re, ast
import numpy as np
import matplotlib.pyplot as plt

import scienceplots  # noqa: F401  (style登録のため)

# スライドなら「bright」や「high-vis」がおすすめ
plt.style.use(["science", "bright", "no-latex"])  # no-latexは環境依存回避に有効
plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "legend.fontsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
})

# -----------------------------
# Read fitted parameter vector x and blocks_map from your saved file
# -----------------------------
def load_params(path: str):
    header_lines = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break

    header = "\n".join(header_lines)

    # Try to parse blocks_map from header text
    m = re.search(r"blocks_map\s*=\s*(\{.*\})", header, flags=re.DOTALL)
    if not m:
        m = re.search(r"blocks_map\s*[:]\s*(\{.*\})", header, flags=re.DOTALL)
    if not m:
        raise RuntimeError("Could not find blocks_map={...} in the header of the params file.")

    blocks_map = ast.literal_eval(m.group(1))

    arr = np.loadtxt(path, comments="#")
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise RuntimeError("Parameter file format must be 2 columns: idx value (as saved by your script).")
    x = arr[:, 1].astype(float)
    return x, blocks_map, header

# -----------------------------
# Fibonacci sphere sampling (approx. uniform directions)
# -----------------------------
def fibonacci_sphere(npts: int):
    i = np.arange(npts, dtype=float)
    phi = (1 + 5**0.5) / 2  # golden ratio
    theta = 2.0 * np.pi * i / phi
    z = 1.0 - 2.0 * (i + 0.5) / npts
    r = np.sqrt(np.clip(1.0 - z*z, 0.0, 1.0))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

# -----------------------------
# Invariants as functions of unit vector n=(nx,ny,nz)
# -----------------------------
def inv4_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    I_4_0_0 = nx4 + ny4 + nz4
    I_2_2_0 = nx2*ny2 + ny2*nz2 + nz2*nx2
    return np.array([I_4_0_0, I_2_2_0], float)

def inv6_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    I_6_0_0 = nx6 + ny6 + nz6
    I_4_2_0 = nx4*(ny2+nz2) + ny4*(nx2+nz2) + nz4*(nx2+ny2)
    I_2_2_2 = nx2*ny2*nz2
    return np.array([I_6_0_0, I_4_2_0, I_2_2_2], float)

def inv8_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    I_8_0_0 = nx8 + ny8 + nz8
    I_6_2_0 = nx6*(ny2+nz2) + ny6*(nx2+nz2) + nz6*(nx2+ny2)
    I_4_4_0 = nx4*ny4 + ny4*nz4 + nz4*nx4
    I_4_2_2 = nx4*ny2*nz2 + ny4*nx2*nz2 + nz4*nx2*ny2
    return np.array([I_8_0_0, I_6_2_0, I_4_4_0, I_4_2_2], float)

def inv10_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    I_10_0_0 = nx10 + ny10 + nz10
    I_8_2_0  = nx8*(ny2+nz2) + ny8*(nx2+nz2) + nz8*(nx2+ny2)
    I_6_4_0  = nx6*(ny4+nz4) + ny6*(nx4+nz4) + nz6*(nx4+ny4)
    I_6_2_2  = nx6*ny2*nz2 + ny6*nx2*nz2 + nz6*nx2*ny2
    I_4_4_2  = nx4*ny4*nz2 + ny4*nz4*nx2 + nz4*nx4*ny2
    return np.array([I_10_0_0, I_8_2_0, I_6_4_0, I_6_2_2, I_4_4_2], float)

def inv12_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    nx12, ny12, nz12 = nx6*nx6, ny6*ny6, nz6*nz6
    I_12_0_0 = nx12 + ny12 + nz12
    I_10_2_0 = nx10*(ny2+nz2) + ny10*(nx2+nz2) + nz10*(nx2+ny2)
    I_8_4_0  = nx8*(ny4+nz4) + ny8*(nx4+nz4) + nz8*(nx4+ny4)
    I_6_6_0  = nx6*ny6 + ny6*nz6 + nz6*nx6
    I_8_2_2  = nx8*ny2*nz2 + ny8*nx2*nz2 + nz8*nx2*ny2
    I_6_4_2  = nx6*(ny4*nz2 + nz4*ny2) + ny6*(nx4*nz2 + nz4*nx2) + nz6*(nx4*ny2 + ny4*nx2)
    I_4_4_4  = nx4*ny4*nz4
    return np.array([I_12_0_0, I_10_2_0, I_8_4_0, I_6_6_0, I_8_2_2, I_6_4_2, I_4_4_4], float)

def inv14_n(n):
    nx, ny, nz = n
    nx2, ny2, nz2 = nx*nx, ny*ny, nz*nz
    nx4, ny4, nz4 = nx2*nx2, ny2*ny2, nz2*nz2
    nx6, ny6, nz6 = nx4*nx2, ny4*ny2, nz4*nz2
    nx8, ny8, nz8 = nx4*nx4, ny4*ny4, nz4*nz4
    nx10, ny10, nz10 = nx8*nx2, ny8*ny2, nz8*nz2
    nx12, ny12, nz12 = nx6*nx6, ny6*ny6, nz6*nz6
    nx14, ny14, nz14 = nx8*nx6, ny8*ny6, nz8*nz6
    I1 = nx14 + ny14 + nz14
    I2 = nx12*(ny2+nz2) + ny12*(nx2+nz2) + nz12*(nx2+ny2)
    I3 = nx10*(ny4+nz4) + ny10*(nx4+nz4) + nz10*(nx4+ny4)
    I4 = nx8*(ny6+nz6) + ny8*(nx6+nz6) + nz8*(nx6+ny6)
    I5 = nx10*ny2*nz2 + ny10*nx2*nz2 + nz10*nx2*ny2
    I6 = nx8*(ny4*nz2 + nz4*ny2) + ny8*(nx4*nz2 + nz4*nx2) + nz8*(nx4*ny2 + ny4*nx2)
    I7 = nx6*ny4*nz4 + ny6*nx4*nz4 + nz6*nx4*ny4
    I8 = nx6*ny6*nz2 + ny6*nz6*nx2 + nz6*nx6*ny2
    return np.array([I1, I2, I3, I4, I5, I6, I7, I8], float)

INV_FUN = {4: inv4_n, 6: inv6_n, 8: inv8_n, 10: inv10_n, 12: inv12_n, 14: inv14_n}

# -----------------------------
# Build temperature-dependent Landau coefficients from x, blocks_map
# matches your UPDATED fit script:
#  - a1(T)  : quadratic  (c0 + c1*T + c2*T^2)        blocks_map[2] length=3
#  - a11(T) : quadratic  (d0 + d1*T + d2*T^2)
#  - a12(T) : quadratic  (e0 + e1*T + e2*T^2)        blocks_map[4] length=6
#  - a111/a112/a123 : linear (f0+f1*T, g0+g1*T, h0+h1*T) blocks_map[6] length=6
# higher orders: T-independent invariant vectors
# -----------------------------
def coeffs_of_T(T, x, blocks_map):
    # order 2 (quadratic)
    i2, ln2 = blocks_map[2]
    if ln2 != 3:
        raise RuntimeError(f"blocks_map[2] expected length=3 (a1 quadratic), got {ln2}")
    c0, c1, c2 = x[i2+0], x[i2+1], x[i2+2]
    a1 = c0 + c1*T + c2*T*T

    # order 4 (quadratic for a11,a12)
    i4, ln4 = blocks_map[4]
    if ln4 != 6:
        raise RuntimeError(f"blocks_map[4] expected length=6 (a11,a12 quadratic), got {ln4}")
    d0, d1, d2 = x[i4+0], x[i4+1], x[i4+2]  # a11
    e0, e1, e2 = x[i4+3], x[i4+4], x[i4+5]  # a12
    a11 = d0 + d1*T + d2*T*T
    a12 = e0 + e1*T + e2*T*T

    # order 6 (linear as before)
    i6, ln6 = blocks_map[6]
    if ln6 != 6:
        raise RuntimeError(f"blocks_map[6] expected length=6 (a111/a112/a123 linear), got {ln6}")
    f0, f1, g0, g1, h0, h1 = x[i6+0], x[i6+1], x[i6+2], x[i6+3], x[i6+4], x[i6+5]
    a111 = f0 + f1*T
    a112 = g0 + g1*T
    a123 = h0 + h1*T

    return a1, a11, a12, a111, a112, a123

def get_theta_const(x, blocks_map, order):
    i0, ln = blocks_map[order]
    return x[i0:i0+ln].astype(float)

# -----------------------------
# 1D minimization along a direction n: P = p n, p>=0
# g(p) = sum_k c_{2k} p^{2k}
# Let q=p^2 >=0, then g(q) = sum_k c_{2k} q^k
# Stationary points satisfy h(q)=c2 + 2c4 q + 3c6 q^2 + ... + K c_{2K} q^{K-1} = 0
# -----------------------------
def min_along_direction(T, n, x, blocks_map, orders_used):
    a1, a11, a12, a111, a112, a123 = coeffs_of_T(T, x, blocks_map)

    c = {}
    c[2] = a1

    I4 = inv4_n(n)
    c[4] = a11*I4[0] + a12*I4[1]

    I6 = inv6_n(n)
    c[6] = a111*I6[0] + a112*I6[1] + a123*I6[2]

    for o in orders_used:
        if o in (2, 4, 6):
            continue
        theta = get_theta_const(x, blocks_map, o)
        invs = INV_FUN[o](n)
        c[o] = float(np.dot(theta, invs))

    max_order = max(c.keys())
    K = max_order // 2
    gq = np.zeros(K+1, float)
    for pwr, coef in c.items():
        gq[pwr//2] = coef

    # derivative in q: h(q)=sum_{k=1..K} k*gq[k]*q^{k-1}
    hcoef = np.array([k*gq[k] for k in range(1, K+1)], float)[::-1]

    cand_q = [0.0]
    if np.any(np.abs(hcoef) > 0):
        roots = np.roots(hcoef)
        for r in roots:
            if abs(r.imag) < 1e-10:
                q = r.real
                if q > 0:
                    cand_q.append(q)

    def eval_g(q):
        qq = 1.0
        s = 0.0
        for k in range(0, K+1):
            s += gq[k] * qq
            qq *= q
        return s

    best_q = 0.0
    best_g = eval_g(0.0)
    for q in cand_q[1:]:
        val = eval_g(q)
        if np.isfinite(val) and val < best_g:
            best_g = val
            best_q = q

    return best_g, float(np.sqrt(best_q))

# -----------------------------
# Main: scan directions, pick global minimum, plot sorted components vs T
# -----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: plot_Pmin_vs_T.py <params_file> [Tmin Tmax nT nDir]")
        sys.exit(1)

    params_file = sys.argv[1]
    Tmin = float(sys.argv[2]) if len(sys.argv) > 2 else 200.0
    Tmax = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0
    nT   = int(sys.argv[4])   if len(sys.argv) > 4 else 151
    nDir = int(sys.argv[5])   if len(sys.argv) > 5 else 2000

    x, blocks_map, header = load_params(params_file)
    orders_used = sorted([int(k) for k in blocks_map.keys()])
    print("orders_used =", orders_used)
    print(f"T range: {Tmin}..{Tmax} (nT={nT}), directions={nDir}")

    # ---- "Curie point(s)" from a1(T)=0: c0 + c1*T + c2*T^2 = 0
    i2, ln2 = blocks_map[2]
    if ln2 != 3:
        raise RuntimeError(f"blocks_map[2] expected length=3, got {ln2}")
    c0, c1, c2 = x[i2+0], x[i2+1], x[i2+2]
    roots = np.roots([c2, c1, c0])  # c2*T^2 + c1*T + c0 = 0
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-10]
    real_roots.sort()
    if real_roots:
        print("a1(T)=0 real roots (candidates):", ", ".join(f"{rt:.6f} K" for rt in real_roots))
    else:
        print("a1(T)=0 has no real roots (with fitted coefficients).")

    dirs = fibonacci_sphere(nDir)
    Ts = np.linspace(Tmin, Tmax, nT)

    Px = np.zeros_like(Ts)
    Py = np.zeros_like(Ts)
    Pz = np.zeros_like(Ts)
    gmin = np.zeros_like(Ts)

    for it, T in enumerate(Ts):
        best_g = np.inf
        best_p = 0.0
        best_n = np.array([1.0, 0.0, 0.0])

        for n in dirs:
            gg, p = min_along_direction(T, n, x, blocks_map, orders_used)
            if gg < best_g:
                best_g, best_p, best_n = gg, p, n

        Pvec = best_p * best_n
        Px[it], Py[it], Pz[it] = np.abs(Pvec[0]), np.abs(Pvec[1]), np.abs(Pvec[2])
        gmin[it] = best_g

        if (it % max(1, nT//10)) == 0:
            print(f"T={T:.1f}: |P|={np.linalg.norm(Pvec):.4g}, "
                  f"P=({Px[it]:.3g},{Py[it]:.3g},{Pz[it]:.3g}), gmin={best_g:.3e}")

    # sort components by magnitude at each T: ascending -> [min, mid, max]
    Pabs = np.vstack([Px, Py, Pz])
    Psorted = np.sort(Pabs, axis=0)
    Pmin_c, Pmid_c, Pmax_c = Psorted[0], Psorted[1], Psorted[2]

    out = np.column_stack([Ts, Px, Py, Pz, Pmax_c, Pmid_c, Pmin_c, gmin])
    np.savetxt("Pmin_vs_T.dat", out, header="T  |Px|  |Py|  |Pz|  Pmax  Pmid  Pmin  gmin")

    # Plot (NO Tc line; components are sorted by magnitude but labeled Px/Py/Pz for 3 curves)
    plt.figure(figsize=(6, 4))
    plt.plot(Ts, Pmax_c*16.02, label=r"$P_x$")
    plt.plot(Ts, Pmid_c*16.02, label=r"$P_y$")
    plt.plot(Ts, Pmin_c*16.02, label=r"$P_z$")
    plt.xlabel("Temperature $T$ (K)")
    plt.ylabel(r"$P_\alpha$ at argmin $g(P_x,P_y,P_z)$ (C/m$^2$)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("Pmin_components_sorted_vs_T.png", dpi=600)

if __name__ == "__main__":
    main()
