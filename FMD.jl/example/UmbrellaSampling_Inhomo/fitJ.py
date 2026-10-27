#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import re
import ast
import matplotlib.pyplot as plt
import os

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
# Invariants (same ordering as your FitLandau.py / plot script)
# ============================================================
def inv4_from_P(Px2, Py2, Pz2):
    return np.stack(
        [
            Px2 * Px2 + Py2 * Py2 + Pz2 * Pz2,
            Px2 * Py2 + Py2 * Pz2 + Pz2 * Px2,
        ],
        axis=-1,
    )

def inv6_from_P(Px2, Py2, Pz2):
    Px4, Py4, Pz4 = Px2 * Px2, Py2 * Py2, Pz2 * Pz2
    return np.stack(
        [
            Px2 * Px4 + Py2 * Py4 + Pz2 * Pz4,
            Px4 * (Py2 + Pz2) + Py4 * (Px2 + Pz2) + Pz4 * (Px2 + Py2),
            Px2 * Py2 * Pz2,
        ],
        axis=-1,
    )

def inv8_from_P(Px2, Py2, Pz2):
    Px4, Py4, Pz4 = Px2 * Px2, Py2 * Py2, Pz2 * Pz2
    Px6, Py6, Pz6 = Px4 * Px2, Py4 * Py2, Pz4 * Pz2
    Px8, Py8, Pz8 = Px4 * Px4, Py4 * Py4, Pz4 * Pz4
    return np.stack(
        [
            Px8 + Py8 + Pz8,
            Px6 * (Py2 + Pz2) + Py6 * (Px2 + Pz2) + Pz6 * (Px2 + Py2),
            Px4 * Py4 + Py4 * Pz4 + Pz4 * Px4,
            Px4 * Py2 * Pz2 + Py4 * Px2 * Pz2 + Pz4 * Px2 * Py2,
        ],
        axis=-1,
    )

def inv10_from_P(Px2, Py2, Pz2):
    Px4, Py4, Pz4 = Px2 * Px2, Py2 * Py2, Pz2 * Pz2
    Px6, Py6, Pz6 = Px4 * Px2, Py4 * Py2, Pz4 * Pz2
    Px8, Py8, Pz8 = Px4 * Px4, Py4 * Py4, Pz4 * Pz4
    Px10, Py10, Pz10 = Px8 * Px2, Py8 * Py2, Pz8 * Pz2
    return np.stack(
        [
            Px10 + Py10 + Pz10,
            Px8 * (Py2 + Pz2) + Py8 * (Px2 + Pz2) + Pz8 * (Px2 + Py2),
            Px6 * (Py4 + Pz4) + Py6 * (Px4 + Pz4) + Pz6 * (Px4 + Py4),
            Px6 * Py2 * Pz2 + Py6 * Px2 * Pz2 + Pz6 * Px2 * Py2,
            Px4 * Py4 * Pz2 + Py4 * Pz4 * Px2 + Pz4 * Px4 * Py2,
        ],
        axis=-1,
    )

def inv12_from_P(Px2, Py2, Pz2):
    Px4, Py4, Pz4 = Px2 * Px2, Py2 * Py2, Pz2 * Pz2
    Px6, Py6, Pz6 = Px4 * Px2, Py4 * Py2, Pz4 * Pz2
    Px8, Py8, Pz8 = Px4 * Px4, Py4 * Py4, Pz4 * Pz4
    Px10, Py10, Pz10 = Px8 * Px2, Py8 * Py2, Pz8 * Pz2
    Px12, Py12, Pz12 = Px6 * Px6, Py6 * Py6, Pz6 * Pz6
    return np.stack(
        [
            Px12 + Py12 + Pz12,
            Px10 * (Py2 + Pz2) + Py10 * (Px2 + Pz2) + Pz10 * (Px2 + Py2),
            Px8 * (Py4 + Pz4) + Py8 * (Px4 + Pz4) + Pz8 * (Px4 + Py4),
            Px6 * Py6 + Py6 * Pz6 + Pz6 * Px6,
            Px8 * Py2 * Pz2 + Py8 * Px2 * Pz2 + Pz8 * Px2 * Py2,
            Px6 * (Py4 * Pz2 + Pz4 * Py2)
            + Py6 * (Px4 * Pz2 + Pz4 * Px2)
            + Pz6 * (Px4 * Py2 + Py4 * Px2),
            Px4 * Py4 * Pz4,
        ],
        axis=-1,
    )

def inv14_from_P(Px2, Py2, Pz2):
    Px4, Py4, Pz4 = Px2 * Px2, Py2 * Py2, Pz2 * Pz2
    Px6, Py6, Pz6 = Px4 * Px2, Py4 * Py2, Pz4 * Pz2
    Px8, Py8, Pz8 = Px4 * Px4, Py4 * Py4, Pz4 * Pz4
    Px10, Py10, Pz10 = Px8 * Px2, Py8 * Py2, Pz8 * Pz2
    Px12, Py12, Pz12 = Px6 * Px6, Py6 * Py6, Pz6 * Pz6
    Px14, Py14, Pz14 = Px8 * Px6, Py8 * Py6, Pz8 * Pz6

    I1 = Px14 + Py14 + Pz14
    I2 = Px12 * (Py2 + Pz2) + Py12 * (Px2 + Pz2) + Pz12 * (Px2 + Py2)
    I3 = Px10 * (Py4 + Pz4) + Py10 * (Px4 + Pz4) + Pz10 * (Px4 + Py4)
    I4 = Px8 * (Py6 + Pz6) + Py8 * (Px6 + Pz6) + Pz8 * (Px6 + Py6)
    I5 = Px10 * Py2 * Pz2 + Py10 * Px2 * Pz2 + Pz10 * Px2 * Py2
    I6 = (
        Px8 * (Py4 * Pz2 + Pz4 * Py2)
        + Py8 * (Px4 * Pz2 + Pz4 * Px2)
        + Pz8 * (Px4 * Py2 + Py4 * Px2)
    )
    I7 = Px6 * Py4 * Pz4 + Py6 * Px4 * Pz4 + Pz6 * Px4 * Py4
    I8 = Px6 * Py6 * Pz2 + Py6 * Pz6 * Px2 + Pz6 * Px6 * Py2
    return np.stack([I1, I2, I3, I4, I5, I6, I7, I8], axis=-1)

INV_FUN_P = {4: inv4_from_P, 6: inv6_from_P, 8: inv8_from_P, 10: inv10_from_P, 12: inv12_from_P, 14: inv14_from_P}

# ============================================================
# landau.dat loader (robust)
# ============================================================
def load_landau_params(file_path: str):
    with open(file_path, "r") as f:
        header_lines = []
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            else:
                break
    header = "\n".join(header_lines)

    m = re.search(r"blocks_map\s*[:=]\s*(\{.*\})", header, flags=re.DOTALL)
    if not m:
        raise ValueError("Header with 'blocks_map={...}' not found.")
    blocks_map = ast.literal_eval(m.group(1))

    data = np.loadtxt(file_path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    idx = data[:, 0].astype(int)
    val = data[:, 1].astype(float)
    order = np.argsort(idx)
    x = val[order]
    return x, blocks_map, header

def theta_of_T(order: int, T: float, x: np.ndarray, blocks_map: dict) -> np.ndarray:
    i0, ln = blocks_map[order]
    p = x[i0:i0 + ln]

    if order == 2:
        if ln != 3:
            raise ValueError(f"order=2 expects len=3, got {ln}")
        return np.array([p[0] + p[1] * T + p[2] * (T * T)], dtype=float)

    if order == 4:
        if ln != 6:
            raise ValueError(f"order=4 expects len=6, got {ln}")
        a11 = p[0] + p[1] * T + p[2] * (T * T)
        a12 = p[3] + p[4] * T + p[5] * (T * T)
        return np.array([a11, a12], dtype=float)

    if order == 6:
        if ln != 6:
            raise ValueError(f"order=6 expects len=6, got {ln}")
        a111 = p[0] + p[1] * T
        a112 = p[2] + p[3] * T
        a123 = p[4] + p[5] * T
        return np.array([a111, a112, a123], dtype=float)

    return p.astype(float)

# ============================================================
# Local Landau energy using ALL orders present in blocks_map
# ============================================================
def floc_landau_all_orders(P, dx, T, x, blocks_map):
    dV = dx ** 3
    Px, Py, Pz = P[..., 0], P[..., 1], P[..., 2]
    Px2, Py2, Pz2 = Px * Px, Py * Py, Pz * Pz

    floc = np.zeros(Px.shape, dtype=np.float64)

    if 2 in blocks_map:
        a1 = theta_of_T(2, T, x, blocks_map)[0]
        floc += a1 * (Px2 + Py2 + Pz2)

    for order in sorted(blocks_map.keys()):
        if order == 2:
            continue
        if order not in INV_FUN_P:
            raise ValueError(f"No invariant function implemented for order={order}")
        theta = theta_of_T(order, T, x, blocks_map)
        invs = INV_FUN_P[order](Px2, Py2, Pz2)
        if invs.shape[-1] != theta.shape[0]:
            raise ValueError(f"order={order}: inv dim {invs.shape[-1]} != theta dim {theta.shape[0]}")
        floc += np.tensordot(invs, theta, axes=([-1], [0]))

    return dV * np.sum(floc)

# ============================================================
# Polarization field: cos mode along z
# ============================================================
GRIDDIM = (3, 3, 12)

def Pfield(mode, wave, amp, griddim=GRIDDIM):
    Nx, Ny, Nz = griddim
    P = np.zeros((Nx, Ny, Nz, 3), dtype=np.float64)
    comp = mode - 1
    z = np.arange(Nz, dtype=np.float64)
    s = np.cos(2.0 * np.pi * wave * z / Nz)
    P[..., comp] = amp * s[None, None, :]
    return P

# ============================================================
# Long-range electrostatics via FFT (periodic Poisson)
# ============================================================
def felec_fft(P, dx, EPS0, epsb=1.0):
    Nx, Ny, Nz, _ = P.shape
    dV = dx ** 3
    N = Nx * Ny * Nz

    Px, Py, Pz = P[..., 0], P[..., 1], P[..., 2]
    Phx = np.fft.fftn(Px)
    Phy = np.fft.fftn(Py)
    Phz = np.fft.fftn(Pz)

    kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=dx)

    kx3 = kx[:, None, None]
    ky3 = ky[None, :, None]
    kz3 = kz[None, None, :]

    k2 = (kx3 * kx3) + (ky3 * ky3) + (kz3 * kz3)
    kdotP = kx3 * Phx + ky3 * Phy + kz3 * Phz

    phi = np.zeros_like(kdotP, dtype=np.complex128)
    mask = (k2 != 0.0)
    phi[mask] = (-1j) * kdotP[mask] / (EPS0 * epsb * k2[mask])

    Ex = (-1j) * kx3 * phi
    Ey = (-1j) * ky3 * phi
    Ez = (-1j) * kz3 * phi

    E2 = (np.abs(Ex) ** 2 + np.abs(Ey) ** 2 + np.abs(Ez) ** 2)
    return 0.5 * EPS0 * epsb * (dV / N) * np.sum(E2).real

# ============================================================
# Gradient stencil features (LINEAR) for axis bonds: shift=1,2 only
# ============================================================
def grad_features_S_axis(P, shift=1):
    """
    shift=1: nearest neighbor (+1)
    shift=2: 2nd neighbor (+2)
    Returns S=[S1,S2,S3,S4] for that bond-length along x,y,z.
    """
    Px, Py, Pz = P[..., 0], P[..., 1], P[..., 2]

    Px_x = np.roll(Px, shift=-shift, axis=0)
    Py_x = np.roll(Py, shift=-shift, axis=0)
    Pz_x = np.roll(Pz, shift=-shift, axis=0)

    Px_y = np.roll(Px, shift=-shift, axis=1)
    Py_y = np.roll(Py, shift=-shift, axis=1)
    Pz_y = np.roll(Pz, shift=-shift, axis=1)

    Px_z = np.roll(Px, shift=-shift, axis=2)
    Py_z = np.roll(Py, shift=-shift, axis=2)
    Pz_z = np.roll(Pz, shift=-shift, axis=2)

    # +x bond: parallel=Px, perps=Py,Pz
    dpx = Px_x - Px
    dpy = Py_x - Py
    dpz = Pz_x - Pz
    S1 = np.sum(dpx * dpx)
    S2 = np.sum(dpy * dpy + dpz * dpz)
    S3 = np.sum(dpx * dpy + dpx * dpz)
    S4 = np.sum(dpy * dpz)

    # +y bond: parallel=Py, perps=Px,Pz
    dpy = Py_y - Py
    dpx = Px_y - Px
    dpz = Pz_y - Pz
    S1 += np.sum(dpy * dpy)
    S2 += np.sum(dpx * dpx + dpz * dpz)
    S3 += np.sum(dpy * dpx + dpy * dpz)
    S4 += np.sum(dpx * dpz)

    # +z bond: parallel=Pz, perps=Px,Py
    dpz = Pz_z - Pz
    dpx = Px_z - Px
    dpy = Py_z - Py
    S1 += np.sum(dpz * dpz)
    S2 += np.sum(dpx * dpx + dpy * dpy)
    S3 += np.sum(dpz * dpx + dpz * dpy)
    S4 += np.sum(dpx * dpy)

    return np.array([S1, S2, S3, S4], dtype=np.float64)

def grad_features_linear_1nn_2nn(P):
    """
    Linear model features:
      [S1..S4 (1nn), S1..S4 (2nn)]  => 8 features
    """
    S1 = grad_features_S_axis(P, shift=1)
    S2 = grad_features_S_axis(P, shift=2)
    return np.concatenate([S1, S2], axis=0)

# ============================================================
# Main
# ============================================================
def main():
    T_fit = 420.0
    x_landau, blocks_map, _header = load_landau_params("landau.dat")

    files = {
        (1, 1): "US_d1_m1/wham_fes_1d_withV.dat",
        (1, 2): "US_d1_m2/wham_fes_1d_withV.dat",
        (1, 3): "US_d1_m3/wham_fes_1d_withV.dat",
        (1, 4): "US_d1_m4/wham_fes_1d_withV.dat",
        (1, 5): "US_d1_m5/wham_fes_1d_withV.dat",
        (1, 6): "US_d1_m6/wham_fes_1d_withV.dat",
        # (3, 1): "US_d3_m1/wham_fes_1d_withV.dat",
        # (3, 2): "US_d3_m2/wham_fes_1d_withV.dat",
        # (3, 3): "US_d3_m3/wham_fes_1d_withV.dat",
        # (3, 4): "US_d3_m4/wham_fes_1d_withV.dat",
        # (3, 5): "US_d3_m5/wham_fes_1d_withV.dat",
        # (3, 6): "US_d3_m6/wham_fes_1d_withV.dat",
    }

    dx = 16.09434994
    FOURPI_EPS0_INV = 14.399645478
    EPS0 = 1.0 / (4.0 * np.pi * FOURPI_EPS0_INV)
    epsb = 1.0

    # load WHAM
    data = {}
    for (mode, wave), filepath in files.items():
        d = np.loadtxt(filepath)
        amp = d[:, 0]
        F = d[:, 2]
        m = np.isfinite(amp) & np.isfinite(F)
        data[(mode, wave)] = (amp[m], F[m])

    # training set + records
    X_list, y_list = [], []
    plot_records = {}

    for (mode, wave), (amps, F) in data.items():
        i0 = int(np.argmin(np.abs(amps)))
        F_rel = F - F[i0]

        P0 = Pfield(mode, wave, amps[i0])
        Floc0 = floc_landau_all_orders(P0, dx, T_fit, x_landau, blocks_map)
        Felec0 = felec_fft(P0, dx, EPS0=EPS0, epsb=epsb)

        rec = []
        for amp, Fsample in zip(amps, F_rel):
            P = Pfield(mode, wave, amp)

            Floc = floc_landau_all_orders(P, dx, T_fit, x_landau, blocks_map)
            Felec = felec_fft(P, dx, EPS0=EPS0, epsb=epsb)

            y_target = (Fsample) - (Floc - Floc0) - (Felec - Felec0)

            S_all = grad_features_linear_1nn_2nn(P)  # 8 features
            xrow = dx * S_all                         # keep your scaling

            X_list.append(xrow)
            y_list.append(y_target)
            rec.append({"amp": float(amp), "y": float(y_target), "xrow": xrow.copy()})

        plot_records[(mode, wave)] = rec

    X = np.vstack(X_list)                   # (Nsamp, 8)
    y = np.array(y_list, dtype=np.float64)  # (Nsamp,)

    # fit 8 params
    J_fit, residuals, rank, svals = np.linalg.lstsq(X, y, rcond=None)

    print("=== Fit result (linear, 1nn+2nn along axes) ===")
    print("J_fit = [J1..J4,  J1_2..J4_2]")
    print("J_fit =", J_fit)
    print("rank =", rank)
    if residuals.size > 0:
        print("SSE =", float(residuals[0]))

    y_pred_all = X @ J_fit
    rmse = float(np.sqrt(np.mean((y_pred_all - y) ** 2)))
    print("RMSE (target grad energy) =", rmse)

    # plots
    outdir = "plots_amp_vs_G_1nn_2nn_linear"
    os.makedirs(outdir, exist_ok=True)

    for (mode, wave), rec in plot_records.items():
        amps = np.array([r["amp"] for r in rec], dtype=float)
        y_t  = np.array([r["y"]   for r in rec], dtype=float)
        Xc   = np.vstack([r["xrow"] for r in rec])
        y_p  = Xc @ J_fit

        idx = np.argsort(amps)
        amps, y_t, y_p = amps[idx], y_t[idx], y_p[idx]

        plt.figure(figsize=(6,6/np.sqrt(2)))
        plt.plot(amps*16, y_t, marker=".", linestyle="None", label="Data")
        plt.plot(amps*16, y_p, linestyle="-", label="Pred")
        plt.xlabel("Amplitude (C/m$^2$)")
        plt.ylabel("$\Delta G$ (eV)")
        plt.legend()
        plt.ylim(-0.1, 3.0)
        plt.xlim(-0.08, 0.08)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"amp_vs_G_mode{mode}_wave{wave}.png"), dpi=200)
        plt.close()

    print(f"Saved plots to: {outdir}/ (PNG)")

if __name__ == "__main__":
    main()
