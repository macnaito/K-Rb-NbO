#!/usr/bin/env python3
"""Frequency-dependent permittivity from the MD dipole time series (post).

Linear response at fixed cell ("clamped"), tin-foil boundary (the Ewald/PME
convention of FMD.jl), from the gauge-fixed physical dipole M(t)
(`dipole_phys` in log.txt, written every step):

    eps_a(w) = 1 + [ C_a(0) - i*w * Int_0^tmax C_a(t) e^{-iwt} dt ]
                   / (eps0 * V * kB * T)          (per axis a = x, y, z)

    C_a(t) = < dM_a(0) dM_a(t) >,   dM = M - <M>

reported in the eps(w) = eps' - i*eps'' convention (eps'' >= 0 is loss).
The isotropic average (the printed/plotted value) averages the three axis
susceptibilities, equivalent to the familiar 3*eps0*V*kB*T denominator.

The electronic contribution eps_inf - 1 is deliberately IGNORED here: the
shells follow the cores adiabatically, so the equilibrium fluctuations of
M only contain the electronic response *slaved to ionic motion*, and the
frozen-core electronic response would have to be added separately. The
ionic GHz-THz features targeted here are unaffected, but note that eps'
consequently plateaus toward 1 (not eps_inf) above the phonon response.
If you need eps_inf, apply a small external field (params[:efield]),
relax the shells, and take the linear response dM/E at fixed cores.

Usage:
    python3 dielectric.py                    # reads out_prod/log.txt
    python3 dielectric.py --tmax-ps 20 --discard 0.2
"""
import argparse

import numpy as np

E_CHARGE = 1.602176634e-19   # C
EPS0 = 8.8541878128e-12      # F/m
KB = 1.380649e-23            # J/K


def acf(x, nlag):
    """Unbiased autocorrelation <x(0)x(k)>, k = 0..nlag-1, via FFT."""
    n = len(x)
    m = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, m)
    c = np.fft.irfft(f * np.conj(f), m)[:nlag]
    return c / (n - np.arange(nlag))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("logfile", nargs="?", default="out_prod/log.txt")
    p.add_argument("--dt-fs", type=float, default=1.0,
                   help="MD timestep in fs — must match input_prod.jl (default 1.0)")
    p.add_argument("--discard", type=float, default=0.1,
                   help="fraction of the run discarded as re-thermalization (default 0.1)")
    p.add_argument("--tmax-ps", type=float, default=10.0,
                   help="ACF truncation time in ps (default 10; capped at half the run)")
    p.add_argument("--fmin-thz", type=float, default=0.02)
    p.add_argument("--fmax-thz", type=float, default=50.0)
    p.add_argument("--nfreq", type=int, default=200)
    p.add_argument("-o", "--out-prefix", default="dielectric",
                   help="output prefix: <prefix>.csv, <prefix>.png (default dielectric)")
    args = p.parse_args()

    # log.txt columns (0-based): 0 temp [K] | 1-9 lattice [Å] |
    # 10-12 dipole [e·Å] | 13-15 dipole_phys [e·Å] | 16 volume [Å³] |
    # 17 conserved | 18 ncg
    data = np.loadtxt(args.logfile)
    data = data[int(args.discard * len(data)):]
    temp = data[:, 0].mean()
    vol = data[:, 16].mean() * 1e-30                     # m³ (constant in NVT)
    M = data[:, 13:16] * (E_CHARGE * 1e-10)              # C·m
    ps = np.linalg.norm(M.mean(axis=0)) / vol            # spontaneous polarization
    dM = M - M.mean(axis=0)

    dt = args.dt_fs * 1e-15
    nlag = min(int(args.tmax_ps * 1e-12 / dt), len(dM) // 2)
    C = np.stack([acf(dM[:, a], nlag) for a in range(3)])   # (3, nlag), (C·m)²
    t = np.arange(nlag) * dt
    # Cosine roll-off over the last 20% of the window only: it suppresses
    # truncation ringing, while tapering the *whole* window would break
    # the C(0) ↔ -iωL cancellation and shift eps by ~Delta*(tau/tmax).
    t1 = 0.8 * t[-1]
    w = np.ones(nlag)
    tail = t >= t1
    w[tail] = 0.5 * (1.0 + np.cos(np.pi * (t[tail] - t1) / (t[-1] - t1)))

    pref = 1.0 / (EPS0 * vol * KB * temp)
    f_thz = np.geomspace(args.fmin_thz, args.fmax_thz, args.nfreq)
    omega = 2e12 * np.pi * f_thz
    # L_a(w) = Int_0^tmax C_a(t) w(t) e^{-iwt} dt — trapezoid weights folded
    # into a matmul so no (3, nfreq, nlag) temporary is formed.
    w_trap = dt * np.r_[0.5, np.ones(nlag - 2), 0.5]
    L = (C * w * w_trap) @ np.exp(-1j * np.outer(t, omega))   # (3, nfreq)
    chi = pref * (C[:, :1] - 1j * omega[None, :] * L)    # (3, nfreq) axis susceptibilities
    eps = 1.0 + chi.mean(axis=0)                         # isotropic average
    eps_static = 1.0 + pref * C[:, 0]                    # per-axis eps(0)

    print(f"T = {temp:.1f} K,  V = {vol * 1e30:.1f} Å³,  "
          f"{len(dM)} samples ({len(dM) * args.dt_fs / 1e3:.1f} ps), "
          f"ACF truncated at {t[-1] * 1e12:.1f} ps")
    print(f"|<M>|/V = {ps:.4f} C/m²  (≈0 in the paraelectric phase; a large value "
          f"means a polarized run — eps(w) below is then not the paraelectric response)")
    print(f"static eps(0) per axis: {eps_static[0]:.1f} {eps_static[1]:.1f} "
          f"{eps_static[2]:.1f}  (isotropy check), mean {eps_static.mean():.1f}")

    header = ("f_THz,eps_re,eps_im,eps_re_x,eps_im_x,eps_re_y,eps_im_y,"
              "eps_re_z,eps_im_z  # eps = eps' - i*eps''; eps_im = +eps''")
    cols = [f_thz, eps.real, -eps.imag]
    for a in range(3):
        e_a = 1.0 + chi[a]
        cols += [e_a.real, -e_a.imag]
    np.savetxt(f"{args.out_prefix}.csv", np.column_stack(cols),
               delimiter=",", header=header)
    print(f"wrote {args.out_prefix}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipped the plot (CSV is complete)")
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(f_thz, eps.real, label=r"$\varepsilon'$")
    ax.plot(f_thz, -eps.imag, label=r"$\varepsilon''$")
    ax.set_xscale("log")
    ax.set_xlabel("frequency (THz)")
    ax.set_ylabel(r"$\varepsilon(\omega)$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}.png", dpi=180)
    print(f"wrote {args.out_prefix}.png")


if __name__ == "__main__":
    main()
