import ase.io
import numpy as np
import matplotlib.pyplot as plt
import json
import os

with open("parameters.json", "r") as f:
    obj = json.load(f)

TEMP = float(obj["Temp"])
KPAR = float(obj["k_parallel"])  # (使わないが残しておく)
windows = np.array(obj["windows"], dtype=float)

kB = 8.617333262145e-5/27.21138625  # Ha/K
kBT = kB * TEMP

deltaP = 0.00003
Pmax  = 0.01

edges   = np.arange(-Pmax, Pmax + deltaP, deltaP)
centers = 0.5 * (edges[:-1] + edges[1:])
nbins   = len(centers)

hist_i   = np.zeros((len(windows), nbins), dtype=np.float64)
Vhist_i  = np.zeros((len(windows), nbins), dtype=np.float64)
w_i      = np.full((len(windows), nbins), np.inf, dtype=np.float64)  # default +inf
N_i      = np.zeros(len(windows), dtype=np.int64)

for i, cv in enumerate(windows):
    d = f"out_us_{i+1}"
    bias_path = os.path.join(d, "Bias.log")
    log_path  = os.path.join(d, "log.txt")

    if (not os.path.exists(bias_path)) or (not os.path.exists(log_path)):
        print(f"[skip] window {i+1}: missing Bias.log or log.txt")
        continue

    try:
        data_us = np.loadtxt(bias_path, skiprows=1, delimiter = ",")
        data    = np.loadtxt(log_path)
    except Exception as e:
        print(f"[skip] window {i+1}: load error: {e}")
        continue

    if data_us.ndim != 2 or data_us.shape[1] < 2:
        print(f"[skip] window {i+1}: Bias.log format unexpected")
        continue
    if data.ndim != 2 or data.shape[1] < 17:
        print(f"[skip] window {i+1}: log.txt format unexpected")
        continue

    cvtraj   = data_us[:, 0].astype(float)
    biastraj = data_us[:, 1].astype(float)           # assumed Ha
    # volume is column 16 (0-based) of log.txt; a positive index on
    # purpose — the trailing columns grew over time (#136 conserved,
    # #144 ncg) and data[:, -1] silently read conserved as volume.
    volumes  = data[:, 16].astype(float)             # Å^3
    if not np.all(volumes > 0):
        print(f"[skip] window {i+1}: non-positive volume - wrong column?")
        continue

    # align lengths safely (sampling frequency mismatch etc.)
    n = min(len(cvtraj), len(biastraj), len(volumes))
    if n <= 0:
        print(f"[skip] window {i+1}: no samples after alignment")
        continue
    cvtraj   = cvtraj[:n]
    biastraj = biastraj[:n]
    volumes  = volumes[:n]

    # histogram counts
    hist, _ = np.histogram(cvtraj, bins=edges)
    hist_i[i, :] = hist
    N_i[i] = int(hist.sum())
    if N_i[i] == 0:
        print(f"[skip] window {i+1}: empty histogram in range")
        continue

    # volume sum per bin
    Vsum, _ = np.histogram(cvtraj, bins=edges, weights=volumes)
    Vhist_i[i, :] = Vsum

    # ---- bias per bin: mean(beta*U) not sum(beta*U) ----
    betaU = biastraj / kBT
    bsum, _ = np.histogram(cvtraj, bins=edges, weights=betaU)

    w = np.full(nbins, np.inf, dtype=float)
    m = hist > 0
    w[m] = bsum[m] / hist[m]          # mean betaU in each visited bin
    # unvisited bins stay +inf -> exp(f - w)=0 (this window won't contribute there)
    w_i[i, :] = w

# --- keep only valid windows (N_i>0) ---
valid = N_i > 0
if not np.any(valid):
    raise RuntimeError("No valid windows found (no log files or empty histograms).")

windows_used = windows[valid]
hist_i  = hist_i[valid, :]
Vhist_i = Vhist_i[valid, :]
w_i     = w_i[valid, :]
N_i     = N_i[valid]

print(f"Using {len(windows_used)}/{len(windows)} windows.")

# Plot histograms for used windows only
fig, ax = plt.subplots(figsize=(10, 7))
for j in range(hist_i.shape[0]):
    ax.plot(centers, hist_i[j, :], label=f"window {np.where(valid)[0][j]+1}")
ax.set_xlabel("CV")
ax.set_ylabel("counts")
ax.legend(ncol=2, fontsize=9)
fig.tight_layout()
fig.savefig("Hist.png", dpi=600)

# -----------------------------
# WHAM
# -----------------------------
def logsumexp(a, axis=None):
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    # if all -inf in a slice, amax is -inf; exp(-inf - -inf) becomes exp(nan)
    # handle by masking
    with np.errstate(invalid="ignore"):
        out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

nwin = len(windows_used)
H_tot = np.sum(hist_i, axis=0)
V_tot = np.sum(Vhist_i, axis=0)

if H_tot.sum() == 0:
    raise RuntimeError("All histograms are empty. Check ranges (Pmax/deltaP) and CV units.")

f = np.zeros(nwin, dtype=np.float64)          # dimensionless (beta*f_i)
logN = np.log(N_i.astype(np.float64))

max_iter = 20000
tol = 1e-10

for it in range(max_iter):
    # denom_k = sum_i N_i * exp(f_i - w_i(i,k))
    log_terms = logN[:, None] + f[:, None] - w_i

    # bins where all windows have w=inf -> log_terms=-inf -> log_denom=-inf
    log_denom = logsumexp(log_terms, axis=0)
    denom = np.exp(log_denom)

    # p_k (bin probability, not density)
    p = np.zeros_like(H_tot, dtype=float)
    mden = np.isfinite(denom) & (denom > 0)
    p[mden] = H_tot[mden] / denom[mden]

    psum = p.sum()
    if psum <= 0:
        raise RuntimeError("p.sum() <= 0. Overlap may be broken or bias too strong or disconnected components.")
    p /= psum

    logp = np.full_like(p, -np.inf)
    mp = p > 0
    logp[mp] = np.log(p[mp])

    # exp(-f_i) = sum_k p_k * exp(-w_i(i,k))
    logZ = logsumexp(logp[None, :] - w_i, axis=1)
    f_new = -logZ
    f_new -= f_new[0]  # gauge

    df = np.max(np.abs(f_new - f))
    f = f_new
    if df < tol:
        print(f"WHAM converged: iter={it}, max|df|={df:.3e}")
        break
else:
    print(f"WHAM NOT fully converged: reached max_iter={max_iter}, last max|df|={df:.3e}")

# density and free energy
rho = p / deltaP
F = np.full_like(rho, np.inf)
m = rho > 0
F[m] = -kBT * np.log(rho[m])   # Ha

Fmin = np.min(F[np.isfinite(F)])
F -= Fmin

# conditional volume and free-energy density
V_cond = np.full_like(centers, np.nan, dtype=np.float64)
mH = H_tot > 0
V_cond[mH] = V_tot[mH] / H_tot[mH]

g = np.full_like(centers, np.nan, dtype=np.float64)
mGV = np.isfinite(F) & (V_cond > 0)
g[mGV] = F[mGV] / V_cond[mGV]          # Ha/Å^3

# Unit conversions
Ha_to_eV = 27.21138625
Ha_to_kJmol = 2625.499638
HaA3_to_Jm3 = 4.3597447222071e-18 / 1.0e-30  # (J/Ha)/(m^3/Å^3)

F_eV = F * Ha_to_eV
F_kJmol = F * Ha_to_kJmol
g_Jm3 = g * HaA3_to_Jm3
g_GJm3 = g_Jm3 / 1e9

out = np.column_stack([centers, F, F_eV, F_kJmol, rho, V_cond, g, g_Jm3, g_GJm3])
np.savetxt(
    "wham_fes_1d_withV.dat",
    out,
    header=("P(e/Å^2)  F(Ha,rel)  F(eV,rel)  F(kJ/mol,rel)  rho(1/(e/Å^2))  "
            "<V|P>(Å^3)  g=F/<V|P>(Ha/Å^3,rel)  g(J/m^3,rel)  g(GJ/m^3,rel)")
)
print("Saved: wham_fes_1d_withV.dat")

fig, ax = plt.subplots()
ax.plot(centers*16.02, g_GJm3)
ax.set_xlabel("P (C/m$^2$)")
ax.set_ylabel("g(P) = ΔG(P)/<V|P> (GJ/m$^3$) [relative]")
fig.tight_layout()
fig.savefig("wham_g_density.png", dpi=600, bbox_inches="tight")
