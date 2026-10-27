import ase.io
import numpy as np
import matplotlib.pyplot as plt
import json

factor = 3.57106  # (e/bohr^2) -> (e/Å^2)

with open("parameters.json", "r") as f:
    obj = json.load(f)

TEMP = obj["Temp"]
KPAR = obj["k_parallel"]
windows = np.array(obj["windows"])*factor
u = np.array(obj["u"])
u /= np.linalg.norm(u)

# ----------------------------------------------------------------------

# --- reference polarization ---
qnominal = {"Ba": 2.0, "Ti": 4.0, "O": -2.0}
qref = 0.8915391361623672

ref = ase.io.read("./cell.xyz")
q = np.array([qref * qnominal[a.symbol] for a in ref])
r = ref.get_positions()
V = ref.get_volume()
Natom = ref.get_global_number_of_atoms()
assert V > 0, "Volume is zero. Check that cell.xyz contains cell/pbc."

Pref_vec = np.sum(q[:, None] * r, axis=0) / V  # e/Å^2
Pref = np.dot(Pref_vec, u)  # projection only (since CV is along u)

kB = 8.617333262145e-5/27.21138625  # Ha/K
kBT = kB * TEMP                         # Ha
ibetak = KPAR /(kB*TEMP) / factor**2    # (beta*k) in 1/(e/Å^2)^2

deltaP = 0.0001*factor
Pmax  = 0.010*factor

# define bin edges and centers consistently
edges   = np.arange(-Pmax, Pmax + deltaP, deltaP)
centers = 0.5 * (edges[:-1] + edges[1:])
nbins   = len(centers)

hist_i   = np.zeros((len(windows), nbins), dtype=np.float64)
Vhist_i  = np.zeros((len(windows), nbins), dtype=np.float64)  # <-- NEW: sum of volumes per P bin
w_i      = np.zeros((len(windows), nbins), dtype=np.float64)  # stores beta*w(s) (dimensionless)
N_i      = np.zeros(len(windows), dtype=np.int64)

for i, cv in enumerate(windows):
    # log.txt columns (0-based): 0 temp | 1-9 lattice | 10-12 dipole |
    # 13-15 dipole_phys | 16 volume | 17 conserved | 18 ncg. Positive
    # indices on purpose: negative ones silently shift when columns are
    # appended (this script read conserved as volume between #136 and #144).
    data = np.loadtxt(f"out_us_{i+1}/log.txt")
    if data.ndim != 2 or data.shape[1] < 17:
        raise RuntimeError(f"out_us_{i+1}/log.txt: expected >= 17 columns "
                           f"(temp..volume), got shape {data.shape}")
    dipoles = data[1000:, 13:16]
    volumes = data[1000:, 16]
    # A volume is strictly positive; catching a mislabeled column here is
    # what would have caught the #136 layout shift immediately.
    assert np.all(volumes > 0), "non-positive volume - wrong column?"

    Pvec = dipoles / volumes[:, None]   # e/Å^2
    Ppar = Pvec @ u - Pref              # projection and reference shift

    # histogram counts
    hist, _ = np.histogram(Ppar, bins=edges)
    hist_i[i, :] = hist
    N_i[i] = hist.sum()

    # <-- NEW: histogram of volumes in the same bins (sum of V in each P bin)
    Vsum, _ = np.histogram(Ppar, bins=edges, weights=volumes)
    Vhist_i[i, :] = Vsum

    # bias on the same bin centers: beta * (k/2) * (s-cv)^2
    w_i[i, :] = 0.5 * ibetak * (centers - cv)**2

fig, ax = plt.subplots()
for i in range(hist_i.shape[0]):
    ax.plot(centers, hist_i[i, :], label=f"window {i+1}")
fig.savefig("Hist.png", dpi=600)

# -----------------------------
# WHAM (weighted histogram) part
# -----------------------------

def logsumexp(a, axis=None):
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

nwin = len(windows)

H_tot = np.sum(hist_i, axis=0)       # total counts per P bin
V_tot = np.sum(Vhist_i, axis=0)      # <-- NEW: total volume sum per P bin

if np.any(N_i == 0):
    raise RuntimeError("Some windows have N_i == 0 after equilibration cutoff.")
if H_tot.sum() == 0:
    raise RuntimeError("All histograms are empty. Check input/log format and ranges.")

f = np.zeros(nwin, dtype=np.float64)  # dimensionless (beta*f_i)
logN = np.log(N_i.astype(np.float64))

max_iter = 20000
tol = 1e-10

for it in range(max_iter):
    log_terms = logN[:, None] + f[:, None] - w_i
    log_denom = logsumexp(log_terms, axis=0)
    denom = np.exp(log_denom)

    p = H_tot / denom
    psum = p.sum()
    if psum <= 0:
        raise RuntimeError("p.sum() <= 0. Overlap may be broken or biases too strong.")
    p /= psum

    logp = np.full_like(p, -np.inf)
    mask = p > 0
    logp[mask] = np.log(p[mask])

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

# Compute free energy F(P) from density: rho(P)=p_k/deltaP
rho = p / deltaP
F = np.full_like(rho, np.inf)
m = rho > 0
F[m] = -kBT * np.log(rho[m])   # Ha

# Shift minimum to zero (note: this makes F relative; absolute constant is undefined)
Fmin = np.min(F[np.isfinite(F)])
F -= Fmin

# -----------------------------
# NEW: conditional volume and free-energy density
# -----------------------------
V_cond = np.full_like(centers, np.nan, dtype=np.float64)  # <V|P> in Å^3
mH = H_tot > 0
V_cond[mH] = V_tot[mH] / H_tot[mH]

# g(P) = ΔG(P) / <V|P>  (relative free-energy density, Ha/Å^3)
g = np.full_like(centers, np.nan, dtype=np.float64)
mGV = np.isfinite(F) & (V_cond > 0)
g[mGV] = F[mGV] / V_cond[mGV]

# Unit conversions
Ha_to_eV = 27.21138625
Ha_to_kJmol = 2625.499638
HaA3_to_Jm3 = 4.3597447222071e-18 / 1.0e-30  # (J/Ha) / (m^3/Å^3) = J/m^3 per (Ha/Å^3)

F_eV = F * Ha_to_eV
F_kJmol = F * Ha_to_kJmol

g_Jm3 = g * HaA3_to_Jm3
g_GJm3 = g_Jm3 / 1e9

# Save results (now includes <V|P> and g)
out = np.column_stack([centers, F, F_eV, F_kJmol, rho, V_cond, g, g_Jm3, g_GJm3])
np.savetxt(
    "wham_fes_1d_withV.dat",
    out,
    header=("P(e/Å^2)  F(Ha,rel)  F(eV,rel)  F(kJ/mol,rel)  rho(1/(e/Å^2))  "
            "<V|P>(Å^3)  g=F/<V|P>(Ha/Å^3,rel)  g(J/m^3,rel)  g(GJ/m^3,rel)")
)
print("Saved: wham_fes_1d_withV.dat")

# Plot: F per atom as before
fig, ax = plt.subplots()
ax.plot(centers*16.02, F_eV/Natom)
ax.set_xlabel("P (C/m$^2$)")
ax.set_ylabel("Free energy (eV/atom) [relative]")
fig.savefig("wham_fes_1d.png", dpi=600, bbox_inches="tight")

# Plot: g(P) as energy density (GJ/m^3)
fig, ax = plt.subplots()
ax.plot(centers*16.02, g_GJm3)
ax.set_xlabel("P (C/m$^2$)")
ax.set_ylabel("g(P) = ΔG(P)/<V|P> (GJ/m$^3$) [relative]")
fig.savefig("wham_g_density.png", dpi=600, bbox_inches="tight")

# --- NEW: rescale window spacing based on current <P>(cv) mapping ---
Pmean = np.array([np.sum(centers*hist_i[i])/hist_i[i].sum() if hist_i[i].sum()>0 else np.nan
                  for i in range(len(windows))])

m = np.isfinite(Pmean)
x = windows[m]          # current window centers (e/Å^2)
y = Pmean[m]        # <P> at each window (e/Å^2)

# enforce monotonic y(x)
if y[-1] >= y[0]:
    y = np.maximum.accumulate(y)
else:
    y = np.minimum.accumulate(y)

# target: equally spaced in <P>, then invert by interpolation to get new windows
yt = np.linspace(y[0], y[-1], len(windows))
windows_new = np.interp(yt, y, x)  # e/Å^2

# ---- save in e/bohr^2 ----
conv = 1.0 / factor  # (e/Å^2) -> (e/bohr^2)
with open("suggested_windows.json", "w") as fjson:
    json.dump({
        "windows": (windows_new*conv).tolist()
    }, fjson, indent=4)
