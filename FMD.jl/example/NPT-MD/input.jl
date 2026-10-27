using FMD
using FMD: UNITS

# Single-file potential — written by FMD.app[:FitACE] (or :FitSpline).
# loadpotential validates the schema and returns a flat NamedTuple:
# qref, qc, qs, k2, k4, eta_c, eta_s, acep (or splp), elements, rc.
# (ACE radial cutoffs live inside the acep struct itself.)
# The MLIP is optional: a ShellParam.jld2 also loads (shell fields + rc,
# but NO :elements), and npt! then runs the pure shell model — in that
# case supply :elements and the Buckingham a/b/rho/c in params yourself.
pot = FMD.loadpotential("./ACEP.jld2")

dt = 1.0/UNITS[:fs]   # MD timestep: 1 fs in a.u.

params = (;
     # Potential (everything that defines its physical shape).
     pot...,

     # MD-time numerical knobs (NOT stored in the JLD2 — set per job).
     output_dir = "out",
     input = "input.xyz",
     pme_meshsize = 2.0,
     efield = [0.0, 0.0, 0.0],
     cgtol = 2e-5,
     cgmaxiter = 50,
     skin = 0.2,
     timestep = dt,
     tau_t = 20dt,
     tau_p = 500dt,
     # Ringing-free 1st-order alternative to the default MTTK barostat
     # (Issue #140) — uncomment both lines (κ_T here from B ≈ 24 GPa):
     # barostat = :scr,
     # compressibility = UNITS[:GPa] / 24.0,
     output_interval = 10,
     nsteps = 10_000,
     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
     target_temperature = 300.0,
     final_temperature = 300.0,
     initial_temperature = 300.0,
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
