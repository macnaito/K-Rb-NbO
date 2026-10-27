# Stage 1/2 — NPT equilibration of cubic BaTiO₃, pure shell model (no MLIP).
#
# 400 K is comfortably above this potential's Curie point, so the cell
# stays paraelectric cubic and no ferroelectric domains form. The point
# of this run is (a) thermalization and (b) the time-averaged lattice
# ⟨h⟩ under P = 0, which input_prod.jl clamps the cell to.
#
# Run (from this directory, after `python3 makecell.py`):
#     julia --project=../.. input_equil.jl

using FMD
using FMD: UNITS

include("potential.jl")   # -> POT: Azuma 2025, R2SCAN-D4 (Ewald) BaTiO₃ set

dt = 1.0 / UNITS[:fs]     # MD timestep: 1 fs in a.u.

params = (;
    # Potential (everything that defines its physical shape).
    POT...,

    # Job I/O.
    input = "input.xyz",           # written by makecell.py
    output_dir = "out_equil",

    # Numerical knobs.
    pme_meshsize = 2.0,
    efield = [0.0, 0.0, 0.0],
    cgtol = 2e-5,                  # per-step adiabatic shell relaxation
    cgmaxiter = 50,
    skin = 0.2,

    # Time control: 20 ps.
    timestep = dt,
    nsteps = 20_000,
    output_interval = 2000,        # snapshots; log.txt is written every step

    # NPT(400 K, σ = 0). CSVR thermostat + SCR barostat: both first-order,
    # ringing-free relaxation to the targets. The compressibility only
    # sets the SCR cell relaxation rate — an order-of-magnitude bulk
    # modulus is enough.
    thermostat = :csvr,
    tau_t = 100dt,
    barostat = :scr,
    tau_p = 500dt,
    compressibility = UNITS[:GPa] / 160.0,   # κ_T from B ≈ 160 GPa
    target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    initial_temperature = 400.0,
    target_temperature = 400.0,
    seed = 1,

    # backend = :cpu,              # uncomment on machines without CUDA
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
