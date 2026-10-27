using FMD
using FMD: UNITS
using FMD.Calculators: PolarizationUmbrella, PolarizationGauge, save_gauge, load_gauge
using FMD.Cores: readxyz, setelemid!
using Printf
using CUDA
using JSON
import KernelAbstractions as KA

I = parse(Int, ARGS[1])

obj = JSON.parsefile("parameters.json")

temp = obj["Temp"]
kpar = obj["k_parallel"]
khomo = obj["k_homo"]
u = Float64[obj["u"]...]
cv = obj["windows"][I]
gridsize = Tuple(Int.(obj["gridsize"]))   # (Nx, Ny, Nz) of the cell.xyz supercell

# ----------------------------------------------------------------------

# Single-file potential — written by FMD.app[:FitACE].
pot = FMD.loadpotential("./ACEP_BCZT.jld2")

atoms_ref = readxyz(open("cell.xyz", "r"))[1]
setelemid!(atoms_ref, pot.elements)

# Polarisation zero-point.  `cell.xyz` from makecell.py is the ideal cubic
# perovskite — the paraelectric prototype — so it is exactly the structure
# that should count as P = 0, and every window builds the same gauge from
# the same file.  When the MD input is NOT the prototype (restarting from a
# polarised snapshot, seeding a domain wall, ...), build the gauge from the
# prototype frame instead and share one file across the windows:
#
#     g = PolarizationGauge(readxyz(open("prototype.xyz", "r"))[1],
#                           gridsize, pot.qc, pot.qs)
#     save_gauge("gauge.jld2", g)          # once, before launching windows
#     gauge = load_gauge("gauge.jld2")     # in every window
#
# Passing it as :polarization_gauge as well makes the dipole_phys column of
# log.txt use the same zero-point as the bias.
gauge = PolarizationGauge(atoms_ref, gridsize, pot.qc, pot.qs)

backend = CUDA.CUDABackend()

outdir = "out_us_$I"
dt = 1.0/UNITS[:fs]   # MD timestep: 1 fs in a.u.
params = (;
    pot...,                      # shell + acep + elements + rc

    input = "cell.xyz",
    output_dir = outdir,
    pme_meshsize = 2.0,
    efield = [0.0, 0.0, 0.0],
    cgtol = 2e-5,
    cgmaxiter = 50,
    skin = 0.2,
    timestep = dt,
    tau_t = 20dt,
    tau_p = 500dt,

    output_interval = 100,
    nsteps = 100,
    target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    final_temperature = 10.0,
    initial_temperature = 10.0,
    target_temperature = 10.0,

    # Uniform polarization bias: m = (0,0,0) means the LS amplitude `A`
    # equals the macroscopic polarization projected onto `u`.
    polarization_umbrella = PolarizationUmbrella(
        atoms_ref,
        kpar, khomo, cv, u, gridsize, pot.qc, pot.qs;
        gauge = gauge,
        m = (0, 0, 0), backend = backend,
    ),
    polarization_gauge = gauge,
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)

params = (;
    params...,
    input = "$outdir/step_100.xyz",
    output_interval = 1000,
    nsteps = 10_000,
    final_temperature = temp,
    initial_temperature = temp,
    target_temperature = temp,
)

FMD.app[:NPTMD](params)
