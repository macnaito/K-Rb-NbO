# Stage 2/2 — NVT production run for ε(ω): cell clamped at the NPT ⟨h⟩.
#
# The dielectric fluctuation formula (README.md) is derived at fixed cell
# ("clamped" permittivity), and barostat-driven cell fluctuations would
# contaminate the dipole autocorrelation. So the cell is pinned to the
# *time-averaged* lattice of the equilibrated NPT segment — the final
# snapshot's instantaneous cell is a random draw from the cell
# distribution and would bias the strain.
#
# Run (after input_equil.jl has finished):
#     julia --project=../.. input_prod.jl

using FMD
using FMD: UNITS
using FMD.Cores: readxyz
using Printf

include("potential.jl")   # -> POT: Azuma 2025, R2SCAN-D4 (Ewald) BaTiO₃ set

# --- 1. ⟨h⟩ over the second half of the NPT log ------------------------
# log.txt columns: temp | lattice(9, column-major, Å) | dipole(3) |
# dipole_phys(3) | volume | conserved | ncg.
lines = readlines(joinpath("out_equil", "log.txt"))
rows = [parse.(Float64, split(l)) for l in lines[end÷2+1:end]]
all(r -> length(r) == 19, rows) ||
    error("out_equil/log.txt: expected 19 columns per line — the layout above changed?")
h_avg = reshape(sum(r[2:10] for r in rows) ./ length(rows), 3, 3)  # Å; rows = lattice vectors

abc = [sqrt(sum(abs2, h_avg[i, :])) for i in 1:3]
offd = maximum(abs(h_avg[i, j]) for i in 1:3, j in 1:3 if i != j)
@printf("⟨h⟩ over the last %d NPT steps:  |a|=%.4f  |b|=%.4f  |c|=%.4f Å\n",
    length(rows), abc...)
@printf("cubic check:  spread(|a|,|b|,|c|) = %.2e Å,  max|offdiag| = %.2e Å\n",
    maximum(abc) - minimum(abc), offd)

# --- 2. final NPT snapshot, affinely remapped to ⟨h⟩ -------------------
# readxyz stores positions fractionally, so pinning the lattice to ⟨h⟩
# IS the affine remap (fractional coordinates preserved). Velocities are
# not carried over: npt! redraws them from :seed whenever
# initial_temperature > 0, so only species + positions are written.
steps = [parse(Int, match(r"\d+", f).match)
         for f in readdir("out_equil") if occursin(r"^step_\d+\.xyz$", f)]
isempty(steps) &&
    error("no step_<n>.xyz in out_equil/ — input_equil.jl needs nsteps ≥ output_interval")
laststep = maximum(steps)
atoms = readxyz(open(joinpath("out_equil", "step_$(laststep).xyz"), "r"))[1]

syms = atoms.info[:species][atoms.arrays[:elemid]]
cart = atoms.arrays[:pos] * h_avg          # fractional × ⟨h⟩[Å] → Å
open("input_prod.xyz", "w") do io
    println(io, length(syms))
    print(io, "Lattice=\"")
    join(io, (@sprintf("%.10e", x) for x in vec(h_avg')), " ")  # row-major header
    println(io, "\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"")
    for i in eachindex(syms)
        @printf(io, "%-2s %20.12e %20.12e %20.12e\n", syms[i], cart[i, 1], cart[i, 2], cart[i, 3])
    end
end
println("wrote input_prod.xyz (step_$(laststep) remapped to ⟨h⟩)")

# --- 3. NVT production run ---------------------------------------------
dt = 1.0 / UNITS[:fs]

params = (;
    POT...,

    input = "input_prod.xyz",
    output_dir = "out_prod",

    pme_meshsize = 2.0,
    efield = [0.0, 0.0, 0.0],
    cgtol = 2e-5,
    cgmaxiter = 50,
    skin = 0.2,

    # 100 ps → dielectric.py resolves down to ~10 GHz. Extend nsteps for
    # more statistics / lower frequencies; log.txt is written every step.
    timestep = dt,
    nsteps = 100_000,
    output_interval = 10_000,

    # Fixed cell (clamped) + weakly coupled CSVR, so the thermostat does
    # not color the dipole dynamics. The purist's alternative is
    # thermostat = :none (NVE) — then watch the `conserved` column of
    # log.txt for drift.
    thermostat = :csvr,
    tau_t = 1000dt,
    barostat = :none,
    initial_temperature = 400.0,
    target_temperature = 400.0,
    seed = 2,                      # ≠ equil seed → independent velocity draw

    # backend = :cpu,              # uncomment on machines without CUDA
)

params = (; FMD.default_params(Val(:NPTMD), params)..., params...)

FMD.app[:NPTMD](params)
