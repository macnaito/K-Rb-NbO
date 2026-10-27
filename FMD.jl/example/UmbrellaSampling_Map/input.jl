# Domain-wall preparation with a target polarization map.
#
# The bias restrains the local polarization towards a map you supply, so the
# configuration you want is stated directly rather than coaxed out of a field.
# Four stages, each an NPT run continuing from the previous one:
#
#   1. strong restraint at low temperature   — form the pattern
#   2. same restraint at the target temperature — equilibrate while held
#   3. decay k over several runs             — release gradually
#   4. k = 0                                 — unbiased equilibrium MD
#
# Run makecell.py first; it writes prototype.xyz, cell.xyz and target.json.

using FMD
using FMD: UNITS
using FMD.Calculators: PolarizationMapUmbrella, PolarizationGauge,
                       save_gauge, load_gauge, group_p_gauged
using FMD.Cores: readxyz, setelemid!
using CUDA
using JSON
import KernelAbstractions as KA

obj = JSON.parsefile("domain.json")
gridsize = Tuple(Int.(obj["gridsize"]))
ng = prod(gridsize)
domain_sign = Float64.(obj["domain_sign"])
weight = Float64.(obj["weight"])

temp = 300.0
k_hold = 5.0e6        # restraint strength while forming and holding
r_th = 0.0            # flat-bottom width; widening this is the alternative
                      # to decaying k when releasing

pot = FMD.loadpotential("./ACEP_BCZT.jld2")

# The polarization zero-point: the ideal cubic frame.  The SAME gauge must be
# used to measure the target map and to run the bias, otherwise the two refer
# to different origins and nothing reports the discrepancy — so it is written
# once and loaded here.
prototype = readxyz(open("prototype.xyz", "r"))[1]
setelemid!(prototype, pot.elements)
gauge_path = "gauge.jld2"
if !isfile(gauge_path)
    save_gauge(gauge_path, PolarizationGauge(prototype, gridsize, pot.qc, pot.qs))
end
gauge = load_gauge(gauge_path)

atoms0 = readxyz(open("cell.xyz", "r"))[1]
setelemid!(atoms0, pot.elements)

# The target map, measured rather than assumed.
#
# `group_p_gauged` returns exactly the quantity the bias restrains, so building
# the target with it is what keeps the two on the same footing (and the same
# gauge).  Here the seeded cell is measured, the z-polarization is averaged
# over the RESTRAINED interior of each domain — the released band around the
# walls is not part of the domain and would drag the average down — and the map
# is rebuilt from that one number times the domain sign.  Averaging is what
# turns a rough geometric seed into a uniform target.
#
# For production, measure this from a RELAXED single-domain run instead: the
# seed's displacement pattern is only an estimate of the real soft mode.
Pg_seed = group_p_gauged(atoms0, gauge, pot.qc, pot.qs)
interior = findall(>(0), weight)
P_domain = sum(abs.(Pg_seed[interior, 3])) / length(interior)
P_target = zeros(Float64, ng, 3)
P_target[:, 3] .= P_domain .* domain_sign
println("measured domain polarization = ", P_domain,
        "  (target map built from it)")

backend = CUDA.CUDABackend()
dt = 1.0 / UNITS[:fs]

# `sigma` coarse-grains the polarization over neighbouring unit cells before
# it is restrained.  A wall is a few cells wide, so smoothing over ~1 cell
# suppresses single-cell noise without smearing the wall itself.
mkbias(k, stage) = PolarizationMapUmbrella(
    atoms0, k, P_target, gridsize, pot.qc, pot.qs;
    gauge = gauge, weight = weight, sigma = 1.0, r_th = r_th,
    backend = backend, sumr2_log = "map_$(stage)_k$(k).log")

base = (;
    pot...,
    output_dir = "out_form",
    pme_meshsize = 2.0,
    efield = [0.0, 0.0, 0.0],
    cgtol = 2e-5,
    cgmaxiter = 50,
    skin = 0.2,
    timestep = dt,
    tau_t = 20dt,
    tau_p = 500dt,
    target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    output_interval = 500,
    polarization_gauge = gauge,     # dipole_phys reported against the same zero
    polmap_interval = 500,          # local polarization map, polmap_<n>.txt (same gauge)
    backend = :cuda,
)

# --- 1. form the pattern: strong restraint, cold ---------------------------
p = (; base...,
     input = "cell.xyz",
     output_dir = "out_form",
     nsteps = 2_000,
     initial_temperature = 10.0, target_temperature = 10.0,
     final_temperature = 10.0,
     polarization_umbrella = mkbias(k_hold, "form"))
p = (; FMD.default_params(Val(:NPTMD), p)..., p...)
FMD.app[:NPTMD](p)

# --- 2. heat to the target temperature, still held -------------------------
p = (; p...,
     input = "out_form/step_2000.xyz",
     output_dir = "out_hold",
     nsteps = 20_000,
     initial_temperature = temp, target_temperature = temp,
     final_temperature = temp,
     polarization_umbrella = mkbias(k_hold, "hold"))
FMD.app[:NPTMD](p)

# --- 3. release: decay k geometrically -------------------------------------
prev = "out_hold/step_20000.xyz"
for (i, k) in enumerate(k_hold .* [0.3, 0.1, 0.03, 0.01])
    outdir = "out_release_$i"
    p = (; p...,
         input = prev,
         output_dir = outdir,
         nsteps = 10_000,
         polarization_umbrella = mkbias(k, "release$i"))
    FMD.app[:NPTMD](p)
    global prev = "$outdir/step_10000.xyz"
end

# --- 4. unbiased equilibrium MD --------------------------------------------
# Drop the bias key entirely; the gauge stays so dipole_phys remains
# comparable across all four stages.
p_free = (; base...,
          input = prev,
          output_dir = "out_free",
          nsteps = 100_000,
          initial_temperature = temp, target_temperature = temp,
          final_temperature = temp)
p_free = (; FMD.default_params(Val(:NPTMD), p_free)..., p_free...)
FMD.app[:NPTMD](p_free)
