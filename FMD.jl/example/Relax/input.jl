using FMD
using FMD: UNITS

# Structure relaxation with FIRE 2.0.
#
# Finds the equilibrium structure of a fitted potential: the atomic
# positions, and optionally the cell, at which the forces and the stress
# meet the requested tolerances. Typically run between fitting and
# production MD, so the MD starts from the force field's own equilibrium
# rather than from whatever the training data happened to contain.
#
# Two things distinguish this from a "minimise the energy" job and are worth
# understanding before reading the results:
#
#   * Convergence is judged on FORCES and STRESS, never on an energy
#     difference. A soft mode leaves the energy stationary to first order
#     while the forces along it are still finite, so an energy criterion
#     reports convergence on exactly the direction that has not converged.
#
#   * A converged verdict is re-checked with both numerical solvers
#     tightened (the shell CG and the PME mesh). If the criterion does not
#     survive, the run does not accept it at that point — it adopts the
#     tightened solvers and keeps relaxing, then judges again (so an
#     escalated run can still finish converged). Watch for that message in
#     the log: it means the :cgtol / :pme_meshsize you asked for were too
#     loose for your :ftol, and you should set them tighter from the start.
#
# Single-file potential written by FMD.app[:FitACE] (or :FitSpline).
pot = FMD.loadpotential("./ACEP.jld2")

params = (;
     pot...,

     output_dir = "out",
     input = "input.xyz",

     # --- numerical knobs -------------------------------------------------
     # Finer than the NPT-MD example's 2.0. That value is sized for MD, where
     # the forces being resolved are thermal (~1e-2 Ha/Bohr); here the
     # criterion is 1e-4, and the PME mesh error is one of the two solver
     # errors :ftol has to clear. If the log reports that the run had to
     # tighten the solvers mid-flight, this is the knob to tighten first.
     pme_meshsize = 1.0,
     skin = 0.2,
     # Tighter than the MD default: :ftol has to clear the O(cgtol) error the
     # inner shell relaxation leaves in the core forces, by the factor
     # :ftol_cgtol_margin (default 10). The entry guard refuses combinations
     # that cannot work rather than letting the run spin to :maxsteps.
     cgtol = 2e-6,
     cgmaxiter = 50,

     # --- convergence -----------------------------------------------------
     # max|F| in Hartree/Bohr. 1e-4 Ha/Bohr ≈ 5e-3 eV/Å.
     ftol = 1e-4,
     # Reaching :maxsteps is reported as NOT converged and the partially
     # relaxed structure is still written; FMD.app[:Relax] returns 1 so a
     # batch driver can branch on it.
     maxsteps = 2000,

     # --- FIRE step control -----------------------------------------------
     # :dt0 / :dt_max mean what :timestep means in MD (the pseudo-dynamics is
     # mass-weighted), so the MD intuition transfers.
     dt0 = 1.0/UNITS[:fs],
     dt_max = 10.0/UNITS[:fs],
     dmax = 0.1/UNITS[:Å],     # per-step displacement cap

     output_interval = 50,
)

params = (; FMD.default_params(Val(:Relax), params)..., params...)

# Returns 0 when converged, 1 when not.
FMD.app[:Relax](params)

# ---------------------------------------------------------------------------
# Relaxing the cell as well
# ---------------------------------------------------------------------------
#
# Add the block below to let the cell relax together with the atoms (one
# joint FIRE, not alternating atom/cell passes — the two are strongly
# coupled in soft-mode systems and alternating converges slowly into shallow
# minima).
#
#     cell_dof = :iso,          # :none | :iso | :diag | :full
#     target_stress = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     stol = 1e-7,              # max|P(σ − σ_target)|, atomic units
#     emax = 1e-3,              # per-step strain cap
#
# :cell_dof picks which cell degrees of freedom move:
#
#     :none   fixed cell (the default)
#     :iso    hydrostatic only — volume relaxes, cell shape is held
#     :diag   the three axes relax independently, no shear
#     :full   all six components
#
# Note that :iso is NOT "the three diagonal components are free": those are
# :diag, where a, b and c relax separately and a cubic crystal under an
# anisotropic target would stop being cubic. :iso constrains the strain to a
# multiple of the identity.
#
# :target_stress is a STRESS in Voigt-6 order [xx, yy, zz, yz, xz, xy] with
# TENSION POSITIVE, the same convention as npt!. A hydrostatic pressure P
# therefore enters as [-P, -P, -P, 0, 0, 0]; with P in GPa that is
# -P/UNITS[:GPa].
#
# Two limitations to know about before using it:
#
#   * The strain is parameterised as a symmetric tensor, which is what keeps
#     the three rigid-rotation degrees of freedom out of the cell force. A
#     consequence is that a rigid ROTATION of the input cell is not relaxed
#     away: a sheared cell will come back cubic in its metric (L·Lᵀ) while
#     the lattice matrix stays visibly non-diagonal. That is the same
#     structure, expressed in a rotated frame.
#
#   * A NON-ZERO external field (:efield) cannot be combined with a moving
#     cell; the run refuses the combination rather than producing a wrong
#     answer. The field term contributes force but neither energy nor
#     virial, so its strain derivative is missing from the stress and the
#     cell would relax to the wrong shape. The zero default is never in the
#     way, and with :cell_dof = :none a field is fine.
#
# ---------------------------------------------------------------------------
# Reading the result
# ---------------------------------------------------------------------------
#
# out/relaxed.xyz  the final structure, written whether or not it converged
# out/log.txt      per step: step | max|F| | energy | σ(6) | volume | dt |
#                  α | ncg | lattice(9). Units: Ha/Bohr, Ha, GPa, Å³, Å.
#
# A HIGH-SYMMETRY starting structure is a stationary point by symmetry, so
# the run can stop at step 0 having done nothing — and a stationary point
# may be a SADDLE rather than a minimum. The run warns when the input
# already satisfies the criteria.
#
# That is why the bundled input.xyz is a RATTLED cell (0.05 Å rms) rather
# than the ideal cubic perovskite: the ideal cell relaxes to itself and
# would demonstrate nothing, and for a ferroelectric it is a saddle in the
# first place. `make_input.jl` regenerates it from the ideal structure.
#
# The same reasoning applies to any result: to check that what you got is a
# minimum and not a saddle, re-relax from a slightly randomised copy and see
# whether it comes back to the same structure.
#
#     using FMD.Cores
#     atoms = readxyz(open("out/relaxed.xyz", "r"))[1]
#     atoms.arrays[:pos] .+= 1e-3 .* randn(size(atoms.arrays[:pos]))
#
# and write that out as the input of a second run.
#
# Reference: Bitzek et al., Phys. Rev. Lett. 97, 170201 (2006);
# Guénolé et al., Comput. Mater. Sci. 175, 109584 (2020).
