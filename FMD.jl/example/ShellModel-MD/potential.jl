# ----------------------------------------------------------------------
# BaTiO₃ shell-model potential — the r²SCAN-D4 (Ewald) set of
#
#   H. Azuma, T. Ogawa, S. Ogata, R. Kobayashi, M. Uranagase, T. Tsuzuki,
#   and F. Wendler, "Unique temperature-dependence of polarization
#   switching paths in ferroelectric BaTiO₃: A molecular dynamics
#   simulation study", Acta Mater. 296, 121216 (2025).
#   https://doi.org/10.1016/j.actamat.2025.121216
#
# The set was fitted against the Ewald Coulomb solver and must be used
# with it (FMD.jl's Ewald/PME path); Wolf-fitted sets are not
# interchangeable — the fitted short-range terms absorb the solver.
#
# Function form: FMD.jl evaluates the Buckingham part as
#
#   V(r) = a (1 + b·r) exp(-r/ρ) − c/r⁶
#
# Tables published as (A + B·r) exp(-r/ρ) − C/r⁶ map onto it with a = A,
# b = B/A; this set has B ≡ 0, so the zero default of b is kept. ρ = 0
# rows marking non-interacting pairs (a = 0) become ρ = 1 Å below: with
# a = 0 the value never enters the physics — it only has to be positive
# because ρ is a divisor in the kernel.
#
# Units: values are written in eV/Å as published and divided by UNITS
# factors at input time — inside FMD everything is atomic units.
# ----------------------------------------------------------------------

using FMD: UNITS

const POT = let eV = UNITS[:eV], Å = UNITS[:Å]
    (;
        elements = Dict("Ba" => 1, "Ti" => 2, "O" => 3),

        # Core / shell charges [e] (already atomic units).
        # Neutral per formula unit: Σ(qc+qs) = 0.
        qc = [4.784102643942809, 4.4875030954044135, 1.0392355943712448],
        qs = [-2.947525781026954, -1.6139336262971906, -2.609284371712271],

        # Core-shell springs: k2 [eV/Å²], k4 [eV/Å⁴].
        k2 = [309.83074929916336, 326.9064490703737, 46.37182988167516] ./ (eV / Å^2),
        k4 = [0.0, 0.0, 6956.922099508013] ./ (eV / Å^4),

        # Buckingham (shell-shell), pair-symmetric matrices indexed
        # (Ba,Ti,O)×(Ba,Ti,O): a [eV], rho [Å], c [eV·Å⁶].
        a = [0.0                0.0                1568.4172482044366;
             0.0                0.0                3139.4888756274936;
             1568.4172482044366 3139.4888756274936 2571.9734540421173] ./ eV,
        rho = [1.0                1.0                0.3470865265465664;
               1.0                1.0                0.25486635629935617;
               0.3470865265465664 0.25486635629935617 0.355472951917363] ./ Å,
        c = [0.0 0.0 0.0;
             0.0 0.0 0.0;
             0.0 0.0 528.6469973229774] ./ (eV * Å^6),

        # Cutoff of the fit [Å].
        rc = 11.0 / Å,
    )
end
