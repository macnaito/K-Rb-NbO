export calcdipole, calcdipole!, calc_dipole_phys!, total_mfrac_ref

"""
    calcdipole(atoms::Atoms, prms::NamedTuple) -> Vector

Absolute (Cartesian) dipole moment

```
M_abs = Σ_i (qc[s_i] · r_i + qs[s_i] · (r_i + sd_i))
```

where `r_i = pos · L`, `sd_i` is the shell displacement, and `s_i` is
the per-atom species id. Uses `prms[:qc]`, `prms[:qs]`. Under domain
decomposition the owned partial sum is Allreduced, so the returned value
is the whole-system dipole on every rank (**collective**: every rank,
empty ones included, must call it).

Writes nothing — [`calcdipole!`](@ref) is the caching wrapper. Read-only
consumers that must not disturb `atoms.info` (the analytic HVP snapshot,
[`hvp_workspace`](@ref)) use this one.

The absolute dipole is gauge-dependent (depends on the choice of cell
origin); see [`calc_dipole_phys!`](@ref) for the physically meaningful,
cell-deformation-covariant version.
"""
function calcdipole(atoms::Atoms, prms::NamedTuple)
    dom = domain(atoms)
    if length(atoms) == 0
        dipole = zeros(FPPREC, 3)
    else
        # r = pos * lattice: rows of `lattice` are the cell vectors, the
        # convention shared with Cores/NeighborList.jl and the XYZ I/O.
        r = atoms.arrays[:pos] * atoms.info[:lattice]
        sd = atoms.arrays[:shelldisp]
        qcprm = prms[:qc]
        qc = KernelAbstractions.zeros(get_backend(atoms), FPPREC, length(atoms))
        AK.map!(si -> qcprm[si], qc, atoms.arrays[:elemid])
        qsprm = prms[:qs]
        qs = KernelAbstractions.zeros(get_backend(atoms), FPPREC, length(atoms))
        AK.map!(si -> qsprm[si], qs, atoms.arrays[:elemid])

        dipole = AK.sum(qc .* r .+ qs .* (r .+ sd), dims = 1)[:]
    end
    # Owned partial sum → whole-system value on every rank under DD.
    if dom isa MPIDomain
        dipole = allreduce_sum!(dom, Vector(FPPREC.(Array(dipole))))
    end
    return dipole
end

"""
    calcdipole!(atoms::Atoms, prms::NamedTuple) -> Vector

[`calcdipole`](@ref), cached in `atoms.info[:dipole]` and returned.
Collective under domain decomposition.
"""
function calcdipole!(atoms::Atoms, prms::NamedTuple)
    dipole = calcdipole(atoms, prms)
    atoms.info[:dipole] = dipole
    return dipole
end

"""
    total_mfrac_ref(atoms::Atoms, prms::NamedTuple) -> Vector{FPPREC}

Total fractional dipole moment

```
ΣM_frac = Σ_i (qc[s_i] · s_i + qs[s_i] · (s_i + sd_i · L⁻¹))
```

computed once at initialization. Multiplied by `Lᵀ(t)` at every NPT step
this yields a reference dipole that scales covariantly with the cell, so
[`calc_dipole_phys!`](@ref) reports zero whenever atoms move only through
homogeneous cell deformation (and non-zero only for genuine polarisation
changes).
"""
function total_mfrac_ref(atoms::Atoms, prms::NamedTuple)
    L_inv = inv(Array(atoms.info[:lattice]))
    s = Array(atoms.arrays[:pos])
    sd = Array(atoms.arrays[:shelldisp])
    # Cartesian → fractional is the row-wise inverse of r = pos * lattice.
    sd_frac = sd * L_inv

    elemid = Array(atoms.arrays[:elemid])
    qcprm = Array(prms[:qc])
    qsprm = Array(prms[:qs])

    M = zeros(FPPREC, 3)
    for i in eachindex(elemid)
        qc = qcprm[elemid[i]]
        qs = qsprm[elemid[i]]
        for d in 1:3
            M[d] += qc * s[i, d] + qs * (s[i, d] + sd_frac[i, d])
        end
    end
    # Owned partial sum → whole-system value on every rank under DD.
    return allreduce_sum!(domain(atoms), M)
end

"""
    calc_dipole_phys!(atoms::Atoms, prms::NamedTuple,
                      M_frac_ref_total::AbstractVector) -> Vector

Cell-deformation-covariant ("physical") dipole moment

```
M_phys(t) = M_abs(t) − Lᵀ(t) · M_frac_ref_total
```

Pass `M_frac_ref_total = total_mfrac_ref(atoms_init, prms)` taken once at
initialization; this is exactly the gauge that makes the answer zero at
`t = 0` (with the initial lattice) and that stays zero whenever atoms
move only through homogeneous cell deformation.

Stored in `atoms.info[:dipole_phys]` and also returned. This is the value
written to `log.txt` by [`npt!`](@ref FMD.Apps.npt!).
"""
function calc_dipole_phys!(atoms::Atoms, prms::NamedTuple, M_frac_ref_total::AbstractVector)
    M_abs = calcdipole!(atoms, prms)
    L = Array(atoms.info[:lattice])
    # Fractional → Cartesian for a column vector is transpose(L) * m, the
    # counterpart of the row-wise `pos * lattice` used in calcdipole!.
    M_ref_now = transpose(L) * Vector(FPPREC.(M_frac_ref_total))
    dphys = Array(M_abs) .- M_ref_now
    atoms.info[:dipole_phys] = dphys
    return dphys
end
