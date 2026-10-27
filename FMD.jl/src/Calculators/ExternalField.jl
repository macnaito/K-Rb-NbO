using KernelAbstractions
using ..FMD: Atoms

export ExternalField, calculate!

"""
    ExternalField

Constant external electric field `prms[:efield]` (3-vector, atomic units)
applied to cores and shells separately via their charges `qc[s]`, `qs[s]`.
No virial contribution. Use a zero vector to disable.
"""
struct ExternalField end

@kernel inbounds = true function kernel_efield!(::ExternalField, core_forces, shell_forces, species, qc, qs, efield)
    I = @index(Global)

    si = species[I]
    core_forces[I,1] += qc[si] * efield[1]
    core_forces[I,2] += qc[si] * efield[2]
    core_forces[I,3] += qc[si] * efield[3]
    shell_forces[I,1] += qs[si] * efield[1]
    shell_forces[I,2] += qs[si] * efield[2]
    shell_forces[I,3] += qs[si] * efield[3]
end

"""
    calculate!(::ExternalField, atoms::Atoms, prms::NamedTuple)

Add `qc[s] · E` to core forces and `qs[s] · E` to shell forces; no
contribution to energy or virial.
"""
function calculate!(::ExternalField, atoms::Atoms, prms::NamedTuple)
    backend = get_backend(atoms)
    arr = atoms.arrays
    kernel_efield!(backend, 512, length(atoms))(ExternalField(), arr[:forces], arr[:shellforces], arr[:elemid], prms[:qc], prms[:qs], prms[:efield])
end
