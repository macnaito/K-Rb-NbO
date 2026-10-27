using KernelAbstractions
import AcceleratedKernels as AK
using ..FMD: Atoms, FPPREC

"""
    Spring

Anisotropic harmonic + quartic spring binding each shell to its parent
core: `U = ½ k2[s] |Δ|² + (1/24) k4[s] |Δ|⁴` with
`Δ = atoms.arrays[:shelldisp]` (per-element).

Required `params`: `:k2`, `:k4` (each indexed by `elemid`).
"""
struct Spring end

export Spring, calculate!

@kernel inbounds = true function kernel_spring!(::Spring, potential_energies, virials, arrays::NamedTuple, k2, k4)
    I = @index(Global)

    si = arrays[:elemid][I]
    k2i = k2[si]
    k4i = k4[si]

    rx, ry, rz = -arrays[:shelldisp][I,1], -arrays[:shelldisp][I,2], -arrays[:shelldisp][I,3]
    dr2 = rx*rx + ry*ry + rz*rz

    potential_energies[I] += k2i*dr2/2 + k4i*dr2^2/24

    dϕdr_r = k2i + k4i*dr2/6

    arrays[:forces][I,1] -= dϕdr_r * rx
    arrays[:forces][I,2] -= dϕdr_r * ry
    arrays[:forces][I,3] -= dϕdr_r * rz
    arrays[:shellforces][I,1] += dϕdr_r * rx
    arrays[:shellforces][I,2] += dϕdr_r * ry
    arrays[:shellforces][I,3] += dϕdr_r * rz
    virials[I,1] -= dϕdr_r * rx * rx
    virials[I,2] -= dϕdr_r * ry * ry
    virials[I,3] -= dϕdr_r * rz * rz
    virials[I,4] -= dϕdr_r * ry * rz
    virials[I,5] -= dϕdr_r * rx * rz
    virials[I,6] -= dϕdr_r * rx * ry
end

"""
    calculate!(::Spring, atoms::Atoms, prms::NamedTuple)

Add the harmonic + quartic core-shell spring contribution to
`potential_energy`, `virial`, `forces`, and `shellforces`. Uses
`prms[:k2]`, `prms[:k4]` indexed by `atoms.arrays[:elemid]`.
"""
function calculate!(::Spring, atoms::Atoms, prms::NamedTuple)
    backend = get_backend(atoms)

    arrays = (
        forces = atoms.arrays[:forces],
        shellforces = atoms.arrays[:shellforces],
        shelldisp = atoms.arrays[:shelldisp],
        elemid = atoms.arrays[:elemid]
    )

    # An empty rank contributes zeros but must still ride the collective
    # inside add_reduced_ev! (kernel launch / AK.sum on zero atoms are
    # not exercised — same pattern as the PME/Ewald aggregation).
    if length(atoms) > 0
        potential_energies = KernelAbstractions.zeros(backend, FPPREC, length(atoms))
        virials = KernelAbstractions.zeros(backend, FPPREC, length(atoms), 6)

        kernel_spring!(backend, 512, length(atoms))(Spring(), potential_energies, virials, arrays, prms[:k2], prms[:k4])

        add_reduced_ev!(atoms, AK.sum(potential_energies), AK.sum(virials, dims = 1)[1, :])

        if haskey(atoms.arrays, :peratom_virial)
            atoms.arrays[:peratom_virial] .+= virials
        end
    else
        add_reduced_ev!(atoms, zero(FPPREC), zeros(FPPREC, 6))
    end
end
