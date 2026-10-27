using LinearAlgebra
using KernelAbstractions
using ..FMD: FPPREC

export neighborlist

@kernel inbounds = true function kernel_subcount!(counter, @Const(xcart), rc, @Const(nlist_src))
    I = @index(Global)
    rc² = rc * rc
    xi = xcart[I, 1]
    yi = xcart[I, 2]
    zi = xcart[I, 3]
    cnt = Int32(0)
    for slot in axes(nlist_src, 2)
        j = nlist_src[I, slot]
        j < 1 && break
        rx = xi - xcart[j, 1]
        ry = yi - xcart[j, 2]
        rz = zi - xcart[j, 3]
        r² = rx * rx + ry * ry + rz * rz
        cnt += ifelse(r² < rc², Int32(1), Int32(0))
    end
    counter[I] = cnt
end

@kernel inbounds = true function kernel_sublistup!(list, @Const(xcart), rc, @Const(nlist_src))
    I = @index(Global)
    rc² = rc * rc
    xi = xcart[I, 1]
    yi = xcart[I, 2]
    zi = xcart[I, 3]
    cnt = 0
    for slot in axes(nlist_src, 2)
        j = nlist_src[I, slot]
        j < 1 && break
        rx = xi - xcart[j, 1]
        ry = yi - xcart[j, 2]
        rz = zi - xcart[j, 3]
        r² = rx * rx + ry * ry + rz * rz
        if r² < rc²
            list[I, cnt+1] = j
            cnt += 1
        end
    end
end

"""
    neighborlist(atoms::Atoms, nl_src::NeighborList, cutoff, skinratio) -> NeighborList

Derive a smaller-cutoff neighbour list from an existing ghost-based parent
`nl_src` by distance-filtering its entries at
`rc = cutoff·(1 + skinratio)`. The derived list **shares the parent's
[`Ghosts`](@ref)** (no rebuild) — used by `build_mlip_nlists` to extract
the MLIP short-range lists from the Coulomb-cutoff parent without
re-traversing all pairs.
"""
function neighborlist(atoms::Atoms, nl_src::NeighborList, cutoff, skinratio)
    backend = get_backend(atoms)
    N = length(atoms)
    rc = FPPREC(cutoff * (1 + skinratio))
    if rc > nl_src.rc_build * (1 + sqrt(eps(FPPREC)))
        error("neighborlist (sub-filter): effective cutoff $(rc) exceeds the parent " *
              "list build cutoff $(nl_src.rc_build) — pairs outside the parent are " *
              "unreachable (silent pair loss). Build the parent with the larger cutoff.")
    end
    g = nl_src.ghosts
    sync_ghosts!(g, atoms)

    counter = KernelAbstractions.zeros(backend, Int32, N)
    kernel_subcount!(backend, 256, N)(counter, g.xcart, rc, nl_src.list)

    maxcount = N > 0 ? Int(reduce(max, counter)) : 0
    list = KernelAbstractions.zeros(backend, Int32, N, maxcount)
    kernel_sublistup!(backend, 256, N)(list, g.xcart, rc, nl_src.list)

    return NeighborList(list, atoms.arrays[:pos] * atoms.info[:lattice],
                        cutoff * skinratio, rc, g)
end
