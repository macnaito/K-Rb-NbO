using LinearAlgebra
using KernelAbstractions
using ..FMD: FPPREC

export neighborlist

@kernel inbounds = true function kernel_naivecount!(counter, @Const(xcart), rc, ntotal)
    i = @index(Global)
    rc² = rc * rc
    xi = xcart[i, 1]
    yi = xcart[i, 2]
    zi = xcart[i, 3]
    cnt = Int32(0)
    for j in 1:ntotal
        rx = xi - xcart[j, 1]
        ry = yi - xcart[j, 2]
        rz = zi - xcart[j, 3]
        r² = rx * rx + ry * ry + rz * rz
        cnt += ifelse((r² < rc²) & (j != i), Int32(1), Int32(0))
    end
    counter[i] = cnt
end

@kernel inbounds = true function kernel_naivelistup!(list, @Const(xcart), rc, ntotal)
    i = @index(Global)
    rc² = rc * rc
    xi = xcart[i, 1]
    yi = xcart[i, 2]
    zi = xcart[i, 3]
    cnt = 0
    for j in 1:ntotal
        rx = xi - xcart[j, 1]
        ry = yi - xcart[j, 2]
        rz = zi - xcart[j, 3]
        r² = rx * rx + ry * ry + rz * rz
        if (r² < rc²) & (j != i)
            list[i, cnt+1] = j
            cnt += 1
        end
    end
end

"""
    neighborlist(atoms::Atoms, cutoff, skinratio) -> NeighborList

Naive O(N·(N+G)) ghost-based neighbour list — scans every owned atom
against every owned + ghost row without a cell list. The atoms are **not**
re-ordered (fitting datasets rely on stable row order). Handles cells
smaller than the cutoff via multi-layer ghost images. Production MD should
prefer the [`CellList`](@ref)-based [`neighborlist`](@ref) overload.

`rc = cutoff·(1 + skinratio)` is the effective cutoff used in the search.
"""
function neighborlist(atoms::Atoms, cutoff, skinratio)
    domain(atoms) isa MPIDomain &&
        error("neighborlist (naive builder): unsupported under domain decomposition — " *
              "it would treat the rank-local atoms as the whole system (silently " *
              "wrong forces); use the CellList-based builder")
    backend = get_backend(atoms)
    N = length(atoms)
    rc = FPPREC(cutoff * (1 + skinratio))

    g = build_ghosts(atoms, rc)
    sync_ghosts!(g, atoms)
    ntotal = length(g)

    counter = KernelAbstractions.zeros(backend, Int32, N)
    kernel_naivecount!(backend, 256, N)(counter, g.xcart, rc, ntotal)

    maxcount = N > 0 ? Int(reduce(max, counter)) : 0
    list = KernelAbstractions.zeros(backend, Int32, N, maxcount)
    kernel_naivelistup!(backend, 256, N)(list, g.xcart, rc, ntotal)

    return NeighborList(list, atoms.arrays[:pos] * atoms.info[:lattice],
                        cutoff * skinratio, rc, g)
end
