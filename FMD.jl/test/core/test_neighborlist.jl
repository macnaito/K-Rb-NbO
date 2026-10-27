# test/core/test_neighborlist.jl
using Test
using LinearAlgebra
using Random

# ghost_pairmap (the canonical pair-map comparator) lives in
# test/helpers.jl — it is shared by the core, GPU, apps, and MPI tests.

# Fixture: 12 atoms in a triclinic cell (rows = lattice vectors, Bohr).
# rc = 4.5 gives mixed interior grid dimensions (ncells = (2, 2, 1)).
function _make_triclinic_12()
    lat = FMD.FPPREC[10.0 0.0 0.0; 3.0 9.0 0.0; 1.0 2.0 8.0]
    rng = Random.MersenneTwister(7)
    pos = FMD.FPPREC.(rand(rng, 12, 3))
    return make_atoms(lat, pos, repeat([1, 2], 6))
end

# Fixture: nacl_8 with positions shifted out of [0,1) by whole lattice
# vectors — physically identical, exercises the frozen wrap offsets.
function _make_nacl8_unwrapped()
    a = make_nacl_8()
    a.arrays[:pos][1, :] .+= 1
    a.arrays[:pos][2, :] .-= 2
    a.arrays[:pos][3, 1] += 3
    return a
end

@testset "NeighborList" begin

    # ------------------------------------------------------------------
    # naive ↔ CellList cross-check: the two builders construct the same
    # ghost-based pair multiset through entirely different machinery
    # (all-pairs scan over the ghost images vs extended-grid cell binning
    # with a re-sort).
    # ------------------------------------------------------------------
    # rc must exceed the nearest-neighbour distance of each fixture (NaCl:
    # 0.5·a, perovskite: 0.5·a) — an empty pair set would make the
    # equivalence vacuous, which the !isempty guard below rejects.
    @testset "naive ↔ CellList equivalence: $name" for (name, mk, rcof, sk) in (
        ("NaCl 64, rc = 0.45L (ncells = 2)", make_nacl_64, a -> 0.45 * a.info[:lattice][1, 1], 0.0),
        ("NaCl 64, rc = 0.30L (ncells = 3)", make_nacl_64, a -> 0.30 * a.info[:lattice][1, 1], 0.0),
        ("NaCl 64, rc = 0.30L, skin 10%", make_nacl_64, a -> 0.30 * a.info[:lattice][1, 1], 0.1),
        ("NaCl 8, rc = 0.55a (ncells = 1, ±1 images)", make_nacl_8, a -> 0.55 * a.info[:lattice][1, 1], 0.0),
        ("perovskite 40, shuffled", () -> replicate_atoms(make_perovskite_5(), (2, 2, 2); shuffle_order = true),
         a -> 0.55 * a.info[:lattice][1, 1] / 2, 0.0),
        ("triclinic 12", _make_triclinic_12, a -> FMD.FPPREC(4.5), 0.0),
        ("multi-image: 5-atom cell, rc = 1.6a", make_perovskite_5, a -> 1.6 * a.info[:lattice][1, 1], 0.0),
        ("unwrapped NaCl 8", _make_nacl8_unwrapped, a -> 0.55 * NACL_A_BOHR, 0.0),
    )
        atoms = mk()
        rc = FMD.FPPREC(rcof(atoms))

        ref = ghost_pairmap(atoms, neighborlist(atoms, rc, sk))
        @test !isempty(ref)

        # new CellList path (atoms re-sorted; the id-keyed map is order-free).
        # The CellList is built with the padded cutoff, as production does.
        cl, atoms_s, _ = CellList(atoms, rc * (1 + sk))
        nl_cl = neighborlist(atoms_s, cl, rc, sk)
        @test ghost_pairmap(atoms_s, nl_cl) == ref
    end

    @testset "sub-filter rejects a cutoff beyond the parent's" begin
        atoms = make_nacl_64()
        rc = FMD.FPPREC(0.30 * atoms.info[:lattice][1, 1])
        parent = neighborlist(atoms, rc, 0.0)
        # a child cutoff beyond the parent cannot be served by filtering —
        # must error, not silently drop pairs (the production
        # build_mlip_nlists path feeds MLIP cutoffs into the Coulomb parent).
        @test_throws ErrorException neighborlist(atoms, parent, 1.5 * rc, 0.0)
    end

    @testset "CellList rejects a cutoff beyond its build cutoff" begin
        atoms = make_nacl_64()
        rc = FMD.FPPREC(0.30 * atoms.info[:lattice][1, 1])
        cl, atoms_s, _ = CellList(atoms, rc)
        # skin 20% on a CellList built without padding → ghost shell and
        # stencil would be too small; must error, not silently drop pairs.
        @test_throws ErrorException neighborlist(atoms_s, cl, rc, 0.2)
    end

    @testset "CellList re-sort locks setelemid! (Issue #70)" begin
        # The CellList constructor slices via atoms[idx], which stamps
        # :remap_locked into the shared info — the one production path
        # where a post-slice remap would silently desynchronize :elemid.
        atoms = make_nacl_8()
        rc = FMD.FPPREC(0.55 * atoms.info[:lattice][1, 1])
        _, atoms_s, _ = CellList(atoms, rc)
        @test haskey(atoms.info, :remap_locked)
        @test_throws ErrorException setelemid!(atoms_s, Dict("Na" => 1, "Cl" => 2))
        @test_throws ErrorException setelemid!(atoms, Dict("Na" => 1, "Cl" => 2))
    end

    @testset "FP edge: fractional coordinate that wraps to exactly 1.0" begin
        # pos = -eps/4 rounds to a wrapped coordinate of exactly 1.0 in FP.
        # The per-dimension sign clamp must keep the owned atom in an
        # interior cell (else its row is silently empty) and every ghost in
        # a halo cell (else the merged start/count hides owned rows).
        atoms = make_nacl_8()
        atoms.arrays[:pos][1, :] .= FMD.FPPREC[-eps(FMD.FPPREC)/4, 0.25, 0.25]
        rc = FMD.FPPREC(0.55 * atoms.info[:lattice][1, 1])

        ref = ghost_pairmap(atoms, neighborlist(atoms, rc, 0.0))
        cl, atoms_s, _ = CellList(atoms, rc)
        nl_cl = neighborlist(atoms_s, cl, rc, 0.0)
        @test ghost_pairmap(atoms_s, nl_cl) == ref
        @test all(i -> any(>(0), Array(nl_cl.list)[i, :]), 1:length(atoms))
    end

    @testset "ghost list: sub-filter equals directly built list" begin
        atoms = replicate_atoms(make_perovskite_5(), (2, 2, 2))
        a5 = atoms.info[:lattice][1, 1] / 2
        rc_parent = FMD.FPPREC(0.80 * a5)   # nearest (0.5a) + second (a/√2) shells
        rc_child = FMD.FPPREC(0.55 * a5)    # nearest shell only

        parent = neighborlist(atoms, rc_parent, 0.0)
        sub = neighborlist(atoms, parent, rc_child, 0.0)
        direct = neighborlist(atoms, rc_child, 0.0)
        @test !isempty(ghost_pairmap(atoms, direct))
        @test ghost_pairmap(atoms, sub) == ghost_pairmap(atoms, direct)
        # the sub-list shares the parent's ghosts (no rebuild)
        @test sub.ghosts === parent.ghosts
    end

    @testset "isreusable: skin threshold" begin
        atoms2 = deepcopy_atoms(make_nacl_8())
        nl = neighborlist(atoms2, 0.45 * atoms2.info[:lattice][1, 1], 0.05)
        # No move yet — must be reusable.
        @test isreusable(nl, atoms2)

        # Move atom 1 by less / more than half the skin (Cartesian Bohr).
        ilat = inv(Array(atoms2.info[:lattice]))
        small = 0.4 * nl.skin
        big = 1.5 * nl.skin
        atoms2.arrays[:pos][1, :] .+= vec(FMD.FPPREC[small 0 0] * ilat)
        @test isreusable(nl, atoms2)

        atoms2.arrays[:pos][1, :] .+= vec(FMD.FPPREC[big-small 0 0] * ilat)
        @test !isreusable(nl, atoms2)
    end
    @testset "change_arrtyp(::CellList) roundtrip (CPU)" begin
        # Pins the 6-positional-argument reconstruction (a start/count swap
        # or dropped scalar would corrupt every device DD list silently) on
        # the default tier — the GPU+MPI tier exercises the real device
        # conversion.
        atoms2 = deepcopy_atoms(make_nacl_8())
        rc2 = FMD.FPPREC(0.45 * atoms2.info[:lattice][1, 1])
        cl, atoms2, _ = CellList(atoms2, rc2)
        cl2 = Cores.change_arrtyp(cl, Array)
        @test cl2.start == cl.start
        @test cl2.count == cl.count
        @test cl2.ncells === cl.ncells
        @test cl2.halo === cl.halo
        @test cl2.rc_build === cl.rc_build
        g1, g2 = cl.ghosts, cl2.ghosts
        @test g2.n_owned == g1.n_owned && g2.n_ghost == g1.n_ghost
        @test g2.src == g1.src && g2.shift == g1.shift
        @test g2.elemid_ext == g1.elemid_ext
        @test g2.dd === nothing === g1.dd
        # The converted list is functional: the same neighbour list falls out.
        nl1 = neighborlist(atoms2, cl, rc2, 0.0)
        nl2 = neighborlist(atoms2, cl2, rc2, 0.0)
        @test nl1.list == nl2.list
    end
end
