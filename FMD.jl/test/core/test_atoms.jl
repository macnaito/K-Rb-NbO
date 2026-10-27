# test/core/test_atoms.jl
using Test
using LinearAlgebra

@testset "Atoms" begin
    atoms = make_nacl_8()

    @testset "length / volume" begin
        @test length(atoms) == 8
        # 8-atom NaCl cubic cell: volume = a³ in Bohr³.
        @test isapprox(volume(atoms), NACL_A_BOHR^3; rtol = 1e-12)

        # det(lattice) agreement (rows = lattice vectors).
        L = atoms.info[:lattice]
        @test isapprox(volume(atoms), abs(det(L)); rtol = 1e-12)
    end

    @testset "kinetic_energy" begin
        a = deepcopy_atoms(atoms)
        # All masses set to a known value, all velocities to a known vector.
        a.arrays[:masses] .= 2.0
        v = [1.0  0.0  0.0;
             0.0  2.0  0.0;
             0.0  0.0  3.0;
             1.0  1.0  1.0;
             0.0  0.0  0.0;
             0.5 -0.5  0.5;
             1.0 -1.0  1.0;
             0.0  0.0  0.0]
        a.arrays[:velo] .= v
        # Kinetic energy = sum(masses .* velo²) / 2 in this codebase
        # (note: matches Atoms.jl:49 — uses elementwise velo², not row-norm).
        expected = sum(a.arrays[:masses] .* a.arrays[:velo].^2) / 2
        @test isapprox(kinetic_energy(a), expected; rtol = 1e-12)
    end

    @testset "stress" begin
        a = deepcopy_atoms(atoms)
        a.arrays[:masses] .= 1.0
        a.arrays[:velo] .= 0
        a.info[:virial] .= FMD.FPPREC[1.0, 2.0, 3.0, 0.4, 0.5, 0.6]
        s = stress(a, false)
        @test length(s) == 6
        # static stress = -virial / volume (when kinetic flag is false)
        v = volume(a)
        @test isapprox(s[1], -1.0 / v; rtol = 1e-12)
        @test isapprox(s[2], -2.0 / v; rtol = 1e-12)
        @test isapprox(s[3], -3.0 / v; rtol = 1e-12)
        @test isapprox(s[4], -0.4 / v; rtol = 1e-12)
        @test isapprox(s[5], -0.5 / v; rtol = 1e-12)
        @test isapprox(s[6], -0.6 / v; rtol = 1e-12)
    end

    @testset "clear!" begin
        a = deepcopy_atoms(atoms)
        a.info[:potential_energy] .= 7.0
        a.info[:virial] .= 1.0
        a.arrays[:forces] .= 2.0
        a.arrays[:shellforces] .= 3.0
        Cores.clear!(a)
        @test all(==(0), a.info[:potential_energy])
        @test all(==(0), a.info[:virial])
        @test all(==(0), a.arrays[:forces])
        @test all(==(0), a.arrays[:shellforces])
    end

    @testset "getindex slice" begin
        a = deepcopy_atoms(atoms)
        sub = a[[1, 3, 5]]
        @test size(sub.arrays[:pos], 1) == 3
        @test size(sub.arrays[:forces], 1) == 3
        @test sub.arrays[:elemid] == a.arrays[:elemid][[1, 3, 5]]
    end

    @testset "getindex rejects non per-atom array shapes" begin
        # Per-atom arrays must have ndims 1 or 2; anything else must error
        # instead of silently vanishing from the sliced copy.
        a = deepcopy_atoms(atoms)
        a.arrays[:bad3d] = zeros(FMD.FPPREC, length(a), 3, 3)
        @test_throws ErrorException a[[1, 2]]
    end

    @testset "change_arrtyp roundtrip (CPU)" begin
        # Round-trip Array -> Array (no-op) preserves bit-for-bit values.
        b = Cores.change_arrtyp(atoms, Array)
        @test b.arrays[:pos] == atoms.arrays[:pos]
        @test b.arrays[:elemid] == atoms.arrays[:elemid]
        @test b.info[:lattice] == atoms.info[:lattice]
        # Non-isbits info entries (the id → symbol table) pass through.
        @test b.info[:species] == atoms.info[:species]
    end

    @testset "change_arrtyp rejects non-isbits per-atom arrays" begin
        # A String column in arrays would fail the device gather in the
        # CellList re-sort (Issue #54) — refuse it at migration time.
        a = deepcopy_atoms(atoms)
        a.arrays[:labels] = fill("x", length(a))
        @test_throws ErrorException Cores.change_arrtyp(a, Array)
    end

    @testset "setelemid!: remap through info[:species]" begin
        # make_nacl_8 convention: elemid 1 = Na, 2 = Cl.
        a = deepcopy_atoms(atoms)
        @test a.info[:species] == ["Na", "Cl"]
        elemid0 = copy(a.arrays[:elemid])

        # Identity mapping is a no-op; idempotent on repeat.
        setelemid!(a, Dict("Na" => 1, "Cl" => 2))
        @test a.arrays[:elemid] == elemid0
        @test a.info[:species] == ["Na", "Cl"]

        # Swapped mapping flips every id and reorders the table.
        setelemid!(a, Dict("Cl" => 1, "Na" => 2))
        @test a.arrays[:elemid] == 3 .- elemid0
        @test a.info[:species] == ["Cl", "Na"]

        # Superset mapping (potential knows more elements than the frame).
        setelemid!(a, Dict("O" => 1, "Na" => 2, "Cl" => 3))
        @test a.arrays[:elemid] == elemid0 .+ 1
        @test a.info[:species] == ["O", "Na", "Cl"]

        # Loud failures, all before any mutation: uncovered symbol,
        # non-contiguous ids, missing table.
        before = copy(a.arrays[:elemid])
        @test_throws ErrorException setelemid!(a, Dict("Na" => 1))
        @test_throws ErrorException setelemid!(a, Dict("O" => 1, "Na" => 2, "Cl" => 4))
        @test a.arrays[:elemid] == before
        @test a.info[:species] == ["O", "Na", "Cl"]
        bare = Atoms{Array}(Dict{Symbol,Any}(),
                            Dict{Symbol,Array}(:elemid => [1, 2]))
        @test_throws ErrorException setelemid!(bare, Dict("Na" => 1, "Cl" => 2))
    end

    @testset "setelemid! remap lock under info sharing (Issue #70)" begin
        # getindex stamps :remap_locked into the *shared* info, so a later
        # remap on either family member errors instead of rewriting the
        # shared info[:species] table while the sibling's :elemid stays in
        # the old id space (a silent per-element parameter swap).
        a = deepcopy_atoms(atoms)
        sub = a[[1, 2, 3]]
        @test sub.info === a.info
        @test haskey(a.info, :remap_locked)
        before = copy(a.arrays[:elemid])
        @test_throws ErrorException setelemid!(a, Dict("Cl" => 1, "Na" => 2))
        @test_throws ErrorException setelemid!(sub, Dict("Cl" => 1, "Na" => 2))
        # The guard fires before any mutation.
        @test a.arrays[:elemid] == before
        @test a.info[:species] == ["Na", "Cl"]

        # change_arrtyp severs the family (fresh info Dict + fresh arrays):
        # the lock is dropped and the independent copy may be remapped
        # freely — this is the blessed unlock path.
        b = Cores.change_arrtyp(a, Array)
        @test !haskey(b.info, :remap_locked)
        setelemid!(b, Dict("Cl" => 1, "Na" => 2))
        @test b.info[:species] == ["Cl", "Na"]
        @test b.arrays[:elemid] == 3 .- before
        # The copy's remap leaves the original family untouched.
        @test a.info[:species] == ["Na", "Cl"]
        @test a.arrays[:elemid] == before

        # The blessed order still works: remap the pristine frame first,
        # slice afterwards.
        c = deepcopy_atoms(atoms)
        setelemid!(c, Dict("Cl" => 1, "Na" => 2))
        csub = c[c.arrays[:elemid] .== 1]
        @test all(==(1), csub.arrays[:elemid])
        @test csub.info[:species] == ["Cl", "Na"]
    end
end
