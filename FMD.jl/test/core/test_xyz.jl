# test/core/test_xyz.jl
using Test
using LinearAlgebra

@testset "XYZ I/O" begin
    @testset "readxyz: lattice, pos, virial Voigt order" begin
        # Manually construct an extended-XYZ frame with 2 atoms,
        # diagonal lattice, known forces and virial.
        xyz_text = """
        2
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3:forces:R:3 potential_energy=1.5 virial="1 0 0 0 2 0 0 0 3" pbc="T T T"
        Na 1.0 2.0 3.0 0.1 0.2 0.3
        Cl 4.0 5.0 0.5 -0.1 -0.2 -0.3
        """
        io = IOBuffer(xyz_text)
        ds = Cores.readxyz(io)
        @test length(ds) == 1
        atoms = ds[1]
        @test length(atoms) == 2

        # Lattice was given in Å; internally stored in Bohr (rows = lattice vectors).
        @test isapprox(atoms.info[:lattice][1, 1], 6.0 / UNITS[:Å]; rtol = 1e-10)
        @test isapprox(atoms.info[:lattice][2, 2], 6.0 / UNITS[:Å]; rtol = 1e-10)
        @test isapprox(atoms.info[:lattice][3, 3], 6.0 / UNITS[:Å]; rtol = 1e-10)

        # virial 9-vector → Voigt 6 [xx, yy, zz, yz, xz, xy] then divided by UNITS[:eV]
        # (Hartree storage).
        @test length(atoms.info[:virial]) == 6
        @test isapprox(atoms.info[:virial][1], 1.0 / UNITS[:eV]; rtol = 1e-10)
        @test isapprox(atoms.info[:virial][2], 2.0 / UNITS[:eV]; rtol = 1e-10)
        @test isapprox(atoms.info[:virial][3], 3.0 / UNITS[:eV]; rtol = 1e-10)
        for k in 4:6
            @test isapprox(atoms.info[:virial][k], 0; atol = 1e-12)
        end

        @test isapprox(atoms.info[:potential_energy][1], 1.5 / UNITS[:eV]; rtol = 1e-10)

        # Forces converted from eV/Å to Hartree/Bohr.
        @test isapprox(atoms.arrays[:forces][1, 1],
                       0.1 / (UNITS[:eV] / UNITS[:Å]); rtol = 1e-10)

        # Masses auto-populated from ATOMIC_MASSES table (qualified name —
        # Cores does not export the lookup table).
        @test atoms.arrays[:masses][1] ≈ Cores.ATOMIC_MASSES["Na"]
        @test atoms.arrays[:masses][2] ≈ Cores.ATOMIC_MASSES["Cl"]

        # Defaulted arrays present and shaped right.
        for key in (:velo, :shelldisp, :shellforces, :id)
            @test haskey(atoms.arrays, key)
        end
        @test atoms.arrays[:id] == [1, 2]
    end

    @testset "readxyz converts the shelldisp column from Å (Issue #154)" begin
        # writexyz emits shelldisp in Å; the reader must divide it back out.
        # Before Issue #154 :shelldisp was missing from the default
        # unitconversions table, so a restart read shell displacements
        # 0.529× too small (Bohr↔Å) — silently, because the :scf path
        # re-relaxes every step and papers over the bad initial guess.
        xyz_text = """
        2
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3:shelldisp:R:3 pbc="T T T"
        Na 1.0 2.0 3.0 0.10 -0.20 0.30
        Cl 4.0 5.0 0.5 -0.05 0.15 -0.25
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        expected = [0.10 -0.20 0.30; -0.05 0.15 -0.25] ./ UNITS[:Å]
        @test isapprox(atoms.arrays[:shelldisp], expected; rtol = 1e-10)
        # Guard the failure mode explicitly: reading the column verbatim
        # keeps the file's Å numbers, which are UNITS[:Å] ≈ 0.529× the
        # correct Bohr values, so the un-fixed reader is caught.
        @test !isapprox(atoms.arrays[:shelldisp], [0.10 -0.20 0.30; -0.05 0.15 -0.25];
                        rtol = 1e-3)

        # A file without the column still gets zeros (auto-populated after
        # the conversion loop, so the missing key must not error).
        xyz_noshell = """
        1
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        Na 1.0 2.0 3.0
        """
        @test iszero(Cores.readxyz(IOBuffer(xyz_noshell))[1].arrays[:shelldisp])
    end

    @testset "potential energy: energy= / potential_energy= keys (Issue #154)" begin
        frame(hdr) = """
        1
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 $(hdr) pbc="T T T"
        Na 1.0 2.0 3.0
        """
        pe(hdr) = Cores.readxyz(IOBuffer(frame(hdr)))[1].info[:potential_energy][1]

        # extxyz spelling (what write_header emits) and the dataset spelling
        # both parse, into Hartree.
        @test isapprox(pe("energy=-12.25"), -12.25 / UNITS[:eV]; rtol = 1e-10)
        @test isapprox(pe("potential_energy=-12.25"), -12.25 / UNITS[:eV]; rtol = 1e-10)
        # Pre-#154 output padded the value (`%15.8e`); still readable so old
        # trajectories can be restarted.
        @test isapprox(pe("energy=  -1.22500000e+01"), -12.25 / UNITS[:eV]; rtol = 1e-8)
        # Both present → the explicit key wins.
        @test isapprox(pe("potential_energy=-12.25 energy=99.0"),
                       -12.25 / UNITS[:eV]; rtol = 1e-10)
        @test isapprox(pe("energy=99.0 potential_energy=-12.25"),
                       -12.25 / UNITS[:eV]; rtol = 1e-10)
        # A different key that merely ends in `energy=` must not be picked up.
        @test pe("free_energy=99.0") == 0

        # extxyz puts the keys in any order, so either spelling may start the
        # comment line (no leading whitespace to anchor on).
        for key in ("energy", "potential_energy")
            line_start = """
            1
            $(key)=-12.25 Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
            Na 1.0 2.0 3.0
            """
            atoms = Cores.readxyz(IOBuffer(line_start))[1]
            @test isapprox(atoms.info[:potential_energy][1], -12.25 / UNITS[:eV];
                           rtol = 1e-10)
        end
    end

    @testset "readxyz converts the peratomstress column from eV/Å³ (Issue #154)" begin
        # writexyz emits this column in eV/Å³, so the default table divides it
        # back out — arrays below this layer are atomic units throughout.
        xyz_text = """
        1
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3:peratomstress:R:6 pbc="T T T"
        Na 1.0 2.0 3.0 1.0 2.0 3.0 0.1 0.2 0.3
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        @test isapprox(atoms.arrays[:peratomstress],
                       [1.0 2.0 3.0 0.1 0.2 0.3] ./ (UNITS[:eV] / UNITS[:Å]^3);
                       rtol = 1e-10)
        # It is a stress, not the :peratom_virial it was written from
        # (opposite sign, kinetic term folded in), so it must not be aliased
        # onto that key.
        @test !haskey(atoms.arrays, :peratom_virial)
    end

    @testset "species → elemid + info[:species] table (Issue #54)" begin
        xyz_text = """
        3
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        Na 1.0 2.0 3.0
        Cl 4.0 5.0 0.5
        Na 2.0 2.0 2.0
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]

        # Per-atom species strings are never stored; ids are provisional
        # (alphabetical order of the symbols found in the file).
        @test !haskey(atoms.arrays, :species)
        @test atoms.info[:species] == ["Cl", "Na"]
        @test atoms.arrays[:elemid] == [2, 1, 2]
        # Masses follow the per-atom symbols, not the id order.
        @test atoms.arrays[:masses] ≈
              [Cores.ATOMIC_MASSES[s] for s in ("Na", "Cl", "Na")]

        # setelemid! rebinds the provisional ids to the params id space and
        # rewrites the table in mapping order.
        setelemid!(atoms, Dict("Na" => 1, "Cl" => 2))
        @test atoms.info[:species] == ["Na", "Cl"]
        @test atoms.arrays[:elemid] == [1, 2, 1]
    end

    @testset "readxyz rejects an explicit elemid column" begin
        # :elemid is derived from the species column; a file that also
        # carries its own elemid column is ambiguous and must error.
        xyz_text = """
        1
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3:elemid:I:1 pbc="T T T"
        Na 1.0 2.0 3.0 1
        """
        @test_throws ErrorException Cores.readxyz(IOBuffer(xyz_text))
    end

    @testset "writexyz round-trip preserves per-atom species" begin
        xyz_text = """
        3
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        Na 1.0 2.0 3.0
        Cl 4.0 5.0 0.5
        Na 2.0 2.0 2.0
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        # Deliberately permuted vs the alphabetical provisional order.
        elements = Dict("Na" => 1, "Cl" => 2)
        setelemid!(atoms, elements)

        symbols(a) = a.info[:species][Array(a.arrays[:elemid])]
        mktemp() do path, io
            Cores.writexyz(io, atoms, Dict{Symbol,Any}(:elements => elements))
            close(io)
            back = Cores.readxyz(open(path, "r"))[1]
            @test symbols(back) == symbols(atoms) == ["Na", "Cl", "Na"]
            @test isapprox(back.arrays[:pos] * back.info[:lattice],
                           atoms.arrays[:pos] * atoms.info[:lattice]; rtol = 1e-6)
            @test back.arrays[:masses] ≈ atoms.arrays[:masses]
        end
    end

    @testset "writexyz → readxyz round-trip is unit-exact (Issue #154)" begin
        # Every quantity writexyz emits must come back in atomic units, so a
        # restart from step_<n>.xyz reproduces the frame it was written from.
        # This is what shadow XL-BOMD needs: it seeds the auxiliary field ŝ
        # from :shelldisp and never re-relaxes it (src/Apps/XLBOMD.jl:183).
        xyz_text = """
        2
        Lattice="6.0 0.0 0.0 0.0 6.5 0.0 0.0 0.0 7.0" Properties=species:S:1:pos:R:3:shelldisp:R:3:velo:R:3:forces:R:3 potential_energy=-12.25 pbc="T T T"
        Na 1.0 2.0 3.0  0.10 -0.20  0.30  1.0e-4  2.0e-4 -3.0e-4  0.1  0.2  0.3
        Cl 4.0 5.0 0.5 -0.05  0.15 -0.25 -1.0e-4  5.0e-5  2.0e-4 -0.1 -0.2 -0.3
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        elements = Dict("Na" => 1, "Cl" => 2)
        setelemid!(atoms, elements)

        mktemp() do path, io
            Cores.writexyz(io, atoms, Dict{Symbol,Any}(:elements => elements))
            close(io)

            # A bare key=value must not be padded, or a whitespace-splitting
            # extxyz parser (readxyz included) sees an empty value.
            header = readlines(path)[2]
            @test occursin(r"\senergy=-?[0-9]", header)

            back = Cores.readxyz(open(path, "r"))[1]
            # %15.8e keeps ~9 significant digits.
            rt = 1e-7
            @test isapprox(back.info[:lattice], atoms.info[:lattice]; rtol = rt)
            @test isapprox(back.arrays[:pos] * back.info[:lattice],
                           atoms.arrays[:pos] * atoms.info[:lattice]; rtol = rt)
            @test isapprox(back.arrays[:shelldisp], atoms.arrays[:shelldisp]; rtol = rt)
            @test isapprox(back.arrays[:velo], atoms.arrays[:velo]; rtol = rt)
            @test isapprox(back.arrays[:forces], atoms.arrays[:forces]; rtol = rt)
            @test isapprox(back.info[:potential_energy][1],
                           atoms.info[:potential_energy][1]; rtol = rt)
            # Not a no-op check: the written values really are non-zero.
            @test !iszero(back.arrays[:shelldisp])
            @test !iszero(back.arrays[:velo])
            @test !iszero(back.info[:potential_energy][1])
        end
    end

    @testset "writexyz round-trip on a sliced frame (lock stays internal)" begin
        # Slicing stamps :remap_locked into the shared info (Issue #70).
        # write_header whitelists its keys, so the stamp must not leak into
        # the extxyz header, and a re-read frame starts unlocked.
        xyz_text = """
        3
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        Na 1.0 2.0 3.0
        Cl 4.0 5.0 0.5
        Na 2.0 2.0 2.0
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        elements = Dict("Na" => 1, "Cl" => 2)
        setelemid!(atoms, elements)
        sub = atoms[[2, 1, 3]]
        @test haskey(sub.info, :remap_locked)
        symbols(a) = a.info[:species][Array(a.arrays[:elemid])]
        mktemp() do path, io
            Cores.writexyz(io, sub, Dict{Symbol,Any}(:elements => elements))
            close(io)
            back = Cores.readxyz(open(path, "r"))[1]
            @test symbols(back) == symbols(sub) == ["Cl", "Na", "Na"]
            @test !haskey(back.info, :remap_locked)
        end
    end

    @testset "writexyz rejects a frame bound to a different id space" begin
        # A freshly read frame still carries provisional (alphabetical) ids;
        # writing it with a mismatching params[:elements] would silently
        # swap element symbols — must be loud instead.
        xyz_text = """
        2
        Lattice="6.0 0.0 0.0 0.0 6.0 0.0 0.0 0.0 6.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        Na 1.0 2.0 3.0
        Cl 4.0 5.0 0.5
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]  # table = ["Cl", "Na"]
        mktemp() do path, io
            @test_throws ErrorException Cores.writexyz(io, atoms,
                Dict{Symbol,Any}(:elements => Dict("Na" => 1, "Cl" => 2)))
        end
    end

    @testset "fractional pos round-trip with lattice" begin
        # Build a simple non-orthogonal lattice (still rows=vectors) and
        # ensure pos × lattice recovers the input Cartesian coords.
        xyz_text = """
        1
        Lattice="5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0" Properties=species:S:1:pos:R:3 pbc="T T T"
        H 2.5 2.5 2.5
        """
        atoms = Cores.readxyz(IOBuffer(xyz_text))[1]
        cart = atoms.arrays[:pos] * atoms.info[:lattice]
        @test isapprox(cart[1, 1], 2.5 / UNITS[:Å]; rtol = 1e-10)
        @test isapprox(cart[1, 2], 2.5 / UNITS[:Å]; rtol = 1e-10)
        @test isapprox(cart[1, 3], 2.5 / UNITS[:Å]; rtol = 1e-10)
    end
end
