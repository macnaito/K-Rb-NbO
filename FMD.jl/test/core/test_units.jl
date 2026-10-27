# test/core/test_units.jl
using Test

@testset "Units" begin
    # Hartree → eV: NIST CODATA 2018 = 27.211386245988
    @test isapprox(UNITS[:eV], 27.211386245988; rtol = 1e-9)

    # Bohr → Å: NIST CODATA 2018 = 0.529177210903
    @test isapprox(UNITS[:Å], 0.529177210903; rtol = 1e-9)

    # e² / (4πε₀) in eV·Å is a famous constant ≈ 14.3996 eV·Å.
    # In atomic units it equals 1, so Hartree·Bohr → eV·Å multiplies by
    # UNITS[:eV] * UNITS[:Å].
    @test isapprox(UNITS[:eV] * UNITS[:Å], 14.39964544; rtol = 1e-6)

    # 1 K in Hartree: k_B / Hartree = 8.617333e-5 eV/K / 27.211 eV/Hartree
    # ≈ 3.1668e-6 Hartree/K. UNITS[:Kelvin] is the reciprocal: T_atomic = T_K / UNITS[:Kelvin].
    @test isapprox(1 / UNITS[:Kelvin], 3.16681156e-6; rtol = 1e-5)

    # ℏ / Hartree gives the atomic time unit; UNITS[:second] should equal
    # UNITS[:Dirac] / UNITS[:Hartree].
    @test isapprox(UNITS[:second], UNITS[:Dirac] / UNITS[:Hartree]; rtol = 1e-12)

    # CODATA 2018 anchor: atomic unit of time = 2.4188843265857e-17 s
    # (self-consistency `second == Dirac/Hartree` alone would not catch
    # both constants being wrong together).
    @test isapprox(UNITS[:second], 2.4188843265857e-17; rtol = 1e-9)

    # fs is the same atomic time unit expressed in femtoseconds. Exact
    # equality holds because both derive from the shared _AU_TIME_S.
    @test UNITS[:fs] == UNITS[:second] * 1e15
    # 1 fs in a.u. must match the literal the examples used to hardcode.
    @test isapprox(1 / UNITS[:fs], 41.34137333518007; rtol = 1e-12)

    # 1 a.u. of pressure (Hartree/Bohr³) in GPa — CODATA 2018 anchor
    # 2.9421015697e13 Pa (Issue #140). Self-consistency with the shared
    # Hartree/Bohr constants is exact by construction, so pin the
    # external literal.
    @test isapprox(UNITS[:GPa], 2.9421015697e4; rtol = 1e-9)

    # Atomic mass coverage and basic ordering.  ATOMIC_MASSES is defined
    # but not exported by Cores, so we access it via the qualified name.
    AM = Cores.ATOMIC_MASSES
    @test length(AM) >= 100
    for sym in ("H", "C", "O", "Na", "Cl", "K", "Rb", "Nb", "Fe", "Au")
        @test haskey(AM, sym)
        @test AM[sym] > 0
    end
    @test AM["H"] < AM["O"] < AM["Cl"] < AM["Fe"] < AM["Rb"] < AM["Au"]

    # ATOMIC_NUMBERS must cover exactly the same symbols as ATOMIC_MASSES:
    # anything readxyz can name must have a nuclear charge, or a
    # screened-nuclear repulsion term built from it would be undefined for
    # that species (Issue #160 removed the last in-tree consumer; #124 adds
    # the next one).
    AN = Cores.ATOMIC_NUMBERS
    @test Set(keys(AN)) == Set(keys(AM))
    @test all(>(0), values(AN))
    @test AN["H"] == 1 && AN["O"] == 8 && AN["Ti"] == 22 && AN["Ba"] == 56

    # FPPREC must be Float64 (asserted at runtests top-level too).
    @test FMD.FPPREC === Float64
end
