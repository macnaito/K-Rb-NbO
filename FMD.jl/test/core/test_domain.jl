# test/core/test_domain.jl
#
# SerialDomain identities and the ownership/wrap primitives.  The MPIDomain
# counterparts run in the MPI profile (test/mpi/test_domain_mpi.jl).
using Test
using LinearAlgebra
import KernelAbstractions

@testset "Domain layer (serial)" begin
    d = Cores.SerialDomain()

    @testset "reduction helpers are identities" begin
        v = FMD.FPPREC[1.0, 2.0, 3.0]
        @test Cores.allreduce_sum!(d, v) === v
        @test v == FMD.FPPREC[1.0, 2.0, 3.0]
        @test Cores.allreduce_sum(d, 2.5) === 2.5
        @test Cores.allreduce_max(d, 4.0) === 4.0
        @test Cores.allreduce_and(d, false) === false
        @test Cores.allreduce_and(d, true) === true
        @test Cores.isroot(d)
        # Device-transport entry points are no-ops off the GPU: the probe
        # for SerialDomain / CPU backends, and the prepare step for host
        # arrays (neither may touch MPI — serial runs never call MPI.Init).
        cpu = KernelAbstractions.CPU()
        @test Cores.probe_device_mpi!(d, cpu) === nothing
        @test probe_device_mpi!(d, cpu) === nothing   # exported
        @test Cores._device_mpi_prepare!(v) === nothing
    end

    @testset "segmented collectives are single-segment copies" begin
        # Serial P=1: counts has one entry; only that many elements move,
        # padding beyond the count must be untouched (the max(count,1)
        # allocation pattern of the slab PME).
        send = FMD.FPPREC[10.0, 20.0, 30.0]
        recv = fill(FMD.FPPREC(-1.0), 4)
        @test Cores.reduce_scatter_sum!(d, send, recv, Cint[3]) === recv
        @test recv == FMD.FPPREC[10.0, 20.0, 30.0, -1.0]
        recv2 = fill(FMD.FPPREC(-1.0), 4)
        @test Cores.allgatherv!(d, send, recv2, Cint[2]) === recv2
        @test recv2 == FMD.FPPREC[10.0, 20.0, -1.0, -1.0]
        recv3 = fill(FMD.FPPREC(-1.0), 3)
        @test Cores.alltoallv!(d, send, Cint[3], recv3, Cint[3]) === recv3
        @test recv3 == send
        @test alltoallv!(d, send, Cint[3], recv3, Cint[3]) === recv3   # exported
        # A multi-rank counts table on the serial path must refuse, not
        # silently return the first segment as a partial result.
        @test_throws ErrorException Cores.reduce_scatter_sum!(d, send, recv, Cint[2, 1])
        @test_throws ErrorException Cores.allgatherv!(d, send, recv, Cint[1, 2])
        @test_throws ErrorException Cores.alltoallv!(d, send, Cint[1, 2], recv3, Cint[1, 2])
        @test_throws ErrorException Cores.alltoallv!(d, send, Cint[3], recv3, Cint[2])
    end

    @testset "domain() accessor and n_global" begin
        a = make_nacl_8()
        @test Cores.domain(a) isa Cores.SerialDomain
        @test Cores.n_global(Cores.domain(a), a) == 8
        a.info[:domain] = d
        @test Cores.domain(a) === d
    end

    @testset "own_dim / owns / wrapfrac" begin
        @test Cores.own_dim(d, 0.99, 1)
        @test Cores.own_dim(d, 0.0, 3)
        @test Cores.owns(d, 0.1, 0.5, 0.9)
        # own_dim/owns wrap internally: raw unwrapped :pos values are safe.
        @test Cores.owns(d, 2.5, -1.25, 0.9)

        @test Cores.wrapfrac(FMD.FPPREC(0.25)) == 0.25
        @test Cores.wrapfrac(FMD.FPPREC(-0.75)) == 0.25
        @test Cores.wrapfrac(FMD.FPPREC(3.25)) == 0.25
        @test Cores.wrapfrac(FMD.FPPREC(0.0)) == 0.0
        # FP edge: x - floor(x) rounds to exactly 1.0 for tiny negative x —
        # wrapfrac must fold it to 0.0 so `w < 1` (and the last brick's
        # exclusive upper bound) always holds.
        @test Cores.wrapfrac(-eps(FMD.FPPREC) / 4) == 0.0
        @test 0.0 <= Cores.wrapfrac(FMD.FPPREC(-1e-300)) < 1.0
        @test 0.0 <= Cores.wrapfrac(FMD.FPPREC(7.0) - eps(FMD.FPPREC(7.0))) < 1.0
    end

    @testset "migrate is a SerialDomain no-op" begin
        a = make_nacl_8()
        @test Cores.migrate(a) === a
    end

    @testset "change_arrtyp / clear! keep :domain intact" begin
        a = make_nacl_8()
        a.info[:domain] = d
        b = Cores.change_arrtyp(a, Array)
        # Host-side pass-through via the explicit _conv_info method — the
        # same object, never fed to the array constructor.
        @test b.info[:domain] === d
        Cores.clear!(b)
        @test b.info[:domain] === d
        # Pin that the *explicit* AbstractDomain method dispatches (the
        # generic fallback happens to behave identically today via the
        # eltype(::Type) = Any accident — this test fails if the explicit
        # method is removed, keeping the guarantee intentional).
        m = which(Cores._conv_info, Tuple{typeof(identity), Cores.SerialDomain})
        @test m.sig == Tuple{typeof(Cores._conv_info), Any, Cores.AbstractDomain}
    end
end
