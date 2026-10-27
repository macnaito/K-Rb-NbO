# test/mpi/_check_ace_gram.jl
#
# Machine-precision parity of the blockchol-si Gram ASSEMBLY itself
# (Issue #184): `_assemble_block_grams` is called directly — its kwargs are
# the supported test surface, like `budget_bytes` in the CPU tests — with a
# small `panel_bytes` so the column-panel accumulation (gemm + syrk,
# multi-sweep), the node-local shared-window reduce, and the panel-wise
# inter-node leader reduce are all exercised. Two modes:
#   * "flat":  physical node split (`Comm_split_type` — one node in CI, so
#     the leaders' communicator degenerates to a single rank)
#   * "split": artificial two-"node" split (`node_color = rank % 2`), so the
#     inter-node leader reduce and the owner-reads-shared-window path run
#     inside one physical machine; with P = 4 some owners are non-leaders
# The dataset mixes frame sizes (NaCl-64 ×2 + NaCl-8 ×1) so frames with
# Nat ≠ maxNat exist, and with P = 4 one rank holds an empty shard.
# Rank 0 writes every block's dense Gram (upper triangle valid, strict
# lower exactly zero) so the parent can compare across rank counts and
# Reduce topologies.

function check_ace_gram(out, dir, mode, backend, comm)
    Random.seed!(0)
    dataset = Atoms[]
    for k in 1:2
        a = make_nacl_64()
        a.arrays[:pos] .+= 1e-3 * k .* randn(FMD.FPPREC, size(a.arrays[:pos]))
        a.info[:potential_energy] = FMD.FPPREC[1e-2 * k]
        a.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(a.arrays[:forces]))
        a.info[:virial] = FMD.FPPREC[1e-4 * k, 1e-4 * k, 1e-4 * k, 0, 0, 0]
        a.info[:method] = [:smoke]
        push!(dataset, a)
    end
    a8 = make_nacl_8()
    a8.arrays[:pos] .+= 1e-3 .* randn(FMD.FPPREC, size(a8.arrays[:pos]))
    a8.info[:potential_energy] = FMD.FPPREC[3e-2]
    a8.arrays[:forces] .= 1e-3 .* randn(FMD.FPPREC, size(a8.arrays[:forces]))
    a8.info[:virial] = FMD.FPPREC[1e-4, 1e-4, 1e-4, 0, 0, 0]
    a8.info[:method] = [:smoke]
    push!(dataset, a8)

    make_acep() = Calculators.ACEParam(FMD.FPPREC(1.0), FMD.FPPREC(6.0),
                                       2, 4, 4, 3, 3)
    Λ = FMD.FPPREC(2.0e8)   # unused by the DATA-ONLY Gram, kept for spec parity
    base = nacl_params(dataset[1]; alpha = 0.30, ewald_tolerance = 1e-6,
                       rc_factor = 0.45, with_shell = true)
    params = (; base...,
        weight_force  = 1.0,
        weight_stress = 0.1,
        method_weight = Dict(:smoke => 1.0),
        rigid         = (; cp = Λ, c0 = Λ, c00 = Λ, c11 = Λ, c22 = Λ, c33 = Λ,
                           c011 = Λ, c022 = Λ, c033 = Λ),
        nsteps        = 5,
    )
    acep = make_acep()
    sys = Calculators.build_normal_system(acep, dataset, params)

    # panel_bytes: the widest block lands in 3 panels (gemm + syrk +
    # multi-sweep); narrower blocks stay single-panel
    layout = Calculators._ace_block_layout(acep)
    S = sizeof(FMD.FPPREC)
    wmax = maximum(length(blk.live) for blk in layout)
    asm = Calculators._assemble_block_grams(acep, params, sys.precond_ctx;
        comm, panel_bytes = S * wmax * cld(wmax, 3),
        node_color = mode == "split" ? MPI.Comm_rank(comm) % 2 : nothing)

    # collect every block on rank 0: owners contribute their Gram, everyone
    # else zeros (0 + G preserves G bitwise under MPI.SUM for finite values)
    gathered = Vector{Matrix{FMD.FPPREC}}(undef, length(layout))
    for (idx, blk) in enumerate(layout)
        w = length(blk.live)
        G = asm.grams[idx] === nothing ? zeros(FMD.FPPREC, w, w) :
            asm.grams[idx]::Matrix{FMD.FPPREC}
        MPI.Reduce!(G, MPI.SUM, comm; root = 0)
        gathered[idx] = G
    end

    if MPI.Comm_rank(comm) == 0
        JLD2.jldsave(out;
                     grams = gathered,
                     owner = asm.owner,
                     mpisize = MPI.Comm_size(comm))
    end
    return nothing
end
