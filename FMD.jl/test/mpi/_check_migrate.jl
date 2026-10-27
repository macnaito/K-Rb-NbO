# test/mpi/_check_migrate.jl
#
# distribute + migrate conservation laws, per-id bitwise equality (gauge
# preservation), and the loud-failure modes. mode ∈ normal | keyset |
# farjump. The error modes deliberately throw (→ nonzero exit without the
# sentinel when run through _run_single.jl) — the parent asserts the
# failure and the error text; they must NOT run inside a bundle.

function check_migrate(out, dir, mode, backend, comm)
    rank = MPI.Comm_rank(comm)
    P = MPI.Comm_size(comm)

    # Identical full frame on every rank: rattled NaCl-64 with gauge-carrying
    # unwrapped rows and runtime-added per-atom arrays riding along.
    Random.seed!(0)
    full = make_nacl_64()
    full.arrays[:pos] .+= 0.01 .* randn(FMD.FPPREC, size(full.arrays[:pos]))
    full.arrays[:pos][1, :] .+= 1.0        # +1 box in every dimension
    full.arrays[:pos][2, 1] -= 2.0         # -2 boxes in x
    full.arrays[:velo] .= randn(FMD.FPPREC, size(full.arrays[:velo]))
    full.arrays[:peratom_virial] = randn(FMD.FPPREC, length(full), 6)
    full.arrays[:shist1] = copy(full.arrays[:shelldisp])
    orig = deepcopy_atoms(full)

    dom = Cores.MPIDomain(length(full); comm = comm)
    loc = Cores.distribute(full, dom)

    ks = sort!(collect(keys(orig.arrays)))
    ids_orig = Array(orig.arrays[:id])

    # Every local row must be bitwise equal to the reference frame's row of
    # the same :id, for every per-atom array (gauge: :pos unwrapped).
    function rows_equal(a, ref)
        for (r, id) in enumerate(Array(a.arrays[:id]))
            ro = findfirst(==(id), ids_orig)
            for k in ks
                arr = a.arrays[k]
                oa = ref.arrays[k]
                rowa = ndims(arr) == 1 ? arr[r] : arr[r, :]
                rowo = ndims(oa) == 1 ? oa[ro] : oa[ro, :]
                rowa == rowo || return false
            end
        end
        return true
    end

    function id_partition_ok(a)
        idcount = zeros(Int32, dom.nglobal)
        for id in Array(a.arrays[:id])
            idcount[id] += 1
        end
        MPI.Allreduce!(idcount, +, comm)
        return all(==(1), idcount)
    end

    ok_partition0 = id_partition_ok(loc)
    ok_bits0 = MPI.Allreduce(rows_equal(loc, orig), &, comm)

    # distribute rejects a slice (wrong length) loudly — the length check is
    # rank-local and fires before any collective, so try/catch is symmetric.
    ok_argcheck = try
        Cores.distribute(orig[1:4], dom)
        false
    catch e
        occursin("pass the full frame", sprint(showerror, e))
    end
    ok_argcheck = MPI.Allreduce(ok_argcheck, &, comm)

    # migrate with nothing to move: bitwise no-op (same rows, same order).
    loc2 = Cores.migrate(loc)
    ok_noop = MPI.Allreduce(
        Array(loc2.arrays[:id]) == Array(loc.arrays[:id]) && rows_equal(loc2, orig),
        &, comm)

    if mode == "keyset"
        # One rank grows an extra per-atom key — the collective keyset check
        # must fail loudly on every rank (silent row skew otherwise).
        if rank == 0
            loc2.arrays[:extra] = zeros(FMD.FPPREC, length(loc2))
        end
        Cores.migrate(loc2)
        error("unreachable: keyset mismatch was not detected")
    end

    if mode == "farjump"
        # A minimal-image brick offset of |Δ| ≥ 2 cannot be routed by the ±1
        # staged exchange and must error, not be silently misplaced. Needs
        # dims[1] >= 5 so that a +2-brick jump is NOT a legitimate wrap-around
        # move (at dims[1] = 3, +2 ≡ -1 in minimal image and routes fine).
        dom.dims[1] >= 5 || error("farjump mode needs dims[1] >= 5 (got $(dom.dims))")
        if rank == 0 && length(loc2) > 0
            loc2.arrays[:pos][1, 1] += FMD.FPPREC(2.0 / dom.dims[1])
        end
        Cores.migrate(loc2)
        error("unreachable: far jump was not detected")
    end

    # Drift: +60% of a brick width in x (unwrapped add — the gauge moves with
    # it). A large fraction of atoms crosses into the plus neighbour.
    delta = FMD.FPPREC(0.6 / dom.dims[1])
    loc2.arrays[:pos][:, 1] .+= delta
    ids_before = Set(Array(loc2.arrays[:id]))

    loc3 = Cores.migrate(loc2)

    expected = deepcopy_atoms(orig)
    expected.arrays[:pos][:, 1] .+= delta

    ok_partition1 = id_partition_ok(loc3)
    pos3 = loc3.arrays[:pos]
    ok_owned = MPI.Allreduce(
        all(Cores.owns(dom, pos3[i, 1], pos3[i, 2], pos3[i, 3]) for i in axes(pos3, 1)),
        &, comm)
    ok_bits1 = MPI.Allreduce(rows_equal(loc3, expected), &, comm)
    moved_total = MPI.Allreduce(length(setdiff(ids_before, Set(Array(loc3.arrays[:id])))), +, comm)

    if rank == 0
        JLD2.jldsave(out;
                     mpisize = P, dimsx = dom.dims[1],
                     ok_partition0 = ok_partition0, ok_bits0 = ok_bits0,
                     ok_argcheck = ok_argcheck, ok_noop = ok_noop,
                     ok_partition1 = ok_partition1, ok_owned = ok_owned,
                     ok_bits1 = ok_bits1, moved_total = moved_total)
    end
    return nothing
end
