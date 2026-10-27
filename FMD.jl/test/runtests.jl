# test/runtests.jl
#
# Top-level entry for FMD.jl tests.
#
# Profiles (controlled by environment variables):
#   default               core/ + calculators (CPU) + apps smoke
#   FMD_TEST_GPU=1        + gpu/* + Ewald==PME equivalence + npt_short
#                         (errors out if CUDA.functional() == false — Issue #18)
#   FMD_TEST_MPI=1        + mpi/* (subprocess driver, requires mpiexec)
#
# Tests assume FMD.FPPREC === Float64 (set in LocalPreferences.toml).
# On Float32 the finite-difference comparisons cannot reach rtol < 1e-4
# and will fail spuriously.

using Test
using FMD
using FMD.Cores
using FMD.Calculators
using Random
using LinearAlgebra

Random.seed!(0)

@assert FMD.FPPREC === Float64 """
Tests assume Float64 internal precision.  Set
    [FMD]
    precision = "Float64"
in LocalPreferences.toml before running Pkg.test()."""

include(joinpath(@__DIR__, "helpers.jl"))
include(joinpath(@__DIR__, "physical_consistency.jl"))

@testset "FMD.jl" begin
    # Dead-export guard (Issue #48/#78): exporting an undefined name is
    # legal Julia — it never fails at load time and only surfaces as an
    # UndefVarError at the caller's first use.
    @testset "all exported names are defined" begin
        for M in (FMD, FMD.Cores, FMD.Calculators, FMD.Apps)
            for n in names(M)
                n === nameof(M) && continue
                @test isdefined(M, n)
            end
        end
    end

    # Macro-import guard (Issue #127): `using Atomix: @atomic` binds the macro
    # for the whole module, so every file included *after* it can use the macro
    # without importing it. Such a file compiles only by include order and
    # breaks when an unrelated file drops its own (possibly dead) import —
    # src/Calculators/PME.jl was in exactly that state. Both directions are
    # checked: a use needs this file's own import, and an import with no use is
    # dead (the Issue #127 case). Atomix exports nothing, so a bare macro needs
    # `using Atomix: @atomic`; the qualified `Atomix.@atomic` (as in
    # Umbrella.jl) needs the name `Atomix`, i.e. plain `using Atomix`.
    #
    # Text matching, with two known limits — both fail loudly (a false
    # violation), never silently: an import wrapped over two lines is not
    # recognised, and neither is an aliased module (`import Atomix as AX` +
    # `AX.@atomic`). Keep to the two forms above.
    @testset "Atomix imports match macro uses (Issue #127)" begin
        srcdir = joinpath(@__DIR__, "..", "src")
        files = String[]
        for (root, _, fnames) in walkdir(srcdir), n in fnames
            endswith(n, ".jl") && push!(files, joinpath(root, n))
        end
        @test length(files) > 10          # the walk actually found the tree
        # (macro, bare use, this file's own import of it). `@atomic\b` does not
        # match inside `@atomicswap`, so the three stay independent.
        checks = map(("@atomic", "@atomicswap", "@atomicreplace")) do m
            (m, Regex("$(m)\\b"), Regex("^\\s*using\\s+Atomix\\s*:[^\\n]*$(m)\\b", "m"))
        end
        needs_import = String[]           # uses it without importing it here
        dead_import = String[]            # imports Atomix, never uses it
        for f in files
            # Comments go first: a file that only mentions a macro in prose is
            # not a user (PME.jl's own note about @atomic would count as one).
            src = replace(read(f, String), r"#[^\n]*" => "")
            imported = occursin(r"^\s*(using|import)\s+Atomix"m, src)
            rel = relpath(f, srcdir)
            # Strip the import lines, then the qualified uses; what is left of a
            # macro name is a bare use, resolved through an in-scope binding.
            # `CUDA.@atomic` (ACEFastCUDA.jl) is a different macro owned by
            # CUDA — its qualifier resolves through `using CUDA`, so it is
            # stripped here but deliberately NOT counted as an Atomix use.
            body = replace(src, r"^\s*(using|import)\s+Atomix[^\n]*\n"m => "")
            qualified = occursin(r"(?:Atomix|Base)\.@atomic\w*", body)
            bare_body = replace(body, r"(?:Atomix|Base|CUDA)\.@atomic\w*" => "")
            used_any = explicit_any = false
            for (m, use_pat, import_pat) in checks
                used = occursin(use_pat, bare_body)
                imports_m = occursin(import_pat, src)
                used_any |= used
                explicit_any |= imports_m
                used && !imports_m && push!(needs_import, "$(rel) ($(m))")
                imports_m && !used && push!(dead_import, "$(rel) ($(m))")
            end
            qualified && !imported && push!(needs_import, "$(rel) (Atomix.@…)")
            imported && !explicit_any && !used_any && !qualified &&
                push!(dead_import, "$(rel) (using Atomix)")
        end
        @test isempty(needs_import)
        @test isempty(dead_import)
    end

    # Detached-docstring guard (Issue #199): Julia attaches a docstring to the
    # expression *immediately* after it, so a comment or a blank line slipped
    # in between silently turns the docstring into a dangling string literal —
    # it vanishes from `?f` and from the Documenter output with no warning, and
    # then rots unnoticed (Issue #198 was exactly that: group_p's docstring had
    # drifted to describing a dipole moment while the code returned a density).
    # Detection is by parse tree, not text: an attached docstring parses to a
    # `Core.@doc` macrocall, a detached one to a bare `String` statement.
    @testset "no detached docstrings (Issue #199)" begin
        function bare_strings(file)
            found = Tuple{Int,String}[]
            todo = Any[Meta.parseall(read(file, String); filename = file)]
            while !isempty(todo)
                e = pop!(todo)
                e isa Expr || continue
                # Module bodies live in args[3]; :toplevel/:block hold their
                # statements directly.  Nothing else is descended into — a
                # string inside a function body is not a docstring position,
                # so walking there would only invite false positives.
                args = e.head === :module ? e.args[3].args :
                       (e.head in (:toplevel, :block) ? e.args : nothing)
                args === nothing && continue
                line = 0
                for a in args
                    if a isa LineNumberNode
                        line = a.line
                    elseif a isa String
                        push!(found, (line, first(split(strip(a), "\n"))))
                    else
                        push!(todo, a)
                    end
                end
            end
            return found
        end

        srcdir = joinpath(@__DIR__, "..", "src")
        files = String[]
        for (root, _, fnames) in walkdir(srcdir), n in fnames
            endswith(n, ".jl") && push!(files, joinpath(root, n))
        end
        @test length(files) > 10          # the walk actually found the tree
        detached = ["$(relpath(f, srcdir)):$(ln)  $(sig)"
                    for f in files for (ln, sig) in bare_strings(f)]
        @test isempty(detached)
    end

    @testset "Cores" begin
        include(joinpath(@__DIR__, "core", "test_units.jl"))
        include(joinpath(@__DIR__, "core", "test_atoms.jl"))
        include(joinpath(@__DIR__, "core", "test_domain.jl"))
        include(joinpath(@__DIR__, "core", "test_xyz.jl"))
        include(joinpath(@__DIR__, "core", "test_ghosts.jl"))
        include(joinpath(@__DIR__, "core", "test_neighborlist.jl"))
    end

    @testset "Calculators (CPU)" begin
        include(joinpath(@__DIR__, "calculators", "test_coulomb_buckingham.jl"))
        include(joinpath(@__DIR__, "calculators", "test_ewald.jl"))
        include(joinpath(@__DIR__, "calculators", "test_spring.jl"))
        include(joinpath(@__DIR__, "calculators", "test_external_field.jl"))
        include(joinpath(@__DIR__, "calculators", "test_coreshell.jl"))
        include(joinpath(@__DIR__, "calculators", "test_nlh.jl"))
        include(joinpath(@__DIR__, "calculators", "test_spline.jl"))
        include(joinpath(@__DIR__, "calculators", "test_ace.jl"))
        include(joinpath(@__DIR__, "calculators", "test_ace_merge.jl"))
        include(joinpath(@__DIR__, "calculators", "test_ace_precond.jl"))
        include(joinpath(@__DIR__, "calculators", "test_umbrella.jl"))
        include(joinpath(@__DIR__, "calculators", "test_map_umbrella.jl"))
        include(joinpath(@__DIR__, "calculators", "test_polmap.jl"))
        include(joinpath(@__DIR__, "calculators", "test_surface_term.jl"))
        include(joinpath(@__DIR__, "calculators", "test_virial.jl"))
        include(joinpath(@__DIR__, "calculators", "test_pme.jl"))
        include(joinpath(@__DIR__, "calculators", "test_pme_error.jl"))
        # PME is backend-generic (#51): runs on Array here, and additionally
        # on CuArray when FMD_TEST_GPU=1 (gating inside the file).
        include(joinpath(@__DIR__, "calculators", "test_ewald_pme_equivalence.jl"))
        include(joinpath(@__DIR__, "calculators", "test_hvp.jl"))
    end

    @testset "Apps (CPU smoke)" begin
        include(joinpath(@__DIR__, "apps", "test_param_validation.jl"))
        include(joinpath(@__DIR__, "apps", "test_loadpotential.jl"))
        include(joinpath(@__DIR__, "apps", "test_shist.jl"))
        include(joinpath(@__DIR__, "apps", "test_npt_cpu_smoke.jl"))
        include(joinpath(@__DIR__, "apps", "test_polmap_app.jl"))
        include(joinpath(@__DIR__, "apps", "test_relax.jl"))
        include(joinpath(@__DIR__, "apps", "test_xlbomd.jl"))
        include(joinpath(@__DIR__, "apps", "test_csvr.jl"))
        include(joinpath(@__DIR__, "apps", "test_scr.jl"))
        include(joinpath(@__DIR__, "apps", "test_fit_spline_smoke.jl"))
        include(joinpath(@__DIR__, "apps", "test_fit_ace_smoke.jl"))
    end

    # Profile inclusion invariant (Issue #109): the both-flags run
    # (FMD_TEST_GPU=1 FMD_TEST_MPI=1) stands in for the MPI-only profile
    # in the CLAUDE.md full gate ONLY while (a) the MPI tier gate below
    # stays independent of FMD_TEST_GPU and (b) no MPI child script reads
    # FMD_TEST_* — mpiexec children inherit the parent ENV, so an ENV
    # read would make the two profiles diverge. (a) is a structural
    # property of this file (keep it that way); (b) is checked here so
    # the documented gate cannot rot silently.
    @testset "profile inclusion invariant (Issue #109)" begin
        mpi_dir = joinpath(@__DIR__, "mpi")
        files = filter(endswith(".jl"), readdir(mpi_dir; join = true))
        @test !isempty(files)
        for f in files
            @test !occursin("FMD_TEST", read(f, String))
        end
    end

    if get(ENV, "FMD_TEST_GPU", "0") == "1"
        using CUDA
        if CUDA.functional()
            @testset "GPU" begin
                include(joinpath(@__DIR__, "gpu", "test_atoms_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_neighborlist_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_ewald_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_pme_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_ace_spline_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_nlh_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_ace_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_umbrella_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_surface_term_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_hvp_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_xlbomd_gpu.jl"))
                include(joinpath(@__DIR__, "gpu", "test_relax_gpu.jl"))
                include(joinpath(@__DIR__, "apps", "test_npt_short.jl"))
            end
        else
            # Issue #18: FMD_TEST_GPU=1 is an explicit request to run the GPU
            # profile.  Silently skipping would let a zero-coverage run come
            # back green, so fail loudly instead.  The inner @testset records
            # the error (making the suite red) without aborting the rest of
            # the file, so an FMD_TEST_MPI=1 run in the same process still
            # executes its tests.
            @testset "GPU" begin
                error("FMD_TEST_GPU=1 set but CUDA.functional() == false; " *
                      "refusing to skip the explicitly requested GPU tests. " *
                      "Fix the CUDA setup or unset FMD_TEST_GPU. " *
                      "If you ran Pkg.test(), use `julia --project test/runtests.jl` " *
                      "instead (the test sandbox ignores LocalPreferences.toml).")
            end
        end
    else
        @info "Skipping GPU tests (set FMD_TEST_GPU=1 to enable)"
    end

    # Hand the machine back to the subprocess tiers below: this process has
    # just run the whole CPU (and possibly GPU) suite and sits on its peak
    # heap plus a CUDA pool, while the bundles fork P ranks that each need
    # ~2 GB resident. On a 15 GB host that overlap is what pushed a
    # measured run into the OOM killer (see run_mpi_script).
    if get(ENV, "FMD_TEST_MPI", "0") == "1"
        GC.gc(true)
        get(ENV, "FMD_TEST_GPU", "0") == "1" && CUDA.functional() && CUDA.reclaim()
    end

    if get(ENV, "FMD_TEST_GPU", "0") == "1" && get(ENV, "FMD_TEST_MPI", "0") == "1"
        # Third tier: GPU × MPI. Requires a CUDA-aware MPI build — the
        # children pass device buffers straight to MPI (no host staging;
        # see the transport contract in Domain.jl) and probe_device_mpi!
        # makes a non-aware build fail loudly rather than silently skip
        # (Issue #18 philosophy). CUDA was loaded by the GPU block above.
        @testset "GPU+MPI (via subprocess driver)" begin
            if CUDA.functional()
                include(joinpath(@__DIR__, "mpi", "test_bundle_gpu_mpi.jl"))
            else
                error("FMD_TEST_GPU=1 and FMD_TEST_MPI=1 set but " *
                      "CUDA.functional() == false; the GPU tier above already " *
                      "failed — fix the CUDA setup first.")
            end
        end
    end

    # NOTE: keep this gate independent of FMD_TEST_GPU — the CLAUDE.md
    # full gate (default + GPU + both-flags) relies on the both-flags
    # run executing this tier in full (Issue #109; see the "profile
    # inclusion invariant" testset above).
    if get(ENV, "FMD_TEST_MPI", "0") == "1"
        @testset "MPI (via subprocess driver)" begin
            include(joinpath(@__DIR__, "mpi", "test_bundle_mpi.jl"))
        end
    else
        @info "Skipping MPI tests (set FMD_TEST_MPI=1 to enable)"
    end
end
