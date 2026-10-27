# test/mpi/_checks.jl
#
# Check registry for the bundled MPI tier (Issue #113): pulls in the
# shared fixtures and every check-function file, and maps check names to
# (function, mode). Loaded by the child runners (_run_bundle.jl /
# _run_single.jl) — never by the parent drivers.
#
# Contract for a check function:
#   check_<name>(out, dir, mode, backend, comm) -> Nothing
#     out     — this check's JLD2 artifact path (rank 0 writes it)
#     dir     — this check's private workdir (npt trajectory output; most
#               checks ignore it)
#     mode    — registry-selected variant string
#     backend — :cpu | :cuda (GPU binding is the runner's job)
#     comm    — MPI.COMM_WORLD
# Checks are self-contained: fresh fixture + own Random.seed! per call; no
# MPI state or frames are carried from one check to the next.

include(joinpath(@__DIR__, "..", "helpers.jl"))
include(joinpath(@__DIR__, "_npt_dd_common.jl"))
include(joinpath(@__DIR__, "_relax_dd_common.jl"))
include(joinpath(@__DIR__, "_check_allreduce.jl"))
include(joinpath(@__DIR__, "_check_collectives.jl"))
include(joinpath(@__DIR__, "_check_domain.jl"))
include(joinpath(@__DIR__, "_check_migrate.jl"))
include(joinpath(@__DIR__, "_check_ghosts.jl"))
include(joinpath(@__DIR__, "_check_reduce.jl"))
include(joinpath(@__DIR__, "_check_pmeslab.jl"))
include(joinpath(@__DIR__, "_check_coreshell.jl"))
include(joinpath(@__DIR__, "_check_hvp_dd.jl"))
include(joinpath(@__DIR__, "_check_umbrella.jl"))
include(joinpath(@__DIR__, "_check_surface_term.jl"))
include(joinpath(@__DIR__, "_check_npt_dd.jl"))
include(joinpath(@__DIR__, "_check_relax_dd.jl"))
include(joinpath(@__DIR__, "_check_fit.jl"))
include(joinpath(@__DIR__, "_check_ace_gram.jl"))
include(joinpath(@__DIR__, "_check_allreduce_gpu.jl"))
include(joinpath(@__DIR__, "_check_ghosts_gpu.jl"))

const CHECKS = Dict{String,Tuple{Function,String}}(
    "allreduce"         => (check_allreduce, "normal"),
    "allreduce_gpu"     => (check_allreduce_gpu, "normal"),
    "collectives"       => (check_collectives, "normal"),
    "domain"            => (check_domain,    "normal"),
    "migrate/normal"    => (check_migrate,   "normal"),
    "migrate/keyset"    => (check_migrate,   "keyset"),    # _run_single.jl only (aborts)
    "migrate/farjump"   => (check_migrate,   "farjump"),   # _run_single.jl only (aborts)
    "ghosts/normal"     => (check_ghosts,    "normal"),
    "ghosts/bigrc"      => (check_ghosts,    "bigrc"),
    "ghosts/tiny"       => (check_ghosts,    "tiny"),
    "ghosts/deadlock"   => (check_ghosts,    "deadlock"),
    "ghosts_gpu/normal" => (check_ghosts_gpu, "normal"),
    "ghosts_gpu/bigrc"  => (check_ghosts_gpu, "bigrc"),
    "ghosts_gpu/tiny"   => (check_ghosts_gpu, "tiny"),
    "reduce/normal"     => (check_reduce,    "normal"),
    "reduce/tiny"       => (check_reduce,    "tiny"),
    "pmeslab/normal"    => (check_pmeslab,   "normal"),
    "pmeslab/tiny"      => (check_pmeslab,   "tiny"),
    "coreshell"         => (check_coreshell, "normal"),
    "hvp_dd/normal"     => (check_hvp_dd,    "normal"),
    "hvp_dd/tiny"       => (check_hvp_dd,    "tiny"),
    "hvp_dd/surface"    => (check_hvp_dd,    "surface"),
    "hvp_dd/surface_tiny" => (check_hvp_dd,  "surface_tiny"),
    "umbrella"          => (check_umbrella,  "normal"),
    "umbrella/map"      => (check_umbrella,  "map"),
    "surface_term"      => (check_surface_term, "normal"),
    "surface_term/tiny" => (check_surface_term, "tiny"),
    "npt_dd/normal"     => (check_npt_dd,    "normal"),
    "npt_dd/umbrella"   => (check_npt_dd,    "umbrella"),
    "npt_dd/tiny"       => (check_npt_dd,    "tiny"),
    "npt_dd/csvr"       => (check_npt_dd,    "csvr"),
    "npt_dd/scr"        => (check_npt_dd,    "scr"),
    "npt_dd/xlbomd"     => (check_npt_dd,    "xlbomd"),
    "npt_dd/xlbomd_surface" => (check_npt_dd, "xlbomd_surface"),
    "npt_dd/xlbomd_cheb" => (check_npt_dd,   "xlbomd_cheb"),
    "npt_dd/xlbomd_baro" => (check_npt_dd,   "xlbomd_baro"),
    "npt_dd/nlh"        => (check_npt_dd,    "nlh"),
    "npt_dd/nlh_tiny"   => (check_npt_dd,    "nlh_tiny"),
    "relax_dd/normal"   => (check_relax_dd,  "normal"),
    "relax_dd/tiny"     => (check_relax_dd,  "tiny"),
    "relax_dd/cell"     => (check_relax_dd,  "cell"),
    "relax_dd/cell_iso" => (check_relax_dd,  "cell_iso"),
    "fitspline"         => (check_fitspline, "normal"),
    "fitspline/nlh"     => (check_fitspline, "nlh"),
    "fitace"            => (check_fitace,    "normal"),
    "fitace/precond"    => (check_fitace,    "precond"),
    "acegram/flat"      => (check_ace_gram,  "flat"),
    "acegram/split"     => (check_ace_gram,  "split"),
)
