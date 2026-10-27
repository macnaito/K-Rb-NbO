using Documenter
using FMD

makedocs(
    sitename = "FMD.jl",
    authors  = "azmhkr",
    modules  = [FMD, FMD.Cores, FMD.Calculators, FMD.Apps],
    format   = Documenter.HTML(prettyurls = false),
    pages    = [
        "Home"          => "index.md",
        "Workflow"      => "workflow.md",
        "Dataset"       => "dataset.md",
        "Parameters"    => "parameters.md",
        "Theory notes"  => [
            "Overview"                     => "theory/index.md",
            "SCR barostat derivation"      => "theory/scr-barostat.md",
            "MTTK cell DOF counting"       => "theory/mttk-cell-dof.md",
            "Core-shell analytic HVP"      => "theory/coreshell-analytic-hvp.md",
            "Shadow XL-BOMD"               => "theory/xlbomd-shell.md",
            "Polarisation-bias virial"     => "theory/polarization-bias-virial.md",
            "Anisotropic surface term"     => "theory/surface-term.md",
        ],
        "HPC setup"     => "hpc.md",
        "API reference" => [
            "Apps"        => "api/apps.md",
            "Cores"       => "api/cores.md",
            "Calculators" => "api/calculators.md",
        ],
    ],
    warnonly = [:cross_references, :missing_docs, :autodocs_block, :docs_block],
)
