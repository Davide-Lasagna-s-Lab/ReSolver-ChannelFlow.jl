import Documenter
import FFTW
import LinearAlgebra
import NSEBase
import ReSolverChannelFlow

Documenter.DocMeta.setdocmeta!(
    ReSolverChannelFlow,
    :DocTestSetup,
    :(import FFTW; import LinearAlgebra; import NSEBase; using ReSolverChannelFlow);
    recursive = true,
)

Documenter.makedocs(
    sitename = "ReSolverChannelFlow.jl",
    modules = [ReSolverChannelFlow],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Davide-Lasagna-s-Lab.github.io/ReSolver-ChannelFlow.jl/stable/",
    ),
    pages = [
        "Home" => "index.md",
        "Manual" => [
            "Assumptions and Conventions" => "manual/conventions.md",
        ],
        "API Reference" => "api.md",
    ],
    checkdocs = :exports,
)

Documenter.deploydocs(
    repo = "github.com/Davide-Lasagna-s-Lab/ReSolver-ChannelFlow.jl.git",
    devbranch = "main",
)
