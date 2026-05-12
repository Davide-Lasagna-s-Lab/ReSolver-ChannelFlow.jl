using Test

using FFTW, Random, LinearAlgebra

using ChebUtils, FDGrids

using ReSolverChannelFlow

include("test_grid.jl")
include("test_modenumber.jl")
include("test_derivatives.jl")
include("test_shifts.jl")
include("test_norms.jl")
include("test_weighting.jl")
include("test_cartesianprimitive.jl")
