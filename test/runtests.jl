using Test

using FFTW, Random, LinearAlgebra

using ChebUtils, FDGrids

using ReSolverChannelFlow

include("test_grid.jl")
include("test_modenumber.jl")
include("test_ftfield.jl")
include("test_field.jl")
include("test_vectorfield.jl")
include("test_fft.jl")
include("test_projectedfield.jl")
include("test_broadcasting.jl")
include("test_derivatives.jl")
include("test_shifts.jl")
include("test_norms.jl")
include("test_weighting.jl")
include("test_cartesianprimitive.jl")
include("test_projectednse.jl")
