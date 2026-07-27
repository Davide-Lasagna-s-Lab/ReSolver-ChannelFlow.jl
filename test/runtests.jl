using Test
using FFTW
using Random
using LinearAlgebra
using ChebUtils
using FDGrids
using NSEBase
using ReSolverChannelFlow

include("test_grid.jl")
include("test_derivatives.jl")
include("test_shifts.jl")
include("test_norms.jl")
include("test_weighting.jl")
include("test_cartesianprimitive.jl")

# MPI tests
include("MPI/runtests.jl")

# CUDA tests
include("CUDA/runtests.jl")
