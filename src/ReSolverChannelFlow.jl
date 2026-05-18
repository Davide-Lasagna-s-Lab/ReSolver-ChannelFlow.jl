module ReSolverChannelFlow

using FFTW, LinearAlgebra, JLD2

using NSEBase

export ChannelGrid, points, growto, get_fields, save_grid, load_grid
export ModeNumber
export FTField, grid, save_field, load_field
export Field
export VectorField
export ProjectedField, modes, project!, project, expand!, expand
export FFT, IFFT, FFTPlans
export dds!, ddx_x!, ddx_y!, ddx_z!, laplacian!
export shift!, shift, normdiff
export dot, norm, minnormdiff
export FarazmandWeight
export CartesianPrimitive, CartesianPrimitiveNSE, CartesianPrimitiveLNSE, Forward, AdjointDiscrete, AdjointContinuous
export CartesianPrimitiveRotatingNSE, CartesianPrimitiveRotatingLNSE
export CoriolisForce
export ProjectedNSE, construct_equations

include("grid.jl")
include("utility.jl")
include("interface.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("equations.jl")

end
