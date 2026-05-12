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
export dds!, ddx1!, ddx2!, ddx3!, laplacian!
export shift!
export dot, norm, normdiff, minnormdiff
export FarazmandWeight
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE, CartesianPrimitiveALNSE, Forward, AdjointDiscrete, AdjointContinuous
export ProjectedNSE

include("grid.jl")
include("modenumber.jl")
include("interface.jl")
include("derivatives.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("cartesianprimitive.jl")

end
