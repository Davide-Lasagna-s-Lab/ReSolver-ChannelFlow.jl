module ReSolverChannelFlow

using FFTW, LinearAlgebra, JLD2

using NSEBase

export AbstractChannelGrid, ChannelGrid, points, growto, get_fields, save_grid, load_grid
export ModeNumber
export FTField, grid, save_field, load_field
export Field
export VectorField
export ProjectedField, modes, project!, project, expand!, expand
export FFT, IFFT, FFTPlans
export dds!, ddx!, add_homogeneous_laplacian!, laplacian!
export shift!
export dot, norm, normdiff, minnormdiff
export FarazmandWeight
export CartesianPrimitiveNSE, CartesianPrimitiveLNSE, CoriolisForce
export ProjectedNSE

include("grid.jl")
include("modenumber.jl")
include("ftfield.jl")
include("field.jl")
include("derivatives.jl")
include("shifts.jl")
include("norms.jl")
include("weighting.jl")
include("equations.jl")

end
