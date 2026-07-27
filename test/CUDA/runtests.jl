using CUDA

const CUDAExt = Base.get_extension(NSEBase, :CUDAExt)

include("test_derivatives.jl")
include("test_equations.jl")
