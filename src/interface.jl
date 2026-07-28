# NSEBase method extensions for AbstractChannelGrid.
#
# `NSEBase.derivative_matrix` fetches the operator responsible for taking derivatives
# of `ORDER`th along the inhomogeneous direction `STORAGE_DIM`. It is utilised in the
# inhomogeneous methods for NSEBase.dd! and NSEBase.laplacian!.
# 
# `Adapt.adapt_structure` handles to movement of data from host (CPU) to device (GPU),
# allowing the use of CUDA kernels for GPU-accelerations.

"""
    NSEBase.derivative_matrix(g::AbstractChannelGrid,
                       stor_dim,
                      Val(order),
                           mode::NSEBase.OperatorMode)

Return the wall-normal FD matrix for derivative `order` and adjoint flag `adj`.

`MPIExt._dd_over!` calls this on the parent (serial) grid of a
`DecomposedGrid` to obtain the local stencil matrix before applying it to each
rank's wall-normal slab. Defining this method here rather than in a package
extension avoids boilerplate: `MPIExt` is a direct dependency and the
method is always needed for any MPI run. Without it the decomposed derivative
kernel throws a `MethodError` at runtime.
"""
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{1},
                           ::NSEBase.AdjointDiscrete) = g.D₁⁺
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{1},
                           ::NSEBase.Forward) = g.D₁
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{2},
                           ::NSEBase.AdjointDiscrete) = g.D₂⁺
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{2},
                           ::NSEBase.Forward) = g.D₂
NSEBase.derivative_matrix(::AbstractChannelGrid,
                          ::Int,
                          ::Val{ORDER},
                          ::NSEBase.OperatorMode) where {ORDER} =
    throw(ArgumentError("only orders 1 and 2 are available, got order=$ORDER"))


function Adapt.adapt_structure(to, g::ChannelGrid{S}) where {S}
    y   = Adapt.adapt_structure(to, g.y)
    D₁  = Adapt.adapt_structure(to, g.D₁)
    D₂  = Adapt.adapt_structure(to, g.D₂)
    D₁⁺ = Adapt.adapt_structure(to, g.D₁⁺)
    D₂⁺ = Adapt.adapt_structure(to, g.D₂⁺)
    ws  = Adapt.adapt_structure(to, g.ws)
    return ChannelGrid{S, Float32}(y, D₁, D₂, D₁⁺, D₂⁺, ws, Float32(g.α), Float32(g.β))
end
