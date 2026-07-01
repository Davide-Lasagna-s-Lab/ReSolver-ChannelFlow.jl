# NSEBase method extensions for AbstractChannelGrid.
#
# `NSEBase.derivative_matrix` fetches the operator responsible for taking derivatives
# of `ORDER`th along the inhomogeneous direction `STORAGE_DIM`. It is utilised in the
# inhomogeneous methods for NSEBase.dd! and NSEBase.laplacian!.

"""
    NSEBase.derivative_matrix(g::AbstractChannelGrid, stor_dim, Val(order), Val(adj))

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
                           ::Val{true}) = g.D₁⁺
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{1},
                           ::Val{false}) = g.D₁
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{2},
                           ::Val{true}) = g.D₂⁺
NSEBase.derivative_matrix(g::AbstractChannelGrid,
                           ::Int,
                           ::Val{2},
                           ::Val{false}) = g.D₂
NSEBase.derivative_matrix(::AbstractChannelGrid,
                          ::Int,
                          ::Val{ORDER},
                          ::Val{ADJ}) where {ORDER, ADJ} =
    throw(ArgumentError("only orders 1 and 2 are available, got order=$ORDER"))
