# NSEBase and NSEBaseMPIExt method extensions for AbstractChannelGrid.
#
# NSEBase.dd! and NSEBase.inhomogeneous_laplacian! are the hooks NSEBase calls
# for the inhomogeneous (non-FFT) wall-normal direction. NSEBaseMPIExt.derivative_matrix
# is the hook NSEBaseMPIExt calls to obtain the local FD matrix for each MPI rank.
#
# parent() is unwrapped before every mul! call. FTField does not implement
# Base.strides, so views of FTField are not recognised as strided arrays by BLAS;
# unwrapping first ensures mul! dispatches to the FDGrids kernel.

"""
    NSEBase.dd!(out, u, Val(1); adjoint=false)

Apply the wall-normal first-order finite-difference derivative.

The wall-normal direction is storage dimension `CHANNEL_INHOMOGENEOUS_DIMS[1]`.
`adjoint=false` applies `D₁`; `adjoint=true` applies `D₁⁺`, the weighted
discrete adjoint so that `dot(D₁ u, v) == dot(u, D₁⁺ v)` under the
wall-normal quadrature weights.
"""
function NSEBase.dd!(out::NSEBase.FTField{G},
                     u::NSEBase.FTField{G},
                     ::Val{CHANNEL_INHOMOGENEOUS_DIMS[1]};
                     adjoint::Bool=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end

"""
    NSEBaseMPIExt.derivative_matrix(g::AbstractChannelGrid, stor_dim, Val(order), Val(adj))

Return the wall-normal FD matrix for derivative `order` and adjoint flag `adj`.

`NSEBaseMPIExt._dd_over!` calls this on the parent (serial) grid of a
`DecomposedGrid` to obtain the local stencil matrix before applying it to each
rank's wall-normal slab. Defining this method here rather than in a package
extension avoids boilerplate: `NSEBaseMPIExt` is a direct dependency and the
method is always needed for any MPI run. Without it the decomposed derivative
kernel throws a `MethodError` at runtime.
"""
# ! this is not how extensions work, need to fixed in NSEBase.jl #19 since this package shouldn't need to depend directly on MPI
# function NSEBaseMPIExt.derivative_matrix(g::AbstractChannelGrid,
#                                           stor_dim::Int,
#                                           ::Val{ORDER},
#                                           ::Val{ADJ}=Val(false)) where {ORDER, ADJ}
#     stor_dim == CHANNEL_INHOMOGENEOUS_DIMS[1] ||
#         throw(ArgumentError("storage dimension $stor_dim is not the inhomogeneous channel direction"))
#     ORDER == 1 && return ADJ ? g.D₁⁺ : g.D₁
#     ORDER == 2 && return ADJ ? g.D₂⁺ : g.D₂
#     throw(ArgumentError("only orders 1 and 2 are available, got order=$ORDER"))
# end

"""
    NSEBase.inhomogeneous_laplacian!(out, u; adjoint=false)

Apply the wall-normal second-order finite-difference operator `∂²/∂y²` in-place.

The homogeneous (streamwise/spanwise/temporal) Fourier contributions are added
separately by `NSEBase.add_homogeneous_laplacian!`. With `adjoint=true` the
second-derivative matrix is replaced by its weighted discrete adjoint `D₂⁺`.
"""
function NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G},
                                          u::NSEBase.FTField{G};
                                          adjoint::Bool=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end
