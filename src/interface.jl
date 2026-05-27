# Specialised methods for NSEBase.jl types are stored here.

# ── derivative methods ───────────────────────────────────────────────────────
# parent() is called explicitly to unwrap FTField to its plain Array.
# This is necessary because FTField does not implement Base.strides, so views
# of FTField are not recognised as strided arrays by BLAS. Unwrapping first
# ensures mul! dispatches to the fast BLAS / ChebUtils / FDGrids kernels.

"""
    NSEBase.ddx!(out::FTField{G}, u::FTField{G}, ::Val{1}; adjoint=false) where {G<:AbstractChannelGrid}

Apply the wall-normal derivative to a channel-flow Fourier field.

For the default channel layout, the wall-normal direction is storage dimension
`1`, exposed as `CHANNEL_INHOMOGENEOUS_DIMS[1]`. With `adjoint=false`, this
method applies `grid(u).D₁`; with `adjoint=true`, it applies `grid(u).D₁⁺`.

The derivative operator itself is supplied by the grid and must implement
dimension-wise `LinearAlgebra.mul!` along the inhomogeneous dimension.
"""
function NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, ::Val{CHANNEL_INHOMOGENEOUS_DIMS[1]}; adjoint=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end

"""
    NSEBase.inhomogeneous_laplacian!(out::FTField{G}, u::FTField{G}; adjoint=false) where {G<:AbstractChannelGrid}

Apply the wall-normal contribution to the Laplacian of a channel-flow Fourier
field.

With `adjoint=false`, this method applies `grid(u).D₂`; with `adjoint=true`,
it applies `grid(u).D₂⁺`. Homogeneous Fourier contributions are handled by the
generic NSEBase Laplacian routines.
"""
function NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}; adjoint::Bool=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end
