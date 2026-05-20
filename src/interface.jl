# Specialised methods for NSEBase.jl types are stored here.

# ------------------ #
# derivative methods #
# ------------------ #
NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, ::Val{CHANNEL_INHOMOGENEOUS_DIMS[1]}; adjoint=false) where {G<:AbstractChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, u)

NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}; adjoint::Bool=false) where {G<:AbstractChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, u)


