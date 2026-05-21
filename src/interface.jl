# Specialised methods for NSEBase.jl types are stored here.

# ── derivative methods ───────────────────────────────────────────────────────
# parent() is called explicitly to unwrap FTField to its plain Array.
# This is necessary because FTField does not implement Base.strides, so views
# of FTField are not recognised as strided arrays by BLAS. Unwrapping first
# ensures mul! dispatches to the fast BLAS / ChebUtils / FDGrids kernels.

function NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, ::Val{CHANNEL_INHOMOGENEOUS_DIMS[1]}; adjoint=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end

function NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}; adjoint::Bool=false) where {G<:AbstractChannelGrid}
    LinearAlgebra.mul!(parent(out), adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, parent(u), Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
    return out
end
