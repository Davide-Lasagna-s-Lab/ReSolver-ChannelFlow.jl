# Specialised methods for NSEBase.jl types are stored here.


# ------------------- #
# VectorField methods #
# ------------------- #
NSEBase.add_base!(u::NSEBase.VectorField{N, <:NSEBase.FTField{G}}, base) where {N, G<:AbstractChannelGrid} = (u[1][:, 1, 1, 1] .+= base; return u)


# ------------------ #
# derivative methods #
# ------------------ #
NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, ::Val{1}; adjoint=false) where {G<:ChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).Dya : NSEBase.grid(u).Dy, u)

NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}; adjoint::Bool=false) where {G<:ChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).Dy2a : NSEBase.grid(u).Dy2, u)


# ---------------------- #
# ProjectedField methods #
# ---------------------- #
NSEBase.no_of_modes(modes::NTuple{3, Array{ComplexF64, 5}}) = size(modes[1], 2)

NSEBase.get_mode_coefficient(modes::NTuple{3, Array{ComplexF64, 5}}, 
                                  ::AbstractChannelGrid, 
                                 n::Int, 
                                 m::Int, 
                               inh::NTuple{1}, spectral...) = modes[n][inh[1], m, spectral...]
