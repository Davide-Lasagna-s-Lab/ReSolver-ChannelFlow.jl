# Specialised methods for NSEBase.jl types are stored here.


# ------------------- #
# VectorField methods #
# ------------------- #
NSEBase.add_base!(u::VectorField{N, <:FTField{G}}, base) where {N, G<:AbstractChannelGrid} = (u[1][:, 1, 1, 1] .+= base; return u)


# ------------------ #
# derivative methods #
# ------------------ #
NSEBase.ddx!(out::FTField{G}, u::FTField{G}, ::Val{1}; adjoint=false) where {G<:ChannelGrid} =
    mul!(out, adjoint ? grid(u).Dya : grid(u).Dy, u)

NSEBase.inhomogeneous_laplacian!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false) where {G<:ChannelGrid} =
    mul!(out, adjoint ? grid(u).Dy2a : grid(u).Dy2, u)


# ---------------------- #
# ProjectedField methods #
# ---------------------- #
NSEBase.no_of_modes(modes::NTuple{3, Array{ComplexF64, 5}}) = size(modes[1], 2)

NSEBase.get_mode_coefficient(modes::NTuple{3, Array{ComplexF64, 5}}, 
                                  ::AbstractChannelGrid, 
                                 n::Int, 
                                 m::Int, 
                               inh::NTuple{1}, spectral...) = modes[n][inh[1], m, spectral...]