# Specialised methods for NSEBase.jl types are stored here.

# ------------------ #
# derivative methods #
# ------------------ #
# TODO: understand the consequences of using AbstractChannelGrid instead of ChannelGrid here.. the issue could be how this works
# for the MPI code, perhaps we do not need to do anything special for the MPI code, because the mul! code will just work
NSEBase.ddx!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}, ::Val{CHANNEL_INHOMOGENEOUS_DIMS[1]}; adjoint=false) where {G<:ChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).D₁⁺ : NSEBase.grid(u).D₁, u)

NSEBase.inhomogeneous_laplacian!(out::NSEBase.FTField{G}, u::NSEBase.FTField{G}; adjoint::Bool=false) where {G<:ChannelGrid} =
    LinearAlgebra.mul!(out, adjoint ? NSEBase.grid(u).D₂⁺ : NSEBase.grid(u).D₂, u)


# ---------------------- #
# ProjectedField methods #
# ---------------------- #
# TODO: this I do not understand... 
NSEBase.no_of_modes(modes::NTuple{3, Array{ComplexF64, 5}}) = size(modes[1], 2)

NSEBase.get_mode_coefficient(modes::NTuple{3, Array{ComplexF64, 5}}, 
                                  ::AbstractChannelGrid, 
                                 n::Int, 
                                 m::Int, 
                               inh::NTuple{1}, spectral...) = modes[n][inh[1], m, spectral...]
