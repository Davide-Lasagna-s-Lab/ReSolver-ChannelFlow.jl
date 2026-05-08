# Derivative methods for flow fields.
#
# ddx!(out, u, Val(d)) for d ∈ {1,2,3} (homogeneous directions) is provided
# generically by NSEBase via wavenumber_scale.  Only the wall-normal direction
# (d = 4, non-homogeneous) needs an explicit override here.
#
# laplacian! = wall-normal Dy2 multiply + NSEBase's add_homogeneous_laplacian!

# wall-normal (y) derivative — non-homogeneous, uses differentiation matrix
NSEBase.ddx!(out::F, u::F, ::Val{4}; adjoint::Bool=false) where {F<:Union{Field, FTField}} =
    mul!(out, adjoint ? grid(u).Dya : grid(u).Dy, u, Val(4))

function laplacian!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false) where {S, G<:ChannelGrid{S}}
    mul!(out, adjoint ? grid(u).Dy2a : grid(u).Dy2, u, Val(4))
    add_homogeneous_laplacian!(out, u)
    return out
end
