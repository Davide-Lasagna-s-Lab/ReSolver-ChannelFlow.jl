# Symmetry operators on channel fields.

# TODO: add discrete symmetries: σ₁[u,v,w](x,y,z)->[u,v,-w](x,y,-z), σ₂[u,v,w](x,y,z)->[-u,-v,w](-x,-y,z)
# see: Visualizing the geometry of state space in plane Couette flow, Gibson, Halcrow, Cvitanovic, section 2.2

"""
    shift!(u::F, shifts::NTuple{3, Real}) -> F
    shift(u::F, shifts::NTuple{3, Real}) -> F

Return the channel field `u` shifted in streamwise, spanwise, and temporal
directions.

Apply a continuous shift to the field `u` in the streamwise, spanwise, and
temporal directions, given by `shifts[1]`, `shifts[2]`, and `shifts[3]`
respectively. This function modifies the field it is provided, so be careful
when using.
"""
shift!(u::Union{FTField, VectorField, ProjectedField}, shifts) = tshift!(zshift!(xshift!(u, shifts[1]), shifts[2]), shifts[3])
shift(u::Union{FTField, VectorField, ProjectedField}, shifts) = shift!(copy(u), shifts)


# ---------------- #
# streamwise shift #
# ---------------- #
function xshift!(u::VectorField{N, <:FTField}, sx) where {N}
    for n in 1:N
        xshift!(u[n], sx)
    end
    return u
end
xshift!(u::FTField{<:ChannelGrid1D{S}}, sx) where {S} = _perform_xshift!(u, sx, S[1], S[2], S[3], S[4])
# TODO: needs to be added back once extra grid structure has been added back to NSEBase.jl
# xshift!(a::ProjectedField{F}, sx) where {S, F<:FTField{<:ChannelGrid{S}}} = _perform_xshift!(a, sx, size(a, 1), S[2], S[3], S[4])

function _perform_xshift!(field, sx, S1, S2, S3, S4)
    sx == 0 && return field
    for nx in 0:(S2 >> 1)
        val = cis(nx*sx*grid(field).α)
        for nt in -(S4 >> 1):(S4 >> 1), nz in -(S3 >> 1):(S3 >> 1), m in 1:S1
            @inbounds field[m, ModeNumber(nx, nz, nt)] *= val
        end
    end
    return field
end

# -------------- #
# spanwise shift #
# -------------- #
function zshift!(u::VectorField{N, <:FTField}, sz) where {N}
    for n in 1:N
        zshift!(u[n], sz)
    end
    return u
end
zshift!(u::FTField{<:ChannelGrid1D{S}}, sz) where {S} = _perform_zshift!(u, sz, S[1], S[2], S[3], S[4])
# zshift!(a::ProjectedField{F}, sz) where {S, F<:FTField{<:ChannelGrid{S}}} = _perform_zshift!(a, sz, size(a, 1), S[2], S[3], S[4])

function _perform_zshift!(field, sz, S1, S2, S3, S4)
    sz == 0 && return field
    for nz in -(S3 >> 1):(S3 >> 1)
        val = cis(nz*sz*grid(field).β)
        for nt in -(S4 >> 1):(S4 >> 1), nx in 0:(S2 >> 1), m in 1:S1
            @inbounds field[m, ModeNumber(nx, nz, nt)] *= val
        end
    end
    return field
end


# -------------- #
# temporal shift #
# -------------- #
function tshift!(u::VectorField{N, <:FTField}, st) where {N}
    for n in 1:N
        tshift!(u[n], st)
    end
    return u
end
tshift!(u::FTField{<:ChannelGrid1D{S}}, st) where {S} = _perform_tshift!(u, st, S[1], S[2], S[3], S[4])
# tshift!(a::ProjectedField{F}, st) where {S, F<:FTField{<:ChannelGrid{S}}} = _perform_tshift!(a, st, size(a, 1), S[2], S[3], S[4])

function _perform_tshift!(field, st, S1, S2, S3, S4)
    st == 0 && return field
    for nt in -(S4 >> 1):(S4 >> 1)
        val = cispi(2*nt*st)
        for nz in -(S3 >> 1):(S3 >> 1), nx in 0:(S2 >> 1), m in 1:S1
            @inbounds field[m, ModeNumber(nx, nz, nt)] *= val
        end
    end
    return field
end
