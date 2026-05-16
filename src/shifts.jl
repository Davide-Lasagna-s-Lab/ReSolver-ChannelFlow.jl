# Symmetry operators on channel fields.

# TODO: add discrete symmetries: σ₁[u,v,w](x,y,z)->[u,v,-w](x,y,-z), σ₂[u,v,w](x,y,z)->[-u,-v,w](-x,-y,z)
# see: Visualizing the geometry of state space in plane Couette flow, Gibson, Halcrow, Cvitanovic, section 2.2

"""
    shift!(u::Union{FTField, VectorField}, shifts::NTuple{3, Real})
    shift(u::Union{FTField, VectorField}, shifts::NTuple{3, Real})

Return the channel field `u` shifted in streamwise, spanwise, and temporal
directions.

Apply a continuous shift to the field `u` in the streamwise, spanwise, and
temporal directions, given by `shifts[1]`, `shifts[2]`, and `shifts[3]`
respectively. This function modifies the field it is provided, so be careful
when using.
"""
function shift!(u::FTField{G}, shifts::NTuple{3, Real}) where {S, G<:AbstractChannelGrid{S}}
    sx, sz, st = shifts
    
    any(!=(0), shifts) || return u

    for nt in -(S[4] >> 1):(S[4] >> 1)
        vt = st == 0 ? 1 : cispi(2*nt*st)
        for nz in -(S[3] >> 1):(S[3] >> 1)
            vz = sz == 0 ? 1 : cis(nz*sz*grid(u).β)
            for nx in 0:(S[2] >> 1)
                vx = sx == 0 ? 1 : cis(nx*sx*grid(u).α)
                val = vx * vz * vt
                for ny in 1:S[1]
                    @inbounds u[ModeNumber(nx, nz, nt), ny] *= val
                end
            end
        end
    end

    return u
end
shift!(u::VectorField{N}, shifts) where {N} = (for n in 1:N; shift!(u[n], shifts); end; return u)
shift(u, shifts) = shift!(copy(u), shifts)
