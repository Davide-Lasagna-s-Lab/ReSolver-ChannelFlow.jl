# Channel-flow extensions to NSEBase.Field.
# The Field struct and basic interface (parent, size, eltype, similar,
# getindex, setindex!, grid, FFT, IFFT) are provided by NSEBase.

# period-based Field constructor: builds a physical field from a function
# evaluated at the collocation points for a given temporal period

function Field(g::ChannelGrid{S, T}, fun, period::Real; dealias::Bool=false) where {S, T}
    t, x, z, y = points(g, period, padded_size((S[2], S[3], S[1]), Val(dealias)))
    data = fun.(reshape(y, 1, 1, 1, :), reshape(x, 1, :, 1, 1),
                reshape(z, 1, 1, :, 1), reshape(t, :, 1, 1, 1))
    return Field(g, data)
end

Field(g::AbstractChannelGrid{S, T}; dealias::Bool=false) where {S, T} =
    Field(g, (y, x, z, t)->zero(T), 1.0, dealias=dealias)
