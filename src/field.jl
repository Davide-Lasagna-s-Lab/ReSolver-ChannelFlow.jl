# Channel-flow extensions to NSEBase.Field.
# The Field struct and basic interface (parent, size, eltype, similar,
# getindex, setindex!, grid, FFT, IFFT) are provided by NSEBase.

# period-based Field constructor: builds a physical field from a function
# evaluated at the collocation points for a given temporal period
Field(g::ChannelGrid{S, T}, fun, period::Real; dealias::Bool=false) where {S, T} =
    _field_from_function(g, points(g, period, _padded_size((S[2], S[3], S[4]), Val(dealias))),
                         (y, x, z, t)->T(fun(y, x, z, t)))
Field(g::Abstract1DChannelGrid{S, T}; dealias::Bool=false) where {S, T} =
    Field(g, (y, x, z, t)->zero(T), 1.0, dealias=dealias)

function _field_from_function(g, pts, fun)
    y, x, z, t = pts
    data = fun.(reshape(y, :, 1, 1, 1), reshape(x, 1, :, 1, 1),
                reshape(z, 1, 1, :, 1), reshape(t, 1, 1, 1, :))
    return Field(g, data)
end

function _padded_size(shape::NTuple{3, Int}, ::Val{true})
    _pad(n) = (3n) >> 1 + 1 - ((3n) >> 1) & 1
    return ntuple(i -> _pad(shape[i]), 3)
end
_padded_size(sizes::NTuple{3, Int}, ::Val{false}) = sizes


# --------------------- #
# grow-and-transform    #
# --------------------- #
# Standard FFT/IFFT (no growto) and VectorField variants are in NSEBase.
# These variants grow to a target resolution before/after transforming.
function FFT(u::Field{G}, N) where {S, G<:Abstract1DChannelGrid{S}}
    û = growto(FTField(grid(u), rfft(parent(u), [2, 3, 4])./prod(S[2:4])), N)
    return û
end
FFT(u::VectorField{L, P}, N) where {L, P<:Field} = VectorField([FFT(u[n], N) for n in 1:L]...)

function IFFT(û::FTField{G}, N) where {G<:Abstract1DChannelGrid}
    u = Field(growto(grid(û), N))
    parent(u) .= brfft(parent(growto(û, N)), N[1], [2, 3, 4])
    return u
end
IFFT(u::VectorField{L, S}, N) where {L, S<:FTField} = VectorField([IFFT(u[n], N) for n in 1:L]...)
