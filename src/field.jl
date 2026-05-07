# Implementation of physical representation of scalar channel fields

struct Field{G, T, A<:AbstractArray{T, 4}} <: AbstractArray{T, 4}
    grid::G
    data::A

    # generic constructor for halo and CUDA arrays
    function Field(g::G, data::A) where {S, T, G<:Abstract1DChannelGrid{S, T}, A}
        new{G, T, A}(g, T.(data))
    end

    # sequential array constructor
    function Field(g::G, data::Array) where {S, T, G<:Abstract1DChannelGrid{S, T}}
        all(isodd.(size(data)[2:4])) || throw(ArgumentError("data must be odd"))
        new{G, T, Array{T, 4}}(g, T.(data))
    end
end

Field(g::ChannelGrid{S, T}, fun, period::Real; dealias::Bool=false) where {S, T} =
        _field_from_function(g, points(g, period, _padded_size((S[2], S[3], S[4]), Val(dealias))), (y, x, z, t)->T(fun(y, x, z, t)))
Field(g::Abstract1DChannelGrid{S, T}; dealias::Bool=false) where {S, T} = Field(g, (y, x, z, t)->zero(T), 1.0, dealias=dealias)

function _field_from_function(g, pts, fun)
    y, x, z, t = pts
    data = fun.(reshape(y, :, 1, 1, 1), reshape(x, 1, :, 1, 1), reshape(z, 1, 1, :, 1), reshape(t, 1, 1, 1, :))
    return Field(g, data)
end


# ------------- #
# array methods #
# ------------- #
Base.IndexStyle(::Type{<:Field})                      = Base.IndexLinear()
Base.parent(u::Field)                                 = u.data
Base.size(u::Field)                                   = size(parent(u))
Base.eltype(::Field{G, T}) where {G, T}               = T
Base.similar(u::Field, ::Type{T}=eltype(u)) where {T} = Field(similar(grid(u), T), zero(parent(u)))

Base.@propagate_inbounds function Base.getindex(u::Field, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds return parent(u)[i]
end
Base.@propagate_inbounds function Base.setindex!(u::Field, v, i::Int)
    @boundscheck checkbounds(parent(u), i)
    @inbounds parent(u)[i] = v
    return v
end


# --------------- #
# utility methods #
# --------------- #
grid(u::Field) = u.grid

function _padded_size(shape::NTuple{3, Int}, ::Val{true})
    new_shape = zeros(Int, 3)
    for (i, s) in enumerate(shape)
        new_shape[i] = (3*s)>>1 + 1 - ((3*s)>>1)&1
    end
    return tuple(new_shape...)
end
_padded_size(sizes::NTuple{3, Int}, ::Val{false}) = sizes


# --------------------- #
# allocating transforms #
# --------------------- #
function FFT(u::Field{G, T}) where {S, G<:Abstract1DChannelGrid{S}, T}
    û = FTField(grid(u))
    parent(û) .= rfft(parent(u), [2, 3, 4])
    û .*= 1/prod(S[2:4])
    return û
end

function FFT(u::Field{G, T}, N) where {S, G<:Abstract1DChannelGrid{S}, T}
    û = growto(FTField(grid(u), rfft(parent(u), [2, 3, 4])./prod(S[2:4])), N)
    return û
end

FFT(u::VectorField{L, P})    where {L, P<:Field} = VectorField([FFT(u[n])    for n in 1:L]...)
FFT(u::VectorField{L, P}, N) where {L, P<:Field} = VectorField([FFT(u[n], N) for n in 1:L]...)

function IFFT(û::FTField{G}) where {S, G<:Abstract1DChannelGrid{S}}
    u = Field(grid(û))
    parent(u) .= brfft(parent(û), S[2], [2, 3, 4])
    return u
end

function IFFT(û::FTField{G}, N) where {G<:Abstract1DChannelGrid}
    u = Field(growto(grid(û), N))
    parent(u) .= brfft(parent(growto(û, N)), N[1], [2, 3, 4])
    return u
end

IFFT(u::VectorField{L, S})    where {L, S<:FTField} = VectorField([IFFT(u[n])    for n in 1:L]...)
IFFT(u::VectorField{L, S}, N) where {L, S<:FTField} = VectorField([IFFT(u[n], N) for n in 1:L]...)
