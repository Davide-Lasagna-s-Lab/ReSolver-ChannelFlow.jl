# Implementation of the RPCF grid

abstract type ChannelGrid1D{S, T} <: AbstractGrid{T, 4, (2, 3, 4)} end

struct ChannelGrid{S, T, ADJ, D1, D2, D3, D4} <: ChannelGrid1D{S, T}
    y::Vector{T}
    Dy::D1
    Dy2::D2
    Dya::D3
    Dy2a::D4
    ws::Vector{T}
    α::T
    β::T

    ChannelGrid{S, T, ADJ}(y::Vector{T},
                          Dy::AbstractMatrix{T},
                         Dy2::AbstractMatrix{T},
                         Dya::AbstractMatrix{T},
                        Dy2a::AbstractMatrix{T},
                          ws::Vector{T},
                           α::T,
                           β::T) where {S, T, ADJ} =
        new{S, T, ADJ, typeof(Dy), typeof(Dy2), typeof(Dya), typeof(Dy2a)}(y, Dy, Dy2, Dya, Dy2a, ws, α, β)
end

function ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, ::Type{T}=Float64; adjoint_diff::Bool=true) where {T}
    (isodd(Nx) && isodd(Nz) && isodd(Nt)) || throw(ArgumentError("grid must be odd in streamwise, spanwise, and time directions"))
    length(y) == length(ws) == size(Dy, 1) == size(Dy2, 1) || throw(ArgumentError("grid variables not compatible sizes"))
    ws = T.(ws)
    Dy = T.(Dy)
    Dy2 = T.(Dy2)
    Dya  = adjoint_diff ? adjoint(Dy,  ws) : Dy
    Dy2a = adjoint_diff ? adjoint(Dy2, ws) : Dy2
    return ChannelGrid{(length(y), Nx, Nz, Nt), T, adjoint_diff}(T.(y), Dy, Dy2, Dya, Dy2a, ws, T(α), T(β))
end

Base.convert(::Type{T}, g::ChannelGrid{S, T}) where {S, T} = g
Base.convert(::Type{T}, g::ChannelGrid{S, <:Any, false}) where {T, S} =
    ChannelGrid{S, T, false}(T.(g.y), T.(g.Dy), T.(g.Dy2), T.(g.Dya), T.(g.Dy2a), T.(g.ws), T(g.α), T(g.β))
function Base.convert(::Type{T}, g::ChannelGrid{S, <:Any, true}) where {T, S}
    Dy = T.(g.Dy)
    Dy2 = T.(g.Dy2)
    ws = T.(g.ws)
    return ChannelGrid{S, T, true}(T.(g.y), Dy, Dy2, adjoint(Dy, ws), adjoint(Dy2, ws), ws, T(g.α), T(g.β))
end
Base.size(::ChannelGrid{S}) where {S} = S
NSEBase.fft_norm(::ChannelGrid{S}) where {S} = prod(S[2:4])

# get points from grid
NSEBase.points(g::ChannelGrid{S}; dealias=false) where {S} = (                                                          reshape(g.y, :, 1, 1, 1),
                                                              reshape(_equidistant_points(_padded_size(S[2], Val(dealias)), 2π/g.α), 1, :, 1, 1),
                                                              reshape(_equidistant_points(_padded_size(S[3], Val(dealias)), 2π/g.β), 1, 1, :, 1),
                                                              reshape(_equidistant_points(_padded_size(S[4], Val(dealias))),         1, 1, 1, :))
NSEBase.points(g::ChannelGrid, S::NTuple{3, Int})          = (                              reshape(g.y, :, 1, 1, 1),
                                                              reshape(_equidistant_points(S[1], 2π/g.α), 1, :, 1, 1),
                                                              reshape(_equidistant_points(S[2], 2π/g.β), 1, 1, :, 1),
                                                              reshape(_equidistant_points(S[3]),         1, 1, 1, :))

_equidistant_points(N, L) = (0:(N - 1))/(N)*L
_equidistant_points(N)    = (0:(N - 1))/(N)

_padded_size(s::Int, ::Val{true})  = (3*s)>>1 + 1 - ((3*s)>>1)&1
_padded_size(s::Int, ::Val{false}) = s

# grow grid size
growto(g::ChannelGrid{S, T, ADJ}, N::NTuple{3, Int}) where {S, T, ADJ} =
    ChannelGrid{(S[1], N...), T, ADJ}(g.y, get_fields(g)...)

# utility method to make mode generation easier with Resolvent.jl
get_fields(g::ChannelGrid) = (g.Dy, g.Dy2, g.Dya, g.Dy2a, g.ws, g.α, g.β)

# read-write methods
function save_grid(g::ChannelGrid; path="./grid.jld2")
    jldopen(path, "w") do f
        f["grid"] = g
    end
    return nothing
end

function load_grid(path)
    jldopen(path, "r") do f
        return f["grid"]
    end
end
