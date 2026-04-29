# Implementation of the RPCF grid

abstract type Abstract1DChannelGrid{S, T} end

struct ChannelGrid{S, T, D1, D2, D3, D4} <: Abstract1DChannelGrid{S, T}
    y::Vector{T}
    Dy::D1
    Dy2::D2
    Dya::D3
    Dy2a::D4
    ws::Vector{T}
    α::T
    β::T

    ChannelGrid{S, T}(y::Vector{T},
                     Dy::AbstractMatrix{T},
                    Dy2::AbstractMatrix{T},
                    Dya::AbstractMatrix{T},
                   Dy2a::AbstractMatrix{T},
                     ws::Vector{T},
                      α::T,
                      β::T) where {S, T} = new{S, T, typeof(Dy), typeof(Dy2), typeof(Dya), typeof(Dy2a)}(y, Dy, Dy2, Dya, Dy2a, ws, α, β)
end

function ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, ::Type{T}=Float64; adjoint_diff::Bool=true) where {T}
    (isodd(Nx) && isodd(Nz) && isodd(Nt)) || throw(ArgumentError("grid must be odd in streamwise, spanwise, and time directions"))
    length(y) == length(ws) == size(Dy, 1) == size(Dy2, 1) || throw(ArgumentError("grid variables not compatible sizes"))
    ws = T.(ws)
    Dy = T.(Dy)
    Dy2 = T.(Dy2)
    Dya  = adjoint_diff ? adjoint(Dy,  ws) : Dy
    Dy2a = adjoint_diff ? adjoint(Dy2, ws) : Dy2
    return ChannelGrid{(length(y), Nx, Nz, Nt), T}(T.(y), Dy, Dy2, Dya, Dy2a, ws, T(α), T(β))
end

# ! change to convert
Base.similar(g::ChannelGrid{S, T}, ::Type{U}=T) where {S, T, U} =
    U == T ? g : ChannelGrid{S, U}(U.(g.y), U.(g.Dy), U.(g.Dy2), U.(g.Dya), U.(g.Dy2a), U.(g.ws), U(g.α), U(g.β))

# get points from grid
points(g::ChannelGrid{S}, T) where {S}       = (                           g.y,
                                                (0:(S[2] - 1))/(S[2])*(2π/g.α),
                                                (0:(S[3] - 1))/(S[3])*(2π/g.β),
                                                (0:(S[4] - 1))/(S[4])*T)
points(g::ChannelGrid, T, S::NTuple{3, Int}) = (                           g.y,
                                                (0:(S[1] - 1))/(S[1])*(2π/g.α),
                                                (0:(S[2] - 1))/(S[2])*(2π/g.β),
                                                (0:(S[3] - 1))/(S[3])*T)

# grow grid size
growto(g::ChannelGrid{S, T}, N::NTuple{3, Int}) where {S, T} = ChannelGrid{(S[1], N...), T}(g.y, get_fields(g)...)

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
