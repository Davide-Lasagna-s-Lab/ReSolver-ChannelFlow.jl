# Implementation of the RPCF grid

#TODO: pass a tuple to FFTPlans to allow the user select the best first transformed dimension 
# this however may require additional code to understand which is the first transformed dimesion in other code

# S    = (Nt, Nx, Nz, Ny): data is stored as (t, x, z, y)
# Axes = (2, 4, 3, 1): x→dim2, y→dim4, z→dim3, t→dim1
# Hs   = (2, 3): streamwise (rfft) and spanwise (fft) spatial homogeneous dims
# Ht   = 1:      temporal (fft) homogeneous dim
abstract type AbstractChannelGrid{S, T} <: NSEBase.AbstractGrid{T, 4, (2, 4, 3, 1), (2, 3), 1} end

Base.size(::AbstractChannelGrid{S}) where {S} = S

struct ChannelGrid{S, T, D1, D2, D3, D4} <: AbstractChannelGrid{S, T}
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
    return ChannelGrid{(Nt, Nx, Nz, length(y)), T}(T.(y), Dy, Dy2, Dya, Dy2a, ws, T(α), T(β))
end

# ! change to convert
Base.similar(g::ChannelGrid{S, T}, ::Type{U}=T) where {S, T, U} =
    U == T ? g : ChannelGrid{S, U}(U.(g.y), U.(g.Dy), U.(g.Dy2), U.(g.Dya), U.(g.Dy2a), U.(g.ws), U(g.α), U(g.β))

# get points from grid — period-based form for constructing physical fields
points(g::ChannelGrid{S}, T) where {S}       = ((0:(S[1] - 1))/(S[1])*T,
                                                (0:(S[2] - 1))/(S[2])*(2π/g.α),
                                                (0:(S[3] - 1))/(S[3])*(2π/g.β),
                                                                        g.y)
points(g::ChannelGrid, T, S::NTuple{3, Int}) = ((0:(S[3] - 1))/(S[3])*T,
                                                (0:(S[1] - 1))/(S[1])*(2π/g.α),
                                                (0:(S[2] - 1))/(S[2])*(2π/g.β),
                                                                        g.y)

# NSEBase interface: keyword-based points for zero-field construction
function NSEBase.points(g::AbstractChannelGrid{S}; dealias::Bool=false) where {S}
    Nt, Nx, Nz = padded_size((S[1], S[2], S[3]), Val(dealias))
    return ((0:(Nt - 1))/Nt,
            (0:(Nx - 1))/Nx*(2π/g.α),
            (0:(Nz - 1))/Nz*(2π/g.β),
            g.y)
end

# dim 2 = streamwise (x), dim 3 = spanwise (z), dim 1 = time; α = 2π/Lx, β = 2π/Lz
NSEBase.wavenumber_scale(g::AbstractChannelGrid{S, T}, dim::Int) where {S, T} =
    dim == 2 ? g.α : dim == 3 ? g.β : one(T)

# grow grid size
NSEBase.growto(g::ChannelGrid{S, T}, N::NTuple{3, Int}) where {S, T} =
    ChannelGrid{(N[3], N[1], N[2], S[4]), T}(g.y, get_fields(g)...)

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
