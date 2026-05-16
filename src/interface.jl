# Specialised methods for NSEBase.jl types are stored here.


# ------------------- #
# VectorField methods #
# ------------------- #
NSEBase.add_base!(u::VectorField{N, <:FTField{G}}, base) where {N, G<:AbstractChannelGrid} = (u[1][:, 1, 1, 1] .+= base; return u)


# ------------------ #
# derivative methods #
# ------------------ #
NSEBase.ddx!(out::FTField{G}, u::FTField{G}, ::Val{1}; adjoint=false) where {G<:ChannelGrid} =
    mul!(out, adjoint ? grid(u).Dya : grid(u).Dy, u)

NSEBase.inhomogeneous_laplacian!(out::FTField{G}, u::FTField{G}; adjoint::Bool=false) where {G<:ChannelGrid} =
    mul!(out, adjoint ? grid(u).Dy2a : grid(u).Dy2, u)


# ---------------------- #
# ProjectedField methods #
# ---------------------- #
NSEBase.no_of_modes(modes::Array{ComplexF64, 5}) = size(modes, 2)

@inline _channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)
@inline _get_mode(modes, Ny, n, m, nx, nz, nt) = @view(modes[(Ny*(n - 1) + 1):Ny*n, m, nx, nz, nt])


# TODO: implement these generically in NSEBase.jl
NSEBase.project!(a::ProjectedField{G},
                 u::VectorField{N, <:FTField{G}}) where {S, T, G<:AbstractChannelGrid{S, T}, N} = _project!(a, u, S, N, T)
function _project!(a, u, S, N, T)
    a .= zero(T)
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1), n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += _channel_int(_get_mode(modes(a), S[1], n, m, _nx, _nz, _nt), grid(u).ws, u[n][:, _nx, _nz, _nt], S[1])
    end
    return a
end

# ! splitting the modes into three seperate objects for each velocity component is faster
# ! allocations are somehow related to FDGrids.jl, the broadcasting is kind of broken???
function NSEBase.expand!(u::VectorField{N, <:FTField{G}},
                         a::ProjectedField{G}) where {N, S, T, G<:AbstractChannelGrid{S, T}}
    u .= zero(Complex{T})
    @inbounds begin
        for n in 1:N
            @loop_modes S[4] S[3] S[2] for m in axes(a, 1)
                @view(u[n][:, _nx, _nz, _nt]) .+= a[m, _nx, _nz, _nt].*_get_mode(modes(a), S[1], n, m, _nx, _nz, _nt)
                # @views u[n][:, _nx, _nz, _nt] .+= a[m, _nx, _nz, _nt].*mds[n][:, m, _nx, _nz, _nt]
            end
        end
    end
    return u
end

# TODO: low-hanging fruit for migration to NSEBase.jl
function save_field(a::ProjectedField{<:FTField{<:AbstractChannelGrid}}; path="./a.jld2")
    jldopen(path, "w") do f
        f["data"] = parent(a)
    end
    return nothing
end

function load_field(g::AbstractChannelGrid, modes, path)
    # read coefficients of projected field
    data = jldopen(path, "r") do f
        return f["data"]
    end

    return ProjectedField(typeof(FTField(g)), data, modes)
end
