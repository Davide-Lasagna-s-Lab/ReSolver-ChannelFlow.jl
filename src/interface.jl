# Specialised methods for NSEBase.jl types are stored here.

# --------------- #
# FTField methods #
# --------------- #
function growto(u::FTField{G}, N::NTuple{3, Int}) where {S, G<:ChannelGrid1D{S}}
    out = FTField(growto(grid(u), N))
    for ny in 1:S[1], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[4] >> 1):(S[4] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end
growto(u::VectorField{N, <:FTField}, S::NTuple{3, Int}) where {N} = VectorField([growto(u[n], S) for n in 1:N]...)


# ------------------- #
# VectorField methods #
# ------------------- #
NSEBase.add_base!(u::VectorField{N, <:FTField{G}}, base) where {N, G<:ChannelGrid1D} = (u[1][:, 1, 1, 1] .+= base; return u)


# ---------------------- #
# ProjectedField methods #
# ---------------------- #
NSEBase.no_of_modes(modes::Array{ComplexF64, 5}) = size(modes, 2)

@inline _channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)
@inline _get_mode(modes, Ny, n, m, nx, nz, nt) = @view(modes[(Ny*(n - 1) + 1):Ny*n, m, nx, nz, nt])

NSEBase.project!(a::ProjectedField{G},
                 u::VectorField{N, <:FTField{G}}) where {S, T, G<:ChannelGrid1D{S, T}, N} = _project!(a, u, S, N, T)
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
                         a::ProjectedField{G}) where {N, S, T, G<:ChannelGrid1D{S, T}}
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

function dds!(out::ProjectedField{G}, a::ProjectedField{G}) where {S, G<:ChannelGrid1D{S}}
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1)
        @inbounds out[m, _nx, _nz, _nt] = 1im*nt*a[m, _nx, _nz, _nt]
    end
    return out
end

function NSEBase.ProjectedNSE(g::ChannelGrid1D{S, T}, Re; Ro=0, base::Vector=g.y, flags=FFTW.EXHAUSTIVE, mode=AdjointDiscrete()) where {S, T}
    # construct operators
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:4]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    nl = CartesianPrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
    ln = CartesianPrimitiveLNSE{typeof(mode)}(T(Re), T(Ro), plans, scache, pcache)
    return ProjectedNSE(g, 3, nl, ln, T.(base))
end

function save_field(a::ProjectedField{<:FTField{<:ChannelGrid1D}}; path="./a.jld2")
    jldopen(path, "w") do f
        f["data"] = parent(a)
    end
    return nothing
end

function load_field(g::ChannelGrid1D, modes, path)
    # read coefficients of projected field
    data = jldopen(path, "r") do f
        return f["data"]
    end

    return ProjectedField(typeof(FTField(g)), data, modes)
end
