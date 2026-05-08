# Channel-flow extensions to NSEBase.FTField.
# The FTField struct itself lives in NSEBase; this file adds:
#   - ModeNumber indexing (physical wavenumber dispatch)
#   - ProjectedField extensions
#   - VectorField / ProjectedField constructors
#   - growto, save/load helpers

# ---------------- #
# indexing methods #
# ---------------- #
Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:AbstractChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[1])
    @boundscheck checkbounds(u, _nt, _nx, _nz, ny)
    @inbounds val = do_conj ? conj(u[_nt, _nx, _nz, ny]) : u[_nt, _nx, _nz, ny]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::FTField{G}, val, ny::Int, n::ModeNumber) where {S, G<:AbstractChannelGrid{S}}
    T = real(eltype(u))
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[1])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[1] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, _nt, _nx, _nz, ny)
                @inbounds u[_nt,     _nx, _nz,     ny] = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds u[_nt_sym, _nx, _nz_sym, ny] = do_conj ?      val  : conj(val)
    return val
end

Base.@propagate_inbounds function Base.getindex(a::ProjectedField{G}, m::Int, n::ModeNumber) where {S, G<:AbstractChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[1])
    @boundscheck checkbounds(a, m, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(a[m, _nx, _nz, _nt]) : a[m, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{G}, val, m::Int, n::ModeNumber) where {S, G<:AbstractChannelGrid{S}}
    T = real(eltype(a))
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[1])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[1] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(a, m, _nx, _nz, _nt)
                @inbounds a[m, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds a[m, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ------------------------------ #
# vector field grid constructors #
# ------------------------------ #
# VectorField(g, FTField; N) is handled generically by NSEBase.
# This variant passes a temporal period to Field's function constructor.
NSEBase.VectorField(g::AbstractChannelGrid, funcs, period; dealias::Bool=false) = VectorField([Field(g, f, period, dealias=dealias) for f in funcs]...)
NSEBase.add_base!(u::VectorField{N, <:FTField}, base) where {N} = (u[1][1, 1, 1, :] .+= base; return u)


# ---------------- #
# projected fields #
# ---------------- #
NSEBase.ProjectedField(g::AbstractChannelGrid{S, T}, modes) where {S, T} = ProjectedField(g, zeros(Complex{T}, size(modes, 2), (S[2] >> 1) + 1, S[3], S[1]), modes)

@inline _channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)
@inline _get_mode(modes, Ny, n, m, nx, nz, nt) = @view(modes[(Ny*(n - 1) + 1):Ny*n, m, nx, nz, nt])

NSEBase.project!(a::ProjectedField{G},
                 u::VectorField{N, <:FTField{G}}) where {S, T, G<:AbstractChannelGrid{S, T}, N} = _project!(a, u, S, N, T)
function _project!(a, u, S, N, T)
    a .= zero(T)
    @loop_modes S[1] S[3] S[2] for m in axes(a, 1), n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += _channel_int(_get_mode(modes(a), S[4], n, m, _nx, _nz, _nt), grid(u).ws, u[n][_nt, _nx, _nz, :], S[4])
    end
    return a
end

function NSEBase.expand!(u::VectorField{N, <:FTField{G}},
                         a::ProjectedField{G}) where {N, S, T, G<:AbstractChannelGrid{S, T}}
    u .= zero(T)
    @inbounds begin
        for n in 1:N
            @loop_modes S[1] S[3] S[2] for m in axes(a, 1)
                @view(u[n][_nt, _nx, _nz, :]) .+= a[m, _nx, _nz, _nt].*_get_mode(modes(a), S[4], n, m, _nx, _nz, _nt)
            end
        end
    end
    return u
end

function dds!(out::ProjectedField{G}, a::ProjectedField{G}) where {S, G<:AbstractChannelGrid{S}}
    @loop_modes S[1] S[3] S[2] for m in axes(a, 1)
        @inbounds out[m, _nx, _nz, _nt] = 1im*nt*a[m, _nx, _nz, _nt]
    end
    return out
end

function NSEBase.ProjectedNSE(g::AbstractChannelGrid{S, T}, Re; Ro=0, base::Vector=g.y, flags=FFTW.EXHAUSTIVE, mode::M=AdjointDiscrete()) where {S, T, M}
    nl = CartesianPrimitiveNSE(g, Re; Ro=Ro, flags=flags)
    ln = CartesianPrimitiveLNSE(g, Re; Ro=Ro, flags=flags, mode=mode)
    return ProjectedNSE(g, 3, nl, ln, T.(base))
end


# --------------- #
# utility methods #
# --------------- #
function NSEBase.growto(u::FTField{G}, N::NTuple{3, Int}) where {S, G<:AbstractChannelGrid{S}}
    out = FTField(growto(grid(u), N))
    for ny in 1:S[4], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[1] >> 1):(S[1] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end

function NSEBase.growto(u::VectorField{L, <:FTField}, N::NTuple{3, Int}) where {L}
    return VectorField([growto(u[n], N) for n in 1:L]...)
end


# ------------------ #
# read-write methods #
# ------------------ #
function save_field(a::ProjectedField{<:AbstractChannelGrid}; path="./a.jld2")
    jldopen(path, "w") do f
        f["data"] = parent(a)
    end
    return nothing
end

function load_field(g::AbstractChannelGrid, modes, path)
    data = jldopen(path, "r") do f
        return f["data"]
    end
    return ProjectedField(g, data, modes)
end
