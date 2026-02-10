# Implementation of the spectral representation of scalar channel fields

# -------------- #
# spectral field #
# -------------- #
struct FTField{G, T, A<:AbstractArray{Complex{T}, 4}} <: AbstractScalarField{4, Complex{T}}
    grid::G
    data::A

    # generic constructor for halo and CUDA arrays
    function FTField(g::G, data::A) where {S, T, G<:Abstract1DChannelGrid{S, T}, A<:AbstractArray}
        new{G, T, A}(g, Complex{T}.(data))
    end

    # sequential array constructor
    function FTField(g::G, data::Array) where {S, T, G<:Abstract1DChannelGrid{S, T}}
        apply_symmetry!(data)
        data[:, 1, 1, 1] .= real.(data[:, 1, 1, 1])
        new{G, T, Array{Complex{T}, 4}}(g, Complex{T}.(data))
    end
end
FTField(g::G) where {S, T, G<:ChannelGrid{S, T}} = FTField(g, zeros(Complex{T}, S[1], (S[2] >> 1) + 1, S[3], S[4]))

Base.parent(u::FTField)                                          = u.data
Base.eltype(::FTField{G, T}) where {G, T}                        = Complex{T}
Base.similar(u::FTField, ::Type{Complex{T}}=eltype(u)) where {T} = FTField(similar(grid(u), T), zero(parent(u)))

NSEBase.hsize(::FTField{<:Abstract1DChannelGrid{S}}) where {S} = ((S[2] >> 1) + 1, S[3], S[4])


# ---------------- #
# indexing methods #
# ---------------- #
# mode number idexing
Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:Abstract1DChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nx, _nz, _nt]) : u[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::FTField{G, T}, val, ny::Int, n::ModeNumber) where {S, G<:Abstract1DChannelGrid{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[4] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
                @inbounds u[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds u[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end

Base.@propagate_inbounds function Base.getindex(a::ProjectedField{<:FTField{G}}, ny::Int, n::ModeNumber) where {S, G<:Abstract1DChannelGrid{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(a[ny, _nx, _nz, _nt]) : a[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{<:FTField{G}, T}, val, ny::Int, n::ModeNumber) where {S, G<:Abstract1DChannelGrid{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[3] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
                @inbounds a[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds a[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ------------------------------ #
# vector field grid constructors #
# ------------------------------ #
NSEBase.VectorField(g::Abstract1DChannelGrid, ::Type{T}=FTField; N::Int=3, kwargs...) where {T} = VectorField([T(g; kwargs...) for _ in 1:N]...)
NSEBase.VectorField(g::Abstract1DChannelGrid, funcs, period; dealias::Bool=false) = VectorField([Field(g, f, period, dealias=dealias) for f in funcs]...)
NSEBase.add_base!(u::VectorField{N, <:FTField}, base) where {N} = (u[1][:, 1, 1, 1] .+= base; return u)


#---------------- #
# projected stuff #
#---------------- #
NSEBase.ProjectedField(g::Abstract1DChannelGrid{S, T}, modes) where {S, T} = ProjectedField(typeof(FTField(g)), zeros(Complex{T}, size(modes, 2), (S[2] >> 1) + 1, S[3], S[4]), modes)

@inline _channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)
@inline _get_mode(modes, Ny, n, m, nx, nz, nt) = @view(modes[(Ny*(n - 1) + 1):Ny*n, m, nx, nz, nt])

NSEBase.project!(a::ProjectedField{<:FTField{G}},
                 u::VectorField{N, <:FTField{G}}) where {S, T, G<:Abstract1DChannelGrid{S, T}, N} = _project!(a, u, S, N, T)
function _project!(a, u, S, N, T)
    a .= zero(T)
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1), n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += _channel_int(_get_mode(modes(a), S[1], n, m, _nx, _nz, _nt), grid(u).ws, u[n][:, _nx, _nz, _nt], S[1])
    end
    return a
end

function NSEBase.expand!(u::VectorField{N, <:FTField{G}},
                         a::ProjectedField{<:FTField{G}}) where {N, S, G<:Abstract1DChannelGrid{S}}
    u .*= 0
    @loop_modes S[4] S[3] S[2] for n in 1:N, m in axes(a, 1)
        @views @inbounds u[n][:, _nx, _nz, _nt] .+= a[m, _nx, _nz, _nt].*_get_mode(modes(a), S[1], n, m, _nx, _nz, _nt)
    end
    return u
end

function dds!(out::ProjectedField{F}, a::ProjectedField{F}) where {S, F<:FTField{<:Abstract1DChannelGrid{S}}}
    @loop_modes S[4] S[3] S[2] for m in axes(a, 1)
        @inbounds out[m, _nx, _nz, _nt] = 1im*nt*a[m, _nx, _nz, _nt]
    end
    return out
end

function NSEBase.ProjectedNSE(g::Abstract1DChannelGrid{S, T}, Re; Ro=0, base::Vector=g.y, flags=FFTW.EXHAUSTIVE, adjoint=true) where {S, T}
    # construct operators
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:6]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    nl = CartesianPrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
    ln = CartesianPrimitiveLNSE(T(Re), T(Ro), plans, scache, pcache, adjoint)

    return ProjectedNSE(scache[1][1], nl, ln, T.(base))
end


# --------------- #
# utility methods #
# --------------- #
grid(u::FTField) = u.grid
grid(u::VectorField) = grid(u[1])

function growto(u::FTField{G}, N::NTuple{3, Int}) where {S, G<:Abstract1DChannelGrid{S}}
    out = FTField(growto(grid(u), N))
    for ny in 1:S[1], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[4] >> 1):(S[4] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end

function growto(u::VectorField{L, <:FTField}, N::NTuple{3, Int}) where {L}
    v = VectorField(growto(u[1], N), N=L)
    for n in 1:L
        parent(v[n]) .= parent(growto(u[n], N))
    end
    return v
end

function apply_symmetry!(u::AbstractArray{T, 4}) where {T}
    Ny, _, Nz, Nt = size(u)
    for nt in 2:Nt, nz in 2:(Nz >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, nz, nt], u[ny, 1, end-nz+2, end-nt+2])
        u[ny, 1,     nz,       nt]   =      av
        u[ny, 1, end-nz+2, end-nt+2] = conj(av)
    end
    for nz in 2:(Nz >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, nz, 1], u[ny, 1, end-nz+2, 1])
        u[ny, 1,     nz,   1] =      av
        u[ny, 1, end-nz+2, 1] = conj(av)
    end
    for nt in 2:(Nt >> 1) + 1, ny in 1:Ny
        av = _average_complex(u[ny, 1, 1, nt], u[ny, 1, 1, end-nt+2])
        u[ny, 1, 1,     nt]   =      av
        u[ny, 1, 1, end-nt+2] = conj(av)
    end
    return u
end

function _average_complex(z1, z2)
    _re = 0.5*(real(z1) + real(z2))
    _im = 0.5*(imag(z1) - imag(z2))
    return _re + 1im*_im
end


# ------------------ #
# read-write methods #
# ------------------ #
function save_field(a::ProjectedField{<:FTField{<:Abstract1DChannelGrid}}; path="./a.jld2")
    jldopen(path, "w") do f
        f["data"] = parent(a)
    end
    return nothing
end

function load_field(g::Abstract1DChannelGrid, modes, path)
    # read coefficients of projected field
    data = jldopen(path, "r") do f
        return f["data"]
    end

    return ProjectedField(typeof(FTField(g)), data, modes)
end
