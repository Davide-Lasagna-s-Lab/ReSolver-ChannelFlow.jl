# Channel-flow extensions to NSEBase.FTField.
# The FTField struct itself lives in NSEBase; this file adds:
#   - VectorField constructors and add_base!
#   - FTField ModeNumber indexing (channel-flow-specific axis ordering)
#   - ProjectedField extensions (project!, expand!, dds!, constructors)
#   - growto, save/load helpers
#

# ------------------------------ #
# vector field constructors       #
# ------------------------------ #
"""
    add_base!(u::VectorField, base)

Add the wall-normal profile `base` to the zero-wavenumber mode of the first
component of `u` (i.e. the streamwise mean flow).  Modifies `u` in-place and
returns it.
"""
NSEBase.add_base!(u::VectorField{N, <:FTField}, base) where {N} = (u[1][1, 1, 1, :] .+= base; return u)


# ------------------------------------ #
# FTField ModeNumber indexing          #
# (channel-flow axis order: t,x,z,y)   #
# ------------------------------------ #

"""
    u[ny, n::ModeNumber]

Return the spectral coefficient of `u` at wall-normal index `ny` and
wavenumber tuple `n = ModeNumber(nx, nz, nt)`.

The wavenumbers follow `fft_dims(grid(u)) = (2, 3, 1)`, i.e. `(nx, nz, nt)`
for streamwise, spanwise, and temporal directions.  The channel-flow array
layout is `(t, x, z, y)`, so the underlying read is
`parent(u)[_nt, _nx, _nz, ny]` with 1-based FFTW indices.

If `nx < 0` the coefficient is obtained by conjugate symmetry:
`û(-nx, -nz, -nt)` is read and conjugated.
"""
Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:AbstractChannelGrid{S}}
    _nx, _nz, _nt, do_conj = NSEBase._modenumber_to_projected_indices(NSEBase.grid(u), n)
    @boundscheck checkbounds(u, _nt, _nx, _nz, ny)
    @inbounds val = parent(u)[_nt, _nx, _nz, ny]
    return do_conj ? conj(val) : val
end

"""
    u[ny, n::ModeNumber] = val

Write the spectral coefficient `val` at wall-normal index `ny` and wavenumber
tuple `n = ModeNumber(nx, nz, nt)`.

For `nx ≠ 0` this is a single write to `parent(u)[_nt, _nx, _nz, ny]`,
with `conj(val)` stored when `nx < 0` (conjugate-symmetry representation).

For `nx = 0` both `(0, nz, nt)` and `(0, -nz, -nt)` exist in storage and
must remain complex conjugates of each other (Hermitian symmetry of the real
field).  Writing one automatically updates the other.  The zero mode
`(nx, nz, nt) = (0, 0, 0)` is additionally forced to be real.
"""
Base.@propagate_inbounds function Base.setindex!(u::FTField{G}, val, ny::Int, n::ModeNumber{N}) where {S, N, G<:AbstractChannelGrid{S}}
    _nx, _nz, _nt, do_conj = NSEBase._modenumber_to_projected_indices(NSEBase.grid(u), n)
    @boundscheck checkbounds(u, _nt, _nx, _nz, ny)
    if n.ns[1] == 0
        # nx = 0: both (0, nz, nt) and (0, -nz, -nt) live in storage and must
        # remain conjugates.  The zero mode (0,0,0) is also forced real.
        T   = real(eltype(u))
        val = (_nz == 1 && _nt == 1) ? Complex{T}(real(val)) : Complex{T}(val)
        _nz_sym = _nz == 1 ? 1 : S[3] - _nz + 2
        _nt_sym = _nt == 1 ? 1 : S[1] - _nt + 2
        @inbounds parent(u)[_nt,     _nx, _nz,     ny] = do_conj ? conj(val) :      val
        @inbounds parent(u)[_nt_sym, _nx, _nz_sym, ny] = do_conj ?      val  : conj(val)
    else
        @inbounds parent(u)[_nt, _nx, _nz, ny] = do_conj ? conj(val) : val
    end
    return val
end


# ---------------- #
# projected fields #
# ---------------- #
NSEBase.ProjectedField(g::AbstractChannelGrid{S, T}, modes) where {S, T} = NSEBase.ProjectedField(g, zeros(Complex{T}, size(modes, 2), (S[2] >> 1) + 1, S[3], S[1]), modes)

@inline _channel_int(u, ws, v, N) = sum(ws[i]*dot(u[i], v[i]) for i in 1:N)
@inline _get_mode(modes, Ny, n, m, nx, nz, nt) = @view(modes[(Ny*(n - 1) + 1):Ny*n, m, nx, nz, nt])

NSEBase.project!(a::ProjectedField{G},
                 u::VectorField{N, <:FTField{G}}) where {S, T, G<:AbstractChannelGrid{S, T}, N} = _project!(a, u, S, N, T)
function _project!(a, u, S, N, T)
    a .= zero(T)
    @loop_modes S[1] S[3] S[2] for m in axes(a, 1), n in 1:N
        @views @inbounds a[m, _nx, _nz, _nt] += _channel_int(_get_mode(NSEBase.modes(a), S[4], n, m, _nx, _nz, _nt),
                                                                       NSEBase.grid(u).ws, u[n][_nt, _nx, _nz, :], S[4])
    end
    return a
end

function NSEBase.expand!(u::VectorField{N, <:FTField{G}},
                         a::ProjectedField{G}) where {N, S, T, G<:AbstractChannelGrid{S, T}}
    u .= zero(T)
    @inbounds begin
        for n in 1:N
            @loop_modes S[1] S[3] S[2] for m in axes(a, 1)
                @view(u[n][_nt, _nx, _nz, :]) .+= a[m, _nx, _nz, _nt].*_get_mode(NSEBase.modes(a), S[4], n, m, _nx, _nz, _nt)
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

function NSEBase.ProjectedNSE(g::AbstractChannelGrid{S, T}, Re; Ro=0, base::Vector=g.y, flags=FFTW.EXHAUSTIVE, mode::M=NSEBase.AdjointDiscrete()) where {S, T, M}
    nl = NSEBase.CartesianPrimitiveNSE(g, Re; Ro=Ro, flags=flags)
    ln = NSEBase.CartesianPrimitiveLNSE(g, Re; Ro=Ro, flags=flags, mode=mode)
    return NSEBase.ProjectedNSE(g, 3, nl, ln, T.(base))
end


# --------------- #
# utility methods #
# --------------- #
function NSEBase.growto(u::FTField{G}, N::NTuple{3, Int}) where {S, G<:AbstractChannelGrid{S}}
    out = NSEBase.FTField(NSEBase.growto(NSEBase.grid(u), N))
    for ny in 1:S[4], nx in 0:(S[2] >> 1), nz in -(S[3] >> 1):(S[3] >> 1), nt in -(S[1] >> 1):(S[1] >> 1)
        out[ny, ModeNumber(nx, nz, nt)] = u[ny, ModeNumber(nx, nz, nt)]
    end
    return out
end

function NSEBase.growto(u::VectorField{L, <:FTField}, N::NTuple{3, Int}) where {L}
    return NSEBase.VectorField([NSEBase.growto(u[n], N) for n in 1:L]...)
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
    return NSEBase.ProjectedField(g, data, modes)
end
