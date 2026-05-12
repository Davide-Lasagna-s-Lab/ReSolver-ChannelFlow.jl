# Utility object to allow dispatch for different indexing methods on FTField

# -------------------- #
# mode number indexing #
# -------------------- #
struct ModeNumber
    nx::Int
    nz::Int
    nt::Int
end

function _convert_modenumber(n::ModeNumber, Nz, Nt)
    if n.nx >= 0
        _nx = n.nx + 1
        _nz = n.nz >= 0 ? n.nz + 1 : Nz + n.nz + 1
        _nt = n.nt >= 0 ? n.nt + 1 : Nt + n.nt + 1
        do_conj = false
    else
        _nx = -n.nx + 1
        _nz = n.nz > 0 ? Nz - n.nz + 1 : -n.nz + 1
        _nt = n.nt > 0 ? Nt - n.nt + 1 : -n.nt + 1
        do_conj = true
    end
    return _nx, _nz, _nt, do_conj
end

Base.@propagate_inbounds function Base.getindex(u::FTField{G}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid1D{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(u[ny, _nx, _nz, _nt]) : u[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(u::FTField{G}, val, ny::Int, n::ModeNumber) where {S, T, G<:ChannelGrid1D{S, T}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[4] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(u, ny, _nx, _nz, _nt)
                @inbounds u[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds u[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end

Base.@propagate_inbounds function Base.getindex(a::ProjectedField{<:FTField{G}}, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid1D{S}}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
    @inbounds val = do_conj ? conj(a[ny, _nx, _nz, _nt]) : a[ny, _nx, _nz, _nt]
    return val
end
Base.@propagate_inbounds function Base.setindex!(a::ProjectedField{<:FTField{G}, T}, val, ny::Int, n::ModeNumber) where {S, G<:ChannelGrid1D{S}, T}
    _nx, _nz, _nt, do_conj = _convert_modenumber(n, S[3], S[4])
    _nz_sym = _nz != 1 ? S[3] - _nz + 2 : _nz
    _nt_sym = _nt != 1 ? S[3] - _nt + 2 : _nt
    val = (_nx == _nz == _nt == 1) ? Complex{T}(real(val)) : val
    @boundscheck checkbounds(a, ny, _nx, _nz, _nt)
                @inbounds a[ny, _nx, _nz,     _nt]     = do_conj ? conj(val) :      val
    _nx == 1 && @inbounds a[ny, _nx, _nz_sym, _nt_sym] = do_conj ?      val  : conj(val)
    return val
end


# ----------- #
# macro stuff #
# ----------- #
macro loop_modes(Nt, Nz, Nx, expr)
    quote
        for $(esc(:_nt)) in 1:($(esc(Nt)) >> 1) + 1
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1, $(esc(:_nx)) in 1:($(esc(Nx)) >> 1) + 1
                $(esc(:nx)) = $(esc(:_nx)) - 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz)), $(esc(:_nx)) in 1:($(esc(Nx)) >> 1) + 1
                $(esc(:nx)) = $(esc(:_nx)) - 1
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
        end
        for $(esc(:_nt)) in ($(esc(Nt)) >> 1) + 2:$(esc(Nt))
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1, $(esc(:_nx)) in 1:($(esc(Nx)) >> 1) + 1
                $(esc(:nx)) = $(esc(:_nx)) - 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz)), $(esc(:_nx)) in 1:($(esc(Nx)) >> 1) + 1
                $(esc(:nx)) = $(esc(:_nx)) - 1
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
        end
    end
end

macro loop_nznt(Nt, Nz, expr)
    quote
        for $(esc(:_nt)) in 1:($(esc(Nt)) >> 1) + 1
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz))
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - 1
                $(esc(expr))
            end
        end
        for $(esc(:_nt)) in ($(esc(Nt)) >> 1) + 2:$(esc(Nt))
            for $(esc(:_nz)) in 1:($(esc(Nz)) >> 1) + 1
                $(esc(:nz)) = $(esc(:_nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
            for $(esc(:_nz)) in ($(esc(Nz)) >> 1) + 2:$(esc(Nz))
                $(esc(:nz)) = $(esc(:_nz)) - $(esc(Nz)) - 1
                $(esc(:nt)) = $(esc(:_nt)) - $(esc(Nt)) - 1
                $(esc(expr))
            end
        end
    end
end
