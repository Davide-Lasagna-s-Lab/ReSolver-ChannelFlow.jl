# Utility object to allow dispatch for different indexing methods on FTField

# ?: I wonder if I could define a @modenumber macro that converts the indexes at parse time and keeps the nice loops?

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
