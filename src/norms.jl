# Norm definitions and special scaling for channel flow fields.

function minnormdiff(u::Union{FTField{G}, VectorField{D, <:FTField{G}}, ProjectedField{<:FTField{G}}},
                     v::Union{FTField{G}, VectorField{D, <:FTField{G}}, ProjectedField{<:FTField{G}}},
                     N::NTuple{3, Int}=(32, 32, 32),
                  tmp1::FTField{G}=zero(v),
                  tmp2::FTField{G}=zero(v)) where {D, G}
    # minimum values
    min_diff = Inf
    sx_min   = Inf
    sz_min   = Inf
    st_min   = Inf

    # get shift steps
    _sx = (2π/grid(u).α)/N[1]
    _sz = (2π/grid(u).β)/N[2]
    _st = 1/N[3]

    # loop over available z and t shifts
    tmp1 .= v
    for ti in 0:N[3] - 1
        for zi in 0:N[2] - 1
            for xi in 0:N[1] - 1
                diff = normdiff(u, tmp1, (0, 0, 0), tmp2)
                if diff < min_diff
                    min_diff = diff
                    sx_min = _sx*xi
                    sz_min = _sz*zi
                    st_min = _st*ti
                end
                shift!(tmp1, (_sx, 0, 0))
            end
            shift!(tmp1, (0, _sz, 0))
        end
        shift!(tmp1, (0, 0, _st))
    end

    return min_diff, (sx_min, sz_min, st_min)
end
