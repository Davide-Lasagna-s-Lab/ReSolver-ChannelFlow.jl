# Norm definitions and special scaling for channel flow fields.

# ----------- #
# other norms #
# ----------- #
function normdiff(u::FTField{G}, v::FTField{G}, shifts=(0, 0, 0), tmp::FTField{G}=zero(v)) where {S, T, G<:AbstractChannelGrid{S, T}}
    sum = zero(T)
    tmp .= v
    shift!(tmp, shifts)
    @loop_nznt S[4] S[3] for ny in 1:S[1]
        @inbounds sum += grid(u).ws[ny]*abs2(u[ny, 1, _nz, _nt] - tmp[ny, 1, _nz, _nt])
    end
    @loop_nznt S[4] S[3] for _nx in 2:(S[2] >> 1) + 1, ny in 1:S[1]
        @inbounds sum += 2*grid(u).ws[ny]*abs2(u[ny, _nx, _nz, _nt] - tmp[ny, _nx, _nz, _nt])
    end
    return sqrt(sum/2)
end

function normdiff(u::VectorField{N, <:FTField{G}}, v::VectorField{N, <:FTField{G}}, shifts=(0, 0, 0), tmp::FTField{G}=zero(u[1])) where {N, S, T, G<:AbstractChannelGrid{S, T}}
    sum = zero(T)
    for n in 1:N
        sum += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(sum)
end

function normdiff(a::ProjectedField{G}, b::ProjectedField{G}, shifts=(0, 0, 0), tmp::ProjectedField{G}=zero(b)) where {S, T, G<:AbstractChannelGrid{S, T}}
    throw(error("Method does not working for projected fields"))
    sum = zero(T)
    tmp .= b
    shift!(tmp, shifts)
    @loop_nznt S[4] S[3] for m in axes(a, 1)
        @inbounds sum += abs2(a[m, 1, _nz, _nt] - tmp[m, 1, _nz, _nt])
    end
    @loop_nznt S[4] S[3] for _nx in 2:(S[2] >> 1) + 1, m in axes(a, 1)
        @inbounds sum += 2*abs2(a[m, _nx, _nz, _nt] - tmp[m, _nx, _nz, _nt])
    end
    return sqrt(sum/2)
end

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
