# Norm definitions and special scaling for channel flow fields.

# ------------------------------------------------------------------ #
# Extensions of NSEBase inner-product infrastructure                  #
# ------------------------------------------------------------------ #

# Quadrature weights for the wall-normal (non-homogeneous) direction.
# dim is ignored here since channel flow has exactly one non-homogeneous dim.
NSEBase._quadrature_weight(g::AbstractChannelGrid, ::Int, ny::Int) = g.ws[ny]

# dot(FTField, FTField) is provided by NSEBase's @generated implementation,
# which dispatches on AbstractGrid{T, D, Axes, Hs, Ht} and calls
# _quadrature_weight above.

# ----------------------- #
# standard inner products #
# ----------------------- #

function LinearAlgebra.dot(a::ProjectedField{G}, b::ProjectedField{G}) where {S, G<:AbstractChannelGrid{S}}
    sum = zero(real(eltype(a)))
    @loop_nznt S[1] S[3] for m in axes(a, 1)
        @inbounds sum += real(dot(a[m, 1, _nz, _nt], b[m, 1, _nz, _nt]))
    end
    @loop_nznt S[1] S[3] for _nx in 2:(S[2] >> 1) + 1, m in axes(a, 1)
        @inbounds sum += 2*real(dot(a[m, _nx, _nz, _nt], b[m, _nx, _nz, _nt]))
    end
    return sum/2
end


# ----------- #
# other norms #
# ----------- #
function normdiff(u::FTField{G}, v::FTField{G}, shifts=(0, 0, 0), tmp::FTField{G}=zero(v)) where {S, G<:AbstractChannelGrid{S}}
    sum = zero(real(eltype(u)))
    tmp .= v
    shift!(tmp, shifts)
    @loop_nznt S[1] S[3] for ny in 1:S[4]
        @inbounds sum += grid(u).ws[ny]*abs2(u[_nt, 1, _nz, ny] - tmp[_nt, 1, _nz, ny])
    end
    @loop_nznt S[1] S[3] for _nx in 2:(S[2] >> 1) + 1, ny in 1:S[4]
        @inbounds sum += 2*grid(u).ws[ny]*abs2(u[_nt, _nx, _nz, ny] - tmp[_nt, _nx, _nz, ny])
    end
    return sqrt(sum/2)
end

function normdiff(u::VectorField{N, <:FTField{G}}, v::VectorField{N, <:FTField{G}}, shifts=(0, 0, 0), tmp::FTField{G}=zero(u[1])) where {N, G<:AbstractChannelGrid}
    sum = zero(real(eltype(u[1])))
    for n in 1:N
        sum += normdiff(u[n], v[n], shifts, tmp)^2
    end
    return sqrt(sum)
end

function normdiff(a::ProjectedField{G}, b::ProjectedField{G}, shifts=(0, 0, 0), tmp::ProjectedField{G}=zero(b)) where {S, G<:AbstractChannelGrid{S}}
    throw(error("Method does not working for projected fields"))
    sum = zero(real(eltype(a)))
    tmp .= b
    shift!(tmp, shifts)
    @loop_nznt S[1] S[3] for m in axes(a, 1)
        @inbounds sum += abs2(a[m, 1, _nz, _nt] - tmp[m, 1, _nz, _nt])
    end
    @loop_nznt S[1] S[3] for _nx in 2:(S[2] >> 1) + 1, m in axes(a, 1)
        @inbounds sum += 2*abs2(a[m, _nx, _nz, _nt] - tmp[m, _nx, _nz, _nt])
    end
    return sqrt(sum/2)
end

function minnormdiff(u::Union{FTField{G}, VectorField{D, <:FTField{G}}, ProjectedField{G}},
                     v::Union{FTField{G}, VectorField{D, <:FTField{G}}, ProjectedField{G}},
                     N::NTuple{3, Int}=(32, 32, 32),
                  tmp1::FTField{G}=zero(v),
                  tmp2::FTField{G}=zero(v)) where {D, G<:AbstractChannelGrid}
    min_diff = Inf
    sx_min   = Inf
    sz_min   = Inf
    st_min   = Inf

    _sx = (2π/grid(u).α)/N[1]
    _sz = (2π/grid(u).β)/N[2]
    _st = 1/N[3]

    tmp1 .= v
    for ti in 0:N[3] - 1
        for zi in 0:N[2] - 1
            for xi in 0:N[1] - 1
                diff = normdiff(u, tmp1, tmp2)
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
