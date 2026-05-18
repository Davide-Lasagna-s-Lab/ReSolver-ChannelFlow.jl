# Norm weight applied to the dot product of projected fields

# TODO: constructor with grid
struct FarazmandWeight{T<:Real}
    ω::T
    β::T
    α::T
end
Base.getindex(A::FarazmandWeight, nx::Int, nz::Int, nt::Int) = 1/(1 + (A.α*nx)^2 + (A.β*nz)^2 + (A.ω*nt)^2)

function LinearAlgebra.mul!(a::NSEBase.ProjectedField{G}, A::FarazmandWeight) where {S, G<:AbstractChannelGrid{S}}
    Nx, _, Nz, Nt = S
    @loop_modes Nt Nz Nx for m in axes(a, 1)
        @inbounds a[m, _nx, _nz, _nt] *= A[nx, nz, nt]
    end
    return a
end

function LinearAlgebra.dot(a::NSEBase.ProjectedField{G}, A::FarazmandWeight, b::NSEBase.ProjectedField{G}) where {S, G<:AbstractChannelGrid{S}}
    Nx, _, Nz, Nt = S
    sum = 0.0
    @loop_nznt Nt Nz for m in axes(a, 1)
        @inbounds sum += A[0, nz, nt]*real(LinearAlgebra.dot(a[m, 1, _nz, _nt], b[m, 1, _nz, _nt]))
    end
    @loop_nznt Nt Nz for _nx in 2:(Nx >> 1) + 1, m in axes(a, 1)
        @inbounds sum += 2*A[_nx-1, nz, nt]*real(LinearAlgebra.dot(a[m, _nx, _nz, _nt], b[m, _nx, _nz, _nt]))
    end
    return sum/2
end
