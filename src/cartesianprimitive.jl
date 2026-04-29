# Operators for the primitive formulation of the Navier-Stokes equation for
# Couette type flows.

# ---------------------- #
# navier-stokes operator #
# ---------------------- #
mutable struct CartesianPrimitiveNSE{G, T, FFT, A, B} <: NSE{FTField} # TODO: need to check performance when not fully typing FTField!
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, FTField{G, T, A}}}
    const pcache::Vector{VectorField{3,   Field{G, T, B}}}

    CartesianPrimitiveNSE(Re::T,
                          Ro::T,
                       plans::FFT,
                      scache::Vector{<:VectorField{3, <:FTField{G, T, A}}},
                      pcache::Vector{<:VectorField{3,   <:Field{G, T, B}}}) where {T, FFT, G, A, B} =
        new{G, T, FFT, A, B}(Re, Ro, plans, scache, pcache)
end

function CartesianPrimitiveNSE(g::Abstract1DChannelGrid{S, T}, Re; Ro=0, flags=FFTW.EXHAUSTIVE) where {S, T}
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:3]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:4]
    CartesianPrimitiveNSE(T(Re), T(Ro), plans, scache, pcache)
end

function (eq::CartesianPrimitiveNSE)(::Real,
                                    u::VectorField{3, F},
                                  out::VectorField{3, F}) where {F<:FTField}
    # aliases
    dudx = eq.scache[1]
    dudy = eq.scache[2]
    dudz = eq.scache[3]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]

    # compute viscous term
    laplacian!(out, u)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dudx, u)
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # advection term
    eq.plans(U, u)
    eq.plans(dUdx, dudx)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)
    for n in 1:3
        @. dUdx[n] = -U[1]*dUdx[n] - U[2]*dUdy[n] - U[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, dUdx, true) # add the result to the output

    # coriolis term
    @. out[1] += eq.Ro*u[2]
    @. out[2] -= eq.Ro*u[1]

    return out
end


# ---------------------------------- #
# linearised navier-stokes operators #
# ---------------------------------- #
abstract type               Mode end
struct Forward           <: Mode end
struct AdjointDiscrete   <: Mode end
struct AdjointContinuous <: Mode end

mutable struct CartesianPrimitiveLNSE{MODE, T, FFT, S, P} <: LNSE{FTField}
              Re::T
              Ro::T
     const plans::FFT
    const scache::Vector{VectorField{3, S}}
    const pcache::Vector{VectorField{3, P}}

    CartesianPrimitiveLNSE{MODE}(Re::T,
                                 Ro::T,
                              plans::FFT,
                             scache::Vector{<:VectorField{3, S}},
                             pcache::Vector{<:VectorField{3, P}}) where {MODE, T, FFT, S, P} =
        new{MODE, T, FFT, S, P}(Re, Ro, plans, scache, pcache)
end

function CartesianPrimitiveLNSE(g::Abstract1DChannelGrid{S, T}, Re; Ro=0, flags=FFTW.EXHAUSTIVE, mode::M=Forward()) where {S, T, M<:Mode}
    plans = FFTPlans(S, (2, 3, 4), T, flags=flags)
    scache = [VectorField([FTField(g)               for _ in 1:3]...) for _ in 1:4]
    pcache = [VectorField([  Field(g, dealias=true) for _ in 1:3]...) for _ in 1:8]
    return CartesianPrimitiveLNSE{M}(T(Re), T(Ro), plans, scache, pcache)
end

function (eq::CartesianPrimitiveLNSE)(::Real,
                                     u::VectorField{3, F},
                                     v::VectorField{3, F},
                                   out::VectorField{3, F}) where {F<:FTField}
    # aliases
    dudx = eq.scache[1]
    dudy = eq.scache[2]
    dudz = eq.scache[3]
    U    = eq.pcache[1]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]

    # compute base derivatives
    ddx1!(dudx, u)
    ddx2!(dudy, u)
    ddx3!(dudz, u)

    # transform base field
    eq.plans(U, u)
    eq.plans(dUdy, dudy)
    eq.plans(dUdz, dudz)

    # compute the rest
    eq(0, v, out)

    return out
end

function (eq::CartesianPrimitiveLNSE{Forward})(::Real,
                                              v::VectorField{3, F},
                                            out::VectorField{3, F}) where {F<:FTField}
    # aliases
    dudx = eq.scache[1]
    dvdx = eq.scache[2]
    dvdy = eq.scache[3]
    dvdz = eq.scache[4]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    V    = eq.pcache[5]
    dVdx = eq.pcache[6]
    dVdy = eq.pcache[7]
    dVdz = eq.pcache[8]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dvdx, v)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdx, dudx) # has to be recomputed since nonlinear equation overwrites this field with other data
    eq.plans(dVdx, dvdx)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n]  = -U[1]*dVdx[n] - U[2]*dVdy[n] - U[3]*dVdz[n] # overwrite field to save memory
        @. dVdx[n] -=  V[1]*dUdx[n] + V[2]*dUdy[n] + V[3]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, dVdx, true) # add the result to the output

    # coriolis term
    @. out[1] += eq.Ro*v[2]
    @. out[2] -= eq.Ro*v[1]

    return out
end

function (eq::CartesianPrimitiveLNSE{AdjointContinuous})(::Real,
                                                        v::VectorField{3, F},
                                                      out::VectorField{3, F}) where {F<:FTField}
    # aliases
    dudx = eq.scache[1]
    dvdx = eq.scache[2]
    dvdy = eq.scache[3]
    dvdz = eq.scache[4]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    V    = eq.pcache[5]
    dVdx = eq.pcache[6]
    dVdy = eq.pcache[7]
    dVdz = eq.pcache[8]

    # compute viscous term
    laplacian!(out, v)
    out .*= 1/eq.Re

    # compute derivatives
    ddx1!(dvdx, v)
    ddx2!(dvdy, v)
    ddx3!(dvdz, v)

    # advection term
    eq.plans(V, v)
    eq.plans(dUdx, dudx) # has to be recomputed since nonlinear equation overwrites this field with other data
    eq.plans(dVdx, dvdx)
    eq.plans(dVdy, dvdy)
    eq.plans(dVdz, dvdz)
    for n in 1:3
        @. dVdx[n] = U[1]*dVdx[n] + U[2]*dVdy[n] + U[3]*dVdz[n] # overwrite field to save memory
    end
    dVdz .= 0.0
    for i in 1:3
        @. dVdz[1] -= V[i]*dUdx[i] # overwrite field to save memory
        @. dVdz[2] -= V[i]*dUdy[i] # overwrite field to save memory
        @. dVdz[3] -= V[i]*dUdz[i] # overwrite field to save memory
    end
    eq.plans(out, dVdx, true) # add the result to the output
    eq.plans(out, dVdz, true) # add the result to the output

    # coriolis term
    @. out[1] -= eq.Ro*v[2]
    @. out[2] += eq.Ro*v[1]

    return out
end

function (eq::CartesianPrimitiveLNSE{AdjointDiscrete})(::Real,
                                                      v::VectorField{3, F},
                                                    out::VectorField{3, F}) where {F<:FTField}
    # aliases
    dudx = eq.scache[1]
    u1v  = eq.scache[2]
    u2v  = eq.scache[3]
    u3v  = eq.scache[4]
    U    = eq.pcache[1]
    dUdx = eq.pcache[2]
    dUdy = eq.pcache[3]
    dUdz = eq.pcache[4]
    V    = eq.pcache[5]
    U1V  = eq.pcache[6]
    U2V  = eq.pcache[7]
    U3V  = eq.pcache[8]

    # compute viscous term
    laplacian!(out, v, adjoint=true)
    out .*= 1/eq.Re

    # compute products
    eq.plans(V, v)
    for n in 1:3
        @. U1V[n] = U[1]*V[n]
        @. U2V[n] = U[2]*V[n]
        @. U3V[n] = U[3]*V[n]
    end
    eq.plans(u1v, U1V)
    eq.plans(u2v, U2V)
    eq.plans(u3v, U3V)

    # advection term
    eq.plans(dUdx, dudx) # has to be recomputed since nonlinear equation overwrites this field with other data
    for n in 1:3
        out[n] .+= ddx1!(dudx[1], u1v[n]) .- ddx2!(dudx[2], u2v[n], adjoint=true) .+ ddx3!(dudx[3], u3v[n])
    end
    U1V .= 0.0
    for n in 1:3
        @. U1V[1] -= V[n]*dUdx[n] # overwrite field to save memory
        @. U1V[2] -= V[n]*dUdy[n] # overwrite field to save memory
        @. U1V[3] -= V[n]*dUdz[n] # overwrite field to save memory
    end
    eq.plans(out, U1V, true) # add the result to the output

    # coriolis term
    @. out[1] -= eq.Ro*v[2]
    @. out[2] += eq.Ro*v[1]

    return out
end

# ! remove if I can
NSEBase.ndim(::Union{CartesianPrimitiveNSE, CartesianPrimitiveLNSE}) = 3
