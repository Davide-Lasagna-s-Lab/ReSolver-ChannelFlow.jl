# Channel-flow extensions to NSEBase.Field.
# The Field struct and basic interface (parent, size, eltype, similar,
# getindex, setindex!, grid, FFT, IFFT) are provided by NSEBase.

"""
    Field(g::ChannelGrid, fun, period; dealias=false)

Construct a physical-space `Field` by evaluating `fun(y, x, z, t)` at the
collocation points of `g`.  The temporal coordinate spans `[0, period)`.
With `dealias=true` the function is sampled on the 3/2-padded grid to suppress
aliasing before the forward FFT.
"""
function Field(g::ChannelGrid{S, T}, fun, period::Real; dealias::Bool=false) where {S, T}
    t, x, z, y = NSEBase.points(g, period, NSEBase.padded_size((S[2], S[3], S[1]), Val(dealias)))
    data = fun.(reshape(y, 1, 1, 1, :), reshape(x, 1, :, 1, 1),
                reshape(z, 1, 1, :, 1), reshape(t, :, 1, 1, 1))
    return NSEBase.Field(g, data)
end

"""
    Field(g::AbstractChannelGrid; dealias=false)

Return a zero-initialised `Field` on `g`.
"""
Field(g::AbstractChannelGrid{S, T}; dealias::Bool=false) where {S, T} =
    Field(g, (y, x, z, t)->zero(T), 1.0, dealias=dealias)

"""
    VectorField(g::AbstractChannelGrid, funcs, period; dealias=false)

Construct a `VectorField` from a collection of functions `funcs` on `g`,
each evaluated at the physical grid points scaled by `period` for the temporal
direction.  Wraps `NSEBase.Field`'s function constructor, which requires the
temporal period; the generic `VectorField(g, FTField; N)` path does not.
"""
NSEBase.VectorField(g::AbstractChannelGrid, funcs, period; dealias::Bool=false) = 
    NSEBase.VectorField([NSEBase.Field(g, f, period, dealias=dealias) for f in funcs]...)
