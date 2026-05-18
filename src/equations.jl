# Channel-flow body force and convenience constructors.
#  CartesianPrimitiveNSE / CartesianPrimitiveLNSE structs live in NSEBase.

# ----------------------- #
# Coriolis body force     #
# ----------------------- #
"""
    CoriolisForce{T}

Body force implementing solid-body rotation at rotation number `Ro`,
i.e. the Coriolis term `Ro (ẑ × u)` added to the streamwise–wall-normal
momentum equations.

In the forward direction the force adds `+Ro v` to the streamwise equation
and `-Ro u` to the wall-normal equation.  The discrete and continuous adjoints
flip both signs.
"""
struct CoriolisForce{T}
    Ro::T
end

(f::CoriolisForce)(out::NSEBase.VectorField{3}, 
                     v::NSEBase.VectorField{3}, 
                      ::NSEBase.Forward) = begin
    @. out[1] += f.Ro*v[2]
    @. out[2] -= f.Ro*v[1]
end

(f::CoriolisForce)(out::NSEBase.VectorField{3},
                     v::NSEBase.VectorField{3},
                      ::Union{NSEBase.AdjointDiscrete, NSEBase.AdjointContinuous}) = begin
    @. out[1] -= f.Ro*v[2]
    @. out[2] += f.Ro*v[1]
end


# --------------------------------- #
# canonical channel-flow base flows #
# --------------------------------- #
"""
    plane_couette_base(g::AbstractChannelGrid) -> Vector

Return the laminar plane-Couette base-flow profile `U(y) = y` evaluated at
the wall-normal collocation points of `g`.
"""
plane_couette_base(g::AbstractChannelGrid) = copy(g.y)

"""
    plane_poiseuille_base(g::AbstractChannelGrid) -> Vector

Return the laminar plane-Poiseuille base-flow profile `U(y) = 1 - y²`
evaluated at the wall-normal collocation points of `g`.
"""
plane_poiseuille_base(g::AbstractChannelGrid) = @. one(eltype(g.y)) - g.y^2


# ------------------------- #
# flow-level constructors   #
# ------------------------- #
"""
    PlaneCouetteFlow(g, Re; Ro=0, base=(plane_couette_base(g), nothing, nothing),
                     mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)

Construct a [`ProjectedNSE`](@ref) for plane-Couette flow at Reynolds number
`Re` on the channel grid `g`.

# Keyword arguments
- `Ro`: inverse Rotation number. When non-zero a [`CoriolisForce`](@ref) is added.
- `base`: laminar base flow as a 3-tuple `(U, V, W)` with one entry per velocity
  component; use `nothing` for components with no base flow.  Defaults to the
  Couette profile `U(y) = y` in the streamwise slot.
- `mode`: adjoint mode for the linearised operator. Defaults to
  `NSEBase.AdjointDiscrete()`.
- `fftw_flags`: FFTW planner flags. Defaults to `FFTW.EXHAUSTIVE`.
- `dealias`: allocate physical-space caches on the dealiased grid. Defaults to
  `true`.
"""
PlaneCouetteFlow(g::AbstractChannelGrid, Re; Ro=0, base=(plane_couette_base(g), nothing, nothing), mode=NSEBase.AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true) =
    _plane_channel_flow(g, Re, base, _coriolis_force(Ro); mode=mode, fftw_flags=fftw_flags, dealias=dealias)

"""
    PlanePoiseuilleFlow(g, Re; Ro=0, base=(plane_poiseuille_base(g), nothing, nothing),
                        mode=AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true)

Construct a [`ProjectedNSE`](@ref) for plane-Poiseuille flow at Reynolds number
`Re` on the channel grid `g`.

# Keyword arguments
- `Ro`: inverse Rotation number. When non-zero a [`CoriolisForce`](@ref) is added.
- `base`: laminar base flow as a 3-tuple `(U, V, W)` with one entry per velocity
  component; use `nothing` for components with no base flow.  Defaults to the
  Poiseuille profile `U(y) = 1 - y²` in the streamwise slot.
- `mode`: adjoint mode for the linearised operator. Defaults to
  `NSEBase.AdjointDiscrete()`.
- `fftw_flags`: FFTW planner flags. Defaults to `FFTW.EXHAUSTIVE`.
- `dealias`: allocate physical-space caches on the dealiased grid. Defaults to
  `true`.
"""
PlanePoiseuilleFlow(g::AbstractChannelGrid, Re; Ro=0, base=(plane_poiseuille_base(g), nothing, nothing), mode=NSEBase.AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true) =
    _plane_channel_flow(g, Re, base, _coriolis_force(Ro); mode=mode, fftw_flags=fftw_flags, dealias=dealias)

_plane_channel_flow(g::AbstractChannelGrid, Re, base, force; mode, fftw_flags, dealias) =
    NSEBase.construct_equations(g, Re, base, NSEBase.CartesianPrimitive(); force=force, mode=mode, flags=fftw_flags, dealias=dealias)

_coriolis_force(Ro) = iszero(Ro) ? NSEBase.NoForce() : CoriolisForce(Ro)
