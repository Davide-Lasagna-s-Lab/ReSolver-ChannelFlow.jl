# Channel-flow body force and convenience constructors.
#  CartesianPrimitiveNSE / CartesianPrimitiveLNSE structs live in NSEBase.

# ----------------------- #
# Coriolis body force     #
# ----------------------- #
struct CoriolisForce{T}
    Ro::T
end

(f::CoriolisForce)(out::NSEBase.VectorField{3}, v::NSEBase.VectorField{3}, ::NSEBase.Forward) = begin
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
plane_couette_base(g::AbstractChannelGrid) = copy(g.y)
plane_poiseuille_base(g::AbstractChannelGrid) = @. one(eltype(g.y)) - g.y^2


# ------------------------- #
# flow-level constructors   #
# ------------------------- #
PlaneCouetteFlow(g::AbstractChannelGrid, Re; Ro=0, base=plane_couette_base(g), mode=NSEBase.AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true) =
    _plane_channel_flow(g, Re, base, _coriolis_force(Ro); mode=mode, fftw_flags=fftw_flags, dealias=dealias)

PlanePoiseuilleFlow(g::AbstractChannelGrid, Re; Ro=0, base=plane_poiseuille_base(g), mode=NSEBase.AdjointDiscrete(), fftw_flags=FFTW.EXHAUSTIVE, dealias=true) =
    _plane_channel_flow(g, Re, base, _coriolis_force(Ro); mode=mode, fftw_flags=fftw_flags, dealias=dealias)

_plane_channel_flow(g::AbstractChannelGrid, Re, base, force; mode, fftw_flags, dealias) =
    NSEBase.construct_equations(g, Re, base, NSEBase.CartesianPrimitive(); force=force, mode=mode, flags=fftw_flags, dealias=dealias)

_coriolis_force(Ro) = iszero(Ro) ? NSEBase.NoForce() : CoriolisForce(Ro)
