# Channel-flow body force and convenience constructors.
#  CartesianPrimitiveNSE / CartesianPrimitiveLNSE structs live in NSEBase.

# ----------------------- #
# Coriolis body force     #
# ----------------------- #
struct CoriolisForce{T}
    Ro::T
end

(f::CoriolisForce)(out::VectorField{3}, v::VectorField{3}, ::Forward) = begin
    @. out[1] += f.Ro*v[2]
    @. out[2] -= f.Ro*v[1]
end

(f::CoriolisForce)(out::VectorField{3}, v::VectorField{3}, ::Union{AdjointDiscrete, AdjointContinuous}) = begin
    @. out[1] -= f.Ro*v[2]
    @. out[2] += f.Ro*v[1]
end


# ---------------------------------- #
# rotating channel-flow constructors #
# ---------------------------------- #
CartesianPrimitiveRotatingNSE(g::AbstractChannelGrid{S, T}, Re; Ro=0, flags=FFTW.EXHAUSTIVE) where {S, T} =
    NSEBase.CartesianPrimitiveNSE(g, Re; force=CoriolisForce(T(Ro)), flags=flags)

CartesianPrimitiveRotatingLNSE(g::AbstractChannelGrid{S, T}, Re; Ro=0, mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE) where {S, T} =
    NSEBase.CartesianPrimitiveLNSE(g, Re; mode=mode, force=CoriolisForce(T(Ro)), flags=flags)

ProjectedCartesianPrimitiveRotatingNSE(g::AbstractChannelGrid{S, T}, Re, base; Ro=0, mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE) where {S, T} =
    NSEBase.construct_equations(g, Re, base, CartesianPrimitive(); force=CoriolisForce(T(Ro)), mode=mode, flags=flags, dealias=true)
