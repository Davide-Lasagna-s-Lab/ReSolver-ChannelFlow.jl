# Channel-flow body force and convenience constructors.
# CartesianPrimitiveNSE / CartesianPrimitiveLNSE structs live in NSEBase.

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


# ------------------------- #
# channel-flow constructors #
# ------------------------- #
CartesianPrimitiveNSE(g::AbstractChannelGrid, Re; Ro=0, flags=FFTW.EXHAUSTIVE) =
    CartesianPrimitiveNSE(g, Re; force=CoriolisForce(oftype(g.α, Ro)), flags=flags)

CartesianPrimitiveLNSE(g::AbstractChannelGrid, Re; Ro=0, mode=AdjointDiscrete(), flags=FFTW.EXHAUSTIVE) =
    CartesianPrimitiveLNSE(g, Re; mode=mode, force=CoriolisForce(oftype(g.α, Ro)), flags=flags)
