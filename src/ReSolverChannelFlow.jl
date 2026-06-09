module ReSolverChannelFlow

import FFTW
import LinearAlgebra
import NSEBase

export ChannelGrid, AbstractChannelGrid
export CHANNEL_AXES, CHANNEL_FFT_ORDER, CHANNEL_INHOMOGENEOUS_DIMS
export PlaneCouetteFlow, PlanePoiseuilleFlow
export CoriolisForce, ConstantForcing

# Shared field, transform, shift, norm, and equation helpers are provided by NSEBase.
# TODO: add discrete channel symmetries:
# σ₁[u,v,w](x,y,z)->[u,v,-w](x,y,-z)
# σ₂[u,v,w](x,y,z)->[-u,-v,w](-x,-y,z)
# See Gibson, Halcrow, Cvitanovic, "Visualizing the geometry of state space
# in plane Couette flow", section 2.2.

include("grid.jl")
include("interface.jl")
include("equations.jl")

end
