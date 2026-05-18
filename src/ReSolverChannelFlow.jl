module ReSolverChannelFlow

import FFTW
import JLD2
import LinearAlgebra
import NSEBase
import NSEBase: points, growto

export ChannelGrid, get_fields
export points, growto
export FarazmandWeight
export PlaneCouetteFlow, PlanePoiseuilleFlow
export CoriolisForce

# Shared field, transform, shift, norm, and equation helpers are provided by NSEBase.
# TODO: add discrete channel symmetries:
# σ₁[u,v,w](x,y,z)->[u,v,-w](x,y,-z)
# σ₂[u,v,w](x,y,z)->[-u,-v,w](-x,-y,z)
# See Gibson, Halcrow, Cvitanovic, "Visualizing the geometry of state space
# in plane Couette flow", section 2.2.

include("grid.jl")
include("utility.jl")
include("interface.jl")
include("weighting.jl")
include("equations.jl")

end
