# API Reference

This page collects the ChannelFlow-specific API. For the generic spectral-field,
projection, transform, and Navier-Stokes interfaces, see `NSEBase.jl`.

## Grid Layout

```@docs
ReSolverChannelFlow.CHANNEL_AXES
ReSolverChannelFlow.AbstractChannelGrid
ReSolverChannelFlow.ChannelGrid
Base.size(::ReSolverChannelFlow.AbstractChannelGrid)
```

## Grid Hooks

```@docs
NSEBase.points(::ReSolverChannelFlow.ChannelGrid; dealias)
NSEBase.points(::ReSolverChannelFlow.ChannelGrid, ::NTuple{3, Int})
NSEBase.wavenumber_scale(::ReSolverChannelFlow.AbstractChannelGrid, ::Int)
NSEBase.weights(::ReSolverChannelFlow.AbstractChannelGrid)
NSEBase.growto(::ReSolverChannelFlow.ChannelGrid, ::NTuple{3, Int})
```

## Wall-Normal Operators

```@docs
NSEBase.ddx!(::NSEBase.FTField{G}, ::NSEBase.FTField{G}, ::Val{1}; adjoint=false) where {G <: ReSolverChannelFlow.AbstractChannelGrid}
NSEBase.inhomogeneous_laplacian!(::NSEBase.FTField{G}, ::NSEBase.FTField{G}; adjoint::Bool=false) where {G <: ReSolverChannelFlow.AbstractChannelGrid}
```

## Forces

```@docs
ReSolverChannelFlow.CoriolisForce
ReSolverChannelFlow.ConstantForcing
```

## Base Flows

```@docs
ReSolverChannelFlow.plane_couette_base
ReSolverChannelFlow.plane_poiseuille_base
```

## Equation Constructors

```@docs
ReSolverChannelFlow.PlaneCouetteFlow
ReSolverChannelFlow.PlanePoiseuilleFlow
```
