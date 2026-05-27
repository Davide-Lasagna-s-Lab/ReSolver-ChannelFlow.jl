# ReSolverChannelFlow.jl

```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="ReSolverChannelFlow.jl logo" width="680">
</p>
```

`ReSolverChannelFlow.jl` supplies the channel-flow pieces that are specific to
one wall-normal inhomogeneous direction: the grid layout, wall-normal
differentiation hooks, channel quadrature weights, and convenience constructors
for plane Couette and plane Poiseuille flow.

The package deliberately keeps the generic spectral-field machinery in
`NSEBase.jl`. ChannelFlow provides the layout and operators needed to make those
generic algorithms act on plane-channel data.

## Quick Start

```julia
using ChebUtils
using FFTW
using LinearAlgebra
using ReSolverChannelFlow

Ny, Nx, Nz, Nt = 33, 31, 31, 15
D1 = chebdiff(Ny)
D2 = chebddiff(Ny)
ws = chebws(Ny)

g = ChannelGrid(
    chebpts(Ny), Nx, Nz, Nt,
    2π, π,
    D1, D2,
    adjoint(D1, ws), adjoint(D2, ws),
    ws,
)

equations = PlanePoiseuilleFlow(g, 1000; Ro = 0, fftw_flags = FFTW.ESTIMATE)
```

## What To Read

| Page | Purpose |
| --- | --- |
| [Assumptions and Conventions](manual/conventions.md) | Storage order, sizes, transform directions, operator contracts, and flow constructors. |
| [API Reference](api.md) | Public docstrings and method signatures. |
