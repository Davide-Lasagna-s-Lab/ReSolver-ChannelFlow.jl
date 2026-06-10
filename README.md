<p align="center">
  <img src="docs/src/assets/logo.svg" alt="ReSolverChannelFlow.jl logo" width="680">
</p>

[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://Davide-Lasagna-s-Lab.github.io/ReSolver-ChannelFlow.jl/dev/)

# ReSolverChannelFlow.jl

`ReSolverChannelFlow.jl` provides plane-channel grids, wall-normal derivative
hooks, and convenience constructors for Couette and Poiseuille flow models built
on top of `NSEBase.jl`.

The package is aimed at spectral and resolvent workflows with one inhomogeneous
wall-normal direction and homogeneous streamwise, spanwise, and temporal
directions.

## Features

- `ChannelGrid` for plane-channel storage with physical sizes `(Nx, Ny, Nz, Nt)`.
- NSEBase-compatible points, weights, growth, and wall-normal derivative hooks.
- Caller-supplied wall-normal differentiation matrices and weighted adjoints.
- Plane Couette and plane Poiseuille equation constructors.
- Optional rotation through the `Ro` keyword and Poiseuille mean forcing.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Davide-Lasagna-s-Lab/ReSolver-ChannelFlow.jl")
```

## Example

```julia
using FDGrids
using FFTW
using LinearAlgebra
using ReSolverChannelFlow

Ny, Nx, Nz, Nt = 33, 31, 31, 15
g_pts = grid(Ny, -1.0, 1.0, GaussLobattoGrid())
D1  = DiffMatrix(g_pts.xs, 5, 1)
D2  = DiffMatrix(g_pts.xs, 5, 2)
D1⁺ = adjoint(D1, g_pts.ws)
D2⁺ = adjoint(D2, g_pts.ws)

g = ChannelGrid(
    g_pts.xs, Nx, Nz, Nt,
    2π, π,
    D1, D2, D1⁺, D2⁺,
    g_pts.ws,
)

equations = PlaneCouetteFlow(g, 1000; Ro = 0, fftw_flags = FFTW.ESTIMATE)
```

`PlanePoiseuilleFlow` has the same interface, with a default parabolic base flow
and a constant mean pressure-gradient forcing.

## Documentation

The full documentation is available at the
[development documentation site](https://Davide-Lasagna-s-Lab.github.io/ReSolver-ChannelFlow.jl/dev/).

Useful entry points:

- [Assumptions and Conventions](docs/src/manual/conventions.md)
- [API Reference](docs/src/api.md)
