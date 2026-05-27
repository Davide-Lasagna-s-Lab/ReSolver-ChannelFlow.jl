# Assumptions and Conventions

This package implements the plane-channel parts of the ReSolver/NSEBase
interface. It assumes one inhomogeneous wall-normal direction and three
homogeneous Fourier directions.

## Coordinates And Storage

The logical physical coordinates are ordered as `(x, y, z, t)`.

`ChannelGrid` stores the physical, pre-dealiasing size tuple as
`S = (Nx, Ny, Nz, Nt)`, where `Nx`, `Nz`, and `Nt` are the numbers of grid
points in physical space before dealiasing, and `Ny` is the number of
wall-normal collocation points.

Arrays are stored in the order `(y, x, z, t)`. This is encoded by
`CHANNEL_AXES = (2, 1, 3, 4)`, meaning that coordinate `x` lives on array
dimension 2, `y` on dimension 1, `z` on dimension 3, and `t` on dimension 4.

```text
logical coordinates:     x      y      z      t
coordinate index:        1      2      3      4
                          \    /        \      \
storage dimension:         2  1          3      4

array layout:           ( y,     x,     z,     t )
Base.size(g):           ( Ny,    Nx,    Nz,    Nt )
```

The wall-normal coordinate is placed first in storage so that wall-normal
derivative matrices can be applied along contiguous slices of a field.

## Transform Directions

The Fourier transform directions are the homogeneous array dimensions
`CHANNEL_FFT_ORDER = (2, 3, 4)`, corresponding to `(x, z, t)`.

The inhomogeneous dimensions are `CHANNEL_INHOMOGENEOUS_DIMS = (1,)`,
corresponding to `y`.

`NSEBase.points(g)` returns broadcastable coordinate arrays in storage order,
namely `(y, x, z, t)`. The homogeneous coordinates span half-open periodic
intervals:

- `x ∈ [0, 2π/α)`
- `z ∈ [0, 2π/β)`
- `t ∈ [0, 1)`

The wall-normal coordinates are supplied by the caller as `g.y`.

## Size Constraints

`Nx`, `Nz`, and `Nt` must be odd. NSEBase uses these physical sizes before
dealiasing and can grow the homogeneous dimensions to padded sizes when needed.

`Ny` is not grown by `NSEBase.growto`. Dealiasing changes only the homogeneous
directions because the wall-normal direction is represented by caller-supplied
collocation points and differentiation operators.

The supplied wall-normal data must be compatible:

- `length(y) == Ny`
- `length(ws) == Ny`
- `size(D1) == size(D2) == size(D1p) == size(D2p) == (Ny, Ny)`

Here `D1p` and `D2p` denote the adjoint derivative operators passed as `D₁⁺`
and `D₂⁺`.

## Wall-Normal Operators

`ChannelGrid` does not construct differentiation matrices. The caller supplies
the first derivative, second derivative, and their weighted adjoints.

The only operator contract required by ChannelFlow is that each derivative
operator can be applied along the inhomogeneous storage dimension through
`LinearAlgebra.mul!`:

```julia
LinearAlgebra.mul!(out, D, u, Val(CHANNEL_INHOMOGENEOUS_DIMS[1]))
```

This keeps the package independent of the specific wall-normal discretisation.
Dense matrices, Chebyshev operators, and finite-difference operators can all be
used when they satisfy that contract.

## Inner Products

`NSEBase.weights(g)` returns the wall-normal quadrature weights `g.ws`. Generic
NSEBase routines use these weights for the inhomogeneous part of weighted inner
products, projections, and norm computations.

The derivative adjoints `D₁⁺` and `D₂⁺` should be chosen consistently with
these weights. ChannelFlow stores and applies the adjoints; it does not verify
that they are mathematically adjoint to `D₁` and `D₂`.

## Flow Constructors

`PlaneCouetteFlow(g, Re; ...)` builds equations with the default base flow
`U(y) = y`.

`PlanePoiseuilleFlow(g, Re; ...)` builds equations with the default base flow
`U(y) = 1 - y^2` and a constant mean pressure-gradient forcing.

Both constructors accept:

- `Ro`, which adds a `CoriolisForce` when nonzero.
- `base`, a three-entry velocity tuple `(U, V, W)` with `nothing` for absent
  components.
- `mode`, forwarded to the NSEBase linearised equation constructor.
- `fftw_flags`, forwarded to FFTW planning.
- `dealias`, which controls allocation of dealiased physical-space caches.

Poiseuille flow additionally accepts `f`, the amplitude of the constant
streamwise forcing.

## What The Package Does Not Enforce

ChannelFlow does not impose boundary conditions, build wall-normal operators,
or prove that the supplied adjoints match the supplied quadrature. Those choices
belong to the discretisation used to construct `ChannelGrid`.

It also does not change the generic ReSolver/NSEBase conventions for spectral
storage, Hermitian symmetry, projection, or time-stepping. Those behaviours live
in `NSEBase.jl`.
