# Implementation of the channel grid with one inhomogeneous
# (wall-normal) direction.

# -------------------- #
# Grid type hierarchy  #
# -------------------- #
"""
`CHANNEL_AXES`, `CHANNEL_FFT_ORDER`, and `CHANNEL_INHOMOGENEOUS_DIMS` are the
default axis-layout constants for plane-channel grids. They define how the four
physical coordinates `(x, y, z, t)` map onto the axes of a 4D storage array:

    CHANNEL_AXES               = (2, 1, 3, 4)   # coordinate k → array dim CHANNEL_AXES[k]
    CHANNEL_FFT_ORDER          = (2, 3, 4)       # array dims transformed by FFTs
    CHANNEL_INHOMOGENEOUS_DIMS = (1,)            # array dims that are inhomogeneous

With this layout, wall-normal `y` occupies array dimension 1, streamwise `x`
dimension 2, spanwise `z` dimension 3, and temporal `t` dimension 4.
`CHANNEL_FFT_ORDER` lists the array dimensions treated by FFTs (`x`, `z`, `t`),
and `CHANNEL_INHOMOGENEOUS_DIMS` marks array dimension 1 (`y`) as the only
inhomogeneous direction. Alternative layouts can be used by subtyping
`AbstractGrid` directly with different constants.
"""
const CHANNEL_AXES = (2, 1, 3, 4)
const CHANNEL_FFT_ORDER = (2, 3, 4)
const CHANNEL_INHOMOGENEOUS_DIMS = (1,)

"""
    AbstractChannelGrid{S} <: NSEBase.AbstractCartesianGrid3D{Float64, CHANNEL_AXES, CHANNEL_FFT_ORDER}

Abstract supertype for all plane-channel grids using the default axis layout
defined by `CHANNEL_AXES`, `CHANNEL_FFT_ORDER`, and `CHANNEL_INHOMOGENEOUS_DIMS`.

`S` is an `NTuple{4, Int}` giving the physical (pre-dealiasing) size in
physical-coordinate order `(Nx, Ny, Nz, Nt)`, where `Ny` is the wall-normal
resolution and `Nx`, `Nz`, `Nt` are the streamwise, spanwise, and temporal
resolutions. `Base.size(grid)` permutes `S` into array-dimension order for
NSEBase generic code.
"""
abstract type AbstractChannelGrid{S} <: NSEBase.AbstractCartesianGrid3D{Float64, CHANNEL_AXES, CHANNEL_FFT_ORDER} end

"""
    size(g::AbstractChannelGrid{S}) -> NTuple{4, Int}

Return the physical (pre-dealiasing) grid size in array-dimension order. The
type parameter `S = (Nx, Ny, Nz, Nt)` is stored in physical-coordinate order,
so this method permutes it according to `CHANNEL_AXES` and returns
`(Ny, Nx, Nz, Nt)` for the default layout. In distributed contexts a
concrete subtype may override this to return slab-local sizes instead.
"""
Base.size(g::AbstractChannelGrid{S}) where {S} = NSEBase.storage_order(S, g)

# ----------------------------- #
# Concrete grid implementation  #
# ----------------------------- #
"""
    ChannelGrid{S}

Concrete plane-channel grid with one inhomogeneous wall-normal direction and
three homogeneous directions `(x, z, t)`.

The size parameter `S = (Nx, Ny, Nz, Nt)` follows physical-coordinate order.

# Fields
- `y`: wall-normal collocation points.
- `D₁`, `D₂`: first- and second-order wall-normal derivative matrices.
- `D₁⁺`, `D₂⁺`: discrete adjoints of `D₁` and `D₂` with respect to `ws`.
- `ws`: wall-normal quadrature weights.
- `α`: streamwise wavenumber scale `2π/Lx`.
- `β`: spanwise wavenumber scale `2π/Lz`.
"""
struct ChannelGrid{S, Y, D1, D2, D3, D4, W} <: AbstractChannelGrid{S}
    y  :: Y
    D₁ :: D1
    D₂ :: D2
    D₁⁺:: D3
    D₂⁺:: D4
    ws :: W
    α  :: Float64
    β  :: Float64

    function ChannelGrid{S}(  y::AbstractVector,
                              D₁::AbstractMatrix,
                              D₂::AbstractMatrix,
                              D₁⁺::AbstractMatrix,
                              D₂⁺::AbstractMatrix,
                              ws::AbstractVector,
                               α::Real,
                               β::Real) where {S}
        Nx, Ny, Nz, Nt = S
        (isodd(Nx) && isodd(Nz) && isodd(Nt)) || throw(ArgumentError("grid must be odd in streamwise, spanwise, and time directions"))
        length(y) == length(ws) == Ny || throw(ArgumentError("quadrature weights and collocation points have incompatible sizes"))
        size(D₁) == size(D₂) == size(D₁⁺) == size(D₂⁺) == (Ny, Ny) || throw(ArgumentError("differentiation matrices have incompatible sizes"))
        return new{S, typeof(y), typeof(D₁), typeof(D₂), typeof(D₁⁺), typeof(D₂⁺), typeof(ws)}(
            y, D₁, D₂, D₁⁺, D₂⁺, ws, Float64(α), Float64(β))
    end
end


# ------------ #
# Constructors #
# ------------ #
"""
    ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, ws; adjoint_diff=true) -> ChannelGrid

Construct a `ChannelGrid` from wall-normal data and homogeneous grid parameters.

# Arguments
- `y`: wall-normal collocation points (length `Ny`).
- `Nx, Nz, Nt`: pre-dealiasing physical-space point counts in `x`, `z`, `t`.
  All three must be odd (dealiased padded size will be even).
- `α, β`: wavenumber scales `2π/Lx` and `2π/Lz`.
- `D₁, D₂`: `Ny × Ny` first- and second-order wall-normal differentiation matrices.
- `ws`: `Ny` quadrature weights.
- `adjoint_diff`: when `true` (default), compute and store discrete adjoint matrices
  `D₁⁺ = adjoint(D₁, ws)` and `D₂⁺ = adjoint(D₂, ws)`. When `false`,
  `D₁⁺` and `D₂⁺` alias the forward matrices.
"""
function ChannelGrid(  y::AbstractVector,
                      Nx::Int,
                      Nz::Int,
                      Nt::Int,
                       α::Real,
                       β::Real,
                      D₁::AbstractMatrix,
                      D₂::AbstractMatrix,
                      ws::AbstractVector;
            adjoint_diff::Bool=true)
    D₁⁺ = adjoint_diff ? LinearAlgebra.adjoint(D₁, ws) : D₁
    D₂⁺ = adjoint_diff ? LinearAlgebra.adjoint(D₂, ws) : D₂
    return ChannelGrid{(Nx, length(y), Nz, Nt)}(y, D₁, D₂, D₁⁺, D₂⁺, ws, α, β)
end

# ------------------- #
# NSEBase grid hooks  #
# ------------------- #
"""
    points(g::ChannelGrid{S}; dealias=false) -> (y, x, z, t)

Return broadcastable coordinate arrays `(y, x, z, t)` at the physical-space
grid points, in array-dimension order, each shaped to broadcast along its
correct array dimension.

When `dealias=false` (default), the homogeneous directions use the
pre-dealiasing physical resolution stored in `S = (Nx, Ny, Nz, Nt)`.
When `dealias=true`, the homogeneous resolutions are first expanded to the
padded (dealiased) size before computing coordinates.

See also [`points(g, N)`](@ref) for the lower-level method that accepts an
explicit homogeneous size tuple.
"""
NSEBase.points(g::ChannelGrid{S}; dealias=false) where {S} = begin
    if dealias
        # size of the fields on the dealiased grid, in logical array (not physical) order 
        padded_storage_size = NSEBase.get_padded_size(size(g), NSEBase.fft_dims(g))
        NSEBase.points(g, (padded_storage_size[CHANNEL_AXES[1]],
                           padded_storage_size[CHANNEL_AXES[3]],
                           padded_storage_size[CHANNEL_AXES[4]]))
    else
        return NSEBase.points(g, (S[1], S[3], S[4]))
    end
end

"""
    points(g::ChannelGrid, (Nx, Nz, Nt)::NTuple{3, Int}) -> (y, x, z, t)

Return broadcastable coordinate arrays `(y, x, z, t)` in array-dimension order
for a channel grid with homogeneous sizes `N = (Nx, Nz, Nt)`.

Each array is reshaped so that it varies along the correct array dimension as
defined by `CHANNEL_AXES`:
- `x` varies along array dimension `CHANNEL_AXES[1] = 2`, shape `(1, Nx, 1, 1)`.
- `y` varies along array dimension `CHANNEL_AXES[2] = 1`, shape `(Ny, 1, 1, 1)`.
- `z` varies along array dimension `CHANNEL_AXES[3] = 3`, shape `(1, 1, Nz, 1)`.
- `t` varies along array dimension `CHANNEL_AXES[4] = 4`, shape `(1, 1, 1, Nt)`.

The returned arrays can be used directly to initialise physical fields via
broadcasting, e.g. `u .= @. sin(x) * cos(y)`. The streamwise and spanwise
coordinates span `[0, 2π/α)` and `[0, 2π/β)` with `Nx` and `Nz` equally
spaced points; the temporal coordinate spans `[0, 1)` with `Nt` points. The
wall-normal coordinate `y` is taken directly from the collocation points `g.y`.
"""
function NSEBase.points(g::ChannelGrid, (Nx, Nz, Nt)::NTuple{3, Int})
    # Build the 4D reshape dimensions that place a one-dimensional coordinate
    # array along storage dimension `dim` and leave all other dimensions as
    # singletons, so the coordinates broadcast together into full fields.
    _reshape_dims(dim, len) = ntuple(d -> d == dim ? len : 1, 4)
    X = reshape(_equal_points(Nx, 2π/g.α), _reshape_dims(CHANNEL_AXES[1], Nx))
    Y = reshape(g.y,                       _reshape_dims(CHANNEL_AXES[2], length(g.y)))
    Z = reshape(_equal_points(Nz, 2π/g.β), _reshape_dims(CHANNEL_AXES[3], Nz))
    T = reshape(_equal_points(Nt, 1.0),    _reshape_dims(CHANNEL_AXES[4], Nt))
    # Return coordinates in storage order, using CHANNEL_AXES to permute the
    # logical coordinate tuple (X, Y, Z, T).
    return NSEBase.storage_order((X, Y, Z, T), g)
end

"""
    _equal_points(N, L)

Return `N` equally spaced points on the half-open interval `[0, L)`, excluding
the repeated endpoint. This is the physical grid used for periodic homogeneous
directions.

For example, `_equal_points(4, 1)` returns the points
`0, 1/4, 1/2, 3/4`.
"""
_equal_points(N, L) = (0:(N - 1))/(N)*L

#TODO: figure out if this should be array dimension or physical coordinate dimension
"""
    wavenumber_scale(g::AbstractChannelGrid, dim::Int) -> Real

Return the wavenumber scale for physical dimension `dim`, used to convert
integer wavenumber indices to physical wavenumbers `k = n * wavenumber_scale`.

`dim` is an array dimension. The coordinate mapping is defined by `CHANNEL_AXES`:
- `dim = CHANNEL_AXES[1]` (streamwise `x`): returns `g.α = 2π/Lx`.
- `dim = CHANNEL_AXES[2]` (wall-normal `y`): returns `one(g.α)`
  (inhomogeneous; this value should never be used in practice).
- `dim = CHANNEL_AXES[3]` (spanwise `z`): returns `g.β = 2π/Lz`.
- `dim = CHANNEL_AXES[4]` (temporal `t`): returns `one(g.α)` (unit-period scaling).

This method is called by NSEBase generic routines such as `minnormdiff` to
compute shift step sizes in each homogeneous direction.
"""
NSEBase.wavenumber_scale(g::AbstractChannelGrid, dim::Int) =
    dim == CHANNEL_AXES[1] ? g.α : dim == CHANNEL_AXES[3] ? g.β : one(g.α)

"""
    weights(g::AbstractChannelGrid) -> AbstractVector

Return the wall-normal quadrature weights `ws` (length `Ny`).

These weights define the discrete inner product in the inhomogeneous direction:

```math
\\langle u, v \\rangle_y = \\sum_{j=1}^{N_y} w_j \\, u_j^* \\, v_j
```

They are used by `project!` to compute modal coefficients and by `normdiff` to
evaluate weighted norms.
"""
NSEBase.weights(g::AbstractChannelGrid) = g.ws

"""
    growto(g::ChannelGrid{S}, (Nx, Nz, Nt)::NTuple{3, Int}) -> ChannelGrid

Return a new `ChannelGrid` with homogeneous resolutions `(Nx, Nz, Nt)` while
keeping the wall-normal grid (`y`, `D₁`, `D₂`, `D₁⁺`, `D₂⁺`, `ws`) and
wavenumber scales (`α`, `β`) unchanged.

This is used by NSEBase when changing spectral resolution, for example when
doubling the grid for dealiasing (3/2 rule) or when constructing a coarser
grid for a continuation study. Only the homogeneous entries `S[1]`, `S[3]`,
and `S[4]` in the new physical-coordinate size tuple are replaced; `S[2] = Ny`
is preserved from the original grid.
"""
NSEBase.growto(g::ChannelGrid{S}, (Nx, Nz, Nt)::NTuple{3, Int}) where {S} =
    ChannelGrid{(Nx, S[2], Nz, Nt)}(g.y, g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.ws, g.α, g.β)
