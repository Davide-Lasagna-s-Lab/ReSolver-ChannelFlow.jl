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
    AbstractChannelGrid{S} <: AbstractGrid{Float64, 4, CHANNEL_AXES, CHANNEL_FFT_ORDER}

Abstract supertype for all plane-channel grids using the default axis layout
defined by `CHANNEL_AXES`, `CHANNEL_FFT_ORDER`, and `CHANNEL_INHOMOGENEOUS_DIMS`.

`S` is an `NTuple{4, Int}` giving the physical (pre-dealiasing) size in physical
coordinate order `(Nx, Ny, Nz, Nt)`, where `Ny` is the wall-normal resolution
and `Nx`, `Nz`, `Nt` are the streamwise, spanwise, and temporal
resolutions. The `AbstractGrid` type parameters are fixed to `Float64` scalars
stored in 4D arrays with the layout described above.
"""
abstract type AbstractChannelGrid{S} <: NSEBase.AbstractCartesianGrid{Float64, 4, CHANNEL_AXES, CHANNEL_FFT_ORDER} end

# TODO: figure out why this is needed.. it seems to be needed for the MPI code, but maybe there is a better way to do this.
# TODO: fix this to return S in the order represented by CHANNEL_AXES, i.e. (Ny, Nx, Nz, Nt) instead of (Nx, Ny, Nz, Nt)
"""
    size(g::AbstractChannelGrid{S}) -> NTuple{4, Int}

Return the physical (pre-dealiasing) grid size `S = (Nx, Ny, Nz, Nt)` in
physical coordinate order. `S` is encoded as a type parameter, so this method
exposes it at runtime for NSEBase generic code. In distributed contexts a
concrete subtype may override this to return slab-local sizes instead.
"""
Base.size(::AbstractChannelGrid{S}) where {S} = S


# ----------------------------- #
# Concrete grid implementation  #
# ----------------------------- #
"""
    ChannelGrid{S}

Concrete plane-channel grid with one inhomogeneous wall-normal direction and
three homogeneous directions `(x, z, t)`.

The size parameter `S = (Nx, Ny, Nz, Nt)` follows physical coordinate order.

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
    points(g::ChannelGrid{S}; dealias=false) -> (x, y, z, t)

Return broadcastable coordinate arrays `(x, y, z, t)` at the physical-space
grid points, each shaped to broadcast along its correct array dimension.

When `dealias=false` (default), the homogeneous directions use the
pre-dealiasing physical resolution stored in `S = (Nx, Ny, Nz, Nt)`.
When `dealias=true`, the homogeneous resolutions are first expanded to the
padded (dealiased) size before computing coordinates.

See also [`points(g, N)`](@ref) for the lower-level method that accepts an
explicit homogeneous size tuple.
"""
NSEBase.points(g::ChannelGrid{S}; dealias=false) where {S} = begin
    if dealias
        # Convert S from physical coord order (Nx, Ny, Nz, Nt) to array dim order
        # (Ny, Nx, Nz, Nt) so that get_padded_size can use CHANNEL_FFT_ORDER indices.
        Nx, Ny, Nz, Nt = S
        _, Nx′, Nz′, Nt′ = NSEBase.get_padded_size((Ny, Nx, Nz, Nt), NSEBase.fft_dims(g))
        NSEBase.points(g, (Nx′, Nz′, Nt′))
    else
        Nx, _, Nz, Nt = S
        NSEBase.points(g, (Nx, Nz, Nt))
    end
end

"""
    points(g::ChannelGrid, N::NTuple{3, Int}) -> (x, y, z, t)

Return broadcastable coordinate arrays `(x, y, z, t)` in physical coordinate
order for a channel grid with homogeneous sizes `N = (Nx, Nz, Nt)`.

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
    _shape(coord, len) = ntuple(d -> d == CHANNEL_AXES[coord] ? len : 1, 4)
    X = reshape(_equal_pts(Nx, 2π/g.α), _shape(1, Nx))
    Y = reshape(g.y,                    _shape(2, length(g.y)))
    Z = reshape(_equal_pts(Nz, 2π/g.β), _shape(3, Nz))
    T = reshape(_equal_pts(Nt),          _shape(4, Nt))
    return (X, Y, Z, T)
end

_equal_pts(N, L) = (0:(N - 1))/(N)*L
_equal_pts(N)    = (0:(N - 1))/(N)

"""
    wavenumber_scale(g::AbstractChannelGrid, dim::Int) -> Float64

Return the wavenumber scale for physical dimension `dim`, used to convert
integer wavenumber indices to physical wavenumbers `k = n * wavenumber_scale`.

`dim` follows the physical coordinate indexing defined by `CHANNEL_AXES`:
- `dim = CHANNEL_AXES[1]` (streamwise `x`): returns `g.α = 2π/Lx`.
- `dim = CHANNEL_AXES[2]` (wall-normal `y`): returns `1.0` (inhomogeneous;
  this value should never be used in practice).
- `dim = CHANNEL_AXES[3]` (spanwise `z`): returns `g.β = 2π/Lz`.
- `dim = CHANNEL_AXES[4]` (temporal `t`): returns `1.0` (unit-period scaling).

This method is called by NSEBase generic routines such as `minnormdiff` to
compute shift step sizes in each homogeneous direction.
"""
NSEBase.wavenumber_scale(g::AbstractChannelGrid, dim::Int) =
    dim == CHANNEL_AXES[1] ? g.α : dim == CHANNEL_AXES[3] ? g.β : 1.0

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
grid for a continuation study. Only `S[1]`, `S[3]`, and `S[4]` in the new
size tuple are replaced; `S[2] = Ny` is preserved from the original grid.
"""
NSEBase.growto(g::ChannelGrid{S}, (Nx, Nz, Nt)::NTuple{3, Int}) where {S} =
    ChannelGrid{(Nx, S[2], Nz, Nt)}(g.y, g.D₁, g.D₂, g.D₁⁺, g.D₂⁺, g.ws, g.α, g.β)


# ----------- #
# Persistence #
# ----------- #
"""
    save_grid(g::ChannelGrid; path="./grid.jld2")

Serialize `g` to a JLD2 file at `path`. The entire struct — including
differentiation matrices, quadrature weights, and wavenumber scales — is
stored under the key `"grid"`. Load with [`load_grid`](@ref).
"""
function save_grid(g::ChannelGrid; path="./grid.jld2")
    JLD2.jldopen(path, "w") do f
        f["grid"] = g
    end
    return nothing
end

"""
    load_grid(path) -> ChannelGrid

Load a `ChannelGrid` previously saved with [`save_grid`](@ref) from the JLD2
file at `path`. The returned grid is identical to the one that was saved,
including all differentiation matrices and quadrature weights.
"""
function load_grid(path)
    JLD2.jldopen(path, "r") do f
        return f["grid"]
    end
end
