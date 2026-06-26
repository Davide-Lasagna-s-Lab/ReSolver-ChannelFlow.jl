using MPI
using Test

using NSEBase,
      ReSolverChannelFlow,
      FDGrids

# get extension
ext = Base.get_extension(NSEBase, :MPIExt)

# initialise MPI environment
MPI.Init()
comm = MPI.COMM_WORLD
np   = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# define grid variables
Ny = 32; Nx = 9; Nz = 9; Nt = 3
y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
D₁ = DiffMatrix(y, 3, 1)
D₂ = DiffMatrix(y, 3, 2)
D₁⁺ = adjoint(D₁, ws)
D₂⁺ = adjoint(D₂, ws)
α = rand()
β = rand()

# construct FTFields
g = distributed(ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws), comm;
                    decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(1,))
@test FTField(g) isa FTField{<:ext.DecomposedGrid, <:ext.HaloArrays.HaloArray{ComplexF64, np, (1, 0, 0, 0), (8, 5, 9, 3)}}

# construct Fields
# TODO: add test for function based constructor of Field type
u = Field(g, dealias=false)
v = Field(g, dealias=true)
@test u isa Field{<:ext.DecomposedGrid, <:ext.HaloArrays.HaloArray{Float64, np, (1, 0, 0, 0), (8, 9, 9, 3)}}
@test v isa Field{<:ext.DecomposedGrid, <:ext.HaloArrays.HaloArray{Float64, np, (1, 0, 0, 0), (8, 15, 15, 5)}}
@test size(u) == (8, 9, 9, 3)
@test size(v) == (8, 15, 15, 5)

# test transforms
U  = FTField(g); U .= randn(ComplexF64, size(U)...);
U[:, 1, 1, 1] .= real.(U[:, 1, 1, 1]); NSEBase.apply_symmetry!(U)
u  = Field(g, dealias=false)
ud = Field(g, dealias=true)
plans  = FFTPlans(g, dealias=false, flags=FFTW.ESTIMATE)
plansd = FFTPlans(g, dealias=true,  flags=FFTW.ESTIMATE)
@test plans( similar(U), plans( u,  U)) ≈ U
@test plansd(similar(U), plansd(ud, U)) ≈ U
