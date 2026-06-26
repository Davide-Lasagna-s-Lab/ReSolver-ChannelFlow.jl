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

# derivative functions
u_fun(y, x, z, t)      = (1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
dudx_fun(y, x, z, t)   = -4π*(1 - y^2)*sin(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
d2udx2_fun(y, x, z, t) = -(4π)^2*(1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
dudy_fun(y, x, z, t)   = -2*y*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
d2udy2_fun(y, x, z, t) = -2*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*cos(4π*x)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(2π*t))
d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*cos(4π*x)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(2π*t))
dudt_fun(y, x, z, t)   = 2π*((1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*cos(2π*t))/(sin(2π*t)^2 + 1)
lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) + d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t)

# construct grid
Ny = 32; Nx = 15; Nz = 33; Nt = 51
y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
D₁ = DiffMatrix(y, 3, 1)
D₂ = DiffMatrix(y, 3, 2)
D₁⁺ = adjoint(D₁, ws)
D₂⁺ = adjoint(D₂, ws)
α = 2π
β = 5.8
g = distributed(ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws), comm;
                    decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(1,))

# test local data derivatives
u = FFT(Field(g, u_fun))
@test NSEBase.ddx!(FTField(g), u) ≈ FFT(Field(g, dudx_fun))
@test NSEBase.ddz!(FTField(g), u) ≈ FFT(Field(g, dudz_fun))
@test NSEBase.ddt!(FTField(g), u) ≈ FFT(Field(g, dudt_fun))

# test swapping derivatives
out1 = FTField(g)
out2 = FTField(g)
reqs = init_requests!(u)
NSEBase.init_ddy!(out1, u)
NSEBase.init_laplacian!(out2, u)
wait_requests!(reqs)
NSEBase.complete_ddy!(out1, u)
NSEBase.complete_laplacian!(out2, u)
@test out1 ≈ FFT(Field(g, dudy_fun))
@test out2 ≈ FFT(Field(g, lapl_fun))
