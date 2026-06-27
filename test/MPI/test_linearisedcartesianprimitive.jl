using MPI
using Test

using LinearAlgebra

using NSEBase,
      ReSolverChannelFlow,
      FDGrids

# initialise MPI environment
MPI.Init()
comm = MPI.COMM_WORLD
np   = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# define functions
# TODO: add x-dependence
ux_fun(y, x, z, t)       = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(t))
uy_fun(y, x, z, t)       = cos(π*y/2)^4*sin(5.8*z)*cos(sin(t))
uz_fun(y, x, z, t)       = (1 - y^2)*cos(5.8*z)*cos(cos(t))
vx_fun(y, x, z, t)       = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(t))
vy_fun(y, x, z, t)       = (1 - y^2)*exp(cos(5.8*z))*atan(cos(t))
vz_fun(y, x, z, t)       = sin(π*y)*sin(z)^2*cos(t)
wx_fun(y, x, z, t)       = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(t)^2
wy_fun(y, x, z, t)       = cos(π*y/2)*cos(5.8*z)*sin(t)^2
wz_fun(y, x, z, t)       = cos(π*y/2)^4*sin(5.8*z)*atan(cos(t))
duydy_fun(y, x, z, t)    = -2π*sin(π*y/2)*cos(π*y/2)^3*sin(5.8*z)*cos(sin(t))
duzdz_fun(y, x, z, t)    = -5.8*(1 - y^2)*sin(5.8*z)*cos(cos(t))

# construct grid
Ny = 64; Nx = 5; Nz = 15; Nt = 15
y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
D₁ = DiffMatrix(y, 5, 1)
D₂ = DiffMatrix(y, 5, 2)
D₁⁺ = adjoint(D₁, ws)
D₂⁺ = adjoint(D₂, ws)
α = 2π
β = 5.8
g = distributed(ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws), comm;
                    decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(2,))

# define fields
u       = FFT(VectorField(g, ux_fun, uy_fun, uz_fun))
v       = FFT(VectorField(g, vx_fun, vy_fun, vz_fun))
w       = FFT(VectorField(g, wx_fun, wy_fun, wz_fun))

# test perturbed nonlinear equations approximates linearised equations
if rank == 0
    Re = rand()*50
    Ro = rand()
    for dest in 1:(np - 1)
        MPI.Send(Re, comm, dest=dest)
        MPI.Send(Ro, comm, dest=dest)
    end
else
    Re = MPI.Recv(Float64, comm)
    Ro = MPI.Recv(Float64, comm, source=0)
end
op_nl = CartesianPrimitive3DNSE(g,  Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
op_ln = CartesianPrimitive3DLNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=Forward())
op_ad = CartesianPrimitive3DLNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=AdjointDiscrete())
a = op_nl(0.0, u .+ 1e-6.*v, VectorField(g)) - op_nl(0.0, u, VectorField(g))
b = op_ln(0.0, u,   1e-6.*v, VectorField(g))
@test norm(a - b) < 1e-9

# test adjoint identity
a = dot(op_ln(0.0, u, v, VectorField(g)), w)
b = dot(v, op_ad(0.0, u, w, VectorField(g)))
out = MPI.Reduce(a - b, +, comm, root=0)
if rank == 0
    @test abs(out) < 1e-12
end
