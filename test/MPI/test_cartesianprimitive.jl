using MPI
using Test

using NSEBase,
      ReSolverChannelFlow,
      FDGrids

# initialise MPI environment
MPI.Init()
comm = MPI.COMM_WORLD
np   = MPI.Comm_size(comm)
rank = MPI.Comm_rank(comm)

# define nonlinear functions
u_fun(y, x, z, t)      = y + (1 - y^2)*exp(cos(5.8*z))*atan(sin(2π*t))
dudy_fun(y, x, z, t)   = 1 - 2*y*exp(cos(5.8*z))*atan(sin(2π*t))
d2udy2_fun(y, x, z, t) = -2*exp(cos(5.8*z))*atan(sin(2π*t))
dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(2π*t))
d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(2π*t))
v_fun(y, x, z, t)      = cos(π*y/2)^2*exp(sin(5.8*z))*cos(sin(2π*t))
dvdy_fun(y, x, z, t)   = -(π/2)*sin(π*y)*exp(sin(5.8*z))*cos(sin(2π*t))
d2vdy2_fun(y, x, z, t) = -(π^2/2)*cos(π*y)*exp(sin(5.8*z))*cos(sin(2π*t))
dvdz_fun(y, x, z, t)   = 5.8*cos(π*y/2)^2*cos(5.8*z)*exp(sin(5.8*z))*cos(sin(2π*t))
d2vdz2_fun(y, x, z, t) = (5.8^2)*cos(π*y/2)^2*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(sin(2π*t))
w_fun(y, x, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(5.8*z))*cos(2π*t)^2
dwdy_fun(y, x, z, t)   = -(π*sin(π*y)*(1 - y^2) + 2*y*cos(π*y))*exp(sin(5.8*z))*cos(2π*t)^2
d2wdy2_fun(y, x, z, t) = -(π^2*cos(π*y)*(1 - y^2) - 4π*y*sin(π*y) + 2*cos(π*y))*exp(sin(5.8*z))*cos(2π*t)^2
dwdz_fun(y, x, z, t)   = 5.8*cos(π*y)*(1 - y^2)*cos(5.8*z)*exp(sin(5.8*z))*cos(2π*t)^2
d2wdz2_fun(y, x, z, t) = (5.8^2)*cos(π*y)*(1 - y^2)*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(2π*t)^2
u_out_fun(y, x, z, t)  = (d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t))/Re - v_fun(y, x, z, t)*dudy_fun(y, x, z, t) - w_fun(y, x, z, t)*dudz_fun(y, x, z, t) + Ro*v_fun(y, x, z, t)
v_out_fun(y, x, z, t)  = (d2vdy2_fun(y, x, z, t) + d2vdz2_fun(y, x, z, t))/Re - v_fun(y, x, z, t)*dvdy_fun(y, x, z, t) - w_fun(y, x, z, t)*dvdz_fun(y, x, z, t) - Ro*u_fun(y, x, z, t)
w_out_fun(y, x, z, t)  = (d2wdy2_fun(y, x, z, t) + d2wdz2_fun(y, x, z, t))/Re - v_fun(y, x, z, t)*dwdy_fun(y, x, z, t) - w_fun(y, x, z, t)*dwdz_fun(y, x, z, t)

# construct grid
Ny = 128; Nx = 7; Nz = 33; Nt = 51
y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
D₁ = DiffMatrix(y, 9, 1)
D₂ = DiffMatrix(y, 9, 2)
D₁⁺ = adjoint(D₁, ws)
D₂⁺ = adjoint(D₂, ws)
α = 2π
β = 5.8
g = distributed(ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws), comm;
                    decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(4,))

# test nonlinear operator
if rank == 0
    Re = rand()*50
    Ro = rand()
    for dest in 1:(np - 1)
        MPI.Send(Re, comm, dest=dest)
        MPI.Send(Ro, comm, dest=dest)
    end
else
    Re = MPI.Recv(Float64, comm, source=0)
    Ro = MPI.Recv(Float64, comm, source=0)
end
op = CartesianPrimitive3DNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
u = FFT(VectorField(g, u_fun, v_fun, w_fun))
exact = FFT(VectorField(g, u_out_fun, v_out_fun, w_out_fun))
@test op(0.0, u, VectorField(g)) ≈ exact
