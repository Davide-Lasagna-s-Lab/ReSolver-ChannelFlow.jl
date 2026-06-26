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

# construct grid
Ny = 32; Nx = 9; Nz = 11; Nt = 9
y = FDGrids.grid(Ny, -1, 1, MappedGrid(1))[1]
ws = ones(Ny)
D₁ = DiffMatrix(y, 3, 1)
D₂ = DiffMatrix(y, 3, 2)
D₁⁺ = adjoint(D₁, ws)
D₂⁺ = adjoint(D₂, ws)
α = 2π
β = 5.8
g = distributed(ChannelGrid(y, Nx, Nz, Nt, α, β, D₁, D₂, D₁⁺, D₂⁺, ws), comm;
                    decomposed_physical_dims=(:y,), nprocesses=(np,), nhalo=(1,))
Ny_sb = div(Ny, np)

# generate modes and split among processes
using Random
Random.seed!(0)
M = 5
Ψ_sb = ntuple(_ -> zeros(ComplexF64, M, Ny_sb, (Nx >> 1) + 1, Nz, Nt), 3)
if rank == 0
    Ψ = ntuple(_ -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 3)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1
        tmp = (qr(randn(ComplexF64, 3*Ny, M)).Q[:, 1:M])'
        Ψ[1][:, :, nx, nz, nt] .= tmp[:,      1:1*Ny]
        Ψ[2][:, :, nx, nz, nt] .= tmp[:,   Ny+1:2*Ny]
        Ψ[3][:, :, nx, nz, nt] .= tmp[:, 2*Ny+1:3*Ny]
    end
    for dest in 1:(np - 1), n in 1:3
        @views Ψ_sb[n] .= Ψ[n][:, (1+dest*Ny_sb):(dest + 1)*Ny_sb, :, :, :]
        MPI.Send(Ψ_sb[n], comm, dest=dest)
    end
    for n in 1:3
        @views Ψ_sb[n] .= Ψ[n][:, 1:Ny_sb, :, :, :]
    end
else
    for n in 1:3
        MPI.Recv!(Ψ_sb[n], comm, source=0)
    end
end

# generate random coefficients and scatter among processes
a = ProjectedField(g, Ψ_sb)
if rank == 0
    parent(a) .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    for dest in 1:(np - 1)
        MPI.Send(parent(a), comm, dest=dest)
    end
else
    MPI.Recv!(parent(a), comm, source=0)
end

# test project-expand is consistent accross processes
u = VectorField(g)
for n in 1:3, nx in 1:(Nx >> 1) + 1, nz in 1:Nz, nt in 1:Nt
    @views u[n][:, nx, nz, nt] .= transpose(Ψ_sb[n][:, :, nx, nz, nt])*a[:, nx, nz, nt]
end
@test expand(a, NSEBase.LoopGalerkin()) ≈ u
@test expand(a, NSEBase.GemmGalerkin()) ≈ u
@test project(u, Ψ_sb) ≈ a
