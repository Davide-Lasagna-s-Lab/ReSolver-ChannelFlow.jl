@testset "Dot product                 " begin
    # construct grid and modes
    Ny = 8; Nx = 9; Nz = 9; Nt = 9
    g = ChannelGrid(zeros(Ny), Nx, Nz, Nt,
                    2π, 2π,
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny, Ny), zeros(Ny, Ny),
                    zeros(Ny))
    M = 5
    Ψ = ntuple(_ -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)

    # construct projected fields
    a = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); ad = CUDA.cu(a)
    b = ProjectedField(g, randn(M, (Nx >> 1) + 1, Nz, Nt), Ψ); bd = CUDA.cu(b)

    @testset "methods are correct" begin
        # initialise dot product methods
        method_twostage = DotTwoStage(ad)
        method_atomic   = DotAtomic(ad)
        method_shared   = DotShared(ad)

        # test result
        res_host = dot(a, b)
        @test abs(res_host - dot(ad, bd, method_twostage)) < 2e-4
        @test abs(res_host - dot(ad, bd, method_atomic))   < 2e-4
        @test abs(res_host - dot(ad, bd, method_shared))   < 2e-4
    end
end

# function definitions
f1(y, x, z, t) =          (1 - y^2)*exp(cos(z))*cos(sin(2π*t))
f2(y, x, z, t) = cos(π*y)*(1 - y^2)*exp(sin(z))*cos(2π*t)^2

# construct grid
Ny = 32; Nx = 5; Nz = 33; Nt = 51
D₁ = chebdiff(Ny)
D₂ = chebddiff(Ny)
ws = chebws(Ny)
g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                1.0, 1.0,
                D₁,
                D₂,
                D₁,
                D₂,
                ws)

# generate modes
M = Ny
Ψ = ntuple(_ -> zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt), 1)
for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
    Ψ[1][:, :, nx, nz, nt] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
end
for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
    Ψ[1][:, :, 1,     nz,       nt]   .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    Ψ[1][:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[1][:, :, 1, nz, nt])
end
for nz in 2:Nz
    Ψ[1][:, :, 1,     nz,   1] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    Ψ[1][:, :, 1, end-nz+2, 1] .= conj.(Ψ[:, :, 1, nz, 1])
end
for nt in 2:Nt
    Ψ[:, :, 1, 1,     nt]   .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    Ψ[:, :, 1, 1, end-nt+2] .= conj.(Ψ[:, :, 1, 1, nt])
end
Ψ[:, :, 1, 1, 1] .= (Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(Float64, Ny, M)).Q[:, 1:M])'
