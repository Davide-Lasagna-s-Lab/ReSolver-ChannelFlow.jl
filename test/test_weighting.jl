@testset "Norm weighting                        " begin
    # function definitions
    f1(y, x, z, t) =          (1 - y^2)*exp(cos(z))*cos(sin(t))
    f2(y, x, z, t) = cos(π*y)*(1 - y^2)*exp(sin(z))*cos(t)^2

    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    # generate modes
    M = Ny
    Ψ = zeros(ComplexF64, Ny, M, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
        Ψ[:, :, nx, nz, nt] .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
    end
    for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
        Ψ[:, :, 1,     nz,       nt]   .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[:, :, 1, nz, nt])
    end
    for nz in 2:Nz
        Ψ[:, :, 1,     nz,   1] .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nz+2, 1] .= conj.(Ψ[:, :, 1, nz, 1])
    end
    for nt in 2:Nt
        Ψ[:, :, 1, 1,     nt]   .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(ComplexF64, Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, 1, end-nt+2] .= conj.(Ψ[:, :, 1, 1, nt])
    end
    Ψ[:, :, 1, 1, 1] .= Diagonal(1 ./ sqrt.(g.ws))*qr(Diagonal(sqrt.(g.ws))*randn(Float64, Ny, M)).Q[:, 1:M]

    # construct fields
    a = project(FFT(VectorField(g, (f1,), 2π)), Ψ)
    b = project(FFT(VectorField(g, (f2,), 2π)), Ψ)

    # test weighting operation
    A = FarazmandWeight(2π, 2π, 2π)
    c = copy(a)
    for nx in 0:(Nx >> 1), nz in -(Nz >> 1):(Nz >> 1), nt in -(Nt >> 1):(Nt >> 1), m in 1:M
        _nx = nx + 1
        _nz = nz >= 0 ? nz + 1 : Nz + nz + 1
        _nt = nt >= 0 ? nt + 1 : Nt + nt + 1
        c[m, _nx, _nz, _nt] /= 1 + 4π^2*nx^2 + 4π^2*nz^2 + 4π^2*nt^2
    end
    @test mul!(copy(a), A) ≈ c
    @test dot(a, A, b) == dot(a, mul!(copy(b), A))
end
