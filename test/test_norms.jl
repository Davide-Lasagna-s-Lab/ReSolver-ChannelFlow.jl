@testset "Field norms                           " begin
    # function definitions
    f1(y, x, z, t) =          (1 - y^2)*exp(cos(z))*cos(sin(2π*t))
    f2(y, x, z, t) = cos(π*y)*(1 - y^2)*exp(sin(z))*cos(2π*t)^2

    # construct grid
    Ny = 32; Nx = 5; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    # test channel integration
    u = ComplexF64[(y^2)*cos(π*y/2) for y in g.y]
    v = ComplexF64[exp(-5*(y^2)) for y in g.y]
    @test ReSolverChannelFlow._channel_int(u, chebws(Ny), v, Ny) ≈ 0.0530025 rtol=1e-5

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

    # test norms of channel fields
    @test dot(FFT(Field(g, f1)), FFT(Field(g, f2))) ≈ 0.169796 rtol=1e-5
    @test norm(FFT(Field(g, f1)))^2                 ≈ 0.743990 rtol=1e-5
    @test norm(FFT(VectorField(g, f1, f2)))^2       ≈ 0.965370 rtol=1e-5
    @test norm(project(FFT(VectorField(g, f1)), Ψ)) ≈ norm(FFT(VectorField(g, f1)))

    # test norm difference methods
    @test normdiff(FFT(VectorField(g, f1)), FFT(VectorField(g, f2)))^2 ≈ 0.625777 rtol=1e-5
    @test_broken normdiff(project(FFT(VectorField(g, f1)), Ψ), project(FFT(VectorField(g, f2)), Ψ))^2 ≈ 0.625777 rtol=1e-5
    mindiff, s_mins = minnormdiff(FFT(Field(g, f1)), FFT(Field(g, (y, x, z, t)->f1(y, x+π, z-π, t-π/2))), (4, 4, 4))
    # FIXME: the t-shift is not correct?
    @test_broken all(s_mins .≈ (0, π, 0.25)) # minimum norm difference is zero since f1 doesn't depend on x
end
