@testset "Projected NS Operators                " begin
    # construct grid
    Ny = 32; Nx = 15; Nz = 15; Nt = 21
    # g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
    #                 2π, 5.8,
    #                 chebdiff(Ny),
    #                 chebddiff(Ny),
    #                 chebws(Ny),
    #                 adjoint_diff=false)
    y = range(-1, 1, length=Ny)
    g = ChannelGrid(y, Nx, Nz, Nt,
                    2π, 5.8,
                    DiffMatrix(y, 5, 1),
                    DiffMatrix(y, 5, 2),
                    quadweights(y, 1),
                    adjoint_diff=true)

    # construct modes
    M = 3*Ny
    Ψ = zeros(ComplexF64, 3*Ny, M, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 2:(Nx >> 1) + 1
        Ψ[:, :, nx, nz, nt] .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    end
    for nz in 2:(Nz >> 1) + 1, nt in 2:Nt
        Ψ[:, :, 1,     nz,       nt]   .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nz+2, end-nt+2] .= conj.(Ψ[:, :, 1, nz, nt])
    end
    for nz in 2:Nz
        Ψ[:, :, 1,     nz,   1] .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, end-nz+2, 1] .= conj.(Ψ[:, :, 1, nz, 1])
    end
    for nt in 2:Nt
        Ψ[:, :, 1, 1,     nt]   .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
        Ψ[:, :, 1, 1, end-nt+2] .= conj.(Ψ[:, :, 1, 1, nt])
    end
    Ψ[:, :, 1, 1, 1] .= Diagonal(1 ./ repeat(sqrt.(g.ws), 3))*qr(Diagonal(repeat(sqrt.(g.ws), 3))*randn(Float64, 3*Ny, M)).Q[:, 1:M]
    for nt in 1:Nt, nz in 1:Nz, nx in 1:((Nx >> 1) + 1), m in 1:M
        Ψ[     1, m, nx, nz, nt] = 0.0
        Ψ[  Ny  , m, nx, nz, nt] = 0.0
        Ψ[  Ny+1, m, nx, nz, nt] = 0.0
        Ψ[2*Ny  , m, nx, nz, nt] = 0.0
        Ψ[2*Ny+1, m, nx, nz, nt] = 0.0
        Ψ[3*Ny  , m, nx, nz, nt] = 0.0
    end

    # generate fields
    a = ProjectedField(g, Ψ)
    b = ProjectedField(g, Ψ)
    a .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    b .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    a[:, 1, 1, 1] .= real.(a[:, 1, 1, 1])
    b[:, 1, 1, 1] .= real.(b[:, 1, 1, 1])
    ReSolverChannelFlow.apply_symmetry!(a)
    ReSolverChannelFlow.apply_symmetry!(b)
    u = expand!(VectorField(g), a)
    u[1][:, 1, 1, 1] .+= g.y
    v = expand!(VectorField(g), b)

    # test projected equations
    Re = rand()*50
    Ro = rand()
    op_nl = CartesianPrimitiveNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE)
    op_ln = CartesianPrimitiveLNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE, mode=AdjointDiscrete())
    op_pr = ProjectedNSE(g, Re, Ro=Ro, flags=FFTW.ESTIMATE, mode=AdjointDiscrete())
    @test op_pr(similar(a), a)    == project(op_nl(0, u, similar(u)), Ψ)
    @test op_pr(similar(a), a, b) == project(op_ln(0, u, v, similar(u)), Ψ)
end
