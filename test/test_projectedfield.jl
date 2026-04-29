@testset "Projected channel field               " begin
    # construct grid
    Ny = 16; Nx = 15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    ones(Ny),
                    adjoint_diff=false)

    # generate modes
    M = 10
    Ψ = zeros(ComplexF64, 3*Ny, M, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1
        Ψ[:, :, nx, nz, nt] .= qr(randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    end

    # test constructor
    @test ProjectedField(g, Ψ) isa ProjectedField{FTField{typeof(g), Float64, Array{ComplexF64, 4}}, ComplexF64, 4, Array{ComplexF64, 5}}

    # test channel integration
    u = ComplexF64[(y^2)*cos(π*y/2) for y in g.y]
    v = ComplexF64[exp(-5*(y^2)) for y in g.y]
    @test ReSolverChannelFlow._channel_int(u, chebws(Ny), v, Ny) ≈ 0.0530025 rtol=1e-5

    # test project and expand
    a = ProjectedField(g, Ψ)
    parent(a) .= randn(ComplexF64, M, (Nx >> 1) + 1, Nz, Nt)
    u = VectorField(g)
    for nx in 1:(Nx >> 1) + 1, nz in 1:Nz, nt in 1:Nt
        u[1][:, nx, nz, nt] .= Ψ[     1:Ny,   :, nx, nz, nt]*a[:, nx, nz, nt]
        u[2][:, nx, nz, nt] .= Ψ[  Ny+1:2*Ny, :, nx, nz, nt]*a[:, nx, nz, nt]
        u[3][:, nx, nz, nt] .= Ψ[2*Ny+1:3*Ny, :, nx, nz, nt]*a[:, nx, nz, nt]
    end
    @test project(u, Ψ) ≈ a
    @test expand!(zero(u), a) ≈ u
end
