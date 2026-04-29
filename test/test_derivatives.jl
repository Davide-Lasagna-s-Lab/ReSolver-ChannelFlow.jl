@testset "Field derivatives                     " begin
    # function definitions
    u_fun(y, x, z, t)      = (1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(t))
    dudx_fun(y, x, z, t)   = -4π*(1 - y^2)*sin(4π*x)*exp(cos(5.8*z))*atan(sin(t))
    d2udx2_fun(y, x, z, t) = -(4π)^2*(1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(t))
    dudy_fun(y, x, z, t)   = -2*y*cos(4π*x)*exp(cos(5.8*z))*atan(sin(t))
    d2udy2_fun(y, x, z, t) = -2*cos(4π*x)*exp(cos(5.8*z))*atan(sin(t))
    dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*cos(4π*x)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(t))
    d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*cos(4π*x)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(t))
    lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) + d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t)
    duds_fun(y, x, z, t)   = ((1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*cos(t))/(sin(t)^2 + 1)

    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2π, 5.8,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    # test values of derivatives
    u = FFT(Field(g, u_fun, 2π))
    @test ReSolverChannelFlow.ddx1!(     FTField(g), u) ≈ FFT(Field(g, dudx_fun,   2π))
    @test ReSolverChannelFlow.ddx2!(     FTField(g), u) ≈ FFT(Field(g, dudy_fun,   2π))
    @test ReSolverChannelFlow.ddx3!(     FTField(g), u) ≈ FFT(Field(g, dudz_fun,   2π))
    @test ReSolverChannelFlow.laplacian!(FTField(g), u) ≈ FFT(Field(g, lapl_fun,   2π))

    # test time derivative of projected field
    M = 10
    Ψ = zeros(ComplexF64, Ny, M, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1
        Ψ[:, :, nx, nz, nt] .= qr(randn(ComplexF64, Ny, M)).Q[:, 1:M]
    end
    for m in 1:M
        ReSolverChannelFlow.apply_symmetry!(@view(Ψ[:, m, :, :, :]))
        Ψ[:, m, 1, 1, 1] .= real.(Ψ[:, m, 1, 1, 1])
    end
    a = project(FFT(VectorField(g, (u_fun,), 2π)), Ψ)
    @test ReSolverChannelFlow.dds!(similar(a), a) ≈ project(FFT(VectorField(g, (duds_fun,), 2π)), Ψ)

    # test allocation
    fun(dx, a, b) = @allocated dx(a, b)
    @test fun(ReSolverChannelFlow.ddx1!,      FTField(g), u) == 0
    @test fun(ReSolverChannelFlow.ddx2!,      FTField(g), u) == 0
    @test fun(ReSolverChannelFlow.ddx3!,      FTField(g), u) == 0
    @test fun(ReSolverChannelFlow.laplacian!, FTField(g), u) == 0
    @test fun(ReSolverChannelFlow.dds!,       similar(a), a) == 0
end
