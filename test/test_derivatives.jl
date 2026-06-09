@testset "Field derivatives                     " begin
    # function definitions
    u_fun(y, x, z, t)      = (1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudx_fun(y, x, z, t)   = -4π*(1 - y^2)*sin(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udx2_fun(y, x, z, t) = -(4π)^2*(1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudy_fun(y, x, z, t)   = -2*y*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udy2_fun(y, x, z, t) = -2*cos(4π*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*cos(4π*x)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*cos(4π*x)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(2π*t))
    lapl_fun(y, x, z, t)   = d2udx2_fun(y, x, z, t) + d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t)
    dudt_fun(y, x, z, t)   = 2π*((1 - y^2)*cos(4π*x)*exp(cos(5.8*z))*cos(2π*t))/(sin(2π*t)^2 + 1)

    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    D₁ = chebdiff(Ny)
    D₂ = chebddiff(Ny)
    ws = chebws(Ny)
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2π, 5.8,
                    D₁,
                    D₂,
                    D₁,
                    D₂,
                    ws,)

    # test values of derivatives
    u = FFT(Field(g, u_fun))
    @test NSEBase.ddx_1!(    FTField(g), u) ≈ FFT(Field(g, dudx_fun))
    @test NSEBase.ddx_2!(    FTField(g), u) ≈ FFT(Field(g, dudy_fun))
    @test NSEBase.ddx_3!(    FTField(g), u) ≈ FFT(Field(g, dudz_fun))
    @test NSEBase.laplacian!(FTField(g), u) ≈ FFT(Field(g, lapl_fun))

    # test time derivative of projected field
    M = 10
    Ψ₁ = zeros(ComplexF64, M, Ny, (Nx >> 1) + 1, Nz, Nt)
    for nt in 1:Nt, nz in 1:Nz, nx in 1:(Nx >> 1) + 1
        Ψ₁[:, :, nx, nz, nt] .= (qr(randn(ComplexF64, Ny, M)).Q[:, 1:M])'
    end
    for m in 1:M
        NSEBase.apply_symmetry!(@view(Ψ₁[m, :, :, :, :]), (2, 3, 4))
        Ψ₁[m, :, 1, 1, 1] .= real.(Ψ₁[m, :, 1, 1, 1])
    end
    Ψ = (Ψ₁, zero(Ψ₁), zero(Ψ₁))
    a = project(FFT(VectorField(g, u_fun)), Ψ)
    @test NSEBase.ddx_4!(similar(a), a) ≈ project(FFT(VectorField(g, dudt_fun)), Ψ)

    # test allocation
    fun(dx, a, b) = @allocated dx(a, b)
    @test fun(NSEBase.ddx_1!,     FTField(g), u) == 0
    @test fun(NSEBase.ddx_2!,     FTField(g), u) == 0
    @test fun(NSEBase.ddx_3!,     FTField(g), u) == 0
    @test fun(NSEBase.laplacian!, FTField(g), u) == 0
    @test fun(NSEBase.ddx_4!,     similar(a), a) == 0
end
