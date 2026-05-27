@testset "Cartesian primitive NSE               " begin
    # define functions
    u_fun(y, x, z, t)      = y + (1 - y^2)*cos(2.9x)*exp(cos(5.8*z))*atan(sin(2π*t))
    v_fun(y, x, z, t)      = cos(π*y/2)^2*sin(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    w_fun(y, x, z, t)      = cos(π*y)*(1 - y^2)*exp(sin(2.9*x))*exp(sin(5.8*z))*cos(2π*t)^2
    dudx_fun(y, x, z, t)   = −2.9*(1 − y^2)*sin(2.9*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dvdx_fun(y, x, z, t)   = 2.9*cos(π*y/2)^2*cos(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    dwdx_fun(y, x, z, t)   = 2.9*cos(π*y)*(1 − y^2)*cos(2.9*x)*exp(sin(2.9*x))*exp(sin(5.8*z))*cos(2π*t)^2
    d2udx2_fun(y, x, z, t) = −(2.9^2)·(1 − y^2)*cos(2.9*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    d2vdx2_fun(y, x, z, t) = −(2.9^2)*cos(π*y/2)^2*sin(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    d2wdx2_fun(y, x, z, t) = (2.9^2)*cos(π*y)*(1 − y^2)*(cos(2.9*x)^2 − sin(2.9*x))*exp(sin(2.9*x))*exp(sin(5.8*z))*cos(2π*t)^2
    dudy_fun(y, x, z, t)   = 1 - 2*y*cos(2.9*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    dvdy_fun(y, x, z, t)   = -(π/2)*sin(π*y)*sin(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    dwdy_fun(y, x, z, t)   = -(π*sin(π*y)*(1 - y^2) + 2*y*cos(π*y))*exp(sin(2.9*x))*exp(sin(5.8*z))*cos(2π*t)^2
    d2udy2_fun(y, x, z, t) = -2*exp(cos(5.8*z))*cos(2.9*x)*atan(sin(2π*t))
    d2vdy2_fun(y, x, z, t) = -(π^2/2)*cos(π*y)*sin(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    d2wdy2_fun(y, x, z, t) = -(π^2*cos(π*y)*(1 - y^2) - 4π*y*sin(π*y) + 2*cos(π*y))*exp(sin(2.9*x))*exp(sin(5.8*z))*cos(2π*t)^2
    dudz_fun(y, x, z, t)   = -5.8*(1 - y^2)*cos(2.9*x)*sin(5.8*z)*exp(cos(5.8*z))*atan(sin(2π*t))
    dvdz_fun(y, x, z, t)   = 5.8*cos(π*y/2)^2*sin(2.9*x)*cos(5.8*z)*exp(sin(5.8*z))*cos(sin(2π*t))
    dwdz_fun(y, x, z, t)   = 5.8*cos(π*y)*(1 - y^2)*exp(sin(2.9*x))*cos(5.8*z)*exp(sin(5.8*z))*cos(2π*t)^2
    d2udz2_fun(y, x, z, t) = (5.8^2)*(1 - y^2)*cos(2.9*x)*(sin(5.8*z)^2 - cos(5.8*z))*exp(cos(5.8*z))*atan(sin(2π*t))
    d2vdz2_fun(y, x, z, t) = (5.8^2)*cos(π*y/2)^2*sin(2.9*x)*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(sin(2π*t))
    d2wdz2_fun(y, x, z, t) = (5.8^2)*cos(π*y)*(1 - y^2)*exp(sin(2.9*x))*(cos(5.8*z)^2 - sin(5.8*z))*exp(sin(5.8*z))*cos(2π*t)^2
    u_out_fun(y, x, z, t)  = (d2udx2_fun(y, x, z, t) + d2udy2_fun(y, x, z, t) + d2udz2_fun(y, x, z, t))/Re - u_fun(y, x, z, t)*dudx_fun(y, x, z, t) - v_fun(y, x, z, t)*dudy_fun(y, x, z, t) - w_fun(y, x, z, t)*dudz_fun(y, x, z, t) + Ro*v_fun(y, x, z, t)
    v_out_fun(y, x, z, t)  = (d2vdx2_fun(y, x, z, t) + d2vdy2_fun(y, x, z, t) + d2vdz2_fun(y, x, z, t))/Re - u_fun(y, x, z, t)*dvdx_fun(y, x, z, t) - v_fun(y, x, z, t)*dvdy_fun(y, x, z, t) - w_fun(y, x, z, t)*dvdz_fun(y, x, z, t) - Ro*u_fun(y, x, z, t)
    w_out_fun(y, x, z, t)  = (d2wdx2_fun(y, x, z, t) + d2wdy2_fun(y, x, z, t) + d2wdz2_fun(y, x, z, t))/Re - u_fun(y, x, z, t)*dwdx_fun(y, x, z, t) - v_fun(y, x, z, t)*dwdy_fun(y, x, z, t) - w_fun(y, x, z, t)*dwdz_fun(y, x, z, t)

    # construct grid
    Ny = 32; Nx = 33; Nz = 33; Nt = 51
    D₁ = chebdiff(Ny)
    D₂ = chebddiff(Ny)
    ws = chebws(Ny)
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2.9, 5.8,
                    D₁,
                    D₂,
                    adjoint(D₁, ws),
                    adjoint(D₂, ws),
                    ws)

    # test nonlinear operator
    Re = rand()*50
    Ro = rand()
    op = CartesianPrimitive3DNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
    u = FFT(VectorField(g, u_fun, v_fun, w_fun))
    exact = FFT(VectorField(g, u_out_fun, v_out_fun, w_out_fun))
    @test op(0.0, u, similar(u)) ≈ exact
end

@testset "Cartesian primitive linearised NSE    " begin
    # define functions
    ux_fun(y, x, z, t)       = y + (1 - y^2)*cos(2.9*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    uy_fun(y, x, z, t)       = cos(π*y/2)^4*exp(sin(2.9*x))*sin(5.8*z)*cos(sin(2π*t))
    uz_fun(y, x, z, t)       = (1 - y^2)*sin(2.9*x)*cos(5.8*z)*cos(cos(2π*t))
    vx_fun(y, x, z, t)       = cos(π*y/2)^2*sin(2.9*x)*exp(sin(5.8*z))*cos(sin(2π*t))
    vy_fun(y, x, z, t)       = (1 - y^2)*cos(2.9*x)*exp(cos(5.8*z))*atan(cos(2π*t))
    vz_fun(y, x, z, t)       = sin(π*y)*exp(cos(2.9*x))*sin(z)^2*cos(2π*t)
    wx_fun(y, x, z, t)       = cos(π*y)*(1 - y^2)*cos(2.9*x)^2*exp(sin(5.8*z))*cos(2π*t)^2
    wy_fun(y, x, z, t)       = cos(π*y/2)*sin(2.9*x)*cos(5.8*z)*sin(2π*t)^2
    wz_fun(y, x, z, t)       = cos(π*y/2)^4*exp(sin(2.9*x))*sin(5.8*z)*atan(cos(2π*t))
    duxdx_fun(y, x, z, t)    = -2.9*(1 − y^2)*sin(2.9*x)*exp(cos(5.8*z))*atan(sin(2π*t))
    duydy_fun(y, x, z, t)    = -2π*sin(π*y/2)*cos(π*y/2)^3*exp(sin(2.9*x))*sin(5.8*z)*cos(sin(2π*t))
    duzdz_fun(y, x, z, t)    = -5.8*(1 - y^2)*sin(2.9*x)*sin(5.8*z)*cos(cos(2π*t))
    div_u_wx_fun(y, x, z, t) = (duxdx_fun(y, x, z, t) + duydy_fun(y, x, z, t) + duzdz_fun(y, x, z, t))*wx_fun(y, x, z, t)
    div_u_wy_fun(y, x, z, t) = (duxdx_fun(y, x, z, t) + duydy_fun(y, x, z, t) + duzdz_fun(y, x, z, t))*wy_fun(y, x, z, t)
    div_u_wz_fun(y, x, z, t) = (duxdx_fun(y, x, z, t) + duydy_fun(y, x, z, t) + duzdz_fun(y, x, z, t))*wz_fun(y, x, z, t)

    # construct grid
    Ny = 16; Nx = 15; Nz = 15; Nt = 21
    fd_grid = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
    D₁ = DiffMatrix(fd_grid.xs, 3, 1)
    D₂ = DiffMatrix(fd_grid.xs, 3, 2)
    D₁⁺ = adjoint(D₁, fd_grid.ws)
    D₂⁺ = adjoint(D₂, fd_grid.ws)
    g = ChannelGrid(fd_grid.xs, Nx, Nz, Nt,
                    2.9, 5.8,
                    D₁,
                    D₂,
                    D₁⁺,
                    D₂⁺,
                    fd_grid.ws)
    @test g.D₁⁺ isa AdjointDiffMatrix
    @test g.D₂⁺ isa AdjointDiffMatrix

    # define fields
    u       = FFT(VectorField(g, ux_fun, uy_fun, uz_fun))
    v       = FFT(VectorField(g, vx_fun, vy_fun, vz_fun))
    w       = FFT(VectorField(g, wx_fun, wy_fun, wz_fun))
    div_u_w = FFT(VectorField(g, div_u_wx_fun, div_u_wy_fun, div_u_wz_fun))

    # test perturbed nonlinear equations approximates linearised equations
    Re = rand()*50
    Ro = rand()
    op_nl = CartesianPrimitive3DNSE(g,  Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE)
    op_ln = CartesianPrimitive3DLNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=Forward())
    op_ad = CartesianPrimitive3DLNSE(g, Re, force=CoriolisForce(Ro), flags=FFTW.ESTIMATE, mode=AdjointDiscrete())
    a = op_nl(0.0, u .+ 1e-6.*v, similar(u)) - op_nl(0.0, u, similar(u))
    b = op_ln(0.0, u, 1e-6.*v, similar(u))
    @test norm(a - b) < 1e-11

    # test adjoint identity
    # ! this test should be independent of grid size since it is discretely consistent
    @test abs(dot(op_ln(0.0, u, v, similar(u)), w) - dot(v, op_ad(0.0, u, w, similar(u)))) < 1e-12
    # ! use below test for continous adjoint
    # @test abs(dot(op_ln(0.0, u, v, similar(u)), w) - dot(v, op_ad(0.0, u, w, similar(u))) - dot(v, div_u_w)) < 1e-12
end

@testset "Plane Couette flow constructor       " begin
    Ny = 16; Nx = 15; Nz = 15; Nt = 21
    D₁ = chebdiff(Ny)
    D₂ = chebddiff(Ny)
    ws = chebws(Ny)
    D₁⁺ = adjoint(D₁, ws)
    D₂⁺ = adjoint(D₂, ws)
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2.9, 5.8,
                    D₁,
                    D₂,
                    D₁⁺,
                    D₂⁺,
                    ws)

    Re = rand()*50
    Ro = rand()
    op = PlaneCouetteFlow(g, Re, Ro=Ro, fftw_flags=FFTW.ESTIMATE)

    @test g.D₁⁺ === D₁⁺
    @test g.D₂⁺ === D₂⁺
    @test op.base == (copy(g.y), nothing, nothing)
    @test op.nl.force.Ro == Ro
    @test op.ln.force.Ro == Ro
end

@testset "Constant forcing                     " begin
    Ny = 8; Nx = 5; Nz = 5; Nt = 5
    y = collect(range(-1, 1; length=Ny))
    D = zeros(Ny, Ny)
    ws = ones(Ny)
    g = ChannelGrid(y, Nx, Nz, Nt,
                    1.0, 1.0,
                    D,
                    D,
                    D,
                    D,
                    ws)

    out = VectorField(g)
    force = ConstantForcing(2.5)
    k0 = WaveNumberVector(0, 0, 0)

    force(out, nothing, Forward())
    @test all(out[1][k0] .== 2.5 + 0im)
    @test all(out[2][k0] .== 0.0 + 0im)
    @test all(out[3][k0] .== 0.0 + 0im)
    @test all(out[1][WaveNumberVector(1, 0, 0)] .== 0.0 + 0im)

    @test (@allocated force(out, nothing, Forward())) == 0
    @test all(out[1][k0] .== 5.0 + 0im)
end
