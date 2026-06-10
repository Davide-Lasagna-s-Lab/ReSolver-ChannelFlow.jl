@testset "Field grid                            " begin
    # generate random inputs
    Ny  = rand(3:51)
    Nx  = rand(3:2:51)
    Nz  = rand(3:2:51)
    Nt  = rand(3:2:51)
    y   = rand(Float64, Ny)
    Dy  = rand(Float64, (Ny, Ny))
    Dy2 = rand(Float64, (Ny, Ny))
    ws  = rand(Float64, Ny)
    α   = 2
    β   = π

    # test isa
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dy, Dy2, ws)
    @test g isa ChannelGrid{(Nx, Ny, Nz, Nt)}
    @test size(g) == (Ny, Nx, Nz, Nt)

    # test point generation; points returns (y, x, z, t) in array-dimension order
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dy, Dy2, ws)
    pts = points(g, dealias=false)
    @test pts[1][:] == y                                               # y collocation points
    @test pts[2][:]  ≈ range(0, 2π*(1 - 1/Nx), length=Nx)/α          # x coordinate
    @test pts[3][:]  ≈ range(0, 2π*(1 - 1/Nz), length=Nz)/β          # z coordinate
    @test pts[4][:]  ≈ range(0,    (1 - 1/Nt), length=Nt)            # t coordinate
    Nx_new = rand(Nx+2:2:81)
    Nz_new = rand(Nz+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    pts = points(g, (Nx_new, Nz_new, Nt_new))
    @test pts[2][:]  ≈ range(0, 2π*(1 - 1/Nx_new), length=Nx_new)/α  # x coordinate
    @test pts[3][:]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β  # z coordinate
    @test pts[4][:]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)    # t coordinate
    padded_storage_size = NSEBase.get_padded_size(size(g), NSEBase.fft_storage_dims(g))
    padded_homogeneous_size = (padded_storage_size[ReSolverChannelFlow.CHANNEL_AXES[1]],
                               padded_storage_size[ReSolverChannelFlow.CHANNEL_AXES[3]],
                               padded_storage_size[ReSolverChannelFlow.CHANNEL_AXES[4]])
    @test points(g, dealias=true) == points(g, padded_homogeneous_size)

    # test growto
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, Dy, Dy2, ws)
    Nx_new = rand(Nx+2:2:81)
    Nz_new = rand(Nz+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    g_new = growto(g, (Nx_new, Nz_new, Nt_new))
    @test g_new isa ChannelGrid{(Nx_new, Ny, Nz_new, Nt_new)}
    @test size(g_new) == (Ny, Nx_new, Nz_new, Nt_new)
    pts = points(g)
    pts_new = points(g_new)
    @test pts_new[1]    == pts[1]                                       # y unchanged after growto
    @test pts_new[2][:]  ≈ range(0, 2π*(1 - 1/Nx_new), length=Nx_new)/α
    @test pts_new[3][:]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β
    @test pts_new[4][:]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)
end
