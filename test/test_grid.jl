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

    # test similar
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, adjoint_diff=false)
    @test         g           isa ChannelGrid{(Ny, Nx, Nz, Nt), Float64}
    @test convert(Float32, g) isa ChannelGrid{(Ny, Nx, Nz, Nt), Float32}

    # test point generation
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, adjoint_diff=false)
    pts = points(g, dealias=false)
    @test pts[1][:] == y
    @test pts[2][:]  ≈ range(0, 2π*(1 - 1/Nx), length=Nx)/α # precision differences in operations
    @test pts[3][:]  ≈ range(0, 2π*(1 - 1/Nz), length=Nz)/β # precision differences in operations
    @test pts[4][:]  ≈ range(0,    (1 - 1/Nt), length=Nt)   # mean they aren't exactly equal
    Nx_new = rand(Nx+2:2:81)
    Nz_new = rand(Nz+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    pts = points(g, (Nx_new, Nz_new, Nt_new))
    @test pts[2][:]  ≈ range(0, 2π*(1 - 1/Nx_new), length=Nx_new)/α # precision differences in operations
    @test pts[3][:]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β # precision differences in operations
    @test pts[4][:]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)   # mean they aren't exactly equal
    @test points(g, dealias=true) == points(g, (ReSolverChannelFlow._padded_size.((Nx, Nz, Nt), Val(true))))

    # test growto
    g = ChannelGrid(y, Nx, Nz, Nt, α, β, Dy, Dy2, ws, adjoint_diff=false)
    Nx_new = rand(Nx+2:2:81)
    Nz_new = rand(Nz+2:2:81)
    Nt_new = rand(Nt+2:2:81)
    g_new = growto(g, (Nx_new, Nz_new, Nt_new))
    pts = points(g)
    pts_new = points(g_new)
    @test pts_new[1]    == pts[1]
    @test pts_new[2][:]  ≈ range(0, 2π*(1 - 1/Nx_new), length=Nx_new)/α # precision differences in operations
    @test pts_new[3][:]  ≈ range(0, 2π*(1 - 1/Nz_new), length=Nz_new)/β # precision differences in operations
    @test pts_new[4][:]  ≈ range(0,    (1 - 1/Nt_new), length=Nt_new)   # mean they aren't exactly equal
end
