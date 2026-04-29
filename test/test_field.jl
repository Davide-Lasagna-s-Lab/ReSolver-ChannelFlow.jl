@testset "Physical channel field                " begin
    # construct grid
    Ny = 16; Nx = 15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    2π, 2π,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    for dealias in [false, true]
        # test constructors
        u = Field(g, dealias=dealias)
        data_size = ReSolverChannelFlow._padded_size((Nx, Nz, Nt), dealias ? Val(true) : Val(false))
        @test u.data == zeros(Ny, data_size...)
        fun(y, x, z, t) = (1 - y^2)*exp(sin(x))*cos(z)*sin(t)
        u = Field(g, fun, 2π, dealias=dealias)
        pts = points(g, 2π, data_size)
        for ny in 1:Ny, nx in 1:data_size[1], nz in 1:data_size[2], nt in 1:data_size[3]
            @test u.data[ny, nx, nz, nt] == fun(pts[1][ny], pts[2][nx], pts[3][nz], pts[4][nt])
        end

        # test interfaces
        @test eltype(u) == Float64
        @test size(u) == (Ny, data_size...)
        @test similar(u) isa Field{typeof(g), Float64}
        @test similar(u, Float16) isa Field{typeof(similar(g, Float16)), Float16}
        @test copy(u) == u
        @test zero(u) == Field(g, dealias=dealias)
    end
end
