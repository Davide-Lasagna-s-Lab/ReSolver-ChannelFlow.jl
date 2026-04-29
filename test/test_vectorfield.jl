@testset "Vector field                          " begin
    # construct grid
    Ny = 16; Nx = 15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    # test vectorfield construction
    u = @test_nowarn VectorField(g)
    @test u isa VectorField{3, FTField{typeof(g), Float64, Array{ComplexF64, 4}}}
    u = @test_nowarn VectorField(similar(g, Float32))
    @test u isa VectorField{3, FTField{typeof(similar(g, Float32)), Float32, Array{ComplexF32, 4}}}
    u = @test_nowarn VectorField(g, N=2)
    @test u isa VectorField{2, FTField{typeof(g), Float64, Array{ComplexF64, 4}}}
    u = @test_nowarn VectorField(g, Field)
    @test u isa VectorField{3, Field{typeof(g), Float64, Array{Float64, 4}}}
    u = @test_nowarn VectorField(g, Field, dealias=false)
    @test u isa VectorField{3, Field{typeof(g), Float64, Array{Float64, 4}}}
    u = @test_nowarn VectorField(g, Field, dealias=true)
    @test u isa VectorField{3, Field{typeof(g), Float64, Array{Float64, 4}}}
    @test size(u[1]) == (Ny, ReSolverChannelFlow._padded_size((Nx, Nz, Nt), Val(true))...)
    u = @test_nowarn VectorField(g, ((y, x, z, t)->1.0, (y, x, z, t)->(1 - y^2)*cos(2π*z)), 1.0)
    @test u isa VectorField{2, Field{typeof(g), Float64, Array{Float64, 4}}}
    @test all(parent(u[1]) .== 1.0)
    pts = points(g, 1.0)
    @test all(parent(u[2]) .== (1 .- reshape(pts[1], :, 1, 1, 1).^2).*cos.(2π.*reshape(pts[3], 1, 1, :, 1)))
    u = @test_nowarn VectorField(g, ((y, x, z, t)->1.0, (y, x, z, t)->(1 - y^2)*cos(2π*z)), 1.0, dealias=true)
    @test u isa VectorField{2, Field{typeof(g), Float64, Array{Float64, 4}}}
    @test all(parent(u[1]) .== 1.0)
    pts = points(g, 1.0, ReSolverChannelFlow._padded_size((Nx, Nz, Nt), Val(true)))
    @test all(parent(u[2]) .== (1 .- reshape(pts[1], :, 1, 1, 1).^2).*cos.(2π.*reshape(pts[3], 1, 1, :, 1)))

    # test interface
    @test size(u) == (2,)
    @test length(u) == 2
    @test eltype(u) == typeof(Field(g))
    @test similar(u) isa VectorField{2, Field{typeof(g), Float64, Array{Float64, 4}}}
    @test copy(u)[1] == u[1]
    @test copy(u)[2] == u[2]
    @test all(parent(zero(u)[1]) .== 0.0)
    @test all(parent(zero(u)[2]) .== 0.0)

    # test growto
    u = VectorField(g)
    for n in 1:3
        u[n] .= randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)
    end
    v = growto(u, (19, 65, 49))
    for n in 1:3
        @test v[n] == growto(u[n], (19, 65, 49))
    end
end
