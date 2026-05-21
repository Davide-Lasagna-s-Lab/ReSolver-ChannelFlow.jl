@testset "Field symmetry shifts                 " begin
    # define functions
    u_funs       = ((y, x, z, t)->y + (1 - y^2)*cos(x)*cos(z)*cos(2π*t),
                    (y, x, z, t)->-(π/2)*cos(π*y/2)^2*cos(x)*sin(z)*sin(2π*t),
                    (y, x, z, t)->(π/2)*sin(π*y)*cos(x)*cos(z)*sin(2π*t))
    u_shift_funs = ((y, x, z, t)->y + (1 - y^2)*cos(x + sx)*cos(z + sz)*cos(2π*(t + st)),
                    (y, x, z, t)->-(π/2)*cos(π*y/2)^2*cos(x + sx)*sin(z + sz)*sin(2π*(t + st)),
                    (y, x, z, t)->(π/2)*sin(π*y)*cos(x + sx)*cos(z + sz)*sin(2π*(t + st)))

    # construct grid
    Ny = 16; Nx = 15; Nz = 33; Nt = 33
    D₁ = chebdiff(Ny)
    D₂ = chebddiff(Ny)
    ws = chebws(Ny)
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    D₁,
                    D₂,
                    adjoint(D₁, ws),
                    adjoint(D₂, ws),
                    ws)

    # generate modes
    # M = 10
    # Ψ = zeros(ComplexF64, 3*Ny, M, (Nx >> 1) + 1, Nz, Nt)
    # for nt in 1:Nt, nz in 1:Nz, nx in 1:((Nx >> 1) + 1)
    #     Ψ[:, :, nx, nz, nt] .= qr(randn(ComplexF64, 3*Ny, M)).Q[:, 1:M]
    # end
    # for m in 1:M
    #     ReSolverChannelFlow.apply_symmetry!(@view(Ψ[:, m, :, :, :]))
    #     Ψ[:, m, 1, 1, 1] .= real.(Ψ[:, m, 1, 1, 1])
    # end

    # test shifts
    sx = 2π*rand(); sz = 2π*rand(); st = rand()
    u       = FFT(VectorField(g, u_funs...      ))
    u_shift = FFT(VectorField(g, u_shift_funs...))
    @test shift!(     u,  (0,  0,  0))  === u
    @test shift!(copy(u), (sx, sz, st)) ≈   u_shift atol=1e-12
    # TODO: these tests needs to be added back once NSEBase.jl is better
    # a       = project(u,       Ψ)
    # a_shift = project(u_shift, Ψ)
    # @test shift!(     a,  (0,  0,  0),) === a
    # @test shift!(copy(a), (sx, sz, st)) ≈   a_shift atol=1e-12
end
