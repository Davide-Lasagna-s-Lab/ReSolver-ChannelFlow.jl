@testset "FFT transforms                        " begin
    # construct grid
    Ny = 16; Nx = 7; Nz = 15; Nt = 5
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    adjoint_diff=false)

    # create plans
    plans  = @test_nowarn ReSolverChannelFlow.FFTPlans((Ny, Nx, Nz, Nt), (2, 3, 4), dealias=false, flags=FFTW.ESTIMATE)
    plansd = @test_nowarn ReSolverChannelFlow.FFTPlans((Ny, Nx, Nz, Nt), (2, 3, 4), dealias=true,  flags=FFTW.ESTIMATE)

    # randon signal
    U  = FTField(g, rand(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt))
    Ud = growto(U, ReSolverChannelFlow._padded_size((Nx, Nz, Nt), Val(true)))
    u  = Field(g)
    ud = Field(g, dealias=true)

    # test transforms
    @test  plans(similar(U),  plans(u,  U))       ≈   U
    @test plansd(similar(U), plansd(ud, U))       ≈   U
    @test plansd(   copy(U), plansd(ud, U), true) ≈ 2*U

    # test allocations
    funf(plan, A, B) = @allocated plan(A, B)
    funb(plan, B, A) = @allocated plan(B, A, false)
    funf(plans, ud,  U) # required to avoid allocation when first called (for some unknown reason)
    funb(plans,  U, ud)
    @test funf(plans, ud,  U) == 0
    @test funb(plans,  U, ud) == 0

    # test allocating transforms
    @test FFT(plans(similar(u), U)) ≈ U
    @test IFFT(U) == plans(similar(u), U)
    @test FFT(plans(similar(u), U), ReSolverChannelFlow._padded_size((Nx, Nz, Nt), Val(true))) ≈ Ud
    @test IFFT(U, ReSolverChannelFlow._padded_size((Nx, Nz, Nt), Val(true))) == plansd(similar(ud), U)

    # vector field transforms
    U = VectorField(g)
    for n in 1:3
        parent(U[n]) .= randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)
        parent(U[n])[:, 1, 1, 1] .= real.(parent(U[n])[:, 1, 1, 1])
        ReSolverChannelFlow.apply_symmetry!(parent(U[n]))
    end
    u = VectorField(g, Field)
    U_new = plans(similar(U), plans(u, U))
    for n in 1:3
        @test U_new[n] ≈ U[n]
    end
    @test IFFT(U) == u
    u_new = IFFT(U, (11, 21, 15))
    for n in 1:3
        @test parent(u_new[n]) ≈ parent(IFFT(growto(U[n], (11, 21, 15))))
    end
    U_new = FFT(u)
    for n in 1:3
        @test U_new[n] ≈ U[n]
    end
    U_new = FFT(u, (11, 21, 15))
    for n in 1:3
        @test U_new[n] ≈ growto(U[n], (11, 21, 15))
    end
end
