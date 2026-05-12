@testset "Mode number index conversion          " begin
    # test mode number conversion
    Nx = 23 # Nx can be even or odd
    Nz = 23 # Nz has to be odd
    Nt = 23 # Nt has to be odd
    nxs = collect(0:(Nx >> 1) + 1)
    nzs = [collect(0:(Nz >> 1)); collect(-(Nt >> 1):-1)]
    nts = [collect(0:(Nt >> 1)); collect(-(Nt >> 1):-1)]
    for _nx in 1:(Nx >> 1) + 1, _nz in 1:Nz, _nt in 1:Nt
        n = ModeNumber(nxs[_nx], nzs[_nz], nts[_nt])
        @test ReSolverChannelFlow._convert_modenumber(n, Nz, Nt) == (_nx, _nz, _nt, false)
    end
    nxs = collect(0:-1:-(Nx >> 1))
    nzs = [[0]; collect(-1:-1:-(Nz >> 1)); collect((Nz >> 1:-1:1))]
    nts = [[0]; collect(-1:-1:-(Nt >> 1)); collect((Nt >> 1:-1:1))]
    for _nx in 2:(Nx >> 1) + 1, _nz in 1:Nz, _nt in 1:Nt
        n = ModeNumber(nxs[_nx], nzs[_nz], nts[_nt])
        @test ReSolverChannelFlow._convert_modenumber(n, Nz, Nt) == (_nx, _nz, _nt, true)
    end
end

@testset "Mode number indexing in FTField       " begin
    # construct grid
    Ny = 16; Nx=15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    Float64,
                    adjoint_diff=false)

    # test mode number indexing
    A = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)
    A_new = ReSolverChannelFlow.NSEBase.apply_symmetry!(ReSolverChannelFlow.NSEBase.normalise_mean!(A, (2, 3, 4)), (2, 3, 4))
    u = FTField(g, A)
    for ny in 1:Ny
        @test u[ny, ModeNumber(0, 0, 0)] == A_new[ny, 1, 1, 1]
    end
    for ny in 1:Ny, nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0, 0,  nt)] == A_new[ny, 1, 1,    nt+1]
        @test u[ny, ModeNumber(0, 0, -nt)] == A_new[ny, 1, 1, Nt-nt+1]
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1)
        @test u[ny, ModeNumber(0,  nz, 0)] ==      A_new[ny, 1, nz+1, 1]
        @test u[ny, ModeNumber(0, -nz, 0)] == conj(A_new[ny, 1, nz+1, 1])
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1), nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0,  nz,  nt)] ==      A_new[ny, 1, nz+1,    nt+1]
        @test u[ny, ModeNumber(0,  nz, -nt)] ==      A_new[ny, 1, nz+1, Nt-nt+1]
        @test u[ny, ModeNumber(0, -nz,  nt)] == conj(A_new[ny, 1, nz+1, Nt-nt+1])
        @test u[ny, ModeNumber(0, -nz, -nt)] == conj(A_new[ny, 1, nz+1,    nt+1])
    end
    for ny in 1:Ny, nx in 1:(Nx >> 1), nz in -(Nz >> 1):(Nz >> 1), nt in (Nt >> 1):(Nt >> 1)
        _nz = nz >= 0 ? nz+1 : Nz+nz+1
        _nt = nt >= 0 ? nt+1 : Nt+nt+1
        @test u[ny, ModeNumber( nx, nz, nt)] == A_new[ny, nx+1,         _nz,      _nt]
        _nz = nz <= 0 ? -nz+1 : Nz-nz+1
        _nt = nt <= 0 ? -nt+1 : Nt-nt+1
        @test u[ny, ModeNumber(-nx, nz, nt)] == conj(A_new[ny, nx+1, _nz, _nt])
    end
end
