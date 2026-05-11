@testset "Mode number index conversion          " begin
    Nx = 23 # Nx can be even or odd
    Nz = 23 # Nz has to be odd
    Nt = 23 # Nt has to be odd
    Ny = 5
    y   = rand(Ny)
    Dy  = rand(Ny, Ny)
    Dy2 = rand(Ny, Ny)
    ws  = rand(Ny)
    g = ChannelGrid(y, Nx, Nz, Nt, 1.0, 1.0, Dy, Dy2, ws, adjoint_diff=false)

    # non-negative rfft wavenumber — do_conj = false
    nxs = collect(0:(Nx >> 1) + 1)
    nzs = [collect(0:(Nz >> 1)); collect(-(Nz >> 1):-1)]
    nts = [collect(0:(Nt >> 1)); collect(-(Nt >> 1):-1)]
    for _nx in 1:(Nx >> 1) + 1, _nz in 1:Nz, _nt in 1:Nt
        n = ModeNumber(nxs[_nx], nzs[_nz], nts[_nt])
        @test NSEBase._modenumber_to_projected_indices(g, n) == (_nx, _nz, _nt, false)
    end

    # negative rfft wavenumber — do_conj = true, conjugate indices returned
    nxs = collect(0:-1:-(Nx >> 1))
    nzs = [[0]; collect(-1:-1:-(Nz >> 1)); collect((Nz >> 1:-1:1))]
    nts = [[0]; collect(-1:-1:-(Nt >> 1)); collect((Nt >> 1:-1:1))]
    for _nx in 2:(Nx >> 1) + 1, _nz in 1:Nz, _nt in 1:Nt
        n = ModeNumber(nxs[_nx], nzs[_nz], nts[_nt])
        @test NSEBase._modenumber_to_projected_indices(g, n) == (_nx, _nz, _nt, true)
    end
end
