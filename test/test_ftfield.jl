@testset "Spectral channel field                " begin
    # construct grid
    # T = rand([Float64, Float32, Float16])
    T = Float32
    Ny = 16; Nx=15; Nz = 33; Nt = 33
    g = ChannelGrid(chebpts(Ny), Nx, Nz, Nt,
                    1.0, 1.0,
                    chebdiff(Ny),
                    chebddiff(Ny),
                    chebws(Ny),
                    T, adjoint_diff=false)

    # some random variables
    A = randn(ComplexF64, Ny, (Nx >> 1) + 1, Nz, Nt)

    # test constructor
    @test_nowarn FTField(g, A)
    @test_nowarn FTField(g)

    # test interfaces
    @test eltype(FTField(g)) == Complex{T}
    @test size(FTField(g)) == (Ny, (Nx >> 1) + 1, Nz, Nt)
    @test similar(FTField(g, A)) isa FTField{typeof(g), T, Array{Complex{T}, 4}}
    @test similar(FTField(g, A), ComplexF64) isa FTField{typeof(similar(g, Float64)), Float64, Array{ComplexF64, 4}}
    @test copy(FTField(g, A)) == FTField(g, A)
    @test zero(FTField(g, A)) == FTField(g, zero(A))
    @test abs(FTField(g, A)) == FTField(g, abs.(Complex{T}.(A)))

    # test array indexing
    u = FTField(g, A)
    for ny in 1:Ny, nx in 1:(Nx >> 1) + 1, nz in 1:Nz, nt in 1:Nt
        @test u[ny, nx, nz, nt] == Complex{T}(A[ny, nx, nz, nt])
    end

    # test mode number indexing
    for ny in 1:Ny
        @test u[ny, ModeNumber(0, 0, 0)] == Complex{T}(A[ny, 1, 1, 1])
    end
    for ny in 1:Ny, nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0, 0,  nt)] == Complex{T}(A[ny, 1, 1,    nt+1])
        @test u[ny, ModeNumber(0, 0, -nt)] == Complex{T}(A[ny, 1, 1, Nt-nt+1])
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1)
        @test u[ny, ModeNumber(0,  nz, 0)] == Complex{T}(     A[ny, 1, nz+1, 1])
        @test u[ny, ModeNumber(0, -nz, 0)] == Complex{T}(conj(A[ny, 1, nz+1, 1]))
    end
    for ny in 1:Ny, nz in 1:(Nz >> 1), nt in 1:(Nt >> 1)
        @test u[ny, ModeNumber(0,  nz,  nt)] == Complex{T}(     A[ny, 1, nz+1,    nt+1])
        @test u[ny, ModeNumber(0,  nz, -nt)] == Complex{T}(     A[ny, 1, nz+1, Nt-nt+1])
        @test u[ny, ModeNumber(0, -nz,  nt)] == Complex{T}(conj(A[ny, 1, nz+1, Nt-nt+1]))
        @test u[ny, ModeNumber(0, -nz, -nt)] == Complex{T}(conj(A[ny, 1, nz+1,    nt+1]))
    end
    for ny in 1:Ny, nx in 1:(Nx >> 1), nz in -(Nz >> 1):(Nz >> 1), nt in (Nt >> 1):(Nt >> 1)
        _nz = nz >= 0 ? nz+1 : Nz+nz+1
        _nt = nt >= 0 ? nt+1 : Nt+nt+1
        @test u[ny, ModeNumber( nx, nz, nt)] == Complex{T}(A[ny, nx+1,         _nz,      _nt])
        _nz = nz <= 0 ? -nz+1 : Nz-nz+1
        _nt = nt <= 0 ? -nt+1 : Nt-nt+1
        @test u[ny, ModeNumber(-nx, nz, nt)] == Complex{T}(conj(A[ny, nx+1, _nz, _nt]))
    end
end
