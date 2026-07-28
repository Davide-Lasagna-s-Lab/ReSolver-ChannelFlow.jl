@testset "GPU equations                         " begin
    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
    M = 2
    y, ws = FDGrids.grid(Ny, -1, 1, MappedGrid(1))
    D₁ = DiffMatrix(y, 3, 1)
    D₂ = DiffMatrix(y, 3, 2)
    g = CUDA.cu(ChannelGrid(y, Nx, Nz, Nt,
                            2π, 5.8,
                            D₁,
                            D₂,
                            adjoint(D₁, ws),
                            adjoint(D₂, ws),
                            ws))

    # construct Couette and Poiseuille equations
    Re = 10
    Ro = Float32(0.5)
    op = @test_nowarn NSEBase.construct_equations(g,
                                                  Re,
                                                 (parent(g).y, nothing, nothing),
                                                  NSEBase.CartesianPrimitive3D();
                                                  force=CoriolisForce(Ro),
                                                   mode=AdjointDiscrete(),
                                                dealias=true)

    # check if types are correct and memory is correct place
    @test op.nl isa CartesianPrimitive3DNSE{Float32, <:FFTPlans, <:FTField{<:CUDAExt.GPUGrid}, <:Field{<:CUDAExt.GPUGrid}}
    @test op.ln isa CartesianPrimitive3DLNSE{AdjointDiscrete, Float32, <:FFTPlans, <:FTField{<:CUDAExt.GPUGrid}, <:Field{<:CUDAExt.GPUGrid}}
    @test op.base isa Tuple{<:CuArray{Float32, 1}, Nothing, Nothing}
    @test op.nl.plans.backend == CUDA.cuFFT
    @test op.ln.plans.backend == CUDA.cuFFT
    @test eltype(op.nl.scache) <: VectorField{3, <:FTField{<:CUDAExt.GPUGrid}}
    @test eltype(op.nl.pcache) <: VectorField{3, <:Field{<:CUDAExt.GPUGrid}}
    @test eltype(op.ln.scache) <: VectorField{3, <:FTField{<:CUDAExt.GPUGrid}}
    @test eltype(op.ln.pcache) <: VectorField{3, <:Field{<:CUDAExt.GPUGrid}}
    @test eltype(op.cache1)    <: FTField{<:CUDAExt.GPUGrid}
    @test eltype(op.cache2)    <: FTField{<:CUDAExt.GPUGrid}

    # check if computation completes
    Ψ   = ntuple(_ -> CUDA.zeros(ComplexF32, M, Ny, (Nx >> 1) + 1, Nz, Nt), 3)
    a   = ProjectedField(g, Ψ)
    b   = ProjectedField(g, Ψ)
    out = ProjectedField(g, Ψ)
    @test_nowarn op(out, a)
    @test_nowarn op(out, a, b)
end
