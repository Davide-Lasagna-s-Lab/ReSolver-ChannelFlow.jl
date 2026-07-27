@testset "GPU equations                         " begin
    # construct grid
    Ny = 32; Nx = 15; Nz = 33; Nt = 51
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
    op_nl = @test_nowarn CartesianPrimitive3DNSE( g, Re, force=CoriolisForce(Ro))
    op_ln = @test_nowarn CartesianPrimitive3DLNSE(g, Re, force=CoriolisForce(Ro), mode=AdjointDiscrete())

    # check if types are correct and memory is correct place
    @test op_nl isa CartesianPrimitive3DNSE{Float32, <:FFTPlans, <:FTField{<:CUDAExt.GPUGrid}, <:Field{<:CUDAExt.GPUGrid}}
    @test op_ln isa CartesianPrimitive3DLNSE{AdjointDiscrete, Float32, <:FFTPlans, <:FTField{<:CUDAExt.GPUGrid}, <:Field{<:CUDAExt.GPUGrid}}
    @test op_nl.plans.backend == CUDA.cuFFT
    @test op_ln.plans.backend == CUDA.cuFFT
    @test eltype(op_nl.scache) <: VectorField{3, <:FTField{<:CUDAExt.GPUGrid}}
    @test eltype(op_nl.pcache) <: VectorField{3, <:Field{<:CUDAExt.GPUGrid}}
    @test eltype(op_ln.scache) <: VectorField{3, <:FTField{<:CUDAExt.GPUGrid}}
    @test eltype(op_ln.pcache) <: VectorField{3, <:Field{<:CUDAExt.GPUGrid}}

    # check if computation completes
    u = VectorField(g)
    v = VectorField(g)
    out = VectorField(g)
    @test_nowarn op_nl(0.0, u, out)
    @test_nowarn op_ln(0.0, u, v, out)
end
