# Run MPI specific tests

using MPI, Test

files = [
         ("test_fields.jl",                       "MPI Fields             ", 4),
         ("test_derivatives.jl",                  "MPI Field derivatives  ", 4),
         ("test_cartesianprimitive.jl",           "MPI Nonlinear operator ", 4),
         ("test_linearisedcartesianprimitive.jl", "MPI Linearised operator", 4),
         ("test_galerkin.jl",                     "MPI Field projection   ", 4),
         ]

cmd(file, nprocs) = addenv(`$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, file))`)

@testset "$(rpad("$testname", 38))" for (f, testname, nprocs) in files
    p = run(ignorestatus(cmd(f, nprocs)))
    @test success(p)
end
