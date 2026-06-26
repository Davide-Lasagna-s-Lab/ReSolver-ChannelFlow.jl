# Run MPI specific tests

using MPI, Test

files = [
         ("test_fields.jl",                       "Fields             ", 4),
         ("test_derivatives.jl",                  "Field derivatives  ", 4),
         ("test_cartesianprimitive.jl",           "Nonlinear operator ", 4),
         ("test_linearisedcartesianprimitive.jl", "Linearised operator", 4),
         ("test_galerkin.jl",                     "Field projection   ", 4),
         ]

cmd(file, nprocs) = addenv(`$(mpiexec()) -n $nprocs $(Base.julia_cmd()) --startup-file=no $(joinpath(@__DIR__, file))`)

@testset "$(rpad("$testname", 38))" for (f, testname, nprocs) in files
    p = run(ignorestatus(cmd(f, nprocs)))
    @test success(p)
end
