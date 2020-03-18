using KernelBenchmarks
using Test

@testset "KernelBenchmarks.jl" begin
    # Write your own tests here.

    A = rand(Float32, 100000 * Threads.nthreads())

    iterations = 10

    # Standard loads/stores
    args = (A, Val{8}(), Val{false}())
    kw = (iterations = iterations,)

    println("Sequential Read")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_read, args...; kw...)
    println("Sequential Write")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_write, args...; kw...)
    println("Sequential ReadWrite")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_readwrite, args...; kw...)

    println("Random Read")
    KernelBenchmarks.threaded(KernelBenchmarks.random_read, args...; kw...)
    println("Random Write")
    KernelBenchmarks.threaded(KernelBenchmarks.random_write, args...; kw...)
    println("Random ReadWrite")
    KernelBenchmarks.threaded(KernelBenchmarks.random_readwrite, args...; kw...)

    # Nontemporal loads/stores
    args = (A, Val{8}(), Val{true}())
    kw = (iterations = iterations,)

    println("Sequential Read")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_read, args...; kw...)
    println("Sequential Write")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_write, args...; kw...)
    println("Sequential ReadWrite")
    KernelBenchmarks.threaded(KernelBenchmarks.sequential_readwrite, args...; kw...)

    println("Random Read")
    KernelBenchmarks.threaded(KernelBenchmarks.random_read, args...; kw...)
    println("Random Write")
    KernelBenchmarks.threaded(KernelBenchmarks.random_write, args...; kw...)
    println("Random ReadWrite")
    KernelBenchmarks.threaded(KernelBenchmarks.random_readwrite, args...; kw...)
end
