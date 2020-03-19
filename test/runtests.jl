using KernelBenchmarks
using Test

# Required for Codegen Test
using InteractiveUtils
using SIMD
using MaxLFSR

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

@testset "Testing Codegen" begin
    # Here, we just emit a lot of the codegen to catch if anything goes out of date.

    types = (SubArray{Float32,1,Array{Float32,1},Tuple{UnitRange{Int64}},true}, Val{8}, Val{true}, Val{true})
    kw = (syntax = :intel, debuginfo = :none)
    code_native(devnull, KernelBenchmarks.sequential_read, types; kw...)
    code_native(devnull, KernelBenchmarks.sequential_write, types; kw...)
    code_native(devnull, KernelBenchmarks.sequential_readwrite, types; kw...)

    types = (Base.ReinterpretArray{Vec{8,Float64},1,Float64,SubArray{Float64,1,Array{Float64,1},Tuple{UnitRange{Int64}},true}}, LFSR, Val{true}, Val{true})
    code_native(devnull, KernelBenchmarks.random_read, types; kw...)
    code_native(devnull, KernelBenchmarks.random_write, types; kw...)
    code_native(devnull, KernelBenchmarks.random_readwrite, types; kw...)
end
