using KernelBenchmarks
using Test

using Cassette
using SIMD
import ProgressMeter

# Setup a Cassette Context that records `vload` and `vstore` options.
Cassette.@context TraceCtx

function Cassette.overdub(ctx::TraceCtx, ::typeof(SIMD.vload), x...)
    push!(ctx.metadata.loads, x)
    return SIMD.vload(x...)
end

function Cassette.overdub(ctx::TraceCtx, ::typeof(SIMD.vstore), x...)
    push!(ctx.metadata.stores, x)
    return SIMD.vstore(x...)
end

struct TraceMeta
    loads::Vector{Any}
    stores::Vector{Any}
end
TraceMeta() = TraceMeta(Any[], Any[])

function pointercheck(pointers, vectype, unroll, iter, A)
    # Are all pointers unique?
    @test allunique(pointers)

    # Are they grouped by the unroll amount?
    for group in Iterators.partition(pointers, unroll)
        @test all(isequal(sizeof(vectype)), diff(group))
    end

    # If the iterator is Sequential, then the pointers should be sorted.
    # Otherwise, they shouldn't be sorted.
    if iter == KernelBenchmarks.Sequential
        @test issorted(pointers)
    elseif iter == KernelBenchmarks.PseudoRandom
        @test !issorted(pointers)
    # Throw an error to try to keep this test from breaking.
    else
        error()
    end

    # Generate all of the addresses we'd expect to write.
    base = convert(Int, pointer(A))
    len = div(sizeof(A), sizeof(vectype))
    expected = [base + sizeof(vectype) * (i-1) for i in 1:len]
    @test sort(pointers) == expected
end

#####
##### Test Set
#####

@testset "KernelBenchmarks.jl" begin
    # Test over a very large range of parameters.
    # I'm sorry Julia compiler :(

    # Allocate a decent size array to serve as our test bed.
    A = rand(Float32, 2^11)
    B = rand(Float64, 2^10)

    # Setup iteration space
    kernels = [ReadOnly, WriteOnly, ReadWrite]
    loadtypes = [Standard, Nontemporal]
    storetypes = [Standard, Nontemporal]
    iterators = [Sequential, PseudoRandom]
    eltypes = [Float32, Float64]
    widths = [1,2,4,8,16]
    unroll = [1,2,4,8]

    iter = Iterators.product(
        kernels,
        loadtypes,
        storetypes,
        iterators,
        eltypes,
        widths,
        unroll
    )

    ProgressMeter.@showprogress 1 "Testing ... " for tup in iter
        # unpack tuple
        kernel      = tup[1]
        loadtype    = tup[2]
        storetype   = tup[3]
        iterator    = tup[4]
        eltype      = tup[5]
        vectorsize  = tup[6]
        unroll      = tup[7]

        # Invalid combination
        eltype == Float64 && unroll == 16 && continue

        kp = KernelBenchmarks.KernelParam(
            kernel = kernel,
            loadtype = loadtype,
            storetype = storetype,
            iterator = iterator,
            eltype = eltype,
            vectorsize = vectorsize,
            unroll = unroll,
        )

        V = eltype == Float32 ? A : B

        # Get a record of all the stores
        ctx = TraceCtx(metadata = TraceMeta())
        Cassette.overdub(ctx, KernelBenchmarks.execute!, V, kp)
        trace = ctx.metadata

        ### Setup
        vectype = Vec{vectorsize,eltype}
        loads = trace.loads
        stores = trace.stores

        if kernel == ReadOnly
            @test isempty(stores)
        elseif kernel == WriteOnly
            @test isempty(loads)
        elseif kernel == ReadWrite
            @test length(stores) == length(loads)
        end

        # Check load type
        if in(kernel, (ReadOnly, ReadWrite))
            # Are all the loads the correct vector type
            @test all(isequal(vectype), getindex.(loads, 1))

            # All loads should be aligned
            @test all(isequal(Val(true)), getindex.(loads, 3))

            # Test for standard/nontemporal stores
            if loadtype == KernelBenchmarks.Standard
                @test all(isequal(Val(false)), getindex.(loads, 4))
            else
                @test all(isequal(Val(true)), getindex.(loads, 4))
            end

            # Pointer Check
            pointers = convert.(Int, getindex.(loads, 2))
            pointercheck(pointers, vectype, unroll, iterator, V)
        end

        if in(kernel, (WriteOnly, ReadWrite))
            # Are all the stores the correct vector type
            @test all(isequal(vectype), typeof.(getindex.(stores, 1)))

            # All stores should be aligned
            @test all(isequal(Val(true)), getindex.(stores, 3))

            # Test for standard/nontemporal stores
            if storetype == KernelBenchmarks.Standard
                @test all(isequal(Val(false)), getindex.(stores, 4))
            else
                @test all(isequal(Val(true)), getindex.(stores, 4))
            end

            # Pointer Check
            pointers = convert.(Int, getindex.(stores, 2))
            pointercheck(pointers, vectype, unroll, iterator, V)
        end
    end
end

@testset "Testing Coverage" begin
    for unroll in (1,2,4,8)
        A = ones(Float32, 2^10)
        KernelBenchmarks.random_write(A, Val(8), Val(false), Val(true), Val(unroll))
        @test all(isone, A)
    end

    for unroll in (1,2,4,8)
        A = zeros(Float32, 2^10)
        KernelBenchmarks.random_readwrite(A, Val(8), Val(false), Val(true), Val(unroll))
        @test all(isone, A)
    end
end

