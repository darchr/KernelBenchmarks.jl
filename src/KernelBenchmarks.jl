module KernelBenchmarks

export  KernelParam,
        # kernel types
        ReadOnly,
        WriteOnly,
        ReadWrite,
        # load/store types
        Standard,
        Nontemporal,
        # itertors
        Sequential,
        PseudoRandom,
        # exported function
        execute!,
        # interactive
        introspect

##### stdlib
using InteractiveUtils
import REPL
using REPL.TerminalMenus

##### "Internal" Packages
# These are packages I've developed for testing purposes and not part of the
# greater Julia ecosystem.
using MaxLFSR

##### External Packages
using DataFrames

# For vector load intrinsics
using SIMD

include("threaded.jl")
include("kernels.jl")
include("introspection.jl")
include("deprecated.jl")

function run(A)
    df = DataFrame()
    num_threads = [1, 4, 8, 10, 16, 20]
    kernels = [ReadOnly, WriteOnly]
    unroll = 8
    loops = [1, 2, 4, 8]

    iter = Iterators.product(num_threads, loops, kernels)
    for tup in iter
        nthreads = tup[1]
        nloops = tup[2]
        kernel = tup[3]
        num_iterations = kernel == ReadOnly ? 3 : 1

        params = KernelParam(;
            kernel = kernel,
            loadtype = Standard,
            storetype = Nontemporal,
            iterator = PseudoRandom,
            vectorsize = 16,
            unroll = unroll,
            loops = nloops,
        )

        # Warmup iteration.
        threaded(execute!, A, params)

        # Actual run.
        rt = @elapsed threaded(
            execute!,
            A,
            params;
            iterations = num_iterations,
            nthreads = nthreads,
        )

        bw = (num_iterations * sizeof(A)) / (rt)

        stats = Dict(
            :num_threads => nthreads,
            :num_bytes => 64 * unroll * nloops,
            :kernel => string(kernel),
            :bandwidth => bw,
        )
        display(stats),

        push!(df, stats; cols = :union)
    end
    return df
end

end # module
