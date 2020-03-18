#####
##### Kernel for testing if the DRAM cache is inclusive.
#####

function cache_smasher(size_per_thread, f = (sz) -> Vector{Float32}(undef, sz))
    # Determine how big our array should be and instantiate it.
    sz = Threads.nthreads() * (2 ^ size_per_thread)
    A = f(sz)
    # Make sure the OS instantiates the whole array.
    threadme(sequential_write, A, Val{16}())

    # Now, make sure everything in the cache is clean.
    threadme(sequential_read, A, Val{16}())
    return _cache_smasher(A, f)
end

# Allow an entry point for if the array is already allocated.
function _cache_smasher(A, f)
    # Allocate another array that figs in the L3 cache (33 MB)
    #
    # Here, we make an array that is 4 MB.
    B = f(2^17)
    @assert eltype(B) == Float32

    # Pass in another vector to record timings.
    timings = Float64[]
    threadme(_smasher, A, B, timings)
    return timings
end

function _smasher(x, small, timings)
    # We want one thread to iterate over the small array.
    #
    # All other threads blow out the cache
    if Threads.threadid() == 1
        _cycle(small, timings)
    else
        # Sleep so the cycler can get some inital data
        sleep(5)
        for _ in 1:2
            sequential_read(x, Val{16}())
        end
    end
end

function _cycle(x, timings)
    # Time for 20 seconds.
    start = now()
    while now() < start + Second(60)
        t = @elapsed for _ in 1:1000
            sequential_read(x, Val{16}())
        end
        push!(timings, t)
    end
end
