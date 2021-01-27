#####
##### Custom Threading
#####

"""
    threaded(f, A::AbstractArray, args...; [iterations = 1], [nthreads = Threads.nthreads()])

Split `A` into `nthreads` separate, sequential views.
For each view `v`, call `f(v, args...)` on a distinct thread.
This inner call is performed `iterations` times.

**Note**: It is required that `mod(length(A), nthreads) == 0` so that `A` can be evenly
broken into views.
"""
function threaded(f, A::AbstractArray, args...; iterations = 1, nthreads = Threads.nthreads())
    @assert iszero(mod(length(A), nthreads))

    step = div(length(A), nthreads)
    Threads.@threads for i in 1:nthreads
        start = step * (i-1) + 1
        stop = step * i
        x = view(A, start:stop)

        # Run the inner loop
        for j in 1:iterations
            f(x, args...)
        end
    end
    return nothing
end
