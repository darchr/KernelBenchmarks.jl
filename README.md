# KernelBenchmarks

Micro benchmarks for testing the memory-subsystem of computers.

## TLDR

Add this package with
```julia
using Pkg
Pkg.add("https://github.com/hildebrandmw/MaxLFSR.jl")
Pkg.add("https://github.com/darchr/KernelBenchmarks.jl")
```

Next, run Julia.
I'm running on a multi-socket system with multiple NUMA domains, so I will be launching Julia under `numactl`.

```sh
# Export the total number of threads on a single socket
export JULIA_NUM_THREADS=24
numactl --physcpubind=0-23 --membind=0 <path/to/julia>
```
**NOTE**: In order for `numactl` to work correctly, you must be using Julia 1.4.0-rc2 or greater.
This is because Julia 1.3 has legacy code in it that nukes CPU affinity during initialization.

Now, suppose we want to measure the sequential read and write bandwidth of a system, comparing standard loads/stores with nontemporal loads/stores.
We can do that as follows:
```julia
juila> using KernelBenchmarks

julia> using Pkg; Pkg.add("BenchmarkTools"); using BenchmarkTools

julia> A = rand(Float32, 50000000 * Threads.nthreads());

# Number of test iterations to perform.
juila> iterations = 5

# Read bandwidth (GB/s) - standard loads
#
# Use a vector size of 16 for AVX-512 instructions
julia> iterations * sizeof(A) / (1E9 * @belapsed KernelBenchmarks.threaded(
        KernelBenchmarks.sequential_read,
        $A,
        Val{16}(),
        Val(false),
        Val(true);
        iterations = iterations
    ))
108.8788161265169

# Read bandwidth (GB/s) - nontemporal loads
#
# Use a vector size of 16 for AVX-512 instructions
julia> iterations * sizeof(A) / (1E9 * @belapsed KernelBenchmarks.threaded(
        KernelBenchmarks.sequential_read,
        $A,
        Val{16}(),
        Val(true),
        Val(true);
        iterations = iterations
    ))
108.89370284254693

# Write bandwidth (GB/s) - standard stores
julia> iterations * sizeof(A) / (1E9 * @belapsed KernelBenchmarks.threaded(
        KernelBenchmarks.sequential_write,
        $A,
        Val{16}(),
        Val(false),
        Val(true);
        iterations = iterations
    ))
49.17513953894055

# Write bandwidth (GB/s) - nontemporal stores
julia> iterations * sizeof(A) / (1E9 * @belapsed KernelBenchmarks.threaded(
        KernelBenchmarks.sequential_write,
        $A,
        Val{16}(),
        Val(false),
        Val(true);
        iterations = iterations
    ))
79.40457999265031
```

### Questions

#### Benchmarking Persistent Memory?

Look at <https://github.com/darchr/PersistentArrays.jl>.
Use that package to create the array `A` and you're good to go!

#### Using Hugepages

Look at <https://github.com/hildebrandmw/HugepageMmap.jl>.
Use that package to create the array `A`.

## Kernels

Below summarizes the test kernels in this repository.

### `sequential_read`
### `sequential_write`
### `sequential_readwrite`
### `random_read`
### `random_write`
### `random_readwrite`
