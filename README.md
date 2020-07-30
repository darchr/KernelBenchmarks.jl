KernelBenchmarks
================

Micro benchmarks for testing the memory-subsystem of computers.

Installation
------------

Add this package with
```julia
using Pkg
Pkg.add("https://github.com/hildebrandmw/MaxLFSR.jl")
Pkg.add("https://github.com/darchr/KernelBenchmarks.jl")
```

Starting Julia
--------------

If you want to use multiple threads, you will need to start Julia with the `JULIA_NUM_THREADS` environmental variable set.
This can easily be done with
```sh
export JULIA_NUM_THREADS=8
<path/to/julia>
```
or as a one-liner:
```sh
JULIA_NUM_THREADS=8 <path/to/julia>
```

I'm running on a multi-socket system with multiple NUMA domains, so I will be launching Julia under `numactl`.

```sh
# Export the total number of threads on a single socket
export JULIA_NUM_THREADS=24
numactl --physcpubind=0-23 --membind=0 <path/to/julia>
```
**NOTE**: In order for `numactl` to work correctly, you must be using Julia 1.4.0-rc2 or greater.
This is because Julia 1.3 has legacy code in it that nukes CPU affinity during initialization.

Usage
-------

#### Step 1 - Build yourself an Array
```julia
A = rand(Float32, 2^20 * Threads.nthreads());
```

#### Step 2 - Select your kernel parameters

```julia
KernelParam{K,L,S,I,T,N,U}
```
Top level type governing the creation of a microbenchmark kernel.
Construct using Keywords:

```julia
KernelParam(; [kw...])
```

The available keywords are:

* `kernel`: Controls the read/write behavior of the kernel. Options are:
    - `ReadOnly`: Only generate loads.
    - `WriteOnly`: Only generate stores.
    - `ReadWrite`: Generate a load followed by a store.
    Default: `ReadOnly`

* `loadtype`: Controls whether standard or nontemporal loads are emitted. Options are:
    - `Standard`: Use standard loads.
    - `Nontemporal`: Use nontemporal loads.
    Default: `Standard`

* `storetype`: Controls whether standard or nontemporal stores are emitter. Options are:
    - `Standard`: Use standard stores.
    - `Nontemporal`: Use nontemporal stores.
    Default: `Standard`

* `iterator`: Controls how the underlying array is accessed. Optiosn are:
    - `Sequential`: Touch every index sequentially.
    - `PseudoRandom`: Use a MaxLFSR to touch each index in a pseudo-random order.
    Default: `Sequential`

* `eltype`: The element type of the underlying array. Default: `Float32`.

* `vectorsize`: The size of the vector to use. This determines whether 128 bit, 256 bit,
    or 512 bit vector instructions are used. The default is `16` to enable AVX512
    instructions for `Float32` entries.

* `unroll`: The number of times the innermost load/store operation is performed.
    Especially useful for the pseudo-random benchmarks to get read/write granularity larger
    than 64 B. Default: `1`.

For example,
```julia
using KernelBenchmarks
K = KernelParam(
    kernel = ReadOnly,
    iterator = PseudoRandom,
    eltype = Float32,
    vectorsize = 8,
    unroll = 8
)
```

#### Step 3 - Run
Running kernels is as simple as calling `execute!`:
```julia
execute!(A::AbstractVector, K::KernelParam)
```
Run the kernel described by `K` on `A`.
The following restrictions apply to `K`:

* `eltype(A) == eltype(K)`
* `A` must be aligned to 64 bytes (i.e., aligned to a cache line)
* The lengthh of `A` must be a multiple of `unroll * vectorsize * Treads.nthreads()`

#### Step 4 - Multithread

If you want to parallelize the kernel across multiple threads, uses `KernelBenchmarks.threaded`
```
KernelBenchmarks.threaded(execute!, A, K)
```

Examples
--------
Suppose we want to measure the sequential read and write bandwidth of a system, comparing standard loads/stores with nontemporal loads/stores.
We can do that as follows:
```julia
juila> using KernelBenchmarks

julia> using Pkg; Pkg.add("BenchmarkTools"); using BenchmarkTools

julia> A = rand(Float32, 50000000 * Threads.nthreads());

# Number of test iterations to perform.
juila> iterations = 5

#####
##### Read Bandwidth using Standard Loads
#####

julia> K = KernelParam(;
    kernel = ReadOnly,
    loadtype = Standard,
    iterator = Sequential,
    eltype = eltype(A),
    vectorsize = 16,
    unroll = 4
)

# Get the elapsed runtime for the kernel
julia> runtime = @belapsed KernelBenchmarks.threaded(execute!, $A, K; iterations = iterations)
0.234733601

# Compute the read bandwidth in GB/s
julia> (iterations * sizeof(A)) / (1E9 * runtime)
102.24356418406413

#####
##### Read Bandwidth using Nontemporal Loads
#####

julia> K = KernelParam(;
    kernel = ReadOnly,
    loadtype = Nontemporal,
    iterator = Sequential,
    eltype = eltype(A),
    vectorsize = 16,
    unroll = 4
)

# Get the elapsed runtime for the kernel
julia> runtime = @belapsed KernelBenchmarks.threaded(execute!, $A, K; iterations = iterations)
0.235436181

# Compute the read bandwidth in GB/s
julia> (iterations * sizeof(A)) / (1E9 * runtime)
101.93845269686905

#####
##### Write Bandwidth using Standard Stores
#####

julia> K = KernelParam(;
    kernel = WriteOnly,
    storetype = Standard,
    iterator = Sequential,
    eltype = eltype(A),
    vectorsize = 16,
    unroll = 4
)

# Get the elapsed runtime for the kernel
julia> runtime = @belapsed KernelBenchmarks.threaded(execute!, $A, K; iterations = iterations)
0.537969783

# Compute the read bandwidth in GB/s
julia> (iterations * sizeof(A)) / (1E9 * runtime)
44.61217108917064

#####
##### Write Bandwidth using Nontemporal Stores
#####

julia> K = KernelParam(;
    kernel = WriteOnly,
    storetype = Nontemporal,
    iterator = Sequential,
    eltype = eltype(A),
    vectorsize = 16,
    unroll = 4
)

# Get the elapsed runtime for the kernel
julia> runtime = @belapsed KernelBenchmarks.threaded(execute!, $A, K; iterations = iterations)
0.537969783

# Compute the read bandwidth in GB/s
julia> (iterations * sizeof(A)) / (1E9 * runtime)
44.61217108917064
```

Inspecting Native Codegen
-------------------------

Since Julia code is Just-in-Time compiled and specialized on arguments, the kernels in this benchmark suite are really templates that allow for experimentation with array element type, vector width, instruction type etc.
One question we can then ask is how can we verify that the correct assembly instructions are being emitter?

KernelBenchmarks.jl provides an interactive tool `KernelBenchmarks.introspect()` that allows you to

* Construct the types of the innermost functions for each kernel.
* Inspect the generated assembly for each function.
* Obtain a command that you can run to reinspect the code without having to go through the interactive too.

Below is an example:

```julia
Choose a kernel to inspect:
   Read Only
   Write Only
 • Read + Write
Choose an Iterator:
 • Sequential
   Pseudo Random
Choose a Element Type:
 • Float32
   Float64
Choose a vector size
   4
   8
 • 16
Create Array View?
 • Yes
   No
Non-temporal Instructions?
 • Yes
   No
Unroll Kernel?
   1
   2
 • 4
   8
```
This generates the following output:
```
	.text
	mov	rcx, qword ptr [rsi]
	mov	rax, qword ptr [rsi + 8]
	sub	rax, rcx
	jl	L154
	mov	rdx, qword ptr [rdi]
	mov	rsi, qword ptr [rdi + 8]
	mov	rdx, qword ptr [rdx]
	shl	rsi, 2
	shl	rcx, 8
	add	rcx, rsi
	add	rcx, rdx
	add	rcx, -68
	inc	rax
	movabs	rdx, offset .rodata.cst4
	vbroadcastss	zmm0, dword ptr [rdx]
	nop
L64:
	vmovntdqa	zmm1, zmmword ptr [rcx - 192]
	vmovntdqa	zmm2, zmmword ptr [rcx - 128]
	vmovntdqa	zmm3, zmmword ptr [rcx - 64]
	vmovntdqa	zmm4, zmmword ptr [rcx]
	vaddps	zmm1, zmm1, zmm0
	vmovntps	zmmword ptr [rcx - 192], zmm1
	vaddps	zmm1, zmm2, zmm0
	vmovntps	zmmword ptr [rcx - 128], zmm1
	vaddps	zmm1, zmm3, zmm0
	vmovntps	zmmword ptr [rcx - 64], zmm1
	vaddps	zmm1, zmm4, zmm0
	vmovntps	zmmword ptr [rcx], zmm1
	add	rcx, 256
	dec	rax
	jne	L64
L154:
	vzeroupper
	ret
	nop


Run this to recreate
    code_native(
        KernelBenchmarks._execute!,
        (
            Base.ReinterpretArray{KernelBenchmarks.SIMD.Vec{16,Float32},1,Float32,SubArray{Float32,1,Array{Float32,1},Tuple{UnitRange{Int64}},true}},
            UnitRange{Int64},
            KernelParam{ReadWrite,Nontemporal,Nontemporal,Sequential,Float32,16,4}
        );
        syntax=:intel,
        debuginfo=:none
    )
```
The assembly code shown here is the main inner loop, which uses non-temporal AvX-512 instructions as expected!
We also notice that the overhead of introducing a `SubArray` is zero once we enter the main inner loop.
This is important because when multi-threading is performed, we really for `SubArrays` of the main array.
But as we see here, we're free to do so without affecting the important code!

Questions
---------

#### Using Hugepages

Look at <https://github.com/hildebrandmw/HugepageMmap.jl>.
Use that package to create the array `A`.

