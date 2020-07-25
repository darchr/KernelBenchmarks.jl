# Kernel Type
abstract type AbstractKernelType end
struct ReadOnly <: AbstractKernelType end
struct WriteOnly <: AbstractKernelType end
struct ReadWrite <: AbstractKernelType end

# LoadStoreType
abstract type AbstractLoadStoreType end
struct Standard <: AbstractLoadStoreType end
struct Nontemporal <: AbstractLoadStoreType end

# Iteration Style
abstract type AbstractIterationStyle end
struct Sequential <: AbstractIterationStyle end
struct PseudoRandom <: AbstractIterationStyle end

struct KernelParam{
        # Kernel Type
        K <: AbstractKernelType,
        # Load Type
        L <: AbstractLoadStoreType,
        # Store Type
        S <: AbstractLoadStoreType,
        # Iteration Style
        I <: AbstractIterationStyle,
        # Vector element Type
        T,
        # Vector Width (bytes)
        N,
        # Unroll factor
        U
    }

    function KernelParam(;
            kernel = ReadOnly,
            loadtype = Standard,
            storetype = Standard,
            iterator = Sequential,
            eltype = Float32,
            vectorsize = 8,
            unroll = 1
        )

        return new{kernel,loadtype,storetype,iterator,eltype,vectorsize,unroll}()
    end
end

# Convenience accessors
kernel(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = K
kernel(K::KernelParam) = kernel(typeof(K))

loadtype(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = L
loadtype(K::KernelParam) = loadtype(typeof(K))

storetype(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = S
storetype(K::KernelParam) = storetype(typeof(K))

iterator(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = I
iterator(K::KernelParam) = iterator(typeof(K))

Base.eltype(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = T
Base.eltype(K::KernelParam) = eltype(typeof(K))

vectorsize(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = N
vectorsize(K::KernelParam) = vectorsize(typeof(K))

unroll(::Type{KernelParam{K,L,S,I,T,N,U}}) where {K,L,S,I,T,N,U} = U
unroll(K::KernelParam) = unroll(typeof(K))

#####
##### Upper level handling
#####

# The upper level functions reinterpret the top level array as an array of `Vec` elements.
# Performs alignment checking and length checking.
# Constructs the iterator
function execute!(A::AbstractVector, K::KernelParam)
    # Sanity check
    @assert eltype(A) == eltype(K)
    # Alignment check
    @assert iszero(mod(convert(Int, pointer(A)), 64))
    # Size check
    @assert iszero(mod(length(A), unroll(K) * sizeof(eltype(K)) * vectorsize(K)))

    V = reinterpret(K, A)
    itr = makeitr(V, K)

    # Inner high-performance function.
    return _execute!(V, itr, K)
end

Base.reinterpret(K::KernelParam, A::AbstractArray) = reinterpret(Vec{vectorsize(K),eltype(K)}, A)

# iterators
makeitr(A, K::KernelParam) = makeitr(A, iterator(K), unroll(K))

makeitr(A, ::Type{Sequential}, unroll) = 1:div(length(A), unroll)
makeitr(A, ::Type{PseudoRandom}, unroll) = MaxLFSR.LFSR(div(length(A), unroll))

#####
##### Generate the kernel
#####

# Note: This is not exactly an extendable API since we're relying on `@generated` functions.
# In particular, in order for any changes made to the `hbf` function or any functions
# called within to take effect, we need to restart Julia.
#
# This is a limitation of the dependency tracking in Julia.
@generated function _execute!(A::AbstractArray{Vec{N,T}}, itr, K::KernelParam) where {N,T}
    return emit(A, K)
end

function emit(A::Type{<:AbstractArray{Vec{N,T}}}, K::Type{<:KernelParam}) where {N,T}
    @nospecialize

    # Get the header, body, and footer for this kernel
    header, body, footer = hbf(A, K)

    # Emit the rest of the function
    return quote
        $(header...)
        base = Base.unsafe_convert(Ptr{$T}, pointer(A))
        @inbounds for i in itr
            ptr = base + $(unroll(K) * sizeof(Vec{N,T})) * (i-1)
            $(body...)
        end
        $(footer...)
    end
end

# header, body, footer
function hbf(::Type{<:AbstractArray{Vec{N,T}}}, K::Type{<:KernelParam}) where {N,T}
    @nospecialize

    # Check invariants
    @assert N == vectorsize(K)
    @assert T == eltype(K)

    lt = loadtype(K)
    st = storetype(K)

    # defaults
    header = Any[]
    footer = Any[:(return nothing)]

    # Check if we need to emit loads/stores
    if kernel(K) == ReadOnly
        # Header
        header = push!(header, :(s = zero(eltype(A))))

        # Body
        body = emit_loads(Vec{N,T}, unroll(K), loadtype(K))
        reduction = unroll(K) == 1 ? :(s += $(symbols(1)[1])) : :(s += +($(symbols(unroll(K))...)))
        push!(body, reduction)

        # footer
        footer = Any[:(return s)]
    elseif kernel(K) == WriteOnly
        # body
        body = emit_stores(Vec{N,T}, unroll(K), storetype(K), false)
    elseif kernel(K) == ReadWrite
        # body
        loads = emit_loads(Vec{N,T}, unroll(K), loadtype(K))
        stores = emit_stores(Vec{N,T}, unroll(K), storetype(K), true)
        body = vcat(loads, stores)
    else
        error("Unknown Kernel: $(kernel(K))")
    end

    return header, body, footer
end

symbols(N) = [Symbol("v$i") for i in 1:N]
lower(::Type{Nontemporal}) = Val{true}()
lower(::Type{Standard}) = Val{false}()

function emit_loads(vectype::Type{<:Vec}, unroll, loadtype)
    @nospecialize

    syms = symbols(unroll)
    return map(1:unroll) do i
        lhs = syms[i]
        shift = sizeof(vectype) * (i-1)
        return :($lhs = vload($vectype, ptr + $shift, Val{true}(), $(lower(loadtype))))
    end
end

function emit_stores(vectype::Type{<:Vec}, unroll, storetype, follows_load::Bool)
    @nospecialize

    syms = symbols(unroll)
    o = one(vectype)
    return map(1:unroll) do i
        rhs = syms[i]
        shift = sizeof(vectype) * (i-1)
        vec = follows_load ? :($rhs + $o) : o
        return :(vstore($vec, ptr + $shift, Val{true}(), $(lower(storetype))))
    end
end

############################################################################################
# Legacy
############################################################################################

#####
##### Sequential Access Kernels
#####

# Conver things to 'Vals' that aren't already Vals
val(x::Val) = x
val(x) = Val(x)

"""
    sequential_read(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a sequential read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function sequential_read(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val(false),
        aligned = Val(true),
    ) where {T, N}

    kp = KernelParam(;
        kernel = ReadOnly,
        loadtype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = Sequential,
        vectorsize = N,
        unroll = 4
    )
    return execute!(A, kp)
end

"""
    sequential_write(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a sequential read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function sequential_write(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val(false),
        aligned = Val(true),
    ) where {T, N}

    kp = KernelParam(;
        kernel = WriteOnly,
        storetype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = Sequential,
        vectorsize = N,
        unroll = 4
    )
    return execute!(A, kp)
end

"""
    sequential_readwrite(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a sequential read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function sequential_readwrite(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val(false),
        aligned = Val(true),
    ) where {T, N}

    kp = KernelParam(;
        kernel = ReadWrite,
        loadtype = nontemporal == Val(true) ? Nontemporal : Standard,
        storetype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = Sequential,
        vectorsize = N,
        unroll = 4,
    )
    return execute!(A, kp)
end

#####
##### Random Access Kernels
#####

"""
    random_read(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a random read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function random_read(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val(false),
        aligned = Val(true),
        unroll::Val{U} = Val(1),
    ) where {T,N,U}

    kp = KernelParam(;
        kernel = ReadOnly,
        loadtype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = PseudoRandom,
        vectorsize = N,
        unroll = U,
    )
    return execute!(A, kp)
end

"""
    random_write(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a random read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function random_write(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val{false}(),
        aligned = Val(true),
        unroll::Val{U} = Val(1),
    ) where {T,N,U}

    kp = KernelParam(;
        kernel = WriteOnly,
        loadtype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = PseudoRandom,
        vectorsize = N,
        unroll = U,
    )
    return execute!(A, kp)
end

"""
    random_readwrite(A::AbstractArray, ::Val{N}, [nontemporal = Val(false), aligned = Val(true)]

Perform a random read on the contents of `A` using vectors intrinsics of size `N`.

Optional arguments `nontemporal` and `aligned` should be passes as `Val{True}()` or
`Val{False}()` and control if the vector instructions are nontemporal and aligned,
respectively.
"""
function random_readwrite(
        A::AbstractArray{T},
        ::Val{N},
        nontemporal = Val(false),
        aligned = Val(true),
        unroll::Val{U} = Val(1),
    ) where {T,N,U}

    kp = KernelParam(;
        kernel = ReadWrite,
        loadtype = nontemporal == Val(true) ? Nontemporal : Standard,
        eltype = T,
        iterator = PseudoRandom,
        vectorsize = N,
        unroll = U,
    )
    return execute!(A, kp)
end

