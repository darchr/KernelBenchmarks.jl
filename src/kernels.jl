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

    aligned = val(aligned)
    nontemporal = val(nontemporal)

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    s = Vec{N,T}(zero(T))
    @inbounds for i in 1:(unroll*N):length(A)
        _v1 = vload(Vec{N,T}, pointer(A, i),         aligned, nontemporal)
        _v2 = vload(Vec{N,T}, pointer(A, i + N),     aligned, nontemporal)
        _v3 = vload(Vec{N,T}, pointer(A, i + 2 * N), aligned, nontemporal)
        _v4 = vload(Vec{N,T}, pointer(A, i + 3 * N), aligned, nontemporal)

        s += _v1 + _v2 + _v3 + _v4
    end
    return sum(s)
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

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    # We need to do some schenanigans to get LLVM to emit the correct code.
    # Just doing something like
    #    vstore(s, pointer(A, i),         aligned, nontemporal)
    #    vstore(s, pointer(A, i + N),     aligned, nontemporal)
    #    vstore(s, pointer(A, i + 2 * N), aligned, nontemporal)
    #    vstore(s, pointer(A, i + 3 * N), aligned, nontemporal)
    #
    # Results in spurious `mov` instructions between the vector stores for pointer
    # conversions, even though these are really not needed.
    #
    # Instead, we perform the pointer arithmetic manually.
    s = Vec{N,T}(one(T))
    base = pointer(A)
    @inbounds for i in 0:(unroll*N):(length(A) - 1)
        vstore(s, base + sizeof(T) * i,           aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + N),     aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        vstore(s, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)
    end
    return nothing
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

    unroll = 4

    # Make sure we can successfully chunk up this array into SIMD sizes
    @assert iszero(mod(length(A), unroll * N))

    # If we've passed alignment flags, make sure the base pointer of this array is in fact
    # aligned correctly.
    if aligned == Val{true}()
        @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    end

    s = Vec{N,T}(one(T))
    base = pointer(A)
    @inbounds for i in 0:(unroll*N):(length(A) - 1)
        _v1 = vload(Vec{N,T}, base + sizeof(T) * i,           aligned, nontemporal)
        _v2 = vload(Vec{N,T}, base + sizeof(T) * (i + N),     aligned, nontemporal)
        _v3 = vload(Vec{N,T}, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        _v4 = vload(Vec{N,T}, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)

        _u1 = _v1 + s
        _u2 = _v2 + s
        _u3 = _v3 + s
        _u4 = _v4 + s

        vstore(_u1, base + sizeof(T) * i,           aligned, nontemporal)
        vstore(_u2, base + sizeof(T) * (i + N),     aligned, nontemporal)
        vstore(_u3, base + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        vstore(_u4, base + sizeof(T) * (i + (3*N)), aligned, nontemporal)
    end
    return nothing
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
        unroll = Val(1),
    ) where {T,N}

    # Wrap the array in "Vec" elements.
    # This really becomes a no-op, but is helpful for not having to deal with it in the
    # inner kernel.
    return random_read(reinterpret(Vec{N,T}, A), nontemporal, aligned, unroll)
end

function random_read(
        A::AbstractArray{T},
        nontemporal::Val{N} = Val(false),
        aligned = Val(true),
        u::Val{unroll} = Val(1),
    ) where {T <: Vec, N, unroll}

    # Make sure we can the length of the array is a multiple of the unroll factor
    if !iszero(mod(length(A), unroll))
        error("Length of `A` must be a multiple of the unroll factor $unroll")
    end

    # Construct a LFSR
    lfsr = MaxLFSR.LFSR(div(length(A), unroll))
    return random_read(A, lfsr, nontemporal, aligned, u)
end

@generated function random_read(
        A::AbstractArray{Vec{N,T}},
        lfsr::LFSR,
        nontemporal = Val(false),
        aligned = Val(true),
        ::Val{unroll} = Val(1),
    ) where {N,T,unroll}

    syms = [Symbol("v$i") for i in 1:unroll]

    loads = map(1:unroll) do i
        lhs = syms[i]
        return :($lhs = vload(Vec{$N,$T}, ptr + $(sizeof(Vec{N,T}) * (i-1)), aligned, nontemporal))
    end

    # Need to manually check if length is 1 for better codegen.
    reduction = length(syms) == 1 ? :(s += $(syms[1])) : :(s += +($(syms...)))

    return quote
        s = zero(eltype(A))
        base = Base.unsafe_convert(Ptr{T}, pointer(A))
        @inbounds for i in lfsr
            ptr = base + $(unroll * sizeof(eltype(A))) * (i-1)
            $(loads...)
            $reduction
        end
        return s
    end
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
        unroll = Val(1),
    ) where {T,N}

    return random_write(reinterpret(Vec{N,T}, A), nontemporal, aligned, unroll)
end

function random_write(
        A::AbstractArray{T},
        nontemporal::Val{N} = Val{false}(),
        aligned = Val(true),
        u::Val{unroll} = Val(1),
    ) where {T <: Vec, N, unroll}

    # Make sure we can the length of the array is a multiple of the unroll factor
    if !iszero(mod(length(A), unroll))
        error("Length of `A` must be a multiple of the unroll factor $unroll")
    end

    # Construct a LFSR
    lfsr = MaxLFSR.LFSR(div(length(A), unroll))
    return random_write(A, lfsr, nontemporal, aligned, u)
end

@generated function random_write(
        A::AbstractArray{Vec{N,T}},
        lfsr::LFSR,
        nontemporal = Val(false),
        aligned = Val(true),
        ::Val{unroll} = Val(1),
    ) where {N,T,unroll}

    stores = map(1:unroll) do i
        :(vstore(s, ptr + $(sizeof(Vec{N,T}) * (i-1)), aligned, nontemporal))
    end

    return quote
        s = one(eltype(A))
        base = Base.unsafe_convert(Ptr{T}, pointer(A))
        @inbounds for i in lfsr
            ptr = base + $(unroll * sizeof(Vec{N,T})) * (i-1)
            $(stores...)
        end
        return nothing
    end
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
        unroll = Val(1),
    ) where {T,N}

    return random_readwrite(reinterpret(Vec{N,T}, A), nontemporal, aligned, unroll)
end

function random_readwrite(
        A::AbstractArray{T},
        nontemporal::Val{N} = Val(false),
        aligned = Val(true),
        u::Val{unroll} = Val(1),
    ) where {T <: Vec, N, unroll}

    # Make sure we can the length of the array is a multiple of the unroll factor
    if !iszero(mod(length(A), unroll))
        error("Length of `A` must be a multiple of the unroll factor $unroll")
    end

    # Construct a LFSR
    lfsr = MaxLFSR.LFSR(div(length(A), unroll))
    return random_readwrite(A, lfsr, nontemporal, aligned, u)
end

@generated function random_readwrite(
        A::AbstractArray{Vec{N,T}},
        lfsr::LFSR,
        nontemporal = Val(false),
        aligned = Val(true),
        ::Val{unroll} = Val(1),
    ) where {N,T,unroll}

    o = one(Vec{N,T})
    syms = [Symbol("v$i") for i in 1:unroll]
    loads = map(1:unroll) do i
        lhs = syms[i]
        shift = sizeof(Vec{N,T}) * (i-1)
        return :($lhs = vload(Vec{$N,$T}, ptr + $shift, aligned, nontemporal))
    end

    stores = map(1:unroll) do i
        v = syms[i]
        shift = sizeof(Vec{N,T}) * (i-1)
        return :(vstore($v+$o, ptr + $shift, aligned, nontemporal))
    end

    return quote
        base = Base.unsafe_convert(Ptr{T}, pointer(A))
        @inbounds for i in lfsr
            ptr = base + $(unroll * sizeof(Vec{N,T})) * (i-1)
            $(loads...)
            $(stores...)
        end
    end
end

#####
##### Streamed copy from A to B
#####

function sequential_copy!(
        A::AbstractArray{T},
        B::AbstractArray{T},
        valn::Val{N}
    ) where {T, N}

    unroll = 4

    # Alignment and size checking
    @assert iszero(mod(length(A), unroll * N))
    @assert iszero(mod(Int(pointer(A)), sizeof(T) * N))
    @assert iszero(mod(Int(pointer(B)), sizeof(T) * N))

    # Forward to the one without bounds checking so we can checkout out the generated
    # code more easily.
    return unsafe_sequential_copy!(A, B, valn)
end

function unsafe_sequential_copy!(
        A::AbstractArray{T},
        B::AbstractArray{T},
        ::Val{N}
    ) where {N,T}

    # These are streaming stores after all.
    unroll = 4
    aligned = Val{true}()
    nontemporal = Val{true}()

    # Again, this pointer arithmetic thing seems to be necessary to get the best native code.
    pa = pointer(A)
    pb = pointer(B)
    @inbounds for i in 0:(unroll*N):(length(A) - 1)
        _v1 = vload(Vec{N,T}, pb + sizeof(T) * i,           aligned, nontemporal)
        _v2 = vload(Vec{N,T}, pb + sizeof(T) * (i + N),     aligned, nontemporal)
        _v3 = vload(Vec{N,T}, pb + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        _v4 = vload(Vec{N,T}, pb + sizeof(T) * (i + (3*N)), aligned, nontemporal)

        vstore(_v1, pa + sizeof(T) * i,           aligned, nontemporal)
        vstore(_v2, pa + sizeof(T) * (i + N),     aligned, nontemporal)
        vstore(_v3, pa + sizeof(T) * (i + (2*N)), aligned, nontemporal)
        vstore(_v4, pa + sizeof(T) * (i + (3*N)), aligned, nontemporal)
    end
end
