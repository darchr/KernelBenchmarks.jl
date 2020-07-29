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

