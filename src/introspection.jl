supports_unrolling(f) = in(f, (random_read, random_write, random_readwrite))

# Introspection for kernel benchmarks to see native code.
function introspect(A = nothing)
    # Kernel Selection
    options = [
        "sequential_read"       => sequential_read,
        "sequential_write"      => sequential_write,
        "sequential_readwrite"  => sequential_readwrite,
        "random_read"           => random_read,
        "random_write"          => random_write,
        "random_readwrite"      => random_readwrite,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Choose a kernel to inspect: ", menu)
    checkexit(choice) && return nothing

    # Pull out the function
    fn = last(options[choice])

    # Now, ask which element type they's like to use and create
    if isnothing(A)
        options = [
            "Float32" => Float32,
            "Float64" => Float64,
        ]

        menu = RadioMenu(first.(options); pagesize = 10)
        choice = request("Choose a Element Type: ", menu)
        checkexit(choice) && return nothing
        T = last(options[choice])
        A = Vector{T}(undef, 1024)
    end

    ### Choose a vector size
    options = [
        "4" => Val{4}(),
        "8" => Val{8}(),
    ]

    # If we have a 4-byte datatype, we can have a vector width of 16 as well.
    if sizeof(eltype(A)) < 8
        push!(options, "16" => Val{16}())
    end

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Choose a vector size ", menu)
    checkexit(choice) && return nothing
    vectorsize = last(options[choice])

    ### Now, we ask some options for non-temporal and aligned
    options = [
        "Yes" => true,
        "No" => false,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Create Array View? ", menu)
    checkexit(choice) && return nothing
    makeview = last(options[choice])
    if makeview
        # Create a SubArray of the array.
        # What's really important is that we get the correct `SubArray` type.
        A = view(A, 1:div(length(A),2))
    end

    ### Now, we ask some options for non-temporal and aligned
    options = [
        "Yes" => Val(true),
        "No" => Val(false),
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Non-temporal Instructions? ", menu)
    checkexit(choice) && return nothing
    nontemporal = last(options[choice])

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Aligned Instructions? ", menu)
    checkexit(choice) && return nothing
    aligned = last(options[choice])

    # If this function supports unrolling, make this an option.
    extra_args = []
    if supports_unrolling(fn)
        options = [
            "1" => Val(1),
            "2" => Val(2),
            "4" => Val(4),
            "8" => Val(8),
        ]

        menu = RadioMenu(first.(options); pagesize = 10)
        choice = request("Unroll Kernel? ", menu)
        checkexit(choice) && return nothing
        push!(extra_args, last(options[choice]))
    end

    # Now that we have everything ready to go, we create the appropriate calls to the
    # innermost kernels.
    generate_code_native_call(fn, A, vectorsize, nontemporal, aligned, extra_args...)
end

checkexit(x) = (x == -1)

unpack(::Val{N}) where {N} = N
function generate_code_native_call(f, A, vectorsize, nontemporal, aligned, extra_args...)
    # No reason to specialize on these arguments
    @nospecialize f A vectorsize nontemporal aligned

    # If we're doing the sequential benchmarks, we can just basically forward the call
    # directly
    if in(f, (sequential_read, sequential_write, sequential_readwrite))
        types = typeof.((A, vectorsize, nontemporal, aligned))
        return native_call(f, (A, vectorsize, nontemporal, aligned, extra_args...))
    end

    # If `f` is one of the random benchmarks, we have to create an LFSR in order to
    # get to the innermost function call
    if in(f, (random_read, random_write, random_readwrite))
        # Wrap up A in a reinterpreted view and create the LFSR
        V = reinterpret(Vec{unpack(vectorsize),eltype(A)}, A)
        lfsr = MaxLFSR.LFSR(length(V))
        return native_call(f, (V, lfsr, nontemporal, aligned, extra_args...))
    end
end

function native_call(f, args)
    @nospecialize f args
    types = typeof.(args)
    @show types
    call = """
        code_native(KernelBenchmarks.$f, $types; syntax=:intel, debuginfo=:none)
    """
    code_native(f, types; syntax=:intel, debuginfo=:none)
    printstyled(stdout, "\n\nRun this to recreate\n"; color = :green)
    println(call)
    return nothing
end
