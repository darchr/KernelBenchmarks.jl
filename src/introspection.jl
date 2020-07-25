# Introspection for kernel benchmarks to see native code.
function introspect(A = nothing)
    ### Kernel
    options = [
        "Read Only"    => ReadOnly,
        "Write Only"   => WriteOnly,
        "Read + Write" => ReadWrite,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Choose a kernel to inspect: ", menu)
    checkexit(choice) && return nothing

    # Pull out the function
    kernel = last(options[choice])

    ### Iterator
    options = [
        "Sequential" => Sequential,
        "Pseudo Random" => PseudoRandom,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Choose an Iterator: ", menu)
    checkexit(choice) && return nothing

    # Pull out the function
    iterator = last(options[choice])

    ### Element Type
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
        "4" => 4,
        "8" => 8,
    ]

    # If we have a 4-byte datatype, we can have a vector width of 16 as well.
    if sizeof(eltype(A)) < 8
        push!(options, "16" => 16)
    end

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Choose a vector size ", menu)
    checkexit(choice) && return nothing
    vectorsize = last(options[choice])

    ### Create an array view?
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
        "Yes" => Nontemporal,
        "No" => Standard,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Non-temporal Instructions? ", menu)
    checkexit(choice) && return nothing
    nontemporal = last(options[choice])

    # If this function supports unrolling, make this an option.
    options = [
        "1" => 1,
        "2" => 2,
        "4" => 4,
        "8" => 8,
    ]

    menu = RadioMenu(first.(options); pagesize = 10)
    choice = request("Unroll Kernel? ", menu)
    checkexit(choice) && return nothing
    unroll = last(options[choice])

    # Generate the "KernelParam" type
    params = KernelParam(
        kernel = kernel,
        loadtype = nontemporal,
        storetype = nontemporal,
        iterator = iterator,
        eltype = eltype(A),
        vectorsize = vectorsize,
        unroll = unroll
    )

    # Now that we have everything ready to go, we create the appropriate calls to the
    # innermost kernels.
    generate_code_native_call(A, params)
end

checkexit(x) = (x == -1)

function generate_code_native_call(A, K::KernelParam)
    # No reason to specialize on these arguments
    @nospecialize

    V = reinterpret(K, A)
    itr = makeitr(V, K)

    return native_call(_execute!, (V, itr, K))
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
