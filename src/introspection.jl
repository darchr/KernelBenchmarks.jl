# Introspection for kernel benchmarks to see native code.
function code_native(f, ::Type{T}; view = false, nontemporal = false, aligned = true) where {T}
    x = Vector{T}(undef, 1000)
    if view
        x = view(x, 1:10)
    end

end
