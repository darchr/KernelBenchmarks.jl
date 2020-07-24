module KernelBenchmarks

##### stdlib
using InteractiveUtils
import REPL
using REPL.TerminalMenus

##### "Internal" Packages
# These are packages I've developed for testing purposes and not part of the
# greater Julia ecosystem.
using MaxLFSR

##### External Packages

# For vector load intrinsics
using SIMD

include("threaded.jl")
include("kernels.jl")
include("introspection.jl")
include("updated.jl")

end # module
