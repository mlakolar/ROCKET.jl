module ROCKET

using Distributions
using HD, ProximalBase

export
  teInference

include("utils.jl")
include("inference.jl")

end
