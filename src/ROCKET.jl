module ROCKET

using Distributions
using CoordinateDescent, ProximalBase

export
  teInference

include("utils.jl")
include("inference.jl")

end
