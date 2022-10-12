module LocalRegNeuralDE

using ChainRulesCore, CUDA, ComponentArrays, DiffEqBase, DiffEqCallbacks, Functors, Lux,
      OrdinaryDiffEq, Random, SciMLSensitivity, Setfield, Tracker, UnPack, Zygote
import ChainRulesCore as CRC

include("perform_step.jl")
include("utils.jl")
include("models.jl")

export ArrayAndTime
export TDChain, NeuralODE, diffeqsol_to_array

end
