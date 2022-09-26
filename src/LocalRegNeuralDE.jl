module LocalRegNeuralDE

using DiffEqBase, DiffEqCallbacks, ChainRulesCore, CUDA, ComponentArrays, Lux, MacroTools,
      OrdinaryDiffEq, Random, SciMLSensitivity, Setfield, UnPack, Zygote
import ChainRulesCore as CRC

include("perform_step.jl")
include("utils.jl")
include("models.jl")

export ArrayAndTime
export TDChain, NeuralODE, diffeqsol_to_array

end
