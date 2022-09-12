module LocalRegNeuralDEExperiments

using ChainRulesCore, Configurations, CUDA, LocalRegNeuralDE, Lux, Random, Setfield,
      SimpleConfig
import ChainRulesCore as CRC

include("config.jl")
include("models.jl")

export TDChain

end