module LocalRegNeuralDEExperiments

using ChainRulesCore, Configurations, CUDA, LocalRegNeuralDE, Lux, Random, Setfield
import ChainRulesCore as CRC

include("config.jl")
include("models.jl")
include("utils.jl")

export ExperimentConfig
export ExpDecay

end