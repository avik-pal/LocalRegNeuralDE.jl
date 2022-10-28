module LocalRegNeuralDE

using ChainRulesCore, CUDA, ComponentArrays, DiffEqBase, DiffEqCallbacks, FastBroadcast,
      Functors, LinearAlgebra, Lux, MLUtils, NNlib, OrdinaryDiffEq, Random,
      SciMLSensitivity, Setfield, Statistics, StochasticDiffEq, Tracker, UnPack, Zygote
import ChainRulesCore as CRC
import OrdinaryDiffEq: Tsit5ConstantCache, VCABM3ConstantCache
import StochasticDiffEq: FourStageSRIConstantCache, RKMilCommuteConstantCache,
                         LambaEulerHeunConstantCache
import Lux: AbstractExplicitLayer, AbstractExplicitContainerLayer

include("perform_step.jl")
include("utils.jl")

include("layers/common.jl")
include("layers/neural_ode.jl")
include("layers/latent_ode.jl")
include("layers/neural_sde.jl")
include("layers/deq.jl")

export ArrayAndTime
export AugmenterLayer, TDChain, NeuralODE, NeuralDSDE, LatentGRUCell, ReparameterizeLayer
export RegularizedDEQ
export diffeqsol_to_array, diffeqsol_to_timeseries

end
