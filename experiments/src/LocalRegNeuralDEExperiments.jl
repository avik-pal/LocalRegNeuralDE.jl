module LocalRegNeuralDEExperiments

using Configurations, Dates, DeepEquilibriumNetworks, Formatting, Functors, FLoops, JLSO,
      LocalRegNeuralDE, Lux, MLUtils, NNlib, OneHotArrays, Optimisers, OrdinaryDiffEq,
      SciMLSensitivity, Setfield, Statistics, Zygote, Wandb, Random
using Lux: Training
import DEQExperiments

include("config.jl")
include("models.jl")
include("construct.jl")
include("utils.jl")
include("logging.jl")

export ExperimentConfig
export Constant, CosineAnneal, ExponentialDecay, InverseDecay, Step
export accuracy, construct, dataloader, run_training_step, warmup_model
export load_checkpoint, save_checkpoint
export create_logger, get_loggable_values, print_meter, reset_meter!

end