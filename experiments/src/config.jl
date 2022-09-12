@option struct SolverConfig
  ode_solver::String = "tsit5"
  abstol::Float32 = 5.0f-2
  reltol::Float32 = 5.0f-2
end

@option struct ModelConfig
  solver::SolverConfig = SolverConfig()
end

@option struct OptimizerConfig
  lr_scheduler::String = "cosine"
  optimizer::String = "adam"
  learning_rate::Float32 = 0.01f0
  nesterov::Bool = false
  momentum::Float32 = 0.0f0
  weight_decay::Float32 = 0.0f0
  cycle_length::Int = 50000
  lr_step::Vector{Int64} = [1000, 2000, 5000]
  lr_step_decay::Float32 = 0.1f0
end

@option struct TrainConfig
  total_steps::Int = 10000
  evaluate_every::Int = 2500
  resume::String = ""
  evaluate::Bool = false
  checkpoint_dir::String = "checkpoints"
  log_dir::String = "logs"
  expt_subdir::String = ""
  expt_id::String = ""
  print_frequency::Int = 100
end

@option struct DatasetConfig
  augment::Bool = false
  data_root::String = ""
  eval_batchsize::Int = 64
  train_batchsize::Int = 64
end

@option struct ExperimentConfig
  seed::Int = 0
  model::ModelConfig = ModelConfig()
  optimizer::OptimizerConfig = OptimizerConfig()
  train::TrainConfig = TrainConfig()
  dataset::DatasetConfig = DatasetConfig()
end
