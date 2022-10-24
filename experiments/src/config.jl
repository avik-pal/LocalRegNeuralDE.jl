@option struct LossConfig
  w_reg_start::Float32 = 1.0f2
  w_reg_end::Float32 = 1.0f1
  w_reg_decay::String = "exponential"
end

@option struct SolverConfig
  ode_solver::String = "tsit5"
  abstol::Float32 = 5.0f-2
  reltol::Float32 = 5.0f-2
end

@option struct ModelConfig
  model_type::String = "mlp"
  regularize::String = "unbiased"
  regularize_type::String = "error_estimate"
  image_size::Vector{Int64} = [32, 32]
  in_channels::Int = 3
  num_classes::Int = 10
  sde::Bool = false

  # Solver
  solver::SolverConfig = SolverConfig()

  # mlp
  mlp_hidden_state_size::Int = 100
  mlp_num_hidden_layers::Int = 1
  mlp_time_dependent::Bool = true

  # time_series
  ts_in_dims::Int = 37
  ts_hidden_dims::Int = 40
  ts_latent_dims::Int = 50
  ts_node_dims::Int = 20
end

@option struct LRSchedulerConfig
  lr_scheduler::String = "inverse"

  # cosine
  cosine_lr_div_factor::Real = 100
  cosine_cycle_length::Int = 50000
  cosine_dampen::Float32 = 1.0f0

  # step
  step_lr_steps::Vector{Int64} = [1000, 2000, 5000]
  step_lr_step_decay::Float32 = 0.1f0

  # inverse
  inverse_decay_factor::Float32 = 0.0001f0

  # exponential
  exponential_lr_div_factor::Real = 100
end

@option struct OptimizerConfig
  optimizer::String = "adam"
  learning_rate::Float32 = 0.01f0
  nesterov::Bool = false
  momentum::Float32 = 0.0f0
  weight_decay::Float32 = 0.0f0
  scheduler::LRSchedulerConfig = LRSchedulerConfig()
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
  loss::LossConfig = LossConfig()
  model::ModelConfig = ModelConfig()
  optimizer::OptimizerConfig = OptimizerConfig()
  train::TrainConfig = TrainConfig()
  dataset::DatasetConfig = DatasetConfig()
end
