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

@option struct DEQSensitivityConfig
  jfb::Bool = false
  abstol::Float32 = 5.0f-2
  reltol::Float32 = 5.0f-2
  maxiters::Int = 20
end

@option struct DEQSolverConfig
  continuous::Bool = true
  stop_mode::String = "rel_norm"
  ode_solver::String = "vcab3"
  abstol::Float32 = 5.0f-2
  reltol::Float32 = 5.0f-2
  abstol_termination::Float32 = 5.0f-2
  reltol_termination::Float32 = 5.0f-2
end

@option struct DEQModelConfig
  num_classes::Int = 10
  dropout_rate::Float32 = 0.25f0
  group_count::Int = 8
  weight_norm::Bool = true
  downsample_times::Int = 0
  expansion_factor::Int = 5
  image_size::Vector{Int64} = [32, 32]
  num_branches::Int = 2
  big_kernels::Vector{Int64} = [0, 0]
  head_channels::Vector{Int64} = [8, 16]
  num_channels::Vector{Int64} = [24, 24]
  fuse_method::String = "sum"
  final_channelsize::Int = 200
  model_type::String = "vanilla"
  maxiters::Int = 18
  in_channels::Int = 3
  sensealg::DEQSensitivityConfig = DEQSensitivityConfig()
  solver::DEQSolverConfig = DEQSolverConfig()
end

@option struct ModelConfig
  model_type::String = "mlp"  # Options: `mlp`, `time_series`
  regularize::String = "unbiased"
  regularize_type::String = "error_estimate"
  image_size::Vector{Int64} = [32, 32]
  in_channels::Int = 3
  num_classes::Int = 10
  sde::Bool = false

  # DEQ
  deq::DEQModelConfig = DEQModelConfig()

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
