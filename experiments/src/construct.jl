function _get_loss_function_classification(expt::ExperimentConfig, cfg::LossConfig)
  if expt.model.regularize == "none"
    return function cts_loss_function_no_regularization(model, ps, st, (x, y), w_reg)
      y_pred, st_ = Lux.apply(model, x, ps, st)
      ce_loss = logitcrossentropy(y_pred, y)
      reg_val = zero(ce_loss)
      if expt.model.sde
        nfe_drift = st_.neural_dsde.nfe_drift
        nfe_diffusion = st_.neural_dsde.nfe_drift
        nfe = (nfe_drift, nfe_diffusion)
      else
        nfe = st_.neural_ode.nfe
      end

      return (ce_loss, st_, (; y_pred, nfe, ce_loss, reg_val))
    end
  else
    return function cts_loss_function_with_regularization(model, ps, st, (x, y), w_reg)
      y_pred, st_ = Lux.apply(model, x, ps, st)
      ce_loss = logitcrossentropy(y_pred, y)
      if expt.model.sde
        reg_val = st_.neural_dsde.reg_val
        nfe_drift = st_.neural_dsde.nfe_drift
        nfe_diffusion = st_.neural_dsde.nfe_drift
        nfe = (nfe_drift, nfe_diffusion)
      else
        reg_val = st_.neural_ode.reg_val
        nfe = st_.neural_ode.nfe
      end

      return (ce_loss + w_reg * reg_val, st_, (; y_pred, nfe, ce_loss, reg_val))
    end
  end
end

function _get_loss_function_latent_ode(expt::ExperimentConfig, cfg::LossConfig)
  if expt.model.regularize == "none"
    return function ts_loss_function_no_regularization(model, ps, st, (data, mask, dt),
                                                       (w_reg, w_kl))
      x = vcat(data, mask, dt)
      y, st_ = Lux.apply(model, x, ps, st)

      data_ = data .* mask
      pred_ = y .* mask
      ∇pred = pred_ .- data_

      log_likelihood = log_likelihood_loss(∇pred, mask)
      kl_div = kl_divergence(st_.reparam.μ₀, st_.reparam.logσ²)

      loss = -mean(log_likelihood .- w_kl .* kl_div)

      return (loss, st_,
              (; neg_log_likelihood=-mean(log_likelihood), kl_div=mean(kl_div), loss,
               nfe=st_.neural_ode.nfe, reg_val=0.0f0))
    end
  else
    return function ts_loss_function_with_regularization(model, ps, st, (data, mask, dt),
                                                         (w_reg, w_kl))
      x = vcat(data, mask, dt)
      y, st_ = Lux.apply(model, x, ps, st)

      data_ = data .* mask
      pred_ = y .* mask
      ∇pred = pred_ .- data_

      log_likelihood = log_likelihood_loss(∇pred, mask)
      kl_div = kl_divergence(st_.reparam.μ₀, st_.reparam.logσ²)

      loss = -mean(log_likelihood .- w_kl .* kl_div) + w_reg * st_.neural_ode.reg_val

      return (loss, st_,
              (; neg_log_likelihood=-mean(log_likelihood), kl_div=mean(kl_div), loss,
               nfe=st_.neural_ode.nfe, st_.neural_ode.reg_val))
    end
  end
end

function construct(expt::ExperimentConfig, cfg::LossConfig)
  if expt.model.model_type == "time_series"
    lfn = _get_loss_function_latent_ode(expt, cfg)
  else
    lfn = _get_loss_function_classification(expt, cfg)
  end

  sched = if expt.model.model_type != "time_series"
    if cfg.w_reg_decay == "exponential"
      ExponentialDecay(cfg.w_reg_start, cfg.w_reg_end, expt.train.total_steps)
    else
      Constant(cfg.w_reg_start)
    end
  else
    w_reg = if cfg.w_reg_decay == "exponential"
      ExponentialDecay(cfg.w_reg_start, cfg.w_reg_end, expt.train.total_steps)
    else
      Constant(cfg.w_reg_start)
    end
    w_kl = t -> max(0, 1 - 0.99f0^(t - 100))
    (w_reg, w_kl)
  end

  return lfn, sched
end

function construct(expt::ExperimentConfig, cfg::OptimizerConfig)
  if cfg.optimizer == "adam"
    opt = Adam(cfg.learning_rate)
  elseif cfg.optimizer == "adamw"
    opt = AdamW(cfg.learning_rate)
  elseif cfg.optimizer == "adamax"
    opt = AdaMax(cfg.learning_rate)
  elseif cfg.optimizer == "sgd"
    if cfg.nesterov
      opt = Nesterov(cfg.learning_rate, cfg.momentum)
    elseif cfg.momentum == 0
      opt = Descent(cfg.learning_rate)
    else
      opt = Momentum(cfg.learning_rate, cfg.momentum)
    end
  else
    throw(ArgumentError("unknown value for `optimizer` = $(cfg.optimizer). Supported " *
                        "options are: `adam`, `adamax` and `sgd`."))
  end

  if cfg.weight_decay != 0
    opt = OptimiserChain(opt, WeightDecay(cfg.weight_decay))
  end

  if cfg.scheduler.lr_scheduler == "cosine"
    scheduler = CosineAnneal(cfg.learning_rate,
                             cfg.learning_rate / cfg.scheduler.cosine_lr_div_factor,
                             cfg.scheduler.cosine_cycle_length; restart=true,
                             dampen=cfg.scheduler.cosine_dampen)
  elseif cfg.scheduler.lr_scheduler == "constant"
    scheduler = Constant(cfg.learning_rate)
  elseif cfg.scheduler.lr_scheduler == "step"
    scheduler = Step(cfg.learning_rate, cfg.scheduler.step_lr_step_decay,
                     cfg.scheduler.step_lr_steps)
  elseif cfg.scheduler.lr_scheduler == "inverse"
    scheduler = InverseDecay(cfg.learning_rate, cfg.scheduler.inverse_decay_factor)
  elseif cfg.scheduler.lr_scheduler == "exponential"
    scheduler = ExponentialDecay(cfg.learning_rate,
                                 cfg.learning_rate /
                                 cfg.scheduler.exponential_lr_div_factor,
                                 expt.train.total_steps)
  else
    throw(ArgumentError("unknown value for `scheduler` = $(cfg.scheduler.lr_scheduler). " *
                        "Supported options are: `constant`, `step`, `exponential`, " *
                        "`inverse` and `cosine`."))
  end

  return opt, scheduler
end

function _ode_solver(s::String)
  if s == "tsit5"
    return Tsit5()
  end

  throw(ArgumentError("unknown SolverConfig."))
end

function construct(expt::ExperimentConfig, cfg::ModelConfig; kwargs...)
  if cfg.model_type == "mlp" && !cfg.sde
    return _construct_mlp_ode(expt, cfg; kwargs...)
  elseif cfg.model_type == "mlp" && cfg.sde
    return _construct_mlp_sde(expt, cfg; kwargs...)
  elseif cfg.model_type == "time_series"
    return _construct_time_series(expt, cfg; kwargs...)
  elseif cfg.model_type == "cifar10_cnn"
    return _construct_cifar10_cnn(expt, cfg; kwargs...)
  elseif cfg.model_type == "cifar10_deq"
    return _construct_cifar10_deq(expt, cfg, cfg.deq; kwargs...)
  end

  throw(ArgumentError("unknown ModelConfig."))
end

function _construct_mlp_ode(expt::ExperimentConfig, cfg::ModelConfig; kwargs...)
  hsize = cfg.mlp_hidden_state_size
  hsize_next = hsize + cfg.mlp_time_dependent
  insize = prod(cfg.image_size) * cfg.in_channels
  layers = Lux.AbstractExplicitLayer[Dense(insize + cfg.mlp_time_dependent => hsize, tanh)]
  for i in 1:(cfg.mlp_num_hidden_layers - 1)
    push!(layers, Dense(hsize_next => hsize, tanh))
  end
  push!(layers, Dense(hsize_next => insize))
  model = (cfg.mlp_time_dependent ? TDChain : identity)(Chain(layers...))

  return Chain(; flatten=FlattenLayer(),
               neural_ode=NeuralODE(model; solver=_ode_solver(cfg.solver.ode_solver),
                                    reltol=cfg.solver.reltol, abstol=cfg.solver.abstol,
                                    save_start=false, regularize=Symbol(cfg.regularize),
                                    maxiters=10_000,
                                    sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP())),
               sol_to_arr=WrappedFunction(diffeqsol_to_array),
               classifier=Dense(insize * cfg.in_channels => cfg.num_classes))
end

function _construct_mlp_sde(expt::ExperimentConfig, cfg::ModelConfig; kwargs...)
  return Chain(; flatten=FlattenLayer(), downsample=Dense(784 => 32),
               neural_dsde=NeuralDSDE(Chain(Dense(32 => 64, tanh), Dense(64 => 32)),
                                      Dense(32 => 32); reltol=cfg.solver.reltol,
                                      abstol=cfg.solver.abstol, save_start=false,
                                      regularize=Symbol(cfg.regularize), maxiters=10_000),
               sol_to_arr=WrappedFunction(diffeqsol_to_array),
               classifier=Dense(32 => cfg.num_classes))
end

function _construct_cifar10_cnn(expt::ExperimentConfig, cfg::ModelConfig; kwargs...)
  node_core = TDChain(Chain(Chain(Conv((3, 3), 9 => 64; pad=(1, 1)), BatchNorm(64, tanh)),
                            Chain(Conv((3, 3), 65 => 64; pad=(1, 1)), BatchNorm(64, tanh)),
                            Conv((3, 3), 65 => 8; pad=(1, 1)); disable_optimizations=true))
  neural_ode = NeuralODE(node_core; solver=Tsit5(), cfg.solver.reltol, cfg.solver.abstol,
                         save_start=false, regularize=Symbol(cfg.regularize),
                         maxiters=10_000,
                         sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
  return Chain(; augment=AugmenterLayer(Conv((3, 3), 3 => 5; pad=(1, 1)), 3),
               bn=BatchNorm(8), neural_ode, sol_to_arr=WrappedFunction(diffeqsol_to_array),
               classifier=Chain(Conv((3, 3), 8 => 1; pad=(1, 1)), FlattenLayer(),
                                Dense(32 * 32 => 10)))
end

function __replace_deq_with_regularizer(regularize::Symbol)
  __replace_deq_with_regularizer_closure(l) = l
  function __replace_deq_with_regularizer_closure(l::LocalRegNeuralDE.AbstractDEQ)
    return RegularizedDEQ(l; regularize)
  end
  return __replace_deq_with_regularizer_closure
end

function _construct_cifar10_deq(expt::ExperimentConfig, cfg_base::ModelConfig,
                                cfg::DEQModelConfig; kwargs...)
  sensealg = DeepEquilibriumAdjoint(cfg.sensealg.abstol, cfg.sensealg.reltol,
                                    cfg.sensealg.maxiters;
                                    mode=cfg.sensealg.jfb ? :jfb : :vanilla)
  solver = ContinuousDEQSolver(Tsit5(); mode=Symbol(cfg.solver.stop_mode),
                               cfg.solver.abstol, cfg.solver.reltol,
                               cfg.solver.abstol_termination, cfg.solver.reltol_termination)
  model = DEQExperiments.get_model(; cfg.num_channels, cfg.downsample_times,
                                   cfg.num_branches, cfg.expansion_factor, cfg.dropout_rate,
                                   cfg.group_count, cfg.big_kernels, cfg.head_channels,
                                   cfg.fuse_method, cfg.final_channelsize, cfg.num_classes,
                                   cfg.model_type, cfg.maxiters, cfg.image_size,
                                   cfg.weight_norm, cfg.in_channels, solver, sensealg)
  # Calling this a NeuralODE to avoid unnecessary branching in the loss function code
  return Chain(; initial_block=model.layers[1],
               neural_ode=RegularizedDEQ(model.layers[2];
                                         regularize=Symbol(cfg_base.regularize)),
               classifier=model.layers[3])
end

function _construct_time_series(expt::ExperimentConfig, cfg::ModelConfig; saveat, kwargs...)
  gru = Recurrence(LatentGRUCell(cfg.ts_in_dims, cfg.ts_hidden_dims, cfg.ts_latent_dims))
  rec_to_gen = Chain(Dense(2 * cfg.ts_latent_dims => cfg.ts_latent_dims, tanh),
                     Dense(cfg.ts_latent_dims => 2 * cfg.ts_node_dims))
  reparam = ReparameterizeLayer()
  gen_dynamics = Chain(Base.Fix1(broadcast, tanh),
                       Dense(cfg.ts_node_dims => cfg.ts_hidden_dims, tanh),
                       Dense(cfg.ts_hidden_dims => cfg.ts_node_dims, tanh),
                       Dense(cfg.ts_node_dims => cfg.ts_hidden_dims, tanh),
                       Dense(cfg.ts_hidden_dims => cfg.ts_node_dims, tanh),
                       Dense(cfg.ts_node_dims => cfg.ts_hidden_dims, tanh),
                       Dense(cfg.ts_hidden_dims => cfg.ts_node_dims, tanh),
                       Dense(cfg.ts_node_dims => cfg.ts_hidden_dims, tanh),
                       Dense(cfg.ts_hidden_dims => cfg.ts_node_dims, tanh))
  neural_ode = NeuralODE(gen_dynamics; solver=_ode_solver(cfg.solver.ode_solver),
                         reltol=cfg.solver.reltol, abstol=cfg.solver.abstol,
                         regularize=Symbol(cfg.regularize), maxiters=10_000, saveat,
                         sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()))
  diffeqsol_to_array = WrappedFunction(diffeqsol_to_timeseries)
  gen_to_data = Dense(cfg.ts_node_dims, cfg.ts_in_dims)
  return Chain(; gru, rec_to_gen, reparam, neural_ode, diffeqsol_to_array, gen_to_data)
end
