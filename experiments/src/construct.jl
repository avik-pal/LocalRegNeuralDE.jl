function _get_loss_function_classification(expt::ExperimentConfig, cfg::LossConfig)
  if expt.model.regularize == "none"
    return function cls_loss_function_no_regularization(model, ps, st, (x, y), w_reg)
      y_pred, st_ = Lux.apply(model, x, ps, st)
      ce_loss = logitcrossentropy(y_pred, y)
      reg_val = zero(ce_loss)
      nfe = st_.neural_ode.nfe

      return (ce_loss, st_, (; y_pred, nfe, ce_loss, reg_val))
    end
  else
    return function cls_loss_function_with_regularization(model, ps, st, (x, y), w_reg)
      y_pred, st_ = Lux.apply(model, x, ps, st)
      ce_loss = logitcrossentropy(y_pred, y)
      reg_val = st_.neural_ode.reg_val
      nfe = st_.neural_ode.nfe

      return (ce_loss + w_reg * reg_val, st_, (; y_pred, nfe, ce_loss, reg_val))
    end
  end
end

function construct(expt::ExperimentConfig, cfg::LossConfig)
  # TODO(@avik-pal): For time series problems (will need a config.jl update)
  lfn = _get_loss_function_classification(expt, cfg)

  sched = if cfg.w_reg_decay == "exponential"
    ExponentialDecay(cfg.w_reg_start, cfg.w_reg_end, expt.train.total_steps)
  else
    Constant(cfg.w_reg_start)
  end

  return lfn, sched
end

function construct(expt::ExperimentConfig, cfg::OptimizerConfig)
  if cfg.optimizer == "adam"
    opt = Adam(cfg.learning_rate)
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
                        "options are: `adam` and `sgd`."))
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

function construct(expt::ExperimentConfig, cfg::ModelConfig)
  if cfg.model_type == "mlp"
    hsize = cfg.mlp_hidden_state_size
    hsize_next = hsize + cfg.mlp_time_dependent
    insize = prod(cfg.image_size) * cfg.in_channels
    layers = Lux.AbstractExplicitLayer[Dense(insize + cfg.mlp_time_dependent => hsize,
                                             tanh)]
    for i in 1:(cfg.mlp_num_hidden_layers - 1)
      push!(layers, Dense(hsize_next => hsize, tanh))
    end
    push!(layers, Dense(hsize_next => insize))
    model = (cfg.mlp_time_dependent ? TDChain : identity)(Chain(layers...))

    return Chain(; flatten=FlattenLayer(),
                 neural_ode=NeuralODE(model; solver=_ode_solver(cfg.solver.ode_solver),
                                      save_everystep=false, reltol=cfg.solver.reltol,
                                      abstol=cfg.solver.abstol, save_start=false,
                                      regularize=Symbol(cfg.regularize), maxiters=10_000),
                 sol_to_arr=WrappedFunction(diffeqsol_to_array),
                 classifier=Dense(insize * cfg.in_channels => cfg.num_classes))
  end

  throw(ArgumentError("unknown ModelConfig."))
end