using LocalRegNeuralDE, LocalRegNeuralDEExperiments
using ComponentArrays, Lux, MLDatasets, MLUtils, OneHotArrays, Random, Setfield,
      SimpleConfig, Statistics, Wandb
using Lux: Training

function get_dataloaders(; augment, data_root, eval_batchsize, train_batchsize)
  train_dataset = MNIST(Float32, :train)
  train_iter = dataloader(unsqueeze(train_dataset.features; dims=3),
                          onehotbatch(train_dataset.targets, 0:9), train_batchsize, true,
                          false, true)

  eval_dataset = MNIST(Float32, :test)
  eval_iter = dataloader(unsqueeze(eval_dataset.features; dims=3),
                         onehotbatch(eval_dataset.targets, 0:9), eval_batchsize, true,
                         false, true)

  return train_iter, eval_iter
end

function main(filename, args)
  cfg = define_configuration(args, ExperimentConfig, filename)

  return main(splitext(basename(filename))[1], cfg)
end

function main(config_name::String, cfg::ExperimentConfig)
  rng = Xoshiro(cfg.seed)

  model = construct(cfg, cfg.model)

  loss_function, w_reg_sched = construct(cfg, cfg.loss)

  opt, sched = construct(cfg, cfg.optimizer)

  tstate = Training.TrainState(rng, model, opt; transform_variables=gpu)
  @set! tstate.parameters = tstate.parameters |> cpu |> ComponentArray |> gpu

  vjp_rule = Training.ZygoteVJP()

  warmup_model(loss_function, model, tstate.parameters, tstate.states, cfg;
               transform_input=gpu)

  ds_train, ds_test = get_dataloaders(; cfg.dataset.augment, cfg.dataset.data_root,
                                      cfg.dataset.eval_batchsize,
                                      cfg.dataset.train_batchsize)
  _, ds_train_iter = iterate(ds_train)

  expt_name = ("config-$(config_name)_regularizer-$(cfg.model.regularize)" *
               "_seed-$(cfg.seed)_id-$(cfg.train.expt_id)")

  ckpt_dir = joinpath(cfg.train.expt_subdir, cfg.train.checkpoint_dir, expt_name)
  log_dir = joinpath(cfg.train.expt_subdir, cfg.train.log_dir, expt_name)
  if cfg.train.resume == ""
    rpath = joinpath(ckpt_dir, "model_current.jlso")
  else
    rpath = cfg.train.resume
  end

  ckpt = load_checkpoint(rpath)
  if !isnothing(ckpt)
    tstate = ckpt.tstate
    initial_step = ckpt.step
    @info "Training Started from Step: $initial_step"
  else
    initial_step = 1
  end

  loggers = DEQExperiments.create_logger(log_dir, cfg.train.total_steps - initial_step,
                                         cfg.train.total_steps - initial_step, expt_name,
                                         flatten_configuration(cfg))

  best_test_accuracy = 0

  for step in initial_step:(cfg.train.total_steps)
    t = time()
    (x, y), ds_train_iter = iterate(ds_train, ds_train_iter)
    x = x |> gpu
    y = y |> gpu
    data_time = time() - t

    bsize = size(x, ndims(x))

    loss, _, stats, tstate, gs, step_stats = run_training_step(vjp_rule, loss_function,
                                                               (x, y), tstate,
                                                               w_reg_sched(step))

    # LR Update
    Setfield.@set! tstate.optimizer_state = Optimisers.adjust(tstate.optimizer_state,
                                                              sched(step + 1))

    acc1 = accuracy(cpu(stats.y_pred), cpu(y))

    # Logging
    loggers.progress_loggers.train.avg_meters.batch_time(data_time +
                                                         step_stats.fwd_time +
                                                         step_stats.bwd_time +
                                                         step_stats.opt_time, bsize)
    loggers.progress_loggers.train.avg_meters.data_time(data_time, bsize)
    loggers.progress_loggers.train.avg_meters.fwd_time(step_stats.fwd_time, bsize)
    loggers.progress_loggers.train.avg_meters.bwd_time(step_stats.bwd_time, bsize)
    loggers.progress_loggers.train.avg_meters.opt_time(step_stats.opt_time, bsize)
    loggers.progress_loggers.train.avg_meters.loss(loss, bsize)
    loggers.progress_loggers.train.avg_meters.ce_loss(stats.ce_loss, bsize)
    loggers.progress_loggers.train.avg_meters.reg_val(stats.reg_val, bsize)
    loggers.progress_loggers.train.avg_meters.top1(acc1, bsize)
    loggers.progress_loggers.train.avg_meters.top5(-1, bsize)
    loggers.progress_loggers.train.avg_meters.nfe(stats.nfe, bsize)

    if step % cfg.train.print_frequency == 1
      print_meter(loggers.progress_loggers.train.progress, step)
      log_vals = get_loggable_values(loggers.progress_loggers.train.progress)
      loggers.csv_loggers.train(step, log_vals...)
      Wandb.log(loggers.wandb_logger, loggers.log_functions.train(step, log_vals...))
      reset_meter!(loggers.progress_loggers.train.progress)
    end

    # Free memory eagarly
    CUDA.unsafe_free!(x)
    CUDA.unsafe_free!(y)

    if step % cfg.train.evaluate_every == 1 || step == cfg.train.total_steps
      st_eval = Lux.testmode(tstate.states)
      w_reg = w_reg_sched(step)

      for (x, y) in ds_test
        t = time()
        x = x |> gpu
        y = y |> gpu
        data_time = time() - t

        t = time()
        loss, st_, stats = loss_function(model, tstate.parameters, st_eval, (x, y), w_reg)
        fwd_time = time() - t

        bsize = size(x, ndims(x))

        acc1 = accuracy(cpu(stats.y_pred), cpu(y))

        loggers.progress_loggers.eval.avg_meters.batch_time(dtime + fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.data_time(dtime, bsize)
        loggers.progress_loggers.eval.avg_meters.fwd_time(fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.loss(loss, bsize)
        loggers.progress_loggers.eval.avg_meters.ce_loss(stats.ce_loss, bsize)
        loggers.progress_loggers.eval.avg_meters.reg_val(stats.reg_val, bsize)
        loggers.progress_loggers.eval.avg_meters.top1(acc1, bsize)
        loggers.progress_loggers.eval.avg_meters.top5(-1, bsize)
        loggers.progress_loggers.eval.avg_meters.nfe(stats.nfe, bsize)

        # Free memory eagarly
        CUDA.unsafe_free!(x)
        CUDA.unsafe_free!(y)
      end

      print_meter(loggers.progress_loggers.eval.progress, step)
      log_vals = get_loggable_values(loggers.progress_loggers.eval.progress)
      loggers.csv_loggers.eval(step, log_vals...)
      Wandb.log(loggers.wandb_logger, loggers.log_functions.eval(step, log_vals...))
      reset_meter!(loggers.progress_loggers.eval.progress)

      acc = loggers.progress_loggers.eval.avg_meters.top1.average
      is_best = acc >= best_test_accuracy
      if is_best
        best_test_accuracy = acc
      end

      ckpt = (tstate=tstate, step=initial_step)
      save_checkpoint(ckpt; is_best, filename=joinpath(ckpt_dir, "model_$(step).jlso"))
    end
  end

  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS[1], ARGS[2:end])
end
