using LocalRegNeuralDE, LocalRegNeuralDEExperiments
using BSON, ComponentArrays, CUDA, Lux, MLDatasets, MLUtils, OneHotArrays, Optimisers,
      Random, Setfield, SimpleConfig, Statistics, Wandb
using Lux: Training

using LazyArtifacts, Artifacts

CUDA.allowscalar(false)

function get_dataloaders(; augment, data_root, eval_batchsize, train_batchsize)
  afile = joinpath(artifact"physionet", "physionet.bson")
  data = BSON.load(afile)[:data]
  total_obs = size(data[:observed_data], 3)
  train_idx, test_idx = splitobs(shuffleobs(collect(1:total_obs)); at=0.8)
  train_data = []
  test_data = []
  for key in [:observed_data, :observed_mask, :data_to_predict, :mask_predicted_data]
    push!(train_data, data[key][:, :, train_idx])
    push!(test_data, data[key][:, :, test_idx])
  end
  for key in [:observed_tp, :tp_to_predict]
    t = reshape(data[key][:, train_idx], 1, 49, :)
    t = hcat(t[:, 2:end, :] .- t[:, 1:(end - 1), :],
             zeros(eltype(t), size(t, 1), 1, size(t, 3)))
    push!(train_data, t)
    t = reshape(data[key][:, test_idx], 1, 49, :)
    t = hcat(t[:, 2:end, :] .- t[:, 1:(end - 1), :],
             zeros(eltype(t), size(t, 1), 1, size(t, 3)))
    push!(test_data, t)
  end

  train_iter = dataloader(Tuple(train_data), train_batchsize, true, false, true)
  eval_iter = dataloader(Tuple(test_data), eval_batchsize, true, false, false)

  _t = data[:observed_tp][:, train_idx[1]] .|> Float32

  return train_iter, eval_iter, _t
end

function main(filename, args)
  cfg = define_configuration(args, ExperimentConfig, filename)

  return main(splitext(basename(filename))[1], cfg)
end

function main(config_name::String, cfg::ExperimentConfig)
  rng = Xoshiro(cfg.seed)

  ds_train, ds_test, saveat = get_dataloaders(; cfg.dataset.augment, cfg.dataset.data_root,
                                              cfg.dataset.eval_batchsize,
                                              cfg.dataset.train_batchsize)
  _, ds_train_iter = iterate(ds_train)

  model = construct(cfg, cfg.model; saveat)

  loss_function, (w_reg_sched, w_kl_sched) = construct(cfg, cfg.loss)

  opt, sched = construct(cfg, cfg.optimizer)

  # Manually create TrainState since ComponentArrays conversion doesn't smoothly work by
  # default
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray |> gpu
  st = st |> gpu
  opt_state = Optimisers.setup(opt, ps)
  tstate = Training.TrainState(model, ps, st, opt_state, 0)

  vjp_rule = Training.ZygoteVJP()

  warmup_model(loss_function, model, tstate.parameters, tstate.states, cfg, first(ds_train);
               transform_input=gpu)

  expt_name = ("config-$(config_name)_regularizer-$(cfg.model.regularize)" *
               "_seed-$(cfg.seed)_id-$(cfg.train.expt_id)")
  @info expt_name

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

  loggers = create_logger(log_dir, cfg.train.total_steps - initial_step,
                          cfg.train.total_steps - initial_step, expt_name,
                          flatten_configuration(cfg); latent_ode=true)

  best_test_loss = 0

  for step in initial_step:(cfg.train.total_steps)
    t = time()
    (x, m, _, _, dt, _), ds_train_iter = iterate(ds_train, ds_train_iter)
    x = x |> gpu
    m = m |> gpu
    dt = dt |> gpu
    data_time = time() - t

    bsize = size(x, ndims(x))

    loss, _, stats, tstate, gs, step_stats = run_training_step(vjp_rule, loss_function,
                                                               (x, m, dt), tstate,
                                                               (w_reg_sched(step),
                                                                w_kl_sched(step)))

    # LR Update
    Setfield.@set! tstate.optimizer_state = Optimisers.adjust(tstate.optimizer_state,
                                                              sched(step + 1))

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
    loggers.progress_loggers.train.avg_meters.neg_ll_loss(stats.neg_log_likelihood, bsize)
    loggers.progress_loggers.train.avg_meters.kl_div(stats.kl_div, bsize)
    loggers.progress_loggers.train.avg_meters.reg_val(stats.reg_val, bsize)
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
    CUDA.unsafe_free!(m)
    CUDA.unsafe_free!(dt)

    if step % cfg.train.evaluate_every == 1 || step == cfg.train.total_steps
      st_eval = Lux.testmode(tstate.states)
      w_reg = w_reg_sched(step)
      w_kl = w_kl_sched(step)

      for (x, m, _, _, dt, _) in ds_test
        t = time()
        x = x |> gpu
        m = m |> gpu
        dt = dt |> gpu
        data_time = time() - t

        t = time()
        x_ = vcat(x, m, dt)
        y, st_ = model(x_, tstate.parameters, st_eval)
        fwd_time = time() - t

        bsize = size(x, ndims(x))

        loss = sum(sum(abs2, (y .- x) .* m; dims=(1, 2)) ./ sum(m; dims = (1, 2)))

        loggers.progress_loggers.eval.avg_meters.batch_time(data_time + fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.data_time(data_time, bsize)
        loggers.progress_loggers.eval.avg_meters.fwd_time(fwd_time, bsize)
        loggers.progress_loggers.eval.avg_meters.loss(loss, bsize)
        loggers.progress_loggers.eval.avg_meters.reg_val(st_.neural_ode.reg_val, bsize)
        loggers.progress_loggers.eval.avg_meters.nfe(st_.neural_ode.nfe, bsize)

        # Free memory eagarly
        CUDA.unsafe_free!(x)
        CUDA.unsafe_free!(m)
        CUDA.unsafe_free!(dt)
      end

      print_meter(loggers.progress_loggers.eval.progress, step)
      log_vals = get_loggable_values(loggers.progress_loggers.eval.progress)
      loggers.csv_loggers.eval(step, log_vals...)
      Wandb.log(loggers.wandb_logger, loggers.log_functions.eval(step, log_vals...))
      reset_meter!(loggers.progress_loggers.eval.progress)

      loss = loggers.progress_loggers.eval.avg_meters.loss.average
      is_best = loss >= best_test_loss
      if is_best
        best_test_loss = loss
      end

      ckpt = (tstate=tstate, step=initial_step)
      save_checkpoint(ckpt; is_best, filename=joinpath(ckpt_dir, "model_$(step).jlso"))
    end
  end

  Wandb.close(loggers.wandb_logger)

  return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
  main(ARGS[1], ARGS[2:end])
end
