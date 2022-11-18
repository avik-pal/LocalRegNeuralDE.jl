# Average Meter
Base.@kwdef mutable struct AverageMeter
  fmtstr::Any
  val::Float64 = 0.0
  sum::Float64 = 0.0
  count::Int = 0
  average::Float64 = 0
end

function AverageMeter(name::String, fmt::String)
  fmtstr = FormatExpr("$name {1:$fmt} ({2:$fmt})")
  return AverageMeter(; fmtstr)
end

function (meter::AverageMeter)(val, n::Int)
  meter.val = val
  s = val * n
  meter.sum += s
  meter.count += n
  meter.average = meter.sum / meter.count
  return meter.average
end

function reset_meter!(meter::AverageMeter)
  meter.val = 0.0
  meter.sum = 0.0
  meter.count = 0
  meter.average = 0.0
  return meter
end

function print_meter(meter::AverageMeter)
  return printfmt(meter.fmtstr, meter.val, meter.average)
end

# ProgressMeter
struct ProgressMeter{N}
  batch_fmtstr::Any
  meters::NTuple{N, AverageMeter}
end

function ProgressMeter(num_batches::Int, meters::NTuple{N}, prefix::String="") where {N}
  fmt = "%" * string(length(string(num_batches))) * "d"
  prefix = prefix != "" ? endswith(prefix, " ") ? prefix : prefix * " " : ""
  batch_fmtstr = generate_formatter("$prefix[$fmt/" * sprintf1(fmt, num_batches) * "]")
  return ProgressMeter{N}(batch_fmtstr, meters)
end

function reset_meter!(meter::ProgressMeter)
  reset_meter!.(meter.meters)
  return meter
end

function print_meter(meter::ProgressMeter, batch::Int)
  base_str = meter.batch_fmtstr(batch)
  print(base_str)
  foreach(x -> (print("\t"); print_meter(x)), meter.meters[1:end])
  println()
  return nothing
end

get_loggable_values(meter::ProgressMeter) = getproperty.(meter.meters, :average)

# Log to a CSV File
struct CSVLogger{N}
  filename::Any
  fio::Any
end

function CSVLogger(filename, header)
  !isdir(dirname(filename)) && mkpath(dirname(filename))
  fio = open(filename, "w")
  N = length(header)
  println(fio, join(header, ","))
  return CSVLogger{N}(filename, fio)
end

function (csv::CSVLogger)(args...)
  println(csv.fio, join(args, ","))
  return flush(csv.fio)
end

function Base.close(csv::CSVLogger)
  return close(csv.fio)
end

function create_logger(base_dir::String, train_length::Int, eval_length::Int,
                       expt_name::String, config::Dict; latent_ode::Bool=false,
                       sde::Bool=false)
  if !isdir(base_dir)
    @warn "$(base_dir) doesn't exist. Creating a directory."
    mkpath(base_dir)
  end

  @info expt_name
  @info config

  # Wandb Logger
  wandb_logger = WandbLogger(; project="localregneuralde", name=expt_name, config=config)

  # CSV Logger
  train_csv_header = [
    "Step",
    "Batch Time",
    "Data Time",
    "Forward Pass Time",
    "Backward Pass Time",
    "Optimizer Time",
    (latent_ode ? ["Neg Log Likelihood", "KL Divergence"] : ["Cross Entropy Loss"])...,
    "Regularize Value",
    "Net Loss",
    (sde ? ["NFE Drift", "NFE Diffusion"] : ["NFE"])...,
    (latent_ode ? [] : ["Accuracy (Top 1)", "Accuracy (Top 5)"])...,
  ]
  train_loggable_dict(args...) = Dict(zip(.*(("Train/",), train_csv_header), args))
  csv_logger_train = CSVLogger(joinpath(base_dir, "results_train.csv"), train_csv_header)

  eval_csv_header = [
    "Step",
    "Batch Time",
    "Data Time",
    "Forward Pass Time",
    (latent_ode ? ["Neg Log Likelihood", "KL Divergence"] : ["Cross Entropy Loss"])...,
    "Regularize Value",
    "Net Loss",
    (sde ? ["NFE Drift", "NFE Diffusion"] : ["NFE"])...,
    (latent_ode ? [] : ["Accuracy (Top 1)", "Accuracy (Top 5)"])...,
  ]
  eval_loggable_dict(args...) = Dict(zip(.*(("Eval/",), eval_csv_header), args))
  csv_logger_eval = CSVLogger(joinpath(base_dir, "results_eval.csv"), eval_csv_header)

  # Train Logger
  _tloggers = [
    :batch_time => AverageMeter("Batch Time", "6.3f"),
    :data_time => AverageMeter("Data Time", "6.3f"),
    :fwd_time => AverageMeter("Forward Pass Time", "6.3f"),
    :bwd_time => AverageMeter("Backward Pass Time", "6.3f"),
    :opt_time => AverageMeter("Optimizer Time", "6.3f"),
    (latent_ode ?
     [
       :neg_ll_loss => AverageMeter("Neg Log Likelihood", "6.3e"),
       :kl_div => AverageMeter("KL Divergence", "6.3e"),
     ] : [:ce_loss => AverageMeter("Cross Entropy Loss", "6.3e")])...,
    :reg_val => AverageMeter("Regularize Value", "6.3e"),
    :loss => AverageMeter("Net Loss", "6.3f"),
    (sde ?
     [
       :nfe_drift => AverageMeter("NFE Drift", "3.2f"),
       :nfe_diffusion => AverageMeter("NFE Diffusion", "3.2f"),
     ] : [:nfe => AverageMeter("NFE", "3.2f")])...,
    (latent_ode ? [] :
     [
       :top1 => AverageMeter("Accuracy (@1)", "3.2f"),
       :top5 => AverageMeter("Accuracy (@5)", "3.2f"),
     ])...,
  ]

  progress = ProgressMeter(train_length, Tuple(last.(_tloggers)), "Train:")

  train_logger = (progress=progress, avg_meters=(; _tloggers...))

  # Eval Logger
  _tloggers = [
    :batch_time => AverageMeter("Batch Time", "6.3f"),
    :data_time => AverageMeter("Data Time", "6.3f"),
    :fwd_time => AverageMeter("Forward Time", "6.3f"),
    (latent_ode ? [] : [:ce_loss => AverageMeter("Cross Entropy Loss", "6.3e")])...,
    :reg_val => AverageMeter("Regularize Value", "6.3e"),
    :loss => AverageMeter("Net Loss", "6.3f"),
     (sde ?
      [
        :nfe_drift => AverageMeter("NFE Drift", "3.2f"),
        :nfe_diffusion => AverageMeter("NFE Diffusion", "3.2f"),
      ] : [:nfe => AverageMeter("NFE", "3.2f")])...,
      (latent_ode ? [] :
       [
         :top1 => AverageMeter("Accuracy (@1)", "3.2f"),
         :top5 => AverageMeter("Accuracy (@5)", "3.2f"),
       ])...,
  ]

  progress = ProgressMeter(eval_length, Tuple(last.(_tloggers)), "Test:")

  eval_logger = (progress=progress, avg_meters=(; _tloggers...))

  return (csv_loggers=(; train=csv_logger_train, eval=csv_logger_eval),
          wandb_logger=wandb_logger,
          progress_loggers=(; train=train_logger, eval=eval_logger),
          log_functions=(; train=train_loggable_dict, eval=eval_loggable_dict))
end
