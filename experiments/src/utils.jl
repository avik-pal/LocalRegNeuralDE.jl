# Schedulers
struct ExponentialDecay{T <: Real}
  λ₀::T
  λ₁::T
  k::T
  nsteps::Int

  function ExponentialDecay(λ₀::T, λ₁::T, nsteps::Int) where {T <: Real}
    k = log(λ₀ / λ₁) / nsteps
    return new{T}(λ₀, λ₁, k, nsteps)
  end
end

(e::ExponentialDecay)(t) = e.λ₀ * exp(-e.k * t)

struct InverseDecay{T <: Real}
  λ₀::T
  γ::T

  function InverseDecay(λ₀::T, γ::T) where {T <: Real}
    return new{T}(λ₀, γ)
  end
end

(id::InverseDecay)(t) = id.λ₀ / (1 + id.γ * t)

struct Step{T, S}
  λ₀::T
  γ::T
  step_sizes::S

  function Step(λ₀::T, γ::T, step_sizes::S) where {T, S}
    _step_sizes = (S <: Integer) ? [step_sizes] : step_sizes

    return new{T, typeof(_step_sizes)}(λ₀, γ, _step_sizes)
  end
end

(s::Step)(t) = s.λ₀ * s.γ^(searchsortedfirst(s.step_sizes, t - 1) - 1)

struct Constant{T <: Real}
  λ::T
end

(c::Constant)(t) = c.λ

struct CosineAnneal{restart, T, S <: Integer}
  range::T
  offset::T
  dampen::T
  period::S

  function CosineAnneal(λ₀, λ₁, period::Integer; restart=false, dampen=1)
    range = abs(λ₀ - λ₁)
    offset = min(λ₀, λ₁)
    return new{restart, typeof(range), typeof(period)}(range, offset, oftype(λ₁, dampen),
                                                       period)
  end
end

function (s::CosineAnneal{true})(t)
  d = s.dampen^div(t - 1, s.period)
  return (s.range * (1 + cos(pi * mod(t - 1, s.period) / s.period)) / 2 + s.offset) / d
end

function (s::CosineAnneal{false})(t)
  return s.range * (1 + cos(pi * (t - 1) / s.period)) / 2 + s.offset
end

# Losses and Metrics
function accuracy(y_pred::AbstractMatrix, y::AbstractMatrix)
  return sum(argmax.(eachcol(y_pred)) .== onecold(y)) * 100 / size(y, 2)
end

function accuracy(y_pred::AbstractMatrix, y::AbstractMatrix,
                  topk::NTuple{N, <:Int}) where {N}
  maxk = maximum(topk)

  pred_labels = partialsortperm.(eachcol(y_pred), (1:maxk,), rev=true)
  true_labels = onecold(y)

  accuracies = Tuple(sum(map((a, b) -> sum(view(a, 1:k) .== b), pred_labels, true_labels))
                     for k in topk)

  return accuracies .* 100 ./ size(y, ndims(y))
end

logitcrossentropy(y_pred, y; dims=1) = mean(-sum(y .* logsoftmax(y_pred; dims); dims))

mean_absolute_error(y_pred, y) = mean(abs, y_pred .- y)

mean_squared_error(y_pred, y) = mean(abs2, y_pred .- y)

function log_likelihood_loss(∇pred, mask)
  σ_ = 0.01f0
  sample_likelihood = -(∇pred .^ 2) ./ (2 * σ_^2) .- log(σ_) .- log(Float32(2π)) ./ 2
  return vec(sum(sample_likelihood; dims=(1, 2)) ./ sum(mask; dims=(1, 2)))
end

# Holds only for a Standard Gaussian Prior
kl_divergence(μ, logσ²) = vec(mean(exp.(logσ²) .+ (μ .^ 2) .- 1 .- logσ²; dims=1) ./ 2)

# Augment the Training Step
function run_training_step(::Training.ZygoteVJP, objective_function, data,
                           ts::Training.TrainState, args...)
  t = time()
  (loss, st, stats), back = Zygote.pullback(ps -> objective_function(ts.model, ps,
                                                                     ts.states, data,
                                                                     args...),
                                            ts.parameters)
  fwd_time = time() - t

  t = time()
  grads = back((one(loss), nothing, nothing))[1]
  bwd_time = time() - t

  @set! ts.states = st
  t = time()
  ts = Training.apply_gradients(ts, grads)
  opt_time = time() - t

  return loss, st, stats, ts, grads, (; fwd_time, bwd_time, opt_time)
end

# Utilities
function warmup_model(loss_function, model, ps, st, cfg::ExperimentConfig; transform_input)
  x = ones(Float32, cfg.model.image_size..., cfg.model.in_channels, 1) |> transform_input
  y = onehotbatch([1], 0:(cfg.model.num_classes - 1)) |> transform_input

  @info "model warmup started"
  loss_function(model, ps, st, (x, y), 1.0f0)
  @info "forward pass warmup completed"
  Zygote.gradient(p -> first(loss_function(model, ps, st, (x, y), 1.0f0)), ps)
  @info "backward pass warmup completed"

  return nothing
end

_unbatch_data(x::AbstractArray{T, N}) where {T, N} = unsqueeze(selectdim(x, N, 1); dims=N)

function warmup_model(loss_function, model, ps, st, cfg::ExperimentConfig, data;
                      transform_input)
  data = data .|> _unbatch_data .|> transform_input

  @info "model warmup started"
  loss_function(model, ps, st, (data[1], data[2], data[5]), (1.0f0, 1.0f0))
  @info "forward pass warmup completed"
  Zygote.gradient(p -> first(loss_function(model, ps, st, (data[1], data[2], data[5]),
                                           (1.0f0, 1.0f0))), ps)
  @info "backward pass warmup completed"

  return nothing
end

function dataloader(data, batchsize, shuffle::Bool, partial::Bool, repeat_forever::Bool)
  _data = BatchView((shuffle ? shuffleobs(data) : data); batchsize, partial, collate=true)

  _iter = MLUtils.eachobsparallel(_data; executor=FLoops.ThreadedEx(), buffer=true,
                                  channelsize=max(1, Threads.nthreads() ÷ 2))

  return repeat_forever ? Iterators.cycle(_iter) : _iter
end

function dataloader(X, Y, batchsize, shuffle::Bool, partial::Bool, repeat_forever::Bool)
  return dataloader((X, Y), batchsize, shuffle, partial, repeat_forever)
end

# Checkpointing
function _symlink_safe(src, dest)
  rm(dest; force=true)
  return symlink(src, dest)
end

function save_checkpoint(state::NamedTuple; is_best::Bool, filename::String)
  isdir(dirname(filename)) || mkpath(dirname(filename))
  JLSO.save(filename, :state => state)
  is_best && _symlink_safe(filename, joinpath(dirname(filename), "model_best.jlso"))
  _symlink_safe(filename, joinpath(dirname(filename), "model_current.jlso"))
  return nothing
end

function load_checkpoint(fname::String)
  try
    # NOTE(@avik-pal): ispath is failing for symlinks?
    return JLSO.load(fname)[:state]
  catch
    @warn """$fname could not be loaded. This might be because the file is absent or is
             corrupt. Proceeding by returning `nothing`."""
    return nothing
  end
end
