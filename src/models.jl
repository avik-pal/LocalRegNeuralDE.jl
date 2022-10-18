import OrdinaryDiffEq: Tsit5ConstantCache
import Lux: AbstractExplicitLayer, AbstractExplicitContainerLayer

# TODO(@avik-pal): Dont compute `reg_val` when not training.

# Core NeuralODE Models
struct NeuralODE{R, M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
       AbstractExplicitContainerLayer{(:model,)}
  model::M
  solver::So
  sensealg::Se
  tspan::T
  maxiters::Int
  kwargs::K

  function NeuralODE(model::Lux.AbstractExplicitLayer; solver=Tsit5(),
                     sensealg=QuadratureAdjoint(; autojacvec=ZygoteVJP()),
                     tspan=(0.0f0, 1.0f0), regularize::Union{Bool, Symbol}=true,
                     maxiters::Int=1000, kwargs...)
    if regularize isa Bool
      regularize = regularize ? :split_mode : :none
    end
    _check_valid_regularize(regularize)
    if :saveat in keys(kwargs)
      if regularize == :biased
        @warn """using `saveat` with `:biased` is typically not the best idea. This is
                 supported but the regularization happens only around the provided `saveat`
                 values and not around the \"hardest\" parts of the dynamical system."""
      end
    end
    return new{regularize, typeof(model), typeof(solver), typeof(sensealg), typeof(tspan),
               typeof(kwargs)}(model, solver, sensealg, tspan, maxiters, kwargs)
  end
end

function Lux.initialstates(rng::AbstractRNG, layer::NeuralODE)
  randn(rng)
  return (model=Lux.initialstates(rng, layer.model), nfe=-1, reg_val=0.0f0,
          rng=Lux.replicate(rng), training=Val(true))
end

function _get_integrator(sol, t, dudt, tspan, ps, solver, sensealg; kwargs...)
  u = sol(t)
  prob = ODEProblem(dudt, u, tspan, ps)
  integrator = _create_integrator(prob, solver; sensealg, kwargs...)
  return integrator
end

CRC.@non_differentiable _get_integrator(::Any...)

function _solve_neuralode_generic(n::NeuralODE, x::AbstractArray, ps, st::NamedTuple,
                                  saveat; kwargs...)
  st_ = st.model
  function dudt(u, p, t)
    u_, st_ = Lux.apply(n.model, ArrayAndTime(u, t), p, st_)
    return get_array(u_)
  end

  prob = ODEProblem(dudt, x, n.tspan, ps)
  sol = solve(prob, n.solver; n.sensealg, n.maxiters, saveat, kwargs...)

  return sol, st_, dudt
end

function _vanilla_node_fallback(n::NeuralODE, x, ps, st)
  saveat, kwargs = _resolve_saveat_kwargs(Val(:none), n.tspan; n.kwargs...)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, saveat; kwargs...)
  return sol, (; model=st_, nfe=_get_destats(sol), reg_val=0.0f0, st.rng, st.training)
end

(n::NeuralODE)(x, ps, st) = n(x, ps, st, st.training)

(n::NeuralODE{:none})(x, ps, st, ::Val) = _vanilla_node_fallback(n, x, ps, st)

(n::NeuralODE{:unbiased})(x, ps, st, ::Val{false}) = _vanilla_node_fallback(n, x, ps, st)

function (n::NeuralODE{:unbiased})(x, ps, st, ::Val{true})
  rng = Lux.replicate(st.rng)
  (t0, t2) = n.tspan
  t1 = rand(rng, eltype(t2)) * (t2 - t0) + t0
  saveat, kwargs, needs_correction = _resolve_saveat_kwargs(Val(:unbiased), n.tspan, t1;
                                                            n.kwargs...)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, saveat; kwargs...)
  integrator = _get_integrator(sol, t1, dudt, n.tspan, ps, n.solver, n.sensealg; kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps)
  nfe = sol.destats.nf + nf2

  sol = needs_correction ? _CorrectedODESolution(sol, n.kwargs[:saveat], t1) : sol

  return sol, (; model=st_, nfe, reg_val, rng, st.training)
end

(n::NeuralODE{:biased})(x, ps, st, ::Val{false}) = _vanilla_node_fallback(n, x, ps, st)

function (n::NeuralODE{:biased})(x, ps, st, ::Val{true})
  saveat, kwargs = _resolve_saveat_kwargs(Val(:biased), n.tspan; n.kwargs...)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, saveat; kwargs...)
  rng = Lux.replicate(st.rng)
  t1 = rand(rng, sol.t)
  integrator = _get_integrator(sol, t1, dudt, n.tspan, ps, n.solver, n.sensealg; kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps)
  nfe = sol.destats.nf + nf2

  return sol, (; model=st_, nfe, reg_val, rng, st.training)
end

function _resolve_saveat_kwargs(::Val{:none}, tspan; saveat=nothing, kwargs...)
  saveat === nothing && return ([tspan[2]], kwargs)
  return (saveat, kwargs)
end

function _resolve_saveat_kwargs(::Val{:unbiased}, tspan, t1; saveat=nothing, kwargs...)
  saveat === nothing && return ([t1, tspan[2]], kwargs, false)
  saveat = vcat(saveat, [t1])
  return (saveat, kwargs, true)
end

function _resolve_saveat_kwargs(::Val{:biased}, tspan; saveat=nothing, kwargs...)
  saveat === nothing && return ([], kwargs)
  return (saveat, kwargs)
end

CRC.@non_differentiable _resolve_saveat_kwargs(::Any...)

# Time Dependent Chain
struct TDChain{L <: NamedTuple} <: AbstractExplicitContainerLayer{(:layers,)}
  layers::L

  TDChain(c::Chain) = TDChain(c.layers)
  TDChain(; kwargs...) = TDChain((; kwargs...))
  TDChain(layers::NamedTuple) = new{typeof(layers)}(layers)
end

(c::TDChain)((x, t), ps, st) = applytdchain(c.layers, x, t, ps, st)

@generated function applytdchain(layers::NamedTuple{fields}, x::T, t, ps,
                                 st::NamedTuple{fields}) where {fields, T}
  N = length(fields)
  x_symbols = vcat([:x], [gensym("x") for _ in 1:N])
  st_symbols = [gensym("st") for _ in 1:N]
  calls = []

  push!(calls, :(_size = size(x);
                 @set! _size[$(ndims(T) - 1)] = 1))
  if T <: CuArray
    push!(calls, :(_t = CUDA.ones($(eltype(T)), _size) .* t))
  else
    push!(calls, :(_t = ones($(eltype(T)), _size) .* t))
  end

  _getinput(v) = :(_cat($v, _t, Val($(ndims(T) - 1))))
  append!(calls,
          [:(($(x_symbols[i + 1]), $(st_symbols[i])) = Lux.apply(layers.$(fields[i]),
                                                                 $(_getinput(x_symbols[i])),
                                                                 ps.$(fields[i]),
                                                                 st.$(fields[i])))
           for i in 1:N])
  push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))

  _getoutput() = :($(x_symbols[N + 1]), _t)

  push!(calls, :(return $(_getoutput()), st))
  return Expr(:block, calls...)
end

function Lux.apply(l::TDChain, x::ArrayAndTime, ps, st::NamedTuple)
  (y, _), st_ = Lux.apply(l, (get_array(x), get_scalar(x)), ps, st)
  return ArrayAndTime(y, get_scalar(x)), st_
end

# Latent GRU Model
struct LatentGRUCell{U <: AbstractExplicitLayer, R <: AbstractExplicitLayer,
                     S <: AbstractExplicitLayer} <:
       AbstractExplicitContainerLayer{(:update_gate, :reset_gate, :new_state)}
  update_gate::U
  reset_gate::R
  new_state::S
  latent_dim::Int
end

function LatentGRUCell(in_dim::Int, h_dim::Int, latent_dim::Int)
  _in_dim = latent_dim * 2 + in_dim * 2 + 1
  update_gate = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim, sigmoid))
  reset_gate = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim, sigmoid))
  new_state = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim * 2, tanh))

  return LatentGRUCell(update_gate, reset_gate, new_state, latent_dim)
end

function (l::LatentGRUCell)(x::A, ps, st::NamedTuple) where {A <: AbstractMatrix}
  y_mean = zeros_like(x, (l.latent_dim, size(x, 2)))
  y_std = ones_like(x, (l.latent_dim, size(x, 2)))
  return l((x, (y_mean, y_std)), ps, st)
end

@views function (l::LatentGRUCell)((x, (y_mean, y_std)), ps, st)
  y_concat = vcat(y_mean, y_std, x)

  update_gate, st_ug = l.update_gate(y_concat, ps.update_gate, st.update_gate)
  reset_gate, st_rg = l.reset_gate(y_concat, ps.reset_gate, st.reset_gate)

  concat = vcat(y_mean .* reset_gate, y_std .* reset_gate, x)

  new_state, st_ns = l.new_state(concat, ps.new_state, st.new_state)
  new_state_mean = new_state[1:(l.latent_dim), :]
  new_state_std = new_state[(l.latent_dim + 1):end, :]

  new_y_mean = (1 .- update_gate) .* new_state_std .+ update_gate .* y_mean
  new_y_std = (1 .- update_gate) .* new_state_std .+ update_gate .* y_std

  mask = sum(x[(size(x, 1) ÷ 2 + 1):end, :]; dims=1) .> 0

  new_y_mean = mask .* new_y_mean .+ (1 .- mask) .* y_mean
  new_y_std = mask .* new_y_std .+ (1 .- mask) .* y_std

  y = vcat(new_y_mean, new_y_std)
  return (y, (new_y_mean, new_y_std)),
         (; update_gate=st_ug, reset_gate=st_rg, new_state=st_ns)
end

# Reparameterization
struct ReparameterizeLayer <: AbstractExplicitLayer end

function Lux.initialstates(rng::AbstractRNG, ::ReparameterizeLayer)
  randn(rng, 1)
  return (; rng, training=Val(true), μ₀=nothing, logσ²=nothing)
end

@views function (r::ReparameterizeLayer)(x::T, ps,
                                         st::NamedTuple) where {T <: AbstractMatrix}
  y, μ₀, logσ², rng = reparameterize(st.training, x, st.rng)
  return y, (; rng, st.training, μ₀, logσ²)
end

@views function reparameterize(::Val{true}, x, rng::AbstractRNG)
  rng = Lux.replicate(rng)

  latent_dim = size(x, 1) ÷ 2
  μ₀ = x[1:latent_dim, :]
  logσ² = x[(latent_dim + 1):end, :]

  sample = randn_like(rng, x, size(μ₀))

  return μ₀ .+ exp.(logσ² ./ 2) .* sample, μ₀, logσ², rng
end

@views function reparameterize(::Val{false}, x, rng::AbstractRNG)
  latent_dim = size(x, 1) ÷ 2
  μ₀ = x[1:latent_dim, :]
  return μ₀, μ₀, μ₀, rng
end
