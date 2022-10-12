import OrdinaryDiffEq: Tsit5ConstantCache
import Lux: AbstractExplicitContainerLayer

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
      throw(ArgumentError(":saveat is currently not supported in NeuralODE"))
    end
    return new{regularize, typeof(model), typeof(solver), typeof(sensealg), typeof(tspan),
               typeof(kwargs)}(model, solver, sensealg, tspan, maxiters, kwargs)
  end
end

function Lux.initialstates(rng::AbstractRNG, layer::NeuralODE)
  randn(rng)
  return (model=Lux.initialstates(rng, layer.model), nfe=-1, reg_val=0.0f0,
          rng=Lux.replicate(rng))
end

function _get_integrator(sol, t, dudt, tspan, ps, solver, sensealg; kwargs...)
  u = sol(t)
  prob = ODEProblem(dudt, u, tspan, ps)
  integrator = _create_integrator(prob, solver; sensealg, kwargs...)
  return integrator
end

CRC.@non_differentiable _get_integrator(::Any...)

function _solve_neuralode_generic(n::NeuralODE, x::AbstractArray, ps, st::NamedTuple,
                                  saveat)
  st_ = st.model
  function dudt(u, p, t)
    u_, st_ = Lux.apply(n.model, ArrayAndTime(u, t), p, st_)
    return get_array(u_)
  end

  prob = ODEProblem(dudt, x, n.tspan, ps)
  sol = solve(prob, n.solver; n.sensealg, n.maxiters, saveat, n.kwargs...)

  return sol, st_, dudt
end

function (n::NeuralODE{:none})(x, ps, st)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, [n.tspan[2]])
  return sol, (; model=st_, nfe=_get_destats(sol), reg_val=0.0f0, st.rng)
end

function (n::NeuralODE{:unbiased})(x, ps, st)
  rng = Lux.replicate(st.rng)
  (t0, t2) = n.tspan
  t1 = rand(rng, eltype(t2)) * (t2 - t0) + t0
  saveat = [t1, t2]
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, saveat)
  integrator = _get_integrator(sol, t1, dudt, n.tspan, ps, n.solver, n.sensealg;
                               n.kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps)
  nfe = sol.destats.nf + nf2

  return sol, (; model=st_, nfe, reg_val, rng)
end

function (n::NeuralODE{:biased})(x, ps, st)
  rng = Lux.replicate(st.rng)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, [])
  t1 = rand(rng, sol.t)
  integrator = _get_integrator(sol, t1, dudt, n.tspan, ps, n.solver, n.sensealg;
                               n.kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps)
  nfe = sol.destats.nf + nf2

  return sol, (; model=st_, nfe, reg_val, rng)
end

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
