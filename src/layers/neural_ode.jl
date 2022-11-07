struct NeuralODE{R, RT, M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
       AbstractExplicitContainerLayer{(:model,)}
  model::M
  solver::So
  sensealg::Se
  tspan::T
  maxiters::Int
  kwargs::K

  function NeuralODE(model::Lux.AbstractExplicitLayer; solver=Tsit5(),
                     sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                     tspan=(0.0f0, 1.0f0), regularize::Union{Bool, Symbol}=true,
                     maxiters::Int=1000, regularize_type::Symbol=:error_estimate, kwargs...)
    if regularize isa Bool
      regularize = regularize ? :unbiased : :none
    end
    _check_valid_regularize(regularize)
    _check_valid_regularize(regularize_type, (:error_estimate, :stiffness_estimate))
    return new{regularize, regularize_type, typeof(model), typeof(solver), typeof(sensealg),
               typeof(tspan), typeof(kwargs)}(model, solver, sensealg, tspan, maxiters,
                                              kwargs)
  end
end

_get_regularize_type(::NeuralODE{R, RT}) where {R, RT} = Val(RT)

function Lux.initialstates(rng::AbstractRNG, layer::NeuralODE)
  randn(rng)
  return (model=Lux.initialstates(rng, layer.model), nfe=-1, reg_val=0.0f0,
          rng=Lux.replicate(rng), training=Val(true))
end

function _get_ode_integrator(sol, t, dudt, tspan, ps, solver, sensealg; kwargs...)
  u = sol(t)
  prob = ODEProblem(dudt, u, tspan, ps)
  integrator = _create_integrator(prob, solver; sensealg, kwargs...)
  return integrator
end

CRC.@non_differentiable _get_ode_integrator(::Any...)

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
  integrator = _get_ode_integrator(sol, t1, dudt, (t1, t2), ps, Tsit5(), n.sensealg;
                                   kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps,
                                       _get_regularize_type(n))
  nfe = sol.destats.nf + nf2

  sol = needs_correction ? _CorrectedDESolution(sol, n.kwargs[:saveat], t1) : sol

  return sol, (; model=st_, nfe, reg_val, rng, st.training)
end

(n::NeuralODE{:biased})(x, ps, st, ::Val{false}) = _vanilla_node_fallback(n, x, ps, st)

function (n::NeuralODE{:biased})(x, ps, st, ::Val{true})
  saveat, kwargs = _resolve_saveat_kwargs(Val(:biased), n.tspan; n.kwargs...)
  (sol, st_, dudt) = _solve_neuralode_generic(n, x, ps, st, saveat; kwargs...)
  rng = Lux.replicate(st.rng)
  t1 = rand(rng, sol.t)
  integrator = _get_ode_integrator(sol, t1, dudt, (t1, n.tspan[2]), ps, Tsit5(), n.sensealg;
                                   kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps,
                                       _get_regularize_type(n))
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