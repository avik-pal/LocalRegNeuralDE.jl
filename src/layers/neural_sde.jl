struct NeuralDSDE{R, Dr <: AbstractExplicitLayer, Di <: AbstractExplicitLayer, So, Se, T, K
                  } <: AbstractExplicitContainerLayer{(:drift, :diffusion)}
  drift::Dr
  diffusion::Di
  solver::So
  sensealg::Se
  tspan::T
  maxiters::Int
  kwargs::K

  function NeuralDSDE(drift::AbstractExplicitLayer, diffusion::AbstractExplicitLayer;
                      solver=SOSRI(),
                      sensealg=SensitivityADPassThrough(),
                      tspan=(0.0f0, 1.0f0),
                      regularize::Symbol=:unbiased, maxiters::Int=1000, kwargs...)
    _check_valid_regularize(regularize)
    return new{regularize, typeof(drift), typeof(diffusion), typeof(solver),
               typeof(sensealg), typeof(tspan), typeof(kwargs)}(drift, diffusion, solver,
                                                                sensealg, tspan, maxiters,
                                                                kwargs)
  end
end

function Lux.initialstates(rng::AbstractRNG, layer::NeuralDSDE)
  randn(rng)
  return (; drift=Lux.initialstates(rng, layer.drift),
          diffusion=Lux.initialstates(rng, layer.diffusion), nfe_drift=-1, nfe_diffusion=-1,
          reg_val=0.0f0, rng=Lux.replicate(rng), training=Val(true))
end

function _get_dsde_integrator(sol, t, dudt, g, tspan, ps, solver, sensealg; kwargs...)
  u = sol(t)
  prob = SDEProblem(dudt, g, u, tspan, ps)
  integrator = _create_integrator(prob, solver; sensealg, kwargs...)
  return integrator
end

CRC.@non_differentiable _get_dsde_integrator(::Any...)

function _update_nfe!(nfes::Vector, idx::Int64, val)
  nfes[idx] += val
end

CRC.@non_differentiable _update_nfe!(::Any...)

function _solve_neuraldsde_generic(n::NeuralDSDE, x::AbstractArray, ps, st::NamedTuple,
                                   saveat; kwargs...)
  nfes = [0, 0]
  st_1, st_2 = st.drift, st.diffusion

  function dudt(u, p, t)
    _update_nfe!(nfes, 1, 1)
    u_, st_1 = Lux.apply(n.drift, ArrayAndTime(u, t), p.drift, st_1)
    return get_array(u_)
  end

  function g(u, p, t)
    _update_nfe!(nfes, 2, 1)
    u_, st_2 = Lux.apply(n.diffusion, ArrayAndTime(u, t), p.diffusion, st_2)
    return get_array(u_)
  end

  prob = SDEProblem(dudt, g, x, n.tspan, ps)
  sol = solve(prob, n.solver; n.sensealg, n.maxiters, saveat, kwargs...)

  return sol, st_1, st_2, dudt, g, nfes
end

function _vanilla_ndsde_fallback(n::NeuralDSDE, x, ps, st)
  saveat, kwargs = _resolve_saveat_kwargs(Val(:none), n.tspan; n.kwargs...)
  (sol, st_1, st_2, _, _, nfes) = _solve_neuraldsde_generic(n, x, ps, st, saveat; kwargs...)
  return sol,
         (; drift=st_1, diffusion=st_2, nfe_drift=nfes[1], nfe_diffusion=nfes[2],
          reg_val=0.0f0, st.rng, st.training)
end

(n::NeuralDSDE)(x, ps, st) = n(x, ps, st, st.training)

(n::NeuralDSDE{:none})(x, ps, st, ::Val) = _vanilla_ndsde_fallback(n, x, ps, st)

(n::NeuralDSDE{:unbiased})(x, ps, st, ::Val{false}) = _vanilla_ndsde_fallback(n, x, ps, st)

function (n::NeuralDSDE{:unbiased})(x, ps, st, ::Val{true})
  rng = Lux.replicate(st.rng)
  (t0, t2) = n.tspan
  t1 = rand(rng, eltype(t2)) * (t2 - t0) + t0
  saveat, kwargs, needs_correction = _resolve_saveat_kwargs(Val(:unbiased), n.tspan, t1;
                                                            n.kwargs...)
  (sol, st_1, st_2, dudt, g, nfes) = _solve_neuraldsde_generic(n, x, ps, st, saveat;
                                                               kwargs...)
  integrator = _get_dsde_integrator(sol, t1, dudt, g, n.tspan, ps, n.solver, n.sensealg;
                                    kwargs...)
  (_, reg_val, _) = _perform_step(integrator, integrator.cache, ps)

  sol = needs_correction ? _CorrectedDESolution(sol, n.kwargs[:saveat], t1) : sol

  return sol,
         (; drift=st_1, diffusion=st_2, nfe_drift=nfes[1], nfe_diffusion=nfes[2], reg_val,
          rng, st.training)
end

(n::NeuralDSDE{:biased})(x, ps, st, ::Val{false}) = _vanilla_ndsde_fallback(n, x, ps, st)

function (n::NeuralDSDE{:biased})(x, ps, st, ::Val{true})
  saveat, kwargs = _resolve_saveat_kwargs(Val(:biased), n.tspan; n.kwargs...)
  (sol, st_1, st_2, dudt, g, nfes) = _solve_neuraldsde_generic(n, x, ps, st, saveat;
                                                               kwargs...)
  rng = Lux.replicate(st.rng)
  t1 = rand(rng, sol.t)
  integrator = _get_dsde_integrator(sol, t1, dudt, g, n.tspan, ps, n.solver, n.sensealg;
                                    kwargs...)
  (_, reg_val, _) = _perform_step(integrator, integrator.cache, ps)

  return sol,
         (; drift=st_1, diffusion=st_2, nfe_drift=nfes[1], nfe_diffusion=nfes[2], reg_val,
          rng, st.training)
end
