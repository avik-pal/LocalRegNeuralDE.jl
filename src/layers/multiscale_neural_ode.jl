# MultiScale Neural ODE with Input Injection
struct MultiScaleNeuralODE{R, N, Sc, So, T, M <: AbstractExplicitLayer, Se, K, Sp} <:
       AbstractExplicitContainerLayer{(:model,)}
  model::M
  solver::So
  sensealg::Se
  tspan::T
  maxiters::Int
  scales::Sc
  kwargs::K
  split_idxs::Sp

  function MultiScaleNeuralODE(main_layers::Tuple, mapping_layers::Matrix,
                               post_fuse_layer::Union{Nothing, Tuple},
                               scales::NTuple{N, NTuple{L, Int64}}; solver=VCAB3(),
                               sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                               tspan=(0.0f0, 1.0f0), regularize::Symbol=:unbiased,
                               maxiters::Int=1000, kwargs...) where {N, L}
    _check_valid_regularize(regularize)

    l1 = Parallel(nothing, main_layers...)
    l2 = BranchLayer(Lux.Parallel.(+, map(x -> tuple(x...), eachrow(mapping_layers))...)...)
    if post_fuse_layer === nothing
      model = Chain(l1, l2)
    else
      model = Chain(l1, l2, Parallel(nothing, post_fuse_layer...))
    end

    scales = static(scales)
    split_idxs = static(Tuple(vcat(0, cumsum(prod.(scales))...)))

    return new{regularize, N, typeof(scales), typeof(solver), typeof(tspan), typeof(model),
               typeof(sensealg), typeof(kwargs), typeof(split_idxs)}(model, solver,
                                                                     sensealg, tspan,
                                                                     maxiters, scales,
                                                                     kwargs, split_idxs)
  end
end

function Lux.initialstates(rng::AbstractRNG, node::MultiScaleNeuralODE)
  randn(rng)
  return (model=Lux.initialstates(rng, node.model), initial_condition=zeros(Float32, 1, 1),
          nfe=-1, reg_val=0.0f0, rng=Lux.replicate(rng), training=Val(true))
end

@generated function __get_initial_condition(::S, x::AbstractArray{T, N},
                                            st::NamedTuple{fields}) where {S, T, N, fields}
  scales = Static.known(S)
  sz = sum(prod.(scales))
  calls = []
  if :initial_condition ∈ fields
    push!(calls, :(u0 = st.initial_condition))
    push!(calls, :(($sz, size(x, $N)) == size(u0) && return u0, st))
  end
  push!(calls, :(u0 = zeros_like(x, ($(sz), size(x, N)))))
  push!(calls, :(st = merge(st, (initial_condition=u0,))))
  push!(calls, :(return u0, st))
  return Expr(:block, calls...)
end

CRC.@non_differentiable __get_initial_condition(::Any...)

function _get_initial_condition(s, x, st)
  u0, st_ = __get_initial_condition(s[2:end], x, st)
  return vcat(flatten(x), u0), st_
end

function _solve_multiscale_neuralode_generic(n::MultiScaleNeuralODE{R, N},
                                             u0::AbstractArray, x::AbstractArray, ps,
                                             st::NamedTuple, saveat; kwargs...) where {R, N}
  st_ = st.model
  function dudt_(u, p, t)
    u_split = split_and_reshape(u, n.split_idxs, n.scales)
    u_, st_ = Lux.apply(n.model, ((u_split[1], x), u_split[2:N]...), p, st_)
    return u_, st_
  end

  dudt(u, p, t) = mapreduce(flatten, vcat, first(dudt_(u, p, t)))

  prob = ODEProblem(dudt, u0, n.tspan, ps)
  sol = solve(prob, n.solver; n.sensealg, n.maxiters, saveat, kwargs...)

  return sol, st_, dudt, dudt_
end

function _vanilla_multiscale_node_fallback(n::MultiScaleNeuralODE, x, ps, st)
  saveat, kwargs = _resolve_saveat_kwargs(Val(:none), n.tspan; n.kwargs...)
  u0, st_ = _get_initial_condition(n.scales, x, st)
  (sol, st_, dudt, dudt_) = _solve_multiscale_neuralode_generic(n, u0, x, ps, st_, saveat;
                                                                kwargs...)

  y, st_ = dudt_(diffeqsol_to_array(sol), ps, n.tspan[2])

  return y, (; model=st_, nfe=_get_destats(sol), reg_val=0.0f0, st.rng, st.training)
end

(n::MultiScaleNeuralODE)(x, ps, st) = n(x, ps, st, st.training)

function (n::MultiScaleNeuralODE{:none})(x, ps, st, ::Val)
  return _vanilla_multiscale_node_fallback(n, x, ps, st)
end

function (n::MultiScaleNeuralODE{:unbiased})(x, ps, st, ::Val{false})
  return _vanilla_multiscale_node_fallback(n, x, ps, st)
end

function (n::MultiScaleNeuralODE{:unbiased})(x, ps, st, ::Val{true})
  rng = Lux.replicate(st.rng)
  (t0, t2) = n.tspan
  t1 = rand(rng, eltype(t2)) * (t2 - t0) + t0
  saveat, kwargs, needs_correction = _resolve_saveat_kwargs(Val(:unbiased), n.tspan, t1;
                                                            n.kwargs...)

  u0, st_ = _get_initial_condition(n.scales, x, st)
  (sol, st_, dudt, dudt_) = _solve_multiscale_neuralode_generic(n, u0, x, ps, st_, saveat;
                                                                kwargs...)

  integrator = _get_ode_integrator(sol, t1, dudt, (t1, t2), ps, Tsit5(), n.sensealg;
                                   kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps,
                                       Val(:error_estimate))

  nfe = sol.destats.nf + nf2

  y, st_ = dudt_(diffeqsol_to_array(sol), ps, n.tspan[2])

  return y, (; model=st_, nfe, reg_val, rng, st.training)
end

function (n::MultiScaleNeuralODE{:biased})(x, ps, st, ::Val{false})
  return _vanilla_multiscale_node_fallback(n, x, ps, st)
end

function (n::MultiScaleNeuralODE{:biased})(x, ps, st, ::Val{true})
  _, t2 = n.tspan
  rng = Lux.replicate(st.rng)
  saveat, kwargs = _resolve_saveat_kwargs(Val(:biased), n.tspan; n.kwargs...)

  u0, st_ = _get_initial_condition(n.scales, x, st)
  (sol, st_, dudt, dudt_) = _solve_multiscale_neuralode_generic(n, u0, x, ps, st_, saveat;
                                                                kwargs...)
  t1 = rand(rng, sol.t[1:(end - 1)])

  integrator = _get_ode_integrator(sol, t1, dudt, (t1, t2), ps, Tsit5(), n.sensealg;
                                   kwargs...)
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps,
                                       Val(:error_estimate))

  nfe = sol.destats.nf + nf2

  y, st_ = dudt_(diffeqsol_to_array(sol), ps, n.tspan[2])

  return y, (; model=st_, nfe, reg_val, rng, st.training)
end
