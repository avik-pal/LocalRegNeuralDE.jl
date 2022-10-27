import DeepEquilibriumNetworks: AbstractDeepEquilibriumNetwork,
                                AbstractSkipDeepEquilibriumNetwork, ContinuousDEQSolver

const AbstractDEQ = Union{<:AbstractDeepEquilibriumNetwork,
                          <:AbstractSkipDeepEquilibriumNetwork}

struct RegularizedDEQ{R, D <: AbstractDEQ} <: AbstractExplicitContainerLayer{(:deq,)}
  deq::D
end

function RegularizedDEQ(deq::AbstractDEQ; regularize::Symbol=:unbiased)
  _check_valid_regularize(regularize)
  if regularize == :none
    @warn "`regularize` is set to `:none`. Returning the original DEQ."
    return deq
  end
  if !(deq.solver isa ContinuousDEQSolver)
    throw(ArgumentError("regularization for DEQs is only defined when using a continuous " *
                        "solver."))
  end
  return RegularizedDEQ{regularize, typeof(deq)}(deq)
end

_get_deq_ode_solver(m::RegularizedDEQ) = _get_deq_ode_solver(m.deq)
_get_deq_ode_solver(m::AbstractDEQ) = m.solver.alg

function _get_deq_ode_integrator(sol, t, solver; kwargs...)
  prob = remake(sol.prob; u0=sol(t), tspan=(t, typeof(t)(Inf)))
  return init(prob, solver; kwargs...)
end

CRC.@non_differentiable _get_deq_ode_integrator(::Any...)

function Lux.initialstates(rng::AbstractRNG, layer::RegularizedDEQ)
  randn(rng)
  return (deq=Lux.initialstates(rng, layer.deq), nfe=-1, reg_val=0.0f0,
          rng=Lux.replicate(rng), training=Val(true))
end

(n::RegularizedDEQ)(x, ps, st) = n(x, ps, st, st.training, st.deq.fixed_depth)

# Don't compute regularization while inference
function (n::RegularizedDEQ)(x, ps, st, ::Val{false}, ::Val)
  y, st_ = n.deq(x, ps, st.deq)
  return y, (; deq=st_, nfe=st_.solution.nfe, reg_val=zero(eltype(x)), st.rng, st.training)
end

# Regularization is undefined if using fixed depth mode
function (n::RegularizedDEQ{:unbiased})(x, ps, st, ::Val{true}, d::Val{D}) where {D}
  D != 0 && return n(x, ps, st, Val(false), d)

  rng = Lux.replicate(st.rng)
  y, st_ = n.deq(x, ps, st.deq)

  # Only regularize from 0 to 1. We don't know more information for unbiased regularization
  (t0, t2) = (eltype(x)(0), eltype(x)(1))
  t1 = rand(rng, eltype(t2)) * (t2 - t0) + t0

  integrator = _get_deq_ode_integrator(st_.solution.sol.sol, t1, _get_deq_ode_solver(n))
  (_, reg_val, nf2, _) = _perform_step(integrator, integrator.cache, ps)

  return y, (; deq=st_, nfe=st_.solution.nfe + nf2, reg_val, rng, st.training)
end
