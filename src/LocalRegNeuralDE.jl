module LocalRegNeuralDE

using DiffEqBase, DiffEqCallbacks, ChainRulesCore, CUDA, ComponentArrays, Lux,
      OrdinaryDiffEq, Random, SciMLSensitivity, Setfield, UnPack, Zygote
import ChainRulesCore as CRC

_basic_tgrad(u, p, t) = zero(u)

_eltype(x::Number) = typeof(x)
_eltype(x::AbstractArray{T}) where {T} = T
_eltype(x::Tuple) = _eltype(first(x))

diffeqsol_to_array(sol::ODESolution) = sol.u[end]
diffeqsol_to_array(x::AbstractArray) = x

# TODO(@avik-pal): Upstream
Base.similar(ca::ComponentArray, l::Int64) = similar(getdata(ca), l)

function perform_step(integrator, cache::OrdinaryDiffEq.Tsit5ConstantCache, p)
  @unpack t, dt, uprev, u, f = integrator
  @unpack c1, c2, c3, c4, c5, c6, a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65, a71, a72, a73, a74, a75, a76, btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = cache
  k1 = integrator.fsalfirst
  a = dt * a21
  k2 = f(uprev + a * k1, p, t + c1 * dt)
  k3 = f(uprev + dt * (a31 * k1 + a32 * k2), p, t + c2 * dt)
  k4 = f(uprev + dt * (a41 * k1 + a42 * k2 + a43 * k3), p, t + c3 * dt)
  k5 = f(uprev + dt * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4), p, t + c4 * dt)
  g6 = uprev + dt * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5)
  k6 = f(g6, p, t + dt)
  u = uprev + dt * (a71 * k1 + a72 * k2 + a73 * k3 + a74 * k4 + a75 * k5 + a76 * k6)
  integrator.fsallast = f(u, p, t + dt)
  k7 = integrator.fsallast
  integrator.destats.nf += 6
  if integrator.opts.adaptive
    utilde = dt * (btilde1 * k1 +
              btilde2 * k2 +
              btilde3 * k3 +
              btilde4 * k4 +
              btilde5 * k5 +
              btilde6 * k6 +
              btilde7 * k7)
    atmp = DiffEqBase.calculate_residuals(utilde, uprev, u, integrator.opts.abstol,
                                          integrator.opts.reltol,
                                          integrator.opts.internalnorm, t)
    integrator.EEst = sum(abs2, atmp)
  end
  return u, t + dt
end

@inline function DiffEqBase.calculate_residuals(ũ, u₀, u₁, α, ρ, internalnorm, t)
  return ũ ./ (α .+ max.(abs.(u₀), abs.(u₁)) .* ρ)
end

@inline function DiffEqBase.calculate_residuals(ũ::Array{T}, u₀::Array{T}, u₁::Array{T},
                                                α::T2, ρ::Real, internalnorm,
                                                t) where {T <: Number, T2 <: Number}
  return ũ ./ (α .+ max.(abs.(u₀), abs.(u₁)) .* ρ)
end

function _create_integrator(args...; kwargs...)
  return init(args...; kwargs...)
end

CRC.@non_differentiable _create_integrator(::Any...)
CRC.@non_differentiable OrdinaryDiffEq.check_error!(::Any...)

function _check_valid_regularize(regularize)
  VALID_MODES = (:none, :split_mode)
  if !(regularize in VALID_MODES)
    throw(ArgumentError("regularize must be one of $VALID_MODES"))
  end
  return
end

struct NeuralODE{R, M <: Lux.AbstractExplicitLayer, So, Se, T, K} <:
       Lux.AbstractExplicitContainerLayer{(:model,)}
  model::M
  solver::So
  sensealg::Se
  tspan::T
  maxiters::Int
  kwargs::K

  function NeuralODE(model::Lux.AbstractExplicitLayer; solver=Tsit5(),
                     sensealg=InterpolatingAdjoint(; autojacvec=ZygoteVJP()),
                     tspan=(0.0f0, 1.0f0), regularize::Union{Bool, Symbol}=true,
                     maxiters::Int=1000, kwargs...)
    if regularize isa Bool
      regularize = regularize ? :split_mode : :none
    end
    _check_valid_regularize(regularize)
    return new{regularize, typeof(model), typeof(solver), typeof(sensealg), typeof(tspan),
               typeof(kwargs)}(model, solver, sensealg, tspan, maxiters, kwargs)
  end
end

function Lux.initialstates(rng::Random.AbstractRNG, layer::NeuralODE)
  randn(rng)
  return (model=Lux.initialstates(rng, layer.model), nfe=-1, reg_val=0.0f0,
          rng=Lux.replicate(rng))
end

function (n::NeuralODE{:none})(x, ps, st)
  st_ = st.model
  function dudt(u, p, t)
    (u_, _), st_ = n.model((u, t), p, st_)
    return u_
  end

  prob = ODEProblem{false}(ODEFunction{false}(dudt), x, n.tspan, ps)
  sol = solve(prob, n.solver; n.sensealg, n.maxiters, n.kwargs...)
  st = (model=st_, nfe=sol.destats.nf, reg_val=zero(_eltype(x)), rng=st.rng)

  return sol, st
end

function (n::NeuralODE{:split_mode})(x, ps, st)
  rng = Lux.replicate(st.rng)

  st_ = st.model
  function dudt(u, p, t)
    (u_, _), st_ = n.model((u, t), p, st_)
    return u_
  end

  t0, t2 = n.tspan
  t1 = rand(rng, eltype(t2)) * (t2 / oftype(t2, 1.3) - t0) + t0

  prob = ODEProblem{false}(ODEFunction{false}(dudt; tgrad=_basic_tgrad), x, (t0, t1), ps)
  sol_t1 = solve(prob, n.solver; n.sensealg, n.kwargs...)

  kwargs = n.kwargs
  if :maxiters in keys(kwargs)
    maxiters = kwargs[:maxiters]
    @set! kwargs[:maxiters] = kwargs[:maxiters] -
                              (sol_t1.destats.naccept + sol_t1.destats.nreject + 1)
  end

  u0 = diffeqsol_to_array(sol_t1)
  prob = ODEProblem{false}(ODEFunction{false}(dudt; tgrad=_basic_tgrad), x, (t1, t2), ps)
  integrator = _create_integrator(prob, n.solver; n.sensealg, kwargs...)
  u0, t1_ = perform_step(integrator, integrator.cache, ps)
  reg_val = integrator.EEst * integrator.dt

  prob = ODEProblem{false}(ODEFunction{false}(dudt; tgrad=_basic_tgrad), x, (t1_, t2), ps)
  sol = solve(prob, n.solver; n.sensealg, kwargs...)

  st = (; model=st_, nfe=(sol_t1.destats.nf + sol.destats.nf + integrator.destats.nf),
        reg_val, st.rng)

  return sol, st
end

export NeuralODE, diffeqsol_to_array

end