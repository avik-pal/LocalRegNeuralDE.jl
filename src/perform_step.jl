function _perform_step(integrator, cache::Tsit5ConstantCache, p, reg_type::Val)
  @unpack t, dt, uprev, u, f = integrator
  @unpack c1, c2, c3, c4, c5, c6 = cache
  @unpack a21, a31, a32, a41, a42, a43, a51, a52, a53, a54, a61, a62, a63, a64, a65 = cache
  @unpack a71, a72, a73, a74, a75, a76 = cache
  @unpack btilde1, btilde2, btilde3, btilde4, btilde5, btilde6, btilde7 = cache
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
  utilde = dt * (btilde1 * k1 +
            btilde2 * k2 +
            btilde3 * k3 +
            btilde4 * k4 +
            btilde5 * k5 +
            btilde6 * k6 +
            btilde7 * k7)
  reg_val = _compute_regularization_value(cache, reg_type, utilde, uprev, u,
                                          integrator.opts.abstol, integrator.opts.reltol,
                                          k7, k6, u, g6, dt)
  return u, reg_val, 6 + integrator.sol.destats.nf, dt
end

function _compute_regularization_value(::Tsit5ConstantCache, ::Val{:error_estimate}, utilde,
                                       uprev, u, abstol, reltol, k7, k6, g7, g6, dt)
  return (sqrt(sum(abs2, _calculate_residuals(utilde, uprev, u, abstol, reltol)) /
               length(u)) * dt)
end

function _compute_regularization_value(::Tsit5ConstantCache, ::Val{:stiffness_estimate},
                                       utilde, uprev, u, abstol, reltol, k7, k6, g7, g6, dt)
  # NOTE: 3.5068f0 is OrdinaryDiffEq.alg_stability_size(Tsit5())
  return abs(sqrt(mean(abs2, k7 .- k6) ./ (mean(abs2, g7 .- g6) .+ eps(eltype(u))))) ./
         3.5068f0
end

function _perform_step(integrator, cache::FourStageSRIConstantCache, p)
  @unpack t, dt, uprev, u, W, f = integrator
  @unpack a021, a031, a032, a041, a042, a043, a121, a131, a132, a141, a142, a143 = cache
  @unpack b021, b031, b032, b041, b042, b043, b121, b131, b132, b141, b142, b143 = cache
  @unpack c02, c03, c04, c11, c12, c13, c14, α1, α2, α3, α4 = cache
  @unpack beta11, beta12, beta13, beta14, beta21, beta22, beta23, beta24, beta31 = cache
  @unpack beta32, beta33, beta34, beta41, beta42, beta43, beta44 = cache

  sqrt3 = sqrt(3one(eltype(W.dW)))
  chi1 = (W.dW .^ 2 .- abs(dt)) / (2integrator.sqdt) #I_(1,1)/sqrt(h)
  chi2 = (W.dW .+ W.dZ ./ sqrt3) ./ 2 #I_(1,0)/h
  chi3 = (W.dW .^ 3 .- 3 * W.dW * dt) / (6dt) #I_(1,1,1)/h

  k1 = integrator.f(uprev, p, t)
  g1 = integrator.g(uprev, p, t + c11 * dt)

  H01 = uprev + dt * a021 * k1 + b021 * chi2 .* g1
  H11 = uprev + dt * a121 * k1 + integrator.sqdt * b121 * g1

  k2 = integrator.f(H01, p, t + c02 * dt)
  g2 = integrator.g(H11, p, t + c12 * dt)

  H02 = uprev + dt * (a031 * k1 + a032 * k2) + chi2 .* (b031 * g1 + b032 * g2)
  H12 = uprev + dt * (a131 * k1 + a132 * k2) + integrator.sqdt * (b131 * g1 + b132 * g2)

  k3 = integrator.f(H02, p, t + c03 * dt)
  g3 = integrator.g(H12, p, t + c13 * dt)

  H03 = uprev +
        dt * (a041 * k1 + a042 * k2 + a043 * k3) +
        chi2 .* (b041 * g1 + b042 * g2 + b043 * g3)
  H13 = uprev +
        dt * (a141 * k1 + a142 * k2 + a143 * k3) +
        integrator.sqdt * (b141 * g1 + b142 * g2 + b143 * g3)

  k4 = integrator.f(H03, p, t + c04 * dt)
  g4 = integrator.g(H13, p, t + c14 * dt)

  E₂ = chi2 .* (beta31 * g1 + beta32 * g2 + beta33 * g3 + beta34 * g4) +
       chi3 .* (beta41 * g1 + beta42 * g2 + beta43 * g3 + beta44 * g4)

  u = uprev +
      dt * (α1 * k1 + α2 * k2 + α3 * k3 + α4 * k4) +
      E₂ +
      W.dW .* (beta11 * g1 + beta12 * g2 + beta13 * g3 + beta14 * g4) +
      chi1 .* (beta21 * g1 + beta22 * g2 + beta23 * g3 + beta24 * g4)

  E₁ = dt * (k1 + k2 + k3 + k4)

  E₁ = dt .* (k1 .+ k2 .+ k3 .+ k4)

  EEst = sqrt(sum(abs2,
                  _calculate_residuals(E₁, E₂, uprev, u, integrator.opts.abstol,
                                       integrator.opts.reltol, integrator.opts.delta)) /
              length(u))

  return u, EEst * dt, 0, dt
end

function _perform_step(integrator, cache::RKMilCommuteConstantCache, p)
  @unpack t, dt, uprev, u, W, f = integrator
  dW = W.dW
  sqdt = integrator.sqdt
  Jalg = cache.Jalg

  ggprime_norm = 0.0

  J = StochasticDiffEq.get_iterated_I(dt, dW, W.dZ, Jalg)

  mil_correction = zero(u)
  if StochasticDiffEq.alg_interpretation(integrator.alg) == :Ito
    if typeof(dW) <: Number || StochasticDiffEq.is_diagonal_noise(integrator.sol.prob)
      J = J .- 1 // 2 .* abs(dt)
    else
      J -= 1 // 2 .* UniformScaling(abs(dt))
    end
  end

  du1 = integrator.f(uprev, p, t)
  L = integrator.g(uprev, p, t)

  K = uprev + dt * du1

  if StochasticDiffEq.is_diagonal_noise(integrator.sol.prob)
    tmp = (StochasticDiffEq.alg_interpretation(integrator.alg) == :Ito ? K : uprev) .+
          integrator.sqdt .* L
    gtmp = integrator.g(tmp, p, t)
    Dgj = (gtmp - L) / sqdt
    ggprime_norm = integrator.opts.internalnorm(Dgj, t)
    u = @. K + L * dW + Dgj * J
  else
    for j in 1:length(dW)
      if typeof(dW) <: Number
        Kj = K + sqdt * L
      else
        Kj = K + sqdt * @view(L[:, j])
      end
      gtmp = integrator.g(Kj, p, t)
      Dgj = (gtmp - L) / sqdt
      if integrator.opts.adaptive
        ggprime_norm += integrator.opts.internalnorm(Dgj, t)
      end
      if typeof(dW) <: Number
        tmp = Dgj * J
      else
        tmp = Dgj * @view(J[:, j])
      end
      mil_correction += tmp
    end
    tmp = L * dW
    u = uprev + dt * du1 + tmp + mil_correction
  end
  En = integrator.opts.internalnorm(dW, t)^3 * ggprime_norm^2 / 6
  du2 = integrator.f(K, p, t + dt)
  tmp = integrator.opts.internalnorm(integrator.opts.delta * dt * (du2 - du1) / 2, t) + En

  tmp = _calculate_residuals(uprev, u, integrator.opts.abstol, integrator.opts.reltol)
  EEst = integrator.opts.internalnorm(tmp, t)

  return u, EEst * dt, 0, dt
end

@inline function _calculate_residuals(ũ, u₀, u₁, alpha, rho)
  return ũ ./ (alpha .+ max.(abs.(u₀), abs.(u₁)) .* rho)
end

@inline function _calculate_residuals(E₁, E₂, u₀, u₁, α, ρ, δ)
  return (δ .* E₁ .+ E₂) ./ (α .+ max.(abs.(u₀), abs.(u₁)) .* ρ)
end

@inline function _calculate_residuals(u₀, u₁, alpha, rho)
  return (u₁ .- u₀) ./ (alpha .+ max.(abs.(u₀), abs.(u₁)) .* rho)
end
