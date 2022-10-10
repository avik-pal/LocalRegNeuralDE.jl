import OrdinaryDiffEq: Tsit5ConstantCache

function _perform_step(integrator, cache::Tsit5ConstantCache, p)
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
  EEst = sqrt(sum(abs2,
                  _calculate_residuals(utilde, uprev, u, integrator.opts.abstol,
                                       integrator.opts.reltol)) / length(u))
  return u, EEst * dt, 6 + integrator.sol.destats.nf, dt
end

@inline function _calculate_residuals(ũ, u₀, u₁, alpha, rho)
  return ũ ./ (alpha .+ max.(abs.(u₀), abs.(u₁)) .* rho)
end

@inline function _calculate_residuals(ũ::Array{T}, u₀::Array{T}, u₁::Array{T},
                                      alpha::Number, rho::Real) where {T <: Number}
  return ũ ./ (alpha .+ max.(abs.(u₀), abs.(u₁)) .* rho)
end
