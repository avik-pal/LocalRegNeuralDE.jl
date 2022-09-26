struct ExpDecay{T}
  λ₀::T
  λ₁::T
  k::T
  nsteps::Int

  function ExpDecay(λ₀::T, λ₁::T, nsteps::Int) where {T <: Real}
    k = log(λ₀ / λ₁) / nsteps
    return new{T}(λ₀, λ₁, k, nsteps)
  end
end

(e::ExpDecay)(t) = e.λ₀ * exp(-e.k * t)
