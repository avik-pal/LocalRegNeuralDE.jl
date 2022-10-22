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

# Augmentation Layer
struct AugmenterLayer{idx, A <: AbstractExplicitLayer} <:
       AbstractExplicitContainerLayer{(:augment,)}
  augment::A
end

function AugmenterLayer(augment::A, idx) where {A <: AbstractExplicitLayer}
  return AugmenterLayer{idx, A}(augment)
end

function (a::AugmenterLayer{idx})(x, ps, st) where {idx}
  x_, st_ = Lux.apply(a.augment, x, ps, st)
  return _cat(x, x_, Val(idx)), st_
end
