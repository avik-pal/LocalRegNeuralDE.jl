struct TDChain{fake, L <: NamedTuple} <: Lux.AbstractExplicitContainerLayer{(:layers,)}
  layers::L

  function TDChain(c::Chain; fake::Bool=false)
    layers = c.layers
    return new{fake, typeof(layers)}(layers)
  end
end

is_fake(::TDChain{fake}) where {fake} = fake

CRC.@non_differentiable CUDA.ones(::Any...)

(c::TDChain)((x, t), ps, st) = applytdchain(c.layers, Val(is_fake(c)), x, t, ps, st)

_cat(x, y, ::Val{dims}) where {dims} = cat(x, y; dims)

@generated function applytdchain(layers::NamedTuple{fields}, ::Val{fake}, x::T, t, ps,
                                 st::NamedTuple{fields}) where {fake, fields, T}
  N = length(fields)
  x_symbols = vcat([:x], [gensym("x") for _ in 1:N])
  st_symbols = [gensym("st") for _ in 1:N]
  calls = []

  if !fake
    push!(calls, :(_size = size(x);
                   @set! _size[$(ndims(T) - 1)] = 1))
    if T <: CuArray
      push!(calls, :(_t = CUDA.ones($(eltype(T)), _size) .* t))
    else
      push!(calls, :(_t = ones($(eltype(T)), _size) .* t))
    end
  end

  _getinput(v) = !fake ? :(_cat($v, _t, Val($(ndims(T) - 1)))) : v
  append!(calls,
          [:(($(x_symbols[i + 1]), $(st_symbols[i])) = layers.$(fields[i])($(_getinput(x_symbols[i])),
                                                                           ps.$(fields[i]),
                                                                           st.$(fields[i])))
           for i in 1:N])
  push!(calls, :(st = NamedTuple{$fields}((($(Tuple(st_symbols)...),)))))

  _getoutput() = !fake ? :($(x_symbols[N + 1]), _t) : :($(x_symbols[N + 1]), t)

  push!(calls, :(return $(_getoutput()), st))
  return Expr(:block, calls...)
end
