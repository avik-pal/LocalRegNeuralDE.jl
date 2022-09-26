_basic_tgrad(u, p, t) = zero(u)

_eltype(x::Number) = typeof(x)
_eltype(x::AbstractArray{T}) where {T} = T
_eltype(x::Tuple) = _eltype(first(x))

# Array and Time Container for Dispatch
struct ArrayAndTime{A <: AbstractArray, T <: Real}
  array::A
  scalar::T
end

get_array(x::ArrayAndTime) = x.array
get_scalar(x::ArrayAndTime) = x.scalar

function diffeqsol_to_array(sol::ODESolution)
  return diffeqsol_to_array(_single_getindex(sol.u, length(sol.u)))
end
diffeqsol_to_array(x::ArrayAndTime) = get_array(x)
diffeqsol_to_array(x::AbstractArray) = x

_single_getindex(x::AbstractVector, i::Int) = x[i]

# TODO(@avik-pal): Upstream
Base.similar(ca::ComponentArray, l::Int64) = similar(getdata(ca), l)

_create_integrator(args...; kwargs...) = init(args...; kwargs...)

function _check_valid_regularize(regularize)
  VALID_MODES = (:none, :unbiased, :biased)
  if !(regularize in VALID_MODES)
    throw(ArgumentError("regularize must be one of $VALID_MODES"))
  end
  return
end

CRC.@non_differentiable _create_integrator(::Any...)
CRC.@non_differentiable OrdinaryDiffEq.check_error!(::Any...)
CRC.@non_differentiable CUDA.ones(::Any...)

_cat(x, y, ::Val{dims}) where {dims} = cat(x, y; dims)

function Lux.apply(l::Lux.AbstractExplicitLayer, x::ArrayAndTime, ps, st::NamedTuple)
  y, st_ = Lux.apply(l, get_array(x), ps, st)
  return ArrayAndTime(y, get_scalar(x)), st_
end