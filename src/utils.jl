_basic_tgrad(u, p, t) = zero(u)

_eltype(x::Number) = typeof(x)
_eltype(x::AbstractArray{T}) where {T} = T
_eltype(x::Tuple) = _eltype(first(x))

_get_destats(sol::ODESolution) = sol.destats.nf
_get_destats(sol::ODESolution, x::Symbol) = getproperty(sol.destats, x)
_get_destats(sol::TrackedArray) = _get_destats(Tracker.data(sol))

# Array and Time Container for Dispatch
struct ArrayAndTime{A <: AbstractArray, T <: Real}
  array::A
  scalar::T
end

get_array(x::ArrayAndTime) = x.array
get_scalar(x::ArrayAndTime) = x.scalar

function Lux.apply(l::Lux.AbstractExplicitLayer, x::ArrayAndTime, ps, st::NamedTuple)
  y, st_ = Lux.apply(l, get_array(x), ps, st)
  return ArrayAndTime(y, get_scalar(x)), st_
end

struct _CorrectedDESolution{U}
  u::U
end

Base.ndims(sol::_CorrectedDESolution) = ndims(first(sol.u)) + 1

function _CorrectedDESolution(sol::ODESolution, saveat, t1)
  return _CorrectedDESolution(sol.u[t1 .!= sol.t])
end

const _DIFFEQ_SOL_TYPES = Union{ODESolution, _CorrectedDESolution, RODESolution}

diffeqsol_to_array(sol::_DIFFEQ_SOL_TYPES) = sol.u[end]
diffeqsol_to_array(x::ArrayAndTime) = get_array(x)
diffeqsol_to_array(x::AbstractArray) = x
diffeqsol_to_array(x::TrackedArray) = selectdim(x, ndims(x), size(x, ndims(x)))

diffeqsol_to_timeseries(sol::_DIFFEQ_SOL_TYPES) = diffeqsol_to_timeseries(Array, sol)
function diffeqsol_to_timeseries(t::Type{Array}, sol::_DIFFEQ_SOL_TYPES)
  return _cat(unsqueeze.(sol.u; dims=ndims(sol) - 1), Val(ndims(sol) - 1))
end
diffeqsol_to_timeseries(::Type{Tuple}, sol::_DIFFEQ_SOL_TYPES) = Tuple(sol.u)

# TODO(@avik-pal): Upstream
Base.similar(ca::ComponentArray, l::Int64) = similar(getdata(ca), l)

_create_integrator(args...; kwargs...) = init(args...; kwargs...)

function _check_valid_regularize(regularize, valid_modes=(:none, :unbiased, :biased))
  if !(regularize in valid_modes)
    throw(ArgumentError("regularize must be one of $valid_modes"))
  end
  return
end

CRC.@non_differentiable _create_integrator(::Any...)
CRC.@non_differentiable OrdinaryDiffEq.check_error!(::Any...)
CRC.@non_differentiable CUDA.ones(::Any...)

@inline _cat(x, y, ::Val{dims}) where {dims} = cat(x, y; dims)
@inline function _cat(t::Union{<:Tuple, <:AbstractVector}, ::Val{dims}) where {dims}
  return cat(t...; dims)
end

# This is a weird type piracy but is needed to make the NeuralDSDE to work
function LinearAlgebra.mul!(A::AbstractVector{T}, B::AbstractMatrix{T},
                            C::AbstractMatrix{T}, b1::Bool, b2::Bool) where {T}
  LinearAlgebra.mul!(reshape(A, :, 1), B, reshape(C, :, 1), b1, b2)
  return A
end

# FastBroadcast + Zygote.jl
@inline function FastBroadcast.fast_materialize(::SB, ::DB, bc) where {SB, DB}
  return Base.Broadcast.materialize(bc)
end
