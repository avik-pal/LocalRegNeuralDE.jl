struct LatentGRUCell{U <: AbstractExplicitLayer, R <: AbstractExplicitLayer,
                     S <: AbstractExplicitLayer} <:
       AbstractExplicitContainerLayer{(:update_gate, :reset_gate, :new_state)}
  update_gate::U
  reset_gate::R
  new_state::S
  latent_dim::Int
end

function LatentGRUCell(in_dim::Int, h_dim::Int, latent_dim::Int)
  _in_dim = latent_dim * 2 + in_dim * 2 + 1
  update_gate = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim, sigmoid))
  reset_gate = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim, sigmoid))
  new_state = Chain(Dense(_in_dim => h_dim, tanh), Dense(h_dim => latent_dim * 2, tanh))

  return LatentGRUCell(update_gate, reset_gate, new_state, latent_dim)
end

function (l::LatentGRUCell)(x::A, ps, st::NamedTuple) where {A <: AbstractMatrix}
  y_mean = zeros_like(x, (l.latent_dim, size(x, 2)))
  y_std = ones_like(x, (l.latent_dim, size(x, 2)))
  return l((x, (y_mean, y_std)), ps, st)
end

@views function (l::LatentGRUCell)((x, (y_mean, y_std)), ps, st)
  y_concat = vcat(y_mean, y_std, x)

  update_gate, st_ug = l.update_gate(y_concat, ps.update_gate, st.update_gate)
  reset_gate, st_rg = l.reset_gate(y_concat, ps.reset_gate, st.reset_gate)

  concat = vcat(y_mean .* reset_gate, y_std .* reset_gate, x)

  new_state, st_ns = l.new_state(concat, ps.new_state, st.new_state)
  new_state_mean = new_state[1:(l.latent_dim), :]
  new_state_std = new_state[(l.latent_dim + 1):end, :]

  new_y_mean = (1 .- update_gate) .* new_state_std .+ update_gate .* y_mean
  new_y_std = (1 .- update_gate) .* new_state_std .+ update_gate .* y_std

  mask = sum(x[(size(x, 1) ÷ 2 + 1):end, :]; dims=1) .> 0

  new_y_mean = mask .* new_y_mean .+ (1 .- mask) .* y_mean
  new_y_std = mask .* new_y_std .+ (1 .- mask) .* y_std

  y = vcat(new_y_mean, new_y_std)
  return (y, (new_y_mean, new_y_std)),
         (; update_gate=st_ug, reset_gate=st_rg, new_state=st_ns)
end
