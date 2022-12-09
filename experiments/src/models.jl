function normal_initializer(rng::AbstractRNG, dims...; mean::T=0.0f0,
                            std::T=0.01f0) where {T}
  return randn(rng, T, dims...) .* std .+ mean
end

addrelu(x, y) = NNlib.relu.(x .+ y)

reassociate((x1, x2), y) = (x1, (x2, y))

addtuple((x, y)) = x .+ y

function compute_feature_scales(; image_size, downsample_times, num_channels, num_branches)
  image_size_downsampled = div.(image_size, 2^downsample_times)
  scales = [(image_size_downsampled..., num_channels[1])]
  for i in 2:num_branches
    push!(scales, ((scales[end][1:2] .÷ 2)..., num_channels[i]))
  end
  return Tuple(scales)
end

# Building Blocks
function conv1x1(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1, weight_norm=false)
  c = Conv((1, 1), mapping, activation; pad=0, init_weight=normal_initializer, stride,
           use_bias, dilation, groups)
  weight_norm || return c
  return WeightNorm(c, (:weight,), (4,))
end

function conv3x3(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1, weight_norm=false)
  c = Conv((3, 3), mapping, activation; pad=1, init_weight=normal_initializer, stride,
           use_bias, dilation, groups)
  weight_norm || return c
  return WeightNorm(c, (:weight,), (4,))
end

function conv5x5(mapping, activation=identity; stride::Int=1, use_bias=false, dilation=1,
                 groups=1, weight_norm=false)
  c = Conv((5, 5), mapping, activation; pad=2, init_weight=normal_initializer, stride,
           use_bias, dilation, groups)
  weight_norm || return c
  return WeightNorm(c, (:weight,), (4,))
end

function downsample_module(mapping, level_difference, activation; group_count=8)
  in_channels, out_channels = mapping

  function intermediate_mapping(i)
    if in_channels * (2^level_difference) == out_channels
      return (in_channels * (2^(i - 1))) => (in_channels * (2^i))
    else
      return i == level_difference ? in_channels => out_channels :
             in_channels => in_channels
    end
  end

  layers = Lux.AbstractExplicitLayer[]
  for i in 1:level_difference
    in_chs, out_chs = intermediate_mapping(i)
    push!(layers,
          Chain(conv3x3(in_chs => out_chs; stride=2),
                BatchNorm(out_chs, activation; affine=true, track_stats=false)))
  end
  return Chain(layers...; disable_optimizations=true)
end

function upsample_module(mapping, level_difference, activation; group_count=8,
                         upsample_mode::Symbol=:nearest)
  in_channels, out_channels = mapping

  return Chain(conv3x3(in_channels => out_channels),
               BatchNorm(out_channels, activation; affine=true, track_stats=false),
               Upsample(upsample_mode; scale=2^level_difference))
end

struct ResidualBlock{C1, C2, Dr, Do, N1, N2, N3} <:
       Lux.AbstractExplicitContainerLayer{(:conv1, :conv2, :dropout, :downsample, :norm1,
                                           :norm2, :norm3)}
  conv1::C1
  conv2::C2
  dropout::Dr
  downsample::Do
  norm1::N1
  norm2::N2
  norm3::N3
end

function ResidualBlock(mapping; deq_expand::Int=3, num_gn_groups::Int=4,
                       downsample=NoOpLayer(), n_big_kernels::Int=0,
                       dropout_rate::Real=0.0f0, gn_affine::Bool=true,
                       weight_norm::Bool=true)
  in_planes, out_planes = mapping
  inner_planes = out_planes * deq_expand
  conv1 = (n_big_kernels >= 1 ? conv5x5 : conv3x3)(in_planes => inner_planes;
                                                   use_bias=false)
  conv2 = (n_big_kernels >= 2 ? conv5x5 : conv3x3)(inner_planes => out_planes;
                                                   use_bias=false)

  if weight_norm
    conv1 = WeightNorm(conv1, (:weight,), (4,))
    conv2 = WeightNorm(conv2, (:weight,), (4,))
  end

  norm1 = BatchNorm(inner_planes, relu; affine=gn_affine, track_stats=false)
  norm2 = BatchNorm(out_planes; affine=gn_affine, track_stats=false)
  norm3 = BatchNorm(out_planes; affine=gn_affine, track_stats=false)

  dropout = VariationalHiddenDropout(dropout_rate)

  return ResidualBlock(conv1, conv2, dropout, downsample, norm1, norm2, norm3)
end

function (rb::ResidualBlock)((x, y)::Tuple, ps, st)
  x, st_conv1 = rb.conv1(x, ps.conv1, st.conv1)
  x, st_norm1 = rb.norm1(x, ps.norm1, st.norm1)
  x, st_conv2 = rb.conv2(x, ps.conv2, st.conv2)

  x_do, st_downsample = rb.downsample(x, ps.downsample, st.downsample)
  x_dr, st_dropout = rb.dropout(x, ps.dropout, st.dropout)

  y = x_dr .+ y
  y, st_norm2 = rb.norm2(y, ps.norm2, st.norm2)

  y = addrelu(y, x_do)
  y, st_norm3 = rb.norm3(y, ps.norm3, st.norm3)

  return (y,
          (conv1=st_conv1, conv2=st_conv2, dropout=st_dropout, downsample=st_downsample,
           norm1=st_norm1, norm2=st_norm2, norm3=st_norm3))
end

(rb::ResidualBlock)(x::AbstractArray, ps, st) = rb((x, eltype(x)(0)), ps, st)

struct BottleneckBlock{R, C, M} <:
       Lux.AbstractExplicitContainerLayer{(:rescale, :conv, :mapping)}
  rescale::R
  conv::C
  mapping::M
end

function BottleneckBlock(mapping::Pair; expansion::Int=4, bn_track_stats::Bool=true,
                         bn_affine::Bool=true)
  rescale = if first(mapping) != last(mapping) * expansion
    Chain(conv1x1(first(mapping) => last(mapping) * expansion),
          BatchNorm(last(mapping) * expansion; affine=bn_affine,
                    track_stats=bn_track_stats))
  else
    NoOpLayer()
  end

  return BottleneckBlock(rescale, conv1x1(mapping),
                         Chain(BatchNorm(last(mapping), relu; affine=bn_affine,
                                         track_stats=bn_track_stats),
                               conv3x3(last(mapping) => last(mapping)),
                               BatchNorm(last(mapping), relu; affine=bn_affine,
                                         track_stats=bn_track_stats),
                               conv1x1(last(mapping) => last(mapping) * expansion),
                               BatchNorm(last(mapping) * expansion; affine=bn_affine,
                                         track_stats=bn_track_stats);
                               disable_optimizations=true))
end

function (bn::BottleneckBlock)((x, y)::Tuple, ps, st)
  x_r, st_rescale = bn.rescale(x, ps.rescale, st.rescale)
  x_m, st_conv1 = bn.conv(x, ps.conv, st.conv)
  x_m, st_mapping = bn.mapping(y .+ x_m, ps.mapping, st.mapping)

  return (addrelu(x_m, x_r), (rescale=st_rescale, conv=st_conv1, mapping=st_mapping))
end

(bn::BottleneckBlock)(x::AbstractArray, ps, st) = bn((x, eltype(x)(0)), ps, st)

function get_model(; num_channels, downsample_times, num_branches, expansion_factor,
                   dropout_rate, group_count, big_kernels, head_channels, fuse_method,
                   final_channelsize, num_classes, model_type, solver, sensealg, maxiters,
                   image_size, weight_norm, in_channels, abstol, reltol)
  init_channel_size = first(num_channels)

  downsample_layers = Lux.AbstractExplicitLayer[conv3x3(in_channels => init_channel_size;
                                                        stride=(downsample_times >= 1 ? 2 :
                                                                1)),
                                                BatchNorm(init_channel_size, relu;
                                                          affine=true, track_stats=false),
                                                conv3x3(init_channel_size => init_channel_size;
                                                        stride=(downsample_times >= 2 ? 2 :
                                                                1)),
                                                BatchNorm(init_channel_size, relu;
                                                          affine=true, track_stats=false)]
  for i in 3:downsample_times
    push!(downsample_layers,
          Chain(conv3x3(init_channel_size => init_channel_size; stride=2),
                BatchNorm(init_channel_size, NNlib.relu; affine=true, track_stats=false)))
  end
  downsample = Chain(downsample_layers...; disable_optimizations=true)

  if downsample_times == 0 && num_branches <= 2
    stage0 = NoOpLayer()
  else
    stage0 = Chain(conv1x1(init_channel_size => init_channel_size),
                   BatchNorm(init_channel_size, relu; affine=true, track_stats=false))
  end

  initial_layers = Chain(downsample, stage0; disable_optimizations=true)

  main_layers = Tuple(ResidualBlock(num_channels[i] => num_channels[i];
                                    deq_expand=expansion_factor, dropout_rate,
                                    num_gn_groups=group_count, n_big_kernels=big_kernels[i],
                                    weight_norm) for i in 1:(num_branches))

  mapping_layers = Matrix{Lux.AbstractExplicitLayer}(undef, num_branches, num_branches)
  for i in 1:num_branches, j in 1:num_branches
    if i == j
      mapping_layers[i, j] = NoOpLayer()
    elseif i < j
      mapping_layers[i, j] = downsample_module(num_channels[i] => num_channels[j], j - i,
                                               relu; group_count)
    else
      mapping_layers[i, j] = upsample_module(num_channels[i] => num_channels[j], i - j,
                                             relu; group_count, upsample_mode=:nearest)
    end
  end

  post_fuse_layers = Tuple(Chain(conv1x1(num_channels[i] => num_channels[i], relu;
                                         weight_norm),
                                 BatchNorm(num_channels[i]; affine=true, track_stats=false))
                           for i in 1:num_branches)

  increment_modules = Parallel(nothing,
                               [BottleneckBlock(num_channels[i] => head_channels[i];
                                                expansion=4, bn_track_stats=true,
                                                bn_affine=true) for i in 1:num_branches]...)

  downsample_modules = [Chain(conv3x3(head_channels[i] * 4 => head_channels[i + 1] * 4;
                                      stride=2, use_bias=true),
                              BatchNorm(head_channels[i + 1] * 4, relu; affine=true,
                                        track_stats=false)) for i in 1:(num_branches - 1)]
  downsample_modules = PairwiseFusion(fuse_method == "sum" ? (+) :
                                      throw(ArgumentError("unknown `fuse_method` = $(fuse_method)")),
                                      downsample_modules...)

  final_layers = Chain(increment_modules, downsample_modules,
                       conv1x1(head_channels[num_branches] * 4 => final_channelsize;
                               use_bias=true),
                       BatchNorm(final_channelsize, relu; affine=true, track_stats=false),
                       GlobalMeanPool(), FlattenLayer(),
                       Dense(final_channelsize => num_classes); disable_optimizations=true)

  scales = compute_feature_scales(; image_size, downsample_times, num_channels,
                                  num_branches)

  neural_ode = MultiScaleNeuralODE(main_layers, mapping_layers, post_fuse_layers, scales;
                                   solver, maxiters, sensealg, verbose=false, abstol,
                                   reltol)

  return Lux.Chain(; initial_layers, neural_ode, final_layers)
end
