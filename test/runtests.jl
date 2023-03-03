using LocalRegNeuralDE
using Test, TestItemRunner

@testitem "NeuralODE: No Regularization | Time Dependent" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(TDChain(; d1=Dense(3 => 4, gelu), d2=Dense(5 => 2));
                                     regularize=:none, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralODE: No Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2));
                                     regularize=:none, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralODE: Unbiased Regularization | Time Dependent" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(TDChain(; d1=Dense(3 => 4, gelu), d2=Dense(5 => 2));
                                     regularize=:unbiased, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test !iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))

    gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

    @test gs_x === nothing
    @test all(isfinite, getdata(gs_ps))
    @test any(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralODE: Unbiased Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2));
                                     regularize=:unbiased, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test !iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))

    gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

    @test gs_x === nothing
    @test all(isfinite, getdata(gs_ps))
    @test any(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralODE: Biased Regularization | Time Dependent" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(TDChain(; d1=Dense(3 => 4, gelu), d2=Dense(5 => 2));
                                     regularize=:biased, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test !iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))

    gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

    @test gs_x === nothing
    @test all(isfinite, getdata(gs_ps))
    @test any(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralODE: Biased Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  use_cuda = parse(Bool, get(ENV, "USE_CUDA", "false"))

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_ode=NeuralODE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2));
                                     regularize=:biased, tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_ode.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))

  if use_cuda && CUDA.functional(true)
    CUDA.allowscalar(false)

    x = x |> gpu
    ps = ps |> gpu
    st = st |> gpu

    y, st_ = model(x, ps, st)

    @test y isa CuArray{Float32, 2}
    @test !iszero(st_.neural_ode.reg_val)

    gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

    @test all(isfinite, gs_x)
    @test all(!iszero, gs_x)
    @test all(isfinite, getdata(gs_ps))
    @test all(!iszero, getdata(gs_ps))

    gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_ode.reg_val, x, ps)

    @test gs_x === nothing
    @test all(isfinite, getdata(gs_ps))
    @test any(!iszero, getdata(gs_ps))
  end
end

@testitem "NeuralDSDE: No Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_sde=NeuralDSDE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2)),
                                      Dense(2 => 2); regularize=:none,
                                      tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test iszero(st_.neural_sde.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))
end

@testitem "NeuralDSDE: Unbiased Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_sde=NeuralDSDE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2)),
                                      Dense(2 => 2); regularize=:unbiased,
                                      tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_sde.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_sde.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))
end

@testitem "NeuralDSDE: Biased Regularization | No Time Dependence" begin
  using ComponentArrays, CUDA, Lux, NNlib, Random, Zygote

  model = Chain(; d1=Dense(2 => 2, gelu),
                neural_sde=NeuralDSDE(Chain(; d1=Dense(2 => 4, gelu), d2=Dense(4 => 2)),
                                      Dense(2 => 2); regularize=:biased,
                                      tspan=(0.0f0, 1.0f0)),
                to_arr=WrappedFunction(diffeqsol_to_array), d2=Dense(2 => 2))

  rng = MersenneTwister(0)
  x = randn(rng, Float32, 2, 1)
  ps, st = Lux.setup(rng, model)
  ps = ps |> ComponentArray

  y, st_ = model(x, ps, st)

  @test y isa Array{Float32, 2}
  @test !iszero(st_.neural_sde.reg_val)

  gs_x, gs_ps = gradient((x, ps) -> sum(model(x, ps, st)[1]), x, ps)

  @test all(isfinite, gs_x)
  @test all(!iszero, gs_x)
  @test all(isfinite, getdata(gs_ps))
  @test all(!iszero, getdata(gs_ps))

  gs_x, gs_ps = gradient((x, ps) -> model(x, ps, st)[2].neural_sde.reg_val, x, ps)

  @test gs_x === nothing
  @test all(isfinite, getdata(gs_ps))
  @test any(!iszero, getdata(gs_ps))
end

@run_package_tests
