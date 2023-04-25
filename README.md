# LocalRegNeuralDE: Some Black Boxes Were Meant to Remain Closed!

Official Implementation of ICML 2023 Paper [Locally Regularized Neural Differential Equations: Some Black Boxes Were Meant to Remain Closed!](https://arxiv.org/pdf/2303.02262.pdf)

Extension of [RegNeuralDE](https://arxiv.org/abs/2105.03918) to allow for local
regularization.

This is a research repository. Most users should wait for the functionality to be available
through packages like `DiffEqFlux.jl`.

## Summary

* (Randomized) Local Regularization of Neural ODEs yield similar trajectories to global
  regularization.

* Local Regularization is simpler to implement since it doesn’t require
  discretize-then-optimize. Rather it can utilize the more commonly used
  optimize-then-discretize approach.

* Additionally introduces benefits for cases where the “blackbox cannot be opened” entirely
  (like in diffusion models)

## Installation

This package is not registered in General Registry. It can be installed using:

```julia
using Pkg
pkg"add https://github.com/avik-pal/LocalRegNeuralDE.jl"
```
