# LocalRegNeuralDE: Some BlackBoxes were meant to remain closed!

Extension of [RegNeuralDE](https://arxiv.org/abs/2105.03918) to allow for local
regularization.

## Summary

* (Randomized) Local Regularization of Neural ODEs yield similar trajectories to global
  regularization.

* Local Regularization is simpler to implement since it doesn’t require
  discretize-then-optimize. Rather it can utilize the more commonly used
  optimize-then-discretize approach.

* Additionally introduces benefits for cases where the “blackbox cannot be opened” entirely
  (like in diffusion models)

## Detailed Writeup

See [this Google Doc](https://docs.google.com/document/d/1ArAYWk4uix-RILJQRTCF3dIabsCCgtOvB9ZMQbFo1SA/edit?usp=sharing)
(Ask for access. If you have access to this repository, I have probably just missed adding
you to the doc.)

## Installation

This package is not registered in General Registry. Instead it can be installed from the
[ResearchRegistry](https://github.com/avik-pal/ResearchRegistry).

> **Note**
> You might run into issues with Pkg cloning a private repository. If that happens, you
  should start julia as `JULIA_PKG_USE_CLI_GIT=true julia` (or do
  `export JULIA_PKG_USE_CLI_GIT=true` in your shell)

```julia
using Pkg
pkg"registry add https://github.com/avik-pal/ResearchRegistry"
pkg"add LocalRegNeuralDE"
```

