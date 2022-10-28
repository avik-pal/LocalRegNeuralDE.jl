# LocalRegNeuralDEExperiments

## Installation

This package depends on some unregistered packages. Open a Julia REPL and run the following
commands:

```julia
using Pkg

Pkg.add(url="https://github.com/SciML/DeepEquilibriumNetworks.jl", subdir="experiments",
        rev="ap/paper")
Pkg.add(url="https://github.com/SciML/DeepEquilibriumNetworks.jl", rev="ap/saveat")
Pkg.add(path="../")
```