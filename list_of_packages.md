
# List of Julia packages

This is a list of Julia packages that can be used to implement PDE solvers in Julia. It is meant as the the starting point for a scan of the Julia ecosystem for PDE solvers. We will focus on packages that support GPU computing and that can potentially be connected to AI-models.

## Time stepping and ODE solvers

### DifferentialEquations.jl
- **Description**: A comprehensive suite of solvers for ordinary differential equations (ODEs), including support for stiff and non-stiff problems. It has a wide range of algorithms and has good support for GPU computing and automatic differentiation.
- **GitHub**: [https://docs.sciml.ai/DiffEqDocs/stable/](https://docs.sciml.ai/DiffEqDocs/stable/)


## Spatial discretization support

### ParallelStencil.jl
- **Description**: A package for parallel stencil computations, which are commonly used in finite difference methods for PDEs. It provides a high-level interface for defining stencil operations and supports both CPU and GPU computing.
- **GitHub**: [https://github.com/omlins/ParallelStencil.jl](https://github.com/omlins/ParallelStencil.jl)


## Full PDE solvers

### Gridap.jl
- **Description**: A finite element library for the numerical solution of PDEs. It provides a high-level interface for defining PDE problems and supports various types of elements and meshes. It also has support for GPU computing and can be used in conjunction with automatic differentiation tools.
- **GitHub**: [https://github.com/Gridap/Gridap.jl](https://github.com/Gridap/Gridap.jl)

