
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

### ImplicitGlobalGrid.jl
- **Description**: A package for solving PDEs on implicit global grids, which are useful for problems with complex geometries. It provides a high-level interface for defining PDE problems and supports GPU computing. It can be used in conjunction with automatic differentiation tools for optimization and sensitivity analysis.
- **GitHub**: [https://github.com/eth-cscs/ImplicitGlobalGrid.jl](https://github.com/eth-cscs/ImplicitGlobalGrid.jl)

## More general-purpose tools that can be used for PDE solvers

### Tullio.jl
- **Description**: A package for tensor operations that can be used to implement various numerical methods for PDEs. It provides a high-level interface for defining tensor operations and supports both CPU and GPU computing.
- **GitHub**: [https://github.com/mcabbott/Tullio.jl](https://github.com/mcabbott/Tullio.jl)

### CUDA.jl
- **Description**: A package for programming NVIDIA GPUs using CUDA. It provides a high-level interface for defining GPU kernels and managing GPU memory, which can be used to implement custom PDE solvers that are optimized for GPU performance.
- **GitHub**: [https://github.com/JuliaGPU/CUDA.jl](https://github.com/JuliaGPU/CUDA.jl)
- **Note**: We have used sparse matrix operations in CUDA.jl for a simple PDE solver, and it showed good performance on the GPU. It also resulted in clear code that was easy to understand and modify, which is a significant advantage for our use case.

### KernelAbstractions.jl
- **Description**: A package for writing GPU kernels in a high-level, abstract way. It provides a unified interface for writing GPU code that can run on different backends, including CUDA and AMD GPUs. It can be used to implement custom PDE solvers that are optimized for GPU performance while maintaining code clarity and portability.
- **GitHub**: [https://github.com/JuliaGPU/KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl)

## Full PDE solvers

### Gridap.jl
- **Description**: A finite element library for the numerical solution of PDEs. It provides a high-level interface for defining PDE problems and supports various types of elements and meshes. It also has support for GPU computing and can be used in conjunction with automatic differentiation tools.
- **GitHub**: [https://github.com/Gridap/Gridap.jl](https://github.com/Gridap/Gridap.jl)

### Oceananigans.jl
- **Description**: A package for simulating ocean and atmospheric dynamics using finite difference methods. It is designed for high-performance computing and has good support for GPU computing. It can be used to solve a wide range of PDEs related to fluid dynamics and geophysical flows.
- **GitHub**: [https://github.com/CliMA/Oceananigans.jl](https://github.com/CliMA/Oceananigans.jl)

### Trixi.jl
- **Description**: A package for solving hyperbolic PDEs using high-order discontinuous Galerkin methods. It is designed for high-performance computing and has some support for GPU computing. It can be used to solve a wide range of PDEs related to fluid dynamics, electromagnetics, and other fields.
- **GitHub**: [https://github.com/trixi-framework/Trixi.jl](https://github.com/trixi-framework/Trixi.jl)


## Documentation and community resources
- [https://juliagpu.org/](https://juliagpu.org/)
  - A website that provides resources and documentation for GPU computing in Julia, including tutorials, examples, and a list of packages that support GPU computing.
- [https://github.com/JuliaPDE/SurveyofPDEPackages](https://github.com/JuliaPDE/SurveyofPDEPackages)
  - A GitHub repository that provides a survey of Julia packages for PDE solvers, including descriptions, links to    documentation, and examples of usage. It is a good starting point for exploring the Julia ecosystem for PDE solvers.   