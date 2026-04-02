# test_2ddiffusion

## Purpose

This folder compares low-level Julia packages for implementing a 2D diffusion equation solver using an explicit Euler-forward time stepping scheme. It is part of a broader evaluation of the Julia ecosystem for GPU-capable PDE solvers (see `../list_of_packages.md` and `../evaluation_criteria.md`).

The packages compared here are low-level tools that give direct control over compute kernels and memory, in contrast to full PDE frameworks:

- **Plain Julia arrays** — baseline CPU reference implementation
- **CUDA.jl** — direct NVIDIA GPU programming via CUDA kernels
- **KernelAbstractions.jl** — backend-agnostic GPU kernel writing (CUDA, AMD, CPU)
- **ParallelStencil.jl** — high-level stencil operations with CPU/GPU support
- **Tullio.jl** — tensor/index notation with automatic CPU/GPU parallelisation

## Problem

2D diffusion equation on a unit square with periodic or Dirichlet boundary conditions:

```
∂u/∂t = D (∂²u/∂x² + ∂²u/∂y²)
```

Discretised with second-order central differences in space and explicit (Euler-forward) time stepping.

## Approach

Each implementation lives in its own script or notebook. All implementations solve the same problem with the same grid size and number of time steps so that results and timings are directly comparable. The evaluation follows the criteria in `../evaluation_criteria.md`, with emphasis on:

- Ease of implementation and code clarity
- CPU and GPU performance
- Compatibility with automatic differentiation

## Status

| Package              | CPU impl. | GPU impl. | Benchmarked | Notes |
|----------------------|-----------|-----------|-------------|-------|
| Plain Julia          |           |           |             |       |
| CUDA.jl              |           |           |             |       |
| KernelAbstractions.jl|           |           |             |       |
| ParallelStencil.jl   |           |           |             |       |
| Tullio.jl            |           |           |             |       |

Status: **work in progress**
