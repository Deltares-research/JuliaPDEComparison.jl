# Sparse-matrix 2D diffusion solver — CUDA GPU version.
#
# The 2D Laplacian is applied by left- and right-multiplying the field matrix u
# with prebuilt 1D sparse second-derivative matrices transferred to the GPU:
#
#   ∇²u ≈ Dx_gpu * u + u * Dy_gpu'
#
# where Dx (Nx×Nx) and Dy (Ny×Ny) are tridiagonal [1,-2,1]/Δ² matrices with
# zero boundary rows/columns (closed / zero-Dirichlet boundary conditions).
# The CPU sparse matrices are built first, then pushed to the GPU as
# CUDA.CUSPARSE.CuSparseMatrixCSC arrays. The field u is moved to the GPU
# with cu(). CUDA.jl supports both sparse×dense and dense×sparse products,
# so the operator formula is identical to the CPU version.

include("diffusion_common.jl")
include("log_timings.jl")

using CUDA
using CUDA.CUSPARSE
using SparseArrays

# Note: cu() always downcasts Float64 → Float32. All GPU arrays (sparse and dense)
# must share the same precision, so everything in this file is kept in Float32.

"""
    make_1d_diffusion_matrix(N, Δ) -> SparseMatrixCSC{Float64, Int64}

Build the 1D second-derivative sparse matrix of size N×N with the centred
finite-difference stencil `[1, -2, 1] / Δ²` and closed (zero Dirichlet)
boundary conditions. Rows 1 and N are all-zero so the boundary values of
`u` are never updated.
"""
function make_1d_diffusion_matrix(N::Int, Δ::Real)
    rows = Int64[]
    cols = Int64[]
    vals = Float64[]

    inv_Δ² = 1 / Δ^2

    for i in 2:N-1
        push!(rows, i, i,  i)
        push!(cols, i-1, i, i+1)
        push!(vals, inv_Δ², -2inv_Δ², inv_Δ²)
    end

    return sparse(rows, cols, vals, N, N)
end

"""
    diffusion_operator(u, α, Dx, Dy) -> CuMatrix{Float64}

Apply the scaled 2D diffusion operator to field `u` on the GPU using prebuilt
CUDA sparse second-derivative matrices `Dx` and `Dy`:

    L(u) = α * (Dx * u + (Dy * u')')

`Dx` (Nx×Nx) acts on the x-direction (left-multiply, sparse × dense).
`Dy` (Ny×Ny) acts on the y-direction: because CUSPARSE only supports
sparse × dense, the right-multiply `u * Dy'` is rewritten as `(Dy * u')'`,
which is equivalent since `Dy` is symmetric. Both matrices include the 1/Δ² scaling.

# Arguments
- `u`  : 2D field on the GPU, size (Nx, Ny).
- `α`  : scalar scaling coefficient (e.g. `α * D`).
- `Dx` : 1D CUDA sparse second-derivative matrix in x, size (Nx, Nx).
- `Dy` : 1D CUDA sparse second-derivative matrix in y, size (Ny, Ny).
"""
function diffusion_operator(u::CuMatrix, α::Real, Dx::CuSparseMatrix, Dy::CuSparseMatrix)
    return α .* (Dx * u .+ (Dy * u')')
end

"""
    simulate(u₀, α; D, Δx, Δy, nsteps, Δt) -> (u, t)

Run a 2D diffusion simulation on the GPU using Euler-forward time integration
and CUDA sparse matrix products for the Laplacian:

    u^{n+1} = u^n + Δt * α * D * (Dx * u^n + u^n * Dy')

CPU sparse matrices `Dx` and `Dy` are built once, transferred to the GPU as
`CuSparseMatrixCSC`, and reused every step. The initial condition is also
transferred to the GPU.

Returns the final field `u` (on CPU) and the elapsed simulation time `t`.

# Arguments
- `u₀`    : initial condition, a 2D CPU matrix of size (Nx, Ny).
- `α`     : scalar scaling coefficient passed to `diffusion_operator`.
- `D`     : diffusion coefficient (keyword, default from `settings`).
- `Δx`    : grid spacing in x (keyword).
- `Δy`    : grid spacing in y (keyword).
- `nsteps`: number of time steps (keyword, default from `settings`).
- `Δt`    : time step size (keyword). If not provided, the CFL-stable value
            `0.4 * min(Δx, Δy)^2 / (2 * α * D)` is used.
"""
function simulate(
    u₀::AbstractMatrix,
    α::Real;
    D      = settings[:D],
    Δx::Real,
    Δy::Real,
    nsteps = settings[:nsteps],
    Δt::Real = 0.4 * min(Δx, Δy)^2 / (2 * α * D),
)
    Nx, Ny = size(u₀)

    # Build CPU sparse matrices in Float32, then push to GPU via cu().
    # cu() produces CuSparseMatrixCSC{Float32} — the only precision CUSPARSE
    # SpMM supports in this environment. The field u must also be Float32.
    Dx_cpu = Float32.(make_1d_diffusion_matrix(Nx, Δx))
    Dy_cpu = Float32.(make_1d_diffusion_matrix(Ny, Δy))
    Dx = cu(Dx_cpu)
    Dy = cu(Dy_cpu)

    u = cu(Float32.(u₀))   # move field to GPU as Float32 to match sparse matrices
    t = 0.0

    elapsed = @elapsed begin
        for _ in 1:nsteps
            u .+= Δt * α * D .* diffusion_operator(u, 1.0, Dx, Dy)
            t  += Δt
        end
        CUDA.synchronize()   # ensure GPU work is finished before stopping the timer
    end

    df = load_timings()
    log_timing!(df, "SparseGPU", get_backend(u), 1, Nx, Ny, nsteps, elapsed * 1000)

    return Array(u), t   # bring result back to CPU
end

"""
    run_simulation(; Lx, Ly, Nx, Ny, D, nsteps, σ, α) -> (u, sim_time, wall_time)

Set up and run the full 2D diffusion simulation using the CUDA sparse-matrix solver.

Constructs the grid and initial Gaussian-bump condition from `settings` (all
keyword arguments default to `settings` values), transfers data to the GPU,
then calls `simulate` and measures wall-clock time with `@elapsed`.

Exits with an error if no CUDA-capable GPU is available.

Returns the final field `u` (CPU array), the elapsed simulation time `sim_time`,
and the wall-clock runtime `wall_time` in seconds.
"""
function run_simulation(;
    Lx     = settings[:Lx],
    Ly     = settings[:Ly],
    Nx     = settings[:Nx],
    Ny     = settings[:Ny],
    D      = settings[:D],
    nsteps = settings[:nsteps],
    σ      = settings[:σ],
    α      = 1.0,
)
    CUDA.functional() || error("No CUDA-capable GPU found. Cannot run GPU simulation.")

    x, y, Δx, Δy = make_grid(; Lx, Ly, Nx, Ny)
    u₀ = gaussian_bump(x, y; σ)

    wall_time = @elapsed u, sim_time = simulate(u₀, α; D, Δx, Δy, nsteps)

    @info "run_simulation complete" Nx Ny nsteps sim_time wall_time
    return u, sim_time, wall_time
end

function main()
    u, sim_time, wall_time = run_simulation()
    @show sim_time wall_time
end

function main_sweep()
    # Warmup run to trigger JIT compilation; result is overwritten by the first real sweep step
    run_simulation(; Nx = 2^6, Ny = 2^6)

    for p in 6:13
        N = 2^p
        @info "Sweeping" N
        run_simulation(; Nx = N, Ny = N)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_sweep()
end
