# Dense-matrix 2D diffusion solver.
#
# The 2D Laplacian is applied by left- and right-multiplying the field matrix u
# with prebuilt 1D dense second-derivative matrices:
#
#   ∇²u ≈ Dx * u + u * Dy'
#
# where Dx (Nx×Nx) and Dy (Ny×Ny) are tridiagonal [1,-2,1]/Δ² matrices with
# zero boundary rows/columns (closed / zero-Dirichlet boundary conditions).

include("diffusion_common.jl")
include("log_timings.jl")

"""
    make_1d_diffusion_matrix(N, Δ) -> Matrix{Float64}

Build the 1D second-derivative dense matrix of size N×N with the centred
finite-difference stencil `[1, -2, 1] / Δ²` and closed (zero Dirichlet)
boundary conditions. Rows 1 and N are all-zero so the boundary values of
`u` are never updated.
"""
function make_1d_diffusion_matrix(N::Int, Δ::Real)
    D = zeros(Float64, N, N)
    inv_Δ² = 1 / Δ^2

    for i in 2:N-1
        D[i, i-1] =  inv_Δ²
        D[i, i]   = -2inv_Δ²
        D[i, i+1] =  inv_Δ²
    end

    return D
end

"""
    diffusion_operator(u, α, Dx, Dy) -> Matrix{Float64}

Apply the scaled 2D diffusion operator to field `u` using prebuilt 1D dense
second-derivative matrices `Dx` and `Dy`:

    L(u) = α * (Dx * u + u * Dy')

`Dx` (Nx×Nx) acts on the x-direction (left-multiply); `Dy` (Ny×Ny) acts on
the y-direction (right-multiply with transpose). Both matrices must already
include the 1/Δ² scaling.

# Arguments
- `u`  : 2D field of size (Nx, Ny).
- `α`  : scalar scaling coefficient (e.g. `α * D`).
- `Dx` : 1D dense second-derivative matrix in x, size (Nx, Nx).
- `Dy` : 1D dense second-derivative matrix in y, size (Ny, Ny).
"""
function diffusion_operator(u::AbstractMatrix, α::Real, Dx::AbstractMatrix, Dy::AbstractMatrix)
    return α .* (Dx * u .+ u * Dy')
end

"""
    simulate(u₀, α; D, Δx, Δy, nsteps, Δt) -> (u, t)

Run a 2D diffusion simulation using Euler-forward time integration and dense
matrix products for the Laplacian:

    u^{n+1} = u^n + Δt * α * D * (Dx * u^n + u^n * Dy')

The matrices `Dx` and `Dy` are built once before the time loop.
Returns the final field `u` and the elapsed simulation time `t`.

# Arguments
- `u₀`    : initial condition, a 2D matrix of size (Nx, Ny).
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
    Dx = make_1d_diffusion_matrix(Nx, Δx)
    Dy = make_1d_diffusion_matrix(Ny, Δy)

    u = copy(u₀)
    t = 0.0

    elapsed = @elapsed for _ in 1:nsteps
        u .+= Δt * α * D .* diffusion_operator(u, 1.0, Dx, Dy)
        t  += Δt
    end

    df = load_timings()
    log_timing!(df, "DenseMat", get_backend(u₀), Threads.nthreads(), Nx, Ny, nsteps, elapsed * 1000)

    return u, t
end

"""
    run_simulation(; Lx, Ly, Nx, Ny, D, nsteps, σ, α) -> (u, sim_time, wall_time)

Set up and run the full 2D diffusion simulation using the dense-matrix solver.

Constructs the grid and initial Gaussian-bump condition from `settings` (all
keyword arguments default to `settings` values), then calls `simulate` and
measures wall-clock time with `@elapsed`.

Returns the final field `u`, the elapsed simulation time `sim_time`, and the
wall-clock runtime `wall_time` in seconds.
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

    for p in 6:11
        N = 2^p
        @info "Sweeping" N
        run_simulation(; Nx = N, Ny = N)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_sweep()
end
