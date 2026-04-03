# Tullio.jl 2-D diffusion operator and Euler-forward simulator.

include("diffusion_common.jl")

include(joinpath(@__DIR__, "log_timings.jl"))

using Tullio

"""
    diffusion_operator(u, α, Δx, Δy) -> Matrix{Float64}

Apply the scaled 2-D diffusion operator to field `u` using Tullio.jl:

    L(u)[i,j] = α * (∂²u/∂x² + ∂²u/∂y²)

The Laplacian is discretised with second-order centred finite differences.
Periodic boundary conditions are imposed via a one-cell ghost layer before
calling `@tullio`, avoiding the slow built-in `mod` indexing mode.

# Arguments
- `u`  : 2-D field of size (Nx, Ny), with `u[ix, iy]` indexing convention.
- `α`  : scalar scaling coefficient (e.g. `Δt * D`).
- `Δx` : uniform grid spacing in x.
- `Δy` : uniform grid spacing in y.
"""
function diffusion_operator(u::AbstractMatrix, α::Real, Δx::Real, Δy::Real)
    Nx, Ny = size(u)
    inv_Δx² = 1 / Δx^2
    inv_Δy² = 1 / Δy^2

    # Pad with one periodic ghost cell on each side → size (Nx+2, Ny+2)
    u_pad = similar(u, Nx + 2, Ny + 2)
    u_pad[2:Nx+1, 2:Ny+1] .= u
    u_pad[1,      2:Ny+1] .= u[Nx, :]   # left  ghost ← right  edge
    u_pad[Nx+2,   2:Ny+1] .= u[1,  :]   # right ghost ← left   edge
    u_pad[2:Nx+1, 1]      .= u[:, Ny]   # bottom ghost ← top   edge
    u_pad[2:Nx+1, Ny+2]   .= u[:, 1]    # top   ghost ← bottom edge

    # i+_ shifts output to 1-based; Tullio infers i ∈ 2:Nx+1 from u_pad bounds
    @tullio Lu[i+_, j+_] := α * (
        inv_Δx² * (u_pad[i+1, j] - 2u_pad[i, j] + u_pad[i-1, j]) +
        inv_Δy² * (u_pad[i, j+1] - 2u_pad[i, j] + u_pad[i, j-1])
    )

    return Lu
end

"""
    simulate(u₀, α; D, Δx, Δy, nsteps, Δt) -> (u, t)

Run a 2-D diffusion simulation using Euler-forward (explicit) time integration:

    u^{n+1} = u^n + Δt * α * D * ∇²u^n

Returns the final field `u` and the elapsed simulation time `t`.

# Arguments
- `u₀`    : initial condition, a 2-D matrix of size (Nx, Ny).
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
    u = copy(u₀)
    t = 0.0

    elapsed = @elapsed for _ in 1:nsteps
        u .+= diffusion_operator(u, Δt * D, Δx, Δy)
        t  += Δt
    end

    Nx, Ny = size(u₀)
    df = load_timings()
    log_timing!(df, "Tullio", get_backend(u₀), Nx, Ny, nsteps, elapsed * 1000)

    return u, t
end

"""
    run_simulation(; Lx, Ly, Nx, Ny, D, nsteps, σ, α) -> (u, sim_time, wall_time)

Set up and run the full 2-D diffusion simulation using the Tullio solver.

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
