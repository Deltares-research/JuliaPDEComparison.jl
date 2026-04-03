# Native-Julia 2-D diffusion operator and Euler-forward simulator.

include("diffusion_common.jl")

include(joinpath(@__DIR__, "log_timings.jl"))

"""
    diffusion_operator!(Lu, u, α, Δx, Δy)

Apply the scaled 2-D diffusion operator to field `u`, writing the result
into the pre-allocated array `Lu`:

    Lu[i,j] = α * (∂²u/∂x² + ∂²u/∂y²)

The Laplacian is discretised with second-order centred finite differences.
Periodic boundary conditions are used in both directions.

# Arguments
- `Lu` : pre-allocated output array of size (Nx, Ny).
- `u`  : 2-D field of size (Nx, Ny), with `u[ix, iy]` indexing convention.
- `α`  : scalar scaling coefficient (e.g. `Δt * D`).
- `Δx` : uniform grid spacing in x.
- `Δy` : uniform grid spacing in y.
"""
function diffusion_operator!(Lu::AbstractMatrix, u::AbstractMatrix, α::Real, Δx::Real, Δy::Real)
    Nx, Ny = size(u)
    inv_Δx² = 1 / Δx^2
    inv_Δy² = 1 / Δy^2

    @inbounds for iy in 1:Ny
        iy_prev = iy == 1  ? Ny : iy - 1
        iy_next = iy == Ny ?  1 : iy + 1
        for ix in 1:Nx
            ix_prev = ix == 1  ? Nx : ix - 1
            ix_next = ix == Nx ?  1 : ix + 1

            d2u_dx2 = (u[ix_next, iy] - 2u[ix, iy] + u[ix_prev, iy]) * inv_Δx²
            d2u_dy2 = (u[ix, iy_next] - 2u[ix, iy] + u[ix, iy_prev]) * inv_Δy²

            Lu[ix, iy] = α * (d2u_dx2 + d2u_dy2)
        end
    end
end

"""
    simulate(u₀, α; D, Δx, Δy, nsteps, dt) -> (u, t)

Run a 2-D diffusion simulation using Euler-forward (explicit) time integration:

    u^{n+1} = u^n + dt * α * D * ∇²u^n

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
    Nx, Ny = size(u₀)
    u  = copy(u₀)
    Lu = zeros(eltype(u₀), Nx, Ny)
    t  = 0.0

    elapsed = @elapsed for _ in 1:nsteps
        diffusion_operator!(Lu, u, Δt * D, Δx, Δy)
        u .+= Lu
        t  += Δt
    end
    df = load_timings()
    log_timing!(df, "PlainJulia", get_backend(u₀), 1, Nx, Ny, nsteps, elapsed * 1000)

    return u, t
end

"""
    run_simulation(; Lx, Ly, Nx, Ny, D, nsteps, σ, α) -> (u, sim_time, wall_time)

Set up and run the full 2-D diffusion simulation using the native-Julia solver.

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