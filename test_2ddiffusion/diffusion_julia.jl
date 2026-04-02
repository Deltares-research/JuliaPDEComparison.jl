# Native-Julia 2-D diffusion operator and Euler-forward simulator.
# Include diffusion_common.jl first to get `settings`, `make_grid`, and `gaussian_bump`.

"""
    diffusion_operator(u, α, Δx, Δy, D) -> Matrix{Float64}

Apply the scaled 2-D diffusion operator to field `u`:

    L(u)[i,j] = α * D * (∂²u/∂x² + ∂²u/∂y²)

The Laplacian is discretised with second-order centred finite differences.
Periodic boundary conditions are used in both directions.

# Arguments
- `u`  : 2-D field of size (Nx, Ny), with `u[ix, iy]` indexing convention.
- `α`  : scalar scaling coefficient (dimensionless).
- `Δx` : uniform grid spacing in x.
- `Δy` : uniform grid spacing in y.
- `D`  : diffusion coefficient.
"""
function diffusion_operator(u::AbstractMatrix, α::Real, Δx::Real, Δy::Real, D::Real)
    Nx, Ny = size(u)
    Lu = similar(u)

    inv_Δx² = 1 / Δx^2
    inv_Δy² = 1 / Δy^2
    scale   = α * D

    @inbounds for iy in 1:Ny
        iy_prev = iy == 1  ? Ny : iy - 1
        iy_next = iy == Ny ?  1 : iy + 1
        for ix in 1:Nx
            ix_prev = ix == 1  ? Nx : ix - 1
            ix_next = ix == Nx ?  1 : ix + 1

            d2u_dx2 = (u[ix_next, iy] - 2u[ix, iy] + u[ix_prev, iy]) * inv_Δx²
            d2u_dy2 = (u[ix, iy_next] - 2u[ix, iy] + u[ix, iy_prev]) * inv_Δy²

            Lu[ix, iy] = scale * (d2u_dx2 + d2u_dy2)
        end
    end

    return Lu
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
- `dt`    : time step size (keyword). If not provided, the CFL-stable value
            `0.4 * min(Δx, Δy)^2 / (2 * α * D)` is used.
"""
function simulate(
    u₀::AbstractMatrix,
    α::Real;
    D      = settings[:D],
    Δx::Real,
    Δy::Real,
    nsteps = settings[:nsteps],
    dt::Real = 0.4 * min(Δx, Δy)^2 / (2 * α * D),
)
    u = copy(u₀)
    t = 0.0

    for _ in 1:nsteps
        u .+= dt .* diffusion_operator(u, α, Δx, Δy, D)
        t  += dt
    end

    return u, t
end
