# Common settings and utilities shared across all 2D diffusion implementations.

settings = Dict(
    :Lx     => 1.0,    # domain length in x
    :Ly     => 1.0,    # domain length in y  (== Lx for a square domain)
    :Nx     => 128,    # number of grid points in x
    :Ny     => 128,    # number of grid points in y  (== Nx for a square grid)
    :D      => 1.0,    # diffusion coefficient
    :nsteps => 1000,   # number of time steps
    :σ      => 0.1,    # width (std dev) of the initial Gaussian bump
)

"""
    make_grid(; Lx, Ly, Nx, Ny) -> (x, y, Δx, Δy)

Return 1-D coordinate vectors `x` and `y` and the uniform grid spacings
`Δx` and `Δy` for a rectangular domain [0, Lx] × [0, Ly] with `Nx` × `Ny`
cell-centred points.

Defaults produce a square unit-domain grid matching `settings`.
"""
function make_grid(;
    Lx = settings[:Lx],
    Ly = settings[:Ly],
    Nx = settings[:Nx],
    Ny = settings[:Ny],
)
    Δx = Lx / Nx
    Δy = Ly / Ny
    x = range(Δx / 2, Lx - Δx / 2; length = Nx)   # cell centres
    y = range(Δy / 2, Ly - Δy / 2; length = Ny)
    return x, y, Δx, Δy
end

"""
    gaussian_bump(x, y; σ) -> Matrix{Float64}

Return a 2-D field on the grid defined by coordinate vectors `x` (length Nx)
and `y` (length Ny), with a Gaussian bump centred on the domain:

    u[i, j] = exp(-((x[i] - x₀)² + (y[j] - y₀)²) / (2σ²))

The peak value is exactly 1.0 at the centre. The field is stored with the
x-index varying along rows and the y-index along columns, i.e. `u[ix, iy]`.
"""
function gaussian_bump(x, y; σ = settings[:σ])
    x₀ = (first(x) + last(x)) / 2
    y₀ = (first(y) + last(y)) / 2
    return @. exp(-((x - x₀)^2 + (y' - y₀)^2) / (2σ^2))
end
