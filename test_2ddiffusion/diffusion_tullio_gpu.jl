# Tullio.jl 2-D diffusion operator — CUDA GPU version.
#
# Uses the same @tullio stencil as diffusion_tullio.jl, but with CuArrays so
# Tullio auto-dispatches to a CUDA kernel via its KernelAbstractions backend.
#
# Key constraint vs the CPU version: Tullio's KA backend requires 1-based
# output indices. The CPU version uses Lu[i,j] where i ∈ 2:Nx-1 (interior
# only), but that non-1-based range causes KA to fail with "can't handle
# OffsetArrays". The `i+_` fix only works with := (allocating), not = (in-place).
# Instead, Lu is sized (Nx-2)×(Ny-2) and all u accesses are shifted +1 so that
# Lu[i,j] maps to u[i+1,j+1]. Tullio infers range 1:Nx-2 from Lu's size —
# 1-based, no OffsetArray. The interior of u is then updated via a view.
#
# Other constraints:
#   - Float32 required: cu() downcasts Float64 → Float32; scalar α must also
#     be Float32 to avoid a type mismatch inside the KA kernel.
#   - CUDA.synchronize() is required inside @elapsed for accurate GPU timing.

include("diffusion_common.jl")
include(joinpath(@__DIR__, "log_timings.jl"))

using Tullio
using KernelAbstractions   # must be in scope before @tullio for GPU dispatch
using CUDA

"""
    diffusion_operator!(Lu, u, α, Δx, Δy)

Apply the scaled 2-D diffusion operator using Tullio's KA/CUDA backend.
`Lu` must be pre-allocated as size (Nx-2, Ny-2) — the interior only.

GPU constraint: KernelAbstractions requires 1-based output indices. The
`i+_` syntax only works with `:=` (allocating), not `=` (in-place). Instead
we shift all u accesses by +1 so that Lu[i,j] (1-based, i ∈ 1:Nx-2)
corresponds to the interior element u[i+1, j+1]. Tullio infers the iteration
range 1:Nx-2 from Lu's size, which is 1-based — no OffsetArray needed.
"""
function diffusion_operator!(Lu::AbstractMatrix, u::AbstractMatrix,
                              α::Real, Δx::Real, Δy::Real)
    inv_Δx² = 1 / Δx^2
    inv_Δy² = 1 / Δy^2
    # Lu[i,j] ↔ u[i+1, j+1]; neighbours are at ±1 in each direction.
    @tullio Lu[i, j] = α * (
        inv_Δx² * (u[i+2, j+1] - 2u[i+1, j+1] + u[i,   j+1]) +
        inv_Δy² * (u[i+1, j+2] - 2u[i+1, j+1] + u[i+1, j  ])
    )
end

"""
    simulate(u₀, α; D, Δx, Δy, nsteps, Δt) -> (u, t)

Run a 2-D diffusion simulation on the GPU using Euler-forward time integration
and Tullio's KernelAbstractions CUDA backend for the Laplacian stencil.

Returns the final field `u` (on CPU) and the elapsed simulation time `t`.
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

    # Move to GPU as Float32. cu() would silently downcast, but being explicit
    # makes the intent clear. Boundary of u must stay zero (Dirichlet BCs);
    # it is initialised to zero by cu(Float32.(u₀)) since u₀ has zero BCs,
    # and the update step only touches the interior view.
    u  = cu(Float32.(u₀))
    # Lu holds only the interior (Nx-2)×(Ny-2) values; i+_ writes 1-based.
    Lu = CUDA.zeros(Float32, Nx-2, Ny-2)

    # α must be Float32 to match the CuArray element type inside the KA kernel.
    α_f32 = Float32(Δt * D)
    t = 0.0

    u_int = view(u, 2:Nx-1, 2:Ny-1)   # interior view; boundary of u never touched

    elapsed = @elapsed begin
        for _ in 1:nsteps
            diffusion_operator!(Lu, u, α_f32, Δx, Δy)
            u_int .+= Lu    # add Laplacian to interior only; boundary stays zero
            t  += Δt
        end
        CUDA.synchronize()   # flush GPU work queue before stopping timer
    end

    df = load_timings()
    log_timing!(df, "TullioGPU", get_backend(u), 1, Nx, Ny, nsteps, elapsed * 1000)

    return Array(u), t   # bring result back to CPU
end

"""
    run_simulation(; Lx, Ly, Nx, Ny, D, nsteps, σ, α) -> (u, sim_time, wall_time)

Set up and run the full 2-D diffusion simulation using the Tullio GPU solver.
Exits with an error if no CUDA-capable GPU is available.
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
    CUDA.functional() || error("No CUDA-capable GPU found.")

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
    # Warmup: triggers JIT + KA/CUDA kernel compilation (can take 10-30 s)
    run_simulation(; Nx = 2^6, Ny = 2^6)

    for p in 6:13   # 64×64 to 8192×8192, matching diffusion_sparse_gpu.jl
        N = 2^p
        @info "Sweeping" N
        run_simulation(; Nx = N, Ny = N)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_sweep()
end
