include(joinpath(@__DIR__, "log_timings.jl"))

using Plots

const PLOT_FILE = joinpath(@__DIR__, "timings.png")

df = load_timings()

if isempty(df)
    @warn "timings.csv is empty — nothing to plot."
    exit(0)
end

# Each (package, backend) pair becomes a separate series
groups = groupby(df, [:package, :backend])

plt = plot(
    xlabel = "N  (grid points per side)",
    ylabel = "time  (ms)",
    title  = "2-D diffusion: wall-clock time vs grid size",
    xscale = :log2,
    yscale = :log10,
    legend = :topleft,
    minorgrid = true,
)

for g in groups
    pkg     = g.package[1]
    backend = g.backend[1]
    rows    = sort(g, :nx)
    plot!(plt, rows.nx, rows.time_ms;
        label   = "$pkg / $backend",
        marker  = :circle,
        lw      = 2,
    )
end

savefig(plt, PLOT_FILE)
@info "Plot saved" PLOT_FILE
