include(joinpath(@__DIR__, "log_timings.jl"))

using Plots

const PLOT_FILE = joinpath(@__DIR__, "timings.png")

df = load_timings()

if isempty(df)
    @warn "timings.csv is empty — nothing to plot."
    exit(0)
end

# Each (package, backend, nthreads) triple becomes a separate series
groups = groupby(df, [:package, :backend, :nthreads])

# Color per package, marker shape per thread count
packages      = unique(df.package)
thread_counts = sort(unique(df.nthreads))
pkg_colors    = Dict(p => c for (p, c) in zip(packages, palette(:tab10)))
thread_markers = Dict(n => m for (n, m) in zip(thread_counts, [:circle, :square, :diamond, :utriangle, :dtriangle]))

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
    pkg      = g.package[1]
    nthreads = g.nthreads[1]
    rows     = sort(g, :nx)
    label    = nthreads == 1 ? pkg : "$pkg / $(nthreads)t"
    plot!(plt, rows.nx, rows.time_ms;
        label        = label,
        color        = pkg_colors[pkg],
        marker       = thread_markers[nthreads],
        markercolor  = pkg_colors[pkg],
        lw           = 2,
    )
end

savefig(plt, PLOT_FILE)
@info "Plot saved" PLOT_FILE
