using DataFrames
using CSV
using Dates

const TIMINGS_FILE = joinpath(@__DIR__, "timings.csv")

# Columns that uniquely identify a run — used as the upsert key
const KEY_COLS = [:package, :backend, :nx, :ny, :nsteps]

"""
    load_timings() -> DataFrame

Load timings from CSV, or return an empty DataFrame with the correct schema.
"""
function load_timings()::DataFrame
    if isfile(TIMINGS_FILE)
        return CSV.read(TIMINGS_FILE, DataFrame)
    end
    return DataFrame(
        package   = String[],
        backend   = String[],
        nx        = Int[],
        ny        = Int[],
        nsteps    = Int[],
        time_ms   = Float64[],
        timestamp = String[],
    )
end

"""
    log_timing!(df, package, backend, nx, ny, nsteps, time_ms) -> DataFrame

Insert or update the row identified by (package, backend, nx, ny, nsteps).
Writes the updated DataFrame to `timings.csv` and returns it.

# Arguments
- `df`      : DataFrame returned by `load_timings()`
- `package` : e.g. `"Tullio"`, `"PlainJulia"`, `"CUDA"`, `"KernelAbstractions"`, `"ParallelStencil"`
- `backend` : `"CPU"` or `"GPU"`
- `nx`, `ny`: grid dimensions
- `nsteps`  : number of time steps
- `time_ms` : wall-clock time in milliseconds
"""
function log_timing!(df::DataFrame,
                     package::String, backend::String,
                     nx::Int, ny::Int, nsteps::Int,
                     time_ms::Float64)::DataFrame
    timestamp = string(now())
    mask = (df.package .== package) .&
           (df.backend .== backend) .&
           (df.nx      .== nx)      .&
           (df.ny      .== ny)      .&
           (df.nsteps  .== nsteps)

    if any(mask)
        idx = findfirst(mask)
        df[idx, :time_ms]   = time_ms
        df[idx, :timestamp] = timestamp
    else
        push!(df, (; package, backend, nx, ny, nsteps, time_ms, timestamp))
    end

    CSV.write(TIMINGS_FILE, df)
    return df
end
