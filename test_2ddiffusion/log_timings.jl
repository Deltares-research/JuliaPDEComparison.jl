using DataFrames
using CSV
using Dates

const TIMINGS_FILE = joinpath(@__DIR__, "timings.csv")

# Columns that uniquely identify a run — used as the upsert key
const KEY_COLS = [:package, :backend, :nthreads, :nx, :ny, :nsteps]

"""
    get_backend(u) -> String

Return a string identifying the compute backend from the array type of `u`:
`"CUDA"`, `"AMDGPU"`, `"Metal"`, or `"CPU"`.
Does not require importing any GPU package.
"""
function get_backend(u::AbstractArray)::String
    m = string(parentmodule(typeof(u)))
    startswith(m, "CUDA")   && return "CUDA"
    startswith(m, "AMDGPU") && return "AMDGPU"
    startswith(m, "Metal")  && return "Metal"
    return Sys.cpu_info()[1].model
end

"""
    load_timings() -> DataFrame

Load timings from CSV, or return an empty DataFrame with the correct schema.
"""
function load_timings()::DataFrame
    if isfile(TIMINGS_FILE)
        return CSV.read(TIMINGS_FILE, DataFrame)
    end
    return DataFrame(
        package  = String[],
        backend  = String[],
        nthreads = Int[],
        nx       = Int[],
        ny       = Int[],
        nsteps   = Int[],
        time_ms  = Float64[],
        timestamp = DateTime[],
    )
end

"""
    log_timing!(df, package, backend, nthreads, nx, ny, nsteps, time_ms) -> DataFrame

Insert or update the row identified by (package, backend, nthreads, nx, ny, nsteps).
Writes the updated DataFrame to `timings.csv` and returns it.

# Arguments
- `df`       : DataFrame returned by `load_timings()`
- `package`  : e.g. `"Tullio"`, `"PlainJulia"`, `"CUDA"`, `"KernelAbstractions"`, `"ParallelStencil"`
- `backend`  : CPU model string or GPU identifier from `get_backend`
- `nthreads` : number of Julia threads (`Threads.nthreads()`)
- `nx`, `ny` : grid dimensions
- `nsteps`   : number of time steps
- `time_ms`  : wall-clock time in milliseconds
"""
function log_timing!(df::DataFrame,
                     package::String, backend::String, nthreads::Int,
                     nx::Int, ny::Int, nsteps::Int,
                     time_ms::Float64)::DataFrame
    timestamp = now()
    mask = (df.package   .== package)  .&
           (df.backend   .== backend)  .&
           (df.nthreads  .== nthreads) .&
           (df.nx        .== nx)       .&
           (df.ny        .== ny)       .&
           (df.nsteps    .== nsteps)

    if any(mask)
        idx = findfirst(mask)
        df[idx, :time_ms]   = time_ms
        df[idx, :timestamp] = timestamp
    else
        push!(df, (; package, backend, nthreads, nx, ny, nsteps, time_ms, timestamp))
    end

    CSV.write(TIMINGS_FILE, df)
    return df
end
