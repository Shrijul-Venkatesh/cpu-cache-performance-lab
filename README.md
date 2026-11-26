# CPU Cache Performance Lab

Small, reproducible experiments for studying how memory access patterns, NumPy
array layouts, and matrix multiplication interact with CPU caches and threading.
Results are machine-specific; compare runs only when their hardware and software
metadata are understood.

## Setup

Python 3.10 or newer is recommended. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Project layout

- `experiments/01_*.py`, `02_*.py`, and `03_*.py` are the documented entry
  points.
- `experiments/cache_hierarchy/`, `memory_layouts/`, and `matrix_multiply/`
  hold each experiment's kernels, benchmark logic, CLI, and optional perf code.
- `experiments/_benchmarking/` contains shared timing, system, metadata, and CSV
  helpers.
- `plots/` has one plot script per experiment plus `generator.py` to regenerate
  all figures from existing CSVs.
- `tests/` contains focused tests for shared benchmark utilities.

The matrix multiplication experiment can optionally collect Linux hardware
counters using `perf`. Counter collection may be unavailable because of kernel
permissions, virtualization, or unsupported events; the benchmark still runs
without them. Use `--no-perf` to skip counters.

## Quick smoke run

These commands exercise every experiment without allocating the largest
datasets. They write CSV measurements and JSON metadata under `data/`, then
generate plots under `plots/figures/`.

```bash
python experiments/01_cache_hierarchy.py \
  --subexp both --max-size-mb 1 --repeat 3 --output data/01_cache_hierarchy.csv
python plots/01_cache_hierarchy_plot.py \
  --data data/01_cache_hierarchy.csv --out-dir plots/figures

python experiments/02_memory_layouts.py \
  --max-side 128 --repeat 3 --output data/02_memory_layouts.csv
python plots/02_memory_layouts_plot.py \
  --data data/02_memory_layouts.csv --out-dir plots/figures

python experiments/03_matrix_multiply.py \
  --max-side 128 --max-threads 2 --repeat 3 --no-perf \
  --output data/03_matrix_multiply.csv
python plots/03_matrix_multiply_plot.py \
  --data data/03_matrix_multiply.csv --out-dir plots/figures
```

After all three CSVs exist, regenerate all figures with:

```bash
python plots/generator.py --data-dir data --out-dir plots/figures
```

For a broader run, omit the size limits. The defaults benchmark up to 256 MB
arrays in experiment 01 and 4096x4096 matrices in experiments 02 and 03. Larger
cases can require substantial memory and runtime.

All experiment scripts accept `--seed` to make generated data and access
patterns repeatable. Timings are summarized by their median; CSVs also preserve
the individual samples, minimum, maximum, and sample standard deviation. Each
CSV has a companion `*.metadata.json` file with the seed, detected cache sizes,
CPU/OS/Python/NumPy details, and command-line arguments.

## Experiments and interpretation

1. **Cache hierarchy** compares sequential reads, fixed strides, random gathers,
   and dependent pointer chasing over increasing array sizes. The working-set
   column includes the index array for gather and pointer-chase cases. A cache
   eviction buffer is touched before timed samples, but this is a best-effort
   eviction attempt—not a guarantee that every measurement starts cold.
   `latency_per_element_ns` is an amortized cost for sequential and gather
   patterns; pointer chasing is the latency-oriented measurement, though its
   dependent index and data loads mean it is not a single-load latency.
2. **Memory layouts** compares row and column reductions on C- and Fortran-order
   arrays, transposed views, and strided slices. Copy-and-reuse cases include
   the copy cost once and then reuse the contiguous copy for one or four
   reductions.
3. **Matrix multiplication** compares `float32`/`float64`, matrix sizes, and
   BLAS thread limits. `nominal_input_output_GB_s` is a calculation based on
   reading two inputs and writing one output; it is **not measured DRAM
   bandwidth**. `perf` counters are process totals from a separate runner and
   include runner setup, so interpret small-workload counts cautiously.

Cache sizes are read from Linux sysfs when available; otherwise labeled
fallback values are used. Cache capacities and sharing differ by CPU, and
measurements can also vary with CPU frequency scaling, background load,
affinity, BLAS implementation, and operating system. Treat results as
observations on the recorded system, not universal constants.

Cache-experiment throughput counts eight useful bytes per target element. It
does not include index-array reads, cache-line overfetch, or other physical
memory traffic, so it should not be read as measured DRAM bandwidth.

## Generated files

CSV files contain raw timing samples and derived metrics. JSON sidecars contain
run metadata. Plot scripts read those CSVs and create PNGs. These outputs are
not generated during installation; retain or commit the outputs you want to
publish with their matching metadata.
