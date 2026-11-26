import argparse
import platform
from pathlib import Path

import numba
import numpy as np

from .._benchmarking import (
    detect_cache_sizes,
    make_flush_buffer,
    print_cache_info,
    save_csv,
    save_metadata,
    system_metadata,
)
from .benchmark import FIELDNAMES, run_experiment
from .kernels import warmup_kernels


def _print_system_info(cache):
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    print(f"CPU: {cpu_model}")
    print(f"OS: {platform.system()} {platform.release()}")
    print(
        f"Python: {platform.python_version()} | NumPy: {np.__version__} "
        f"| Numba: {numba.__version__}"
    )
    print_cache_info(cache)


def main():
    parser = argparse.ArgumentParser(description="Cache hierarchy benchmark")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--array-mb", type=int, default=64)
    parser.add_argument("--max-size-mb", type=int, default=256)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="data/01_cache_hierarchy.csv")
    parser.add_argument("--subexp", choices=("A", "B", "both"), default="both")
    args = parser.parse_args()
    if min(args.repeat, args.array_mb, args.max_size_mb) < 1:
        parser.error("--repeat, --array-mb, and --max-size-mb must be positive")

    cache = detect_cache_sizes()
    _print_system_info(cache)
    warmup_kernels()
    rows = run_experiment(
        args.subexp,
        args.array_mb,
        args.max_size_mb,
        args.repeat,
        np.random.default_rng(args.seed),
        cache,
        make_flush_buffer(cache),
    )
    save_csv(rows, FIELDNAMES, args.output)
    save_metadata(args.output, system_metadata(
        "01_cache_hierarchy", args.seed, cache, vars(args)
    ))


if __name__ == "__main__":
    main()
