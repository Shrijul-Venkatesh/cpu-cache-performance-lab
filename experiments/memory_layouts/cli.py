import argparse
import platform

import numpy as np

from .._benchmarking import (
    detect_cache_sizes,
    make_flush_buffer,
    print_cache_info,
    save_csv,
    save_metadata,
    system_metadata,
)
from .benchmark import FIELDNAMES, run_sweep


def main():
    parser = argparse.ArgumentParser(description="NumPy memory-layout benchmark")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="data/02_memory_layouts.csv")
    args = parser.parse_args()
    if args.repeat < 1 or args.max_side < 32:
        parser.error("--repeat must be positive and --max-side must be at least 32")

    cache = detect_cache_sizes()
    print(f"OS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()} | NumPy: {np.__version__}")
    print_cache_info(cache)
    rows = run_sweep(
        args.repeat,
        args.max_side,
        np.random.default_rng(args.seed),
        cache,
        make_flush_buffer(cache),
    )
    save_csv(rows, FIELDNAMES, args.output)
    save_metadata(args.output, system_metadata(
        "02_memory_layouts", args.seed, cache, vars(args)
    ))


if __name__ == "__main__":
    main()
