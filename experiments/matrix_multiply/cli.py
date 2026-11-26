import argparse
import platform

import numpy as np
import threadpoolctl

from .._benchmarking import (
    detect_cache_sizes,
    print_cache_info,
    save_csv,
    save_metadata,
    system_metadata,
)
from .benchmark import FIELDNAMES, run_sweep


def main():
    parser = argparse.ArgumentParser(description="Matrix multiplication benchmark")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--perf-repeat", type=int, default=10)
    parser.add_argument("--max-side", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--output", default="data/03_matrix_multiply.csv")
    parser.add_argument("--no-perf", action="store_true")
    parser.add_argument("--max-threads", type=int, default=8)
    args = parser.parse_args()
    if (
        args.repeat < 1
        or args.perf_repeat < 1
        or args.max_side < 64
        or args.max_threads < 1
    ):
        parser.error(
            "repetitions and thread limit must be positive; "
            "max-side must be at least 64"
        )

    cache = detect_cache_sizes()
    print(f"OS: {platform.system()} {platform.release()}")
    print(
        f"Python: {platform.python_version()} | NumPy: {np.__version__} "
        f"| threadpoolctl: {threadpoolctl.__version__}"
    )
    print_cache_info(cache)
    rows = run_sweep(
        args.repeat,
        args.perf_repeat,
        not args.no_perf,
        args.max_threads,
        args.max_side,
        args.seed,
        cache,
    )
    save_csv(rows, FIELDNAMES, args.output)
    save_metadata(args.output, system_metadata(
        "03_matrix_multiply", args.seed, cache, vars(args)
    ))


if __name__ == "__main__":
    main()
