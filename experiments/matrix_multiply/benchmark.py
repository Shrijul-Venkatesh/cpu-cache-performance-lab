import multiprocessing
import os

import numpy as np
from threadpoolctl import threadpool_limits

from .._benchmarking import benchmark, cache_csv_fields
from .perf import check_perf_available, run_perf_for_config


FIELDNAMES = [
    "side", "matrix_size_bytes", "dtype", "n_threads", "flops",
    "time_s", "time_min_s", "time_max_s", "time_stdev_s", "time_samples_s",
    "latency_ms", "gflops_s", "nominal_input_output_GB_s", "working_set_bytes",
    "cache_misses", "cache_references", "llc_load_misses", "instructions",
    "cycles", "perf_repetitions", "perf_scope", "cache_l1_bytes",
    "cache_l2_bytes", "cache_l3_bytes", "cache_line_bytes",
]


def available_thread_counts(max_threads=8):
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
        else multiprocessing.cpu_count()
    )
    return [count for count in (1, 2, 4, 8) if count <= min(available, max_threads)]


def run_sweep(repeat, perf_repeat, use_perf, max_threads, max_side, seed, cache):
    sides = [
        side for side in (64, 128, 256, 512, 1024, 2048, 4096, 8192)
        if side <= max_side
    ]
    thread_counts = available_thread_counts(max_threads)
    rng = np.random.default_rng(seed)
    perf_available = check_perf_available() if use_perf else False
    rows = []

    print(f"Thread limits: {thread_counts} | dtypes: float32, float64")
    for side in sides:
        for dtype in (np.float32, np.float64):
            dtype_name = np.dtype(dtype).name
            left = rng.random((side, side), dtype=dtype)
            right = rng.random((side, side), dtype=dtype)
            output = np.empty((side, side), dtype=dtype)

            np.matmul(left, right, out=output)
            check_side = min(side, 4)
            expected = left[:check_side] @ right[:, :check_side]
            np.testing.assert_allclose(
                output[:check_side, :check_side],
                expected,
                rtol=2e-4,
                atol=2e-5,
            )

            flops = 2 * side**3
            matrix_bytes = side**2 * np.dtype(dtype).itemsize
            nominal_io_bytes = 3 * matrix_bytes
            for threads in thread_counts:
                with threadpool_limits(limits=threads, user_api="blas"):
                    timing = benchmark(
                        lambda a=left, b=right, c=output: np.matmul(a, b, out=c),
                        repeat=repeat,
                    )

                elapsed = timing["time_s"]
                counters = {}
                if perf_available and threads == 1 and side >= 1024:
                    counters = run_perf_for_config(
                        side, dtype_name, threads, repeat=perf_repeat
                    )

                rows.append({
                    "side": side,
                    "matrix_size_bytes": matrix_bytes,
                    "dtype": dtype_name,
                    "n_threads": threads,
                    "flops": flops,
                    **timing,
                    "latency_ms": elapsed * 1000,
                    "gflops_s": flops / elapsed / 1e9,
                    "nominal_input_output_GB_s": nominal_io_bytes / elapsed / 1e9,
                    "working_set_bytes": 3 * matrix_bytes,
                    "cache_misses": counters.get("cache-misses", ""),
                    "cache_references": counters.get("cache-references", ""),
                    "llc_load_misses": counters.get("LLC-load-misses", ""),
                    "instructions": counters.get("instructions", ""),
                    "cycles": counters.get("cycles", ""),
                    "perf_repetitions": perf_repeat if counters else "",
                    "perf_scope": (
                        "whole runner process; includes setup" if counters else ""
                    ),
                    **cache_csv_fields(cache),
                })
                print(
                    f"{side:4} {dtype_name:7} {threads} thread(s): "
                    f"{flops / elapsed / 1e9:8.2f} GFLOPS/s"
                )
    return rows
