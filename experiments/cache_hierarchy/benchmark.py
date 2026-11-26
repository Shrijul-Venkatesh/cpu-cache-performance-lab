from .._benchmarking import benchmark, cache_csv_fields
from .kernels import (
    build_chase_indices,
    kernel_gather,
    kernel_pointer_chase,
    kernel_sequential,
    kernel_strided,
)


FIELDNAMES = [
    "subexp", "pattern", "array_size_bytes", "stride", "n_accessed",
    "time_s", "time_min_s", "time_max_s", "time_stdev_s", "time_samples_s",
    "latency_per_element_ns", "throughput_GB_s", "working_set_bytes",
    "cache_l1_bytes", "cache_l2_bytes", "cache_l3_bytes", "cache_line_bytes",
]

SIZE_SWEEP_PATTERNS = (
    ("sequential", 1),
    ("strided_8", 8),
    ("strided_64", 64),
    ("gather", None),
    ("pointer_chase", None),
)


def _row(
    subexp,
    pattern,
    size_bytes,
    stride,
    accessed,
    timing,
    working_set,
    cache,
):
    elapsed = timing["time_s"]
    return {
        "subexp": subexp,
        "pattern": pattern,
        "array_size_bytes": size_bytes,
        "stride": stride if stride is not None else "N/A",
        "n_accessed": accessed,
        **timing,
        "latency_per_element_ns": elapsed / accessed * 1e9,
        "throughput_GB_s": accessed * 8 / elapsed / 1e9,
        "working_set_bytes": working_set,
        **cache_csv_fields(cache),
    }


def run_stride_sweep(array_size_mb, repeat, rng, cache, flush_buffer):
    size_bytes = array_size_mb * 1024 * 1024
    array = rng.random(size_bytes // 8)
    if array.nbytes <= cache["L3"]:
        print("[warn] stride-sweep array does not exceed detected L3 cache")

    rows = []
    for stride in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        accessed = len(range(0, len(array), stride))
        timing = benchmark(
            lambda s=stride: kernel_strided(array, s),
            repeat=repeat,
            before_each=lambda: kernel_sequential(flush_buffer),
        )
        rows.append(_row(
            "A", "strided", array.nbytes, stride, accessed, timing,
            array.nbytes, cache,
        ))
        elapsed = timing["time_s"]
        print(
            f"  stride={stride:4d}  accessed={accessed:>10,}  "
            f"latency={elapsed / accessed * 1e9:7.2f}ns"
        )
    return rows


def run_size_sweep(repeat, max_size_mb, rng, cache, flush_buffer):
    sizes = [
        2**power for power in range(12, 29)
        if 2**power <= max_size_mb * 1024 * 1024
    ]
    if sizes and sizes[-1] <= cache["L3"]:
        print("[warn] largest dataset may not exceed detected L3 cache")

    rows = []
    for size_bytes in sizes:
        array = rng.random(size_bytes // 8)
        label = f"{size_bytes // 1024}KB" if size_bytes < 1024**2 else (
            f"{size_bytes // 1024**2}MB"
        )
        print(f"\n  Array size: {label}")

        for pattern, stride in SIZE_SWEEP_PATTERNS:
            indices = None
            if pattern == "sequential":
                fn = lambda: kernel_sequential(array)
                accessed = len(array)
            elif stride is not None:
                fn = lambda s=stride: kernel_strided(array, s)
                accessed = len(range(0, len(array), stride))
            elif pattern == "gather":
                indices = rng.permutation(len(array))
                fn = lambda ix=indices: kernel_gather(array, ix)
                accessed = len(indices)
            else:
                indices = build_chase_indices(len(array), rng)
                fn = lambda ix=indices: kernel_pointer_chase(array, ix)
                accessed = len(indices)

            timing = benchmark(
                fn,
                repeat=repeat,
                before_each=lambda: kernel_sequential(flush_buffer),
            )
            rows.append(_row(
                "B",
                pattern,
                size_bytes,
                stride,
                accessed,
                timing,
                array.nbytes + (indices.nbytes if indices is not None else 0),
                cache,
            ))
            elapsed = timing["time_s"]
            print(
                f"    {pattern:<15}  "
                f"latency={elapsed / accessed * 1e9:7.2f}ns"
            )
    return rows


def run_experiment(
    subexp, array_size_mb, max_size_mb, repeat, rng, cache, flush_buffer
):
    rows = []
    if subexp in ("A", "both"):
        print(f"=== Stride sweep ({array_size_mb}MB array) ===")
        rows.extend(run_stride_sweep(
            array_size_mb, repeat, rng, cache, flush_buffer
        ))
    if subexp in ("B", "both"):
        print("\n=== Dataset-size sweep ===")
        rows.extend(run_size_sweep(
            repeat, max_size_mb, rng, cache, flush_buffer
        ))
    return rows
