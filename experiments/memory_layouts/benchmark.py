import numpy as np

from .._benchmarking import benchmark, cache_csv_fields


FIELDNAMES = [
    "variant", "side", "matrix_size_bytes", "layout", "is_contiguous",
    "is_f_contiguous", "stride_row_bytes", "stride_col_bytes", "n_elements",
    "operation_repeats", "time_s", "time_min_s", "time_max_s",
    "time_stdev_s", "time_samples_s", "latency_per_element_ns",
    "throughput_GB_s", "working_set_bytes", "cache_l1_bytes",
    "cache_l2_bytes", "cache_l3_bytes", "cache_line_bytes",
]


def _variant(
    name,
    array,
    axis,
    layout,
    operation_repeats=1,
    fn=None,
    n_elements=None,
):
    return {
        "variant": name,
        "fn": fn or (lambda a=array, ax=axis: np.sum(a, axis=ax)),
        "n_elements": array.size if n_elements is None else n_elements,
        "operation_repeats": operation_repeats,
        "layout": layout,
        "is_contiguous": bool(array.flags["C_CONTIGUOUS"]),
        "is_f_contiguous": bool(array.flags["F_CONTIGUOUS"]),
        "strides_bytes": array.strides,
    }


def make_variants(array, array_f):
    transposed = array.T
    sliced = array[:, ::2]
    variants = [
        _variant("row_sum_c", array, 1, "C"),
        _variant("row_sum_f", array_f, 1, "Fortran"),
        _variant("col_sum_c", array, 0, "C"),
        _variant("col_sum_f", array_f, 0, "Fortran"),
        _variant("transpose_view_sum", transposed, 1, "transpose view"),
        _variant(
            "slice_sum", sliced, 1, "strided slice", n_elements=sliced.size
        ),
    ]

    for count in (1, 4):
        def sum_view(a=transposed, repeats=count):
            result = None
            for _ in range(repeats):
                result = np.sum(a, axis=1)
            return result

        def copy_then_sum(a=array, repeats=count):
            contiguous = np.ascontiguousarray(a.T)
            result = None
            for _ in range(repeats):
                result = np.sum(contiguous, axis=1)
            return result

        variants.extend([
            _variant(
                "transpose_view_reuse",
                transposed,
                1,
                "transpose view",
                count,
                fn=sum_view,
                n_elements=array.size * count,
            ),
            _variant(
                "transpose_copy_reuse",
                array,
                0,
                "copy + contiguous reuse",
                count,
                fn=copy_then_sum,
                n_elements=array.size * count,
            ),
        ])
    return variants


def run_sweep(repeat, max_side, rng, cache, flush_buffer):
    sides = [
        side for side in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
        if side <= max_side
    ]
    rows = []
    for side in sides:
        array = rng.random((side, side))
        array_f = np.asfortranarray(array)
        np.testing.assert_allclose(array.sum(axis=1), array_f.sum(axis=1))
        np.testing.assert_allclose(array.sum(axis=0), array_f.sum(axis=0))
        np.testing.assert_allclose(array.sum(axis=0), array.T.sum(axis=1))
        np.testing.assert_allclose(
            array.sum(axis=0), np.ascontiguousarray(array.T).sum(axis=1)
        )

        def flush_cache():
            np.sum(flush_buffer)

        for variant in make_variants(array, array_f):
            timing = benchmark(
                variant["fn"], repeat=repeat, before_each=flush_cache
            )
            elapsed = timing["time_s"]
            n_elements = variant["n_elements"]
            rows.append({
                "variant": variant["variant"],
                "side": side,
                "matrix_size_bytes": array.nbytes,
                "layout": variant["layout"],
                "is_contiguous": variant["is_contiguous"],
                "is_f_contiguous": variant["is_f_contiguous"],
                "stride_row_bytes": variant["strides_bytes"][0],
                "stride_col_bytes": variant["strides_bytes"][1],
                "n_elements": n_elements,
                "operation_repeats": variant["operation_repeats"],
                **timing,
                "latency_per_element_ns": elapsed / n_elements * 1e9,
                "throughput_GB_s": n_elements * 8 / elapsed / 1e9,
                "working_set_bytes": array.nbytes + array_f.nbytes,
                **cache_csv_fields(cache),
            })
        print(f"Finished {side}x{side} ({array.nbytes / 1024:.0f} KB per array)")
    return rows
