import json
import statistics
import time
from collections.abc import Callable


def benchmark(
    fn: Callable[[], object],
    repeat: int = 10,
    warmup: int = 3,
    before_each: Callable[[], object] | None = None,
) -> dict[str, object]:
    """Return median timing, spread, and all samples after warmup."""
    if repeat < 1:
        raise ValueError("repeat must be at least 1")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")

    for _ in range(warmup):
        fn()

    samples = []
    for _ in range(repeat):
        if before_each is not None:
            before_each()
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)

    return {
        "time_s": statistics.median(samples),
        "time_min_s": min(samples),
        "time_max_s": max(samples),
        "time_stdev_s": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "time_samples_s": json.dumps(samples),
    }
