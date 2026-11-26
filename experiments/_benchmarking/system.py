from pathlib import Path

import numpy as np


DEFAULT_CACHE_BYTES = {
    "L1": 32 * 1024,
    "L2": 256 * 1024,
    "L3": 8 * 1024 * 1024,
    "cache_line_bytes": 64,
}


def _parse_size(size_text: str) -> int:
    suffix = size_text[-1].upper()
    if suffix in {"K", "M"}:
        multiplier = 1024 if suffix == "K" else 1024 * 1024
        return int(size_text[:-1]) * multiplier
    return int(size_text)


def detect_cache_sizes() -> dict[str, int]:
    """Read CPU 0 cache sizes; use fallback values where sysfs is unavailable."""
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    if not base.exists():
        return DEFAULT_CACHE_BYTES.copy()

    cache: dict[str, int] = {}
    for index_dir in sorted(base.glob("index*")):
        try:
            level = int((index_dir / "level").read_text().strip())
            cache_type = (index_dir / "type").read_text().strip().lower()
            size = _parse_size((index_dir / "size").read_text().strip())
            line_size = int((index_dir / "coherency_line_size").read_text().strip())
        except (FileNotFoundError, ValueError):
            continue

        if (level == 1 and cache_type != "data") or (
            level > 1 and cache_type == "instruction"
        ):
            continue
        cache[f"L{level}"] = max(cache.get(f"L{level}", 0), size)
        if level == 1:
            cache["cache_line_bytes"] = line_size

    for key, value in DEFAULT_CACHE_BYTES.items():
        cache.setdefault(key, value)
    return cache


def cache_csv_fields(cache: dict[str, int]) -> dict[str, int]:
    return {
        "cache_l1_bytes": cache["L1"],
        "cache_l2_bytes": cache["L2"],
        "cache_l3_bytes": cache["L3"],
        "cache_line_bytes": cache["cache_line_bytes"],
    }


def make_flush_buffer(cache: dict[str, int]) -> np.ndarray:
    return np.zeros(max(32 * 1024 * 1024, 2 * cache["L3"]) // 8, dtype=np.float64)


def print_cache_info(cache: dict[str, int]) -> None:
    print("\n=== Cache topology ===")
    for level in ("L1", "L2", "L3"):
        print(f"  {level}: {cache[level] // 1024} KB")
    print(f"  Cache line: {cache['cache_line_bytes']} bytes\n")
