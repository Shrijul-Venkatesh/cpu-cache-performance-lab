import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numba
import numpy as np
from threadpoolctl import threadpool_info


def system_metadata(
    experiment: str,
    seed: int,
    cache: dict[str, int],
    arguments: dict[str, object],
) -> dict[str, object]:
    cpu_model = "unknown"
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break

    affinity_count = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    )
    return {
        "experiment": experiment,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "cpu_model": cpu_model,
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_affinity_count": affinity_count,
        "os": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "numba": numba.__version__,
        "threadpools": threadpool_info(),
        "cache_sizes_source": (
            "sysfs" if Path("/sys/devices/system/cpu/cpu0/cache").exists()
            else "fallback_defaults"
        ),
        "cache_bytes": cache,
        "arguments": arguments,
    }


def save_metadata(csv_path: str | Path, metadata: dict[str, object]) -> Path:
    path = Path(csv_path)
    metadata_path = path.with_suffix(path.suffix + ".metadata.json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved metadata → {metadata_path}")
    return metadata_path
