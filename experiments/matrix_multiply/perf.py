import platform
import re
import subprocess
import sys
from pathlib import Path


PERF_EVENTS = (
    "cache-misses,cache-references,LLC-load-misses,LLC-loads,"
    "instructions,cycles"
)


def check_perf_available():
    if platform.system() != "Linux":
        return False
    try:
        result = subprocess.run(
            ["perf", "stat", "-e", "cache-misses", "--", "true"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except FileNotFoundError:
        print("[warn] perf not found; skipping hardware counters")
        return False
    except subprocess.TimeoutExpired:
        print("[warn] perf availability check timed out")
        return False

    if result.returncode != 0:
        print("[warn] perf counters unavailable; check permissions and event support")
        return False
    return True


def run_perf_for_config(side, dtype, threads, repeat=10, warmup=2):
    runner = Path(__file__).with_name("perf_runner.py")
    command = [
        "perf", "stat", "-e", PERF_EVENTS, sys.executable, str(runner),
        "--side", str(side), "--dtype", dtype, "--threads", str(threads),
        "--warmup", str(warmup), "--repeat", str(repeat),
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print(f"[warn] perf measurement failed for {side} {dtype}: {exc}")
        return {}

    if result.returncode != 0:
        print(f"[warn] perf measurement failed for {side} {dtype}: "
              f"{result.stderr.strip()}")
        return {}
    counters = parse_perf_output(result.stderr)
    if not counters:
        print(f"[warn] perf returned no parseable counters for {side} {dtype}")
    elif "not supported" in result.stderr.lower():
        print(f"[warn] some perf events are unsupported for {side} {dtype}")
    return counters


def parse_perf_output(stderr):
    counters = {}
    for line in stderr.splitlines():
        match = re.match(r"\s*([\d,]+)\s+([\w\-:]+)", line)
        if match:
            counters[match.group(2)] = int(match.group(1).replace(",", ""))
    return counters
