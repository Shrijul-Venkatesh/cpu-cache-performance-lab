"""
Plot results from experiment 03: Matrix Multiplication.

Reads:  data/03_matrix_multiply.csv
Writes: figures/03_gflops.png
        figures/03_dtype_comparison.png
        figures/03_thread_scaling.png
        figures/03_cache_misses.png  (only if perf data present)

Usage: python plots/03_matrix_multiply_plot.py [--data PATH] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from _plot_utils import add_cache_boundaries, cache_sizes_from_frame, size_formatter

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

THREAD_COLORS = {1: "#2196F3", 2: "#4CAF50", 4: "#FF9800", 8: "#F44336"}
DTYPE_COLORS = {"float32": "#FF9800", "float64": "#2196F3"}

def plot_gflops(df, out_dir):
    sub = df[df["dtype"] == "float64"]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 03: GFLOPS/s vs Matrix Size (float64)",
                 fontsize=13, fontweight="bold")

    for n_threads in sorted(sub["n_threads"].unique()):
        s = sub[sub["n_threads"] == n_threads].sort_values("working_set_bytes")
        ax.plot(s["working_set_bytes"], s["gflops_s"],
                "o-", color=THREAD_COLORS.get(n_threads, "black"),
                label=f"{n_threads} thread{'s' if n_threads > 1 else ''}",
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working set (A + B + C, bytes)")
    ax.set_ylabel("GFLOPS/s")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    ax.autoscale()
    add_cache_boundaries(ax, cache_sizes_from_frame(sub))
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "03_gflops.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_dtype_comparison(df, out_dir):
    sub = df[df["n_threads"] == 1]
    if sub.empty:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Experiment 03: float32 vs float64 (single-threaded)",
                 fontsize=13, fontweight="bold")

    for dtype in ["float32", "float64"]:
        s = sub[sub["dtype"] == dtype].sort_values("working_set_bytes")
        if s.empty:
            continue
        ax1.plot(s["working_set_bytes"], s["gflops_s"],
                 "o-", color=DTYPE_COLORS[dtype], label=dtype,
                 linewidth=2, markersize=5)
        ax2.plot(s["working_set_bytes"], s["nominal_input_output_GB_s"],
                 "o-", color=DTYPE_COLORS[dtype], label=dtype,
                 linewidth=2, markersize=5)

    for ax, ylabel, title in [
        (ax1, "GFLOPS/s", "Compute throughput"),
        (ax2, "Nominal input/output (GB/s; not DRAM bandwidth)",
         "Algorithmic input/output estimate"),
    ]:
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Working set (A + B + C, bytes)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
        ax.autoscale()
        add_cache_boundaries(ax, cache_sizes_from_frame(sub))
        ax.legend(framealpha=0.9)

    fig.tight_layout()
    out = out_dir / "03_dtype_comparison.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_thread_scaling(df, out_dir):
    largest = df["side"].max()
    sub = df[(df["side"] == largest) & (df["dtype"] == "float64")]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Experiment 03: Thread Scaling ({largest}x{largest} float64)",
                 fontsize=13, fontweight="bold")

    thread_counts = sorted(sub["n_threads"].unique())
    gflops = [sub[sub["n_threads"] == t]["gflops_s"].values[0] for t in thread_counts]
    ideal = [gflops[0] * t for t in thread_counts]

    ax.plot(thread_counts, gflops, "o-", color="#2196F3", linewidth=2,
            markersize=8, label="Measured")
    ax.plot(thread_counts, ideal, "--", color="gray", linewidth=1.5,
            label="Ideal linear scaling")

    ax.set_xlabel("Thread count")
    ax.set_ylabel("GFLOPS/s")
    ax.set_xticks(thread_counts)
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "03_thread_scaling.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_cache_misses(df, out_dir):
    sub = df[df["n_threads"] == 1].copy()
    sub["cache_misses"] = pd.to_numeric(sub["cache_misses"], errors="coerce")
    sub = sub.dropna(subset=["cache_misses"])
    if sub.empty:
        print("[info] No parseable perf hardware-counter data — skipping cache miss plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 03: Perf Process Cache Misses vs Working Set\n"
                 "(single-threaded; includes runner setup)",
                 fontsize=13, fontweight="bold")

    for dtype in ["float32", "float64"]:
        s = sub[sub["dtype"] == dtype].sort_values("working_set_bytes")
        if s.empty:
            continue
        ax.plot(s["working_set_bytes"], s["cache_misses"],
                "o-", color=DTYPE_COLORS[dtype], label=dtype,
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Working set (A + B + C, bytes)")
    ax.set_ylabel("Cache misses (log scale)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    ax.autoscale()
    add_cache_boundaries(ax, cache_sizes_from_frame(sub))
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "03_cache_misses.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment 03 results")
    parser.add_argument("--data", type=str, default="data/03_matrix_multiply.csv")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        parser.error(f"data file not found: {data_path}; run experiments/03_matrix_multiply.py first")

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    plot_gflops(df, out_dir)
    plot_dtype_comparison(df, out_dir)
    plot_thread_scaling(df, out_dir)
    plot_cache_misses(df, out_dir)

    print("\nAll figures saved to", out_dir)


if __name__ == "__main__":
    main()
