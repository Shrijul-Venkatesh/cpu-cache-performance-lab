"""
Plot results from experiment 01: Cache Hierarchy.

Reads:  data/01_cache_hierarchy.csv
Writes: figures/01a_stride_sweep.png
        figures/01b_size_sweep_latency.png
        figures/01b_size_sweep_throughput.png

Usage: python plots/01_cache_hierarchy_plot.py [--data PATH] [--out-dir DIR]
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from _plot_utils import add_cache_boundaries, cache_sizes_from_frame

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
})

PATTERN_COLORS = {
    "sequential":    "#2196F3",
    "strided_8":     "#9C27B0",
    "strided_64":    "#673AB7",
    "gather":        "#FF9800",
    "pointer_chase": "#F44336",
    "strided":       "#9C27B0",
}

PATTERN_LABELS = {
    "sequential":    "Sequential",
    "strided_8":     "Stride 8 elements",
    "strided_64":    "Stride 64 elements",
    "gather":        "Gather (random, pipelined)",
    "pointer_chase": "Pointer chase (serialized)",
    "strided":       "Strided",
}

def plot_stride_sweep(df_a, out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sub-experiment A: Stride Sweep\n"
                 "(fixed array > L3, vary stride length)", fontsize=13, fontweight="bold")

    strides = df_a["stride"].values
    latency = df_a["latency_per_element_ns"].values
    throughput = df_a["throughput_GB_s"].values

    ax1.plot(strides, latency, "o-", color=PATTERN_COLORS["strided"], linewidth=2, markersize=6)
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Stride (elements)")
    ax1.set_ylabel("Latency per element (ns)")
    ax1.set_title("Latency per element vs stride")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))
    cache_line_bytes = (
        int(df_a["cache_line_bytes"].dropna().iloc[0])
        if "cache_line_bytes" in df_a else 64
    )
    line_stride = cache_line_bytes // 8
    ax1.axvline(x=line_stride, color="red", linestyle="--", linewidth=0.9, alpha=0.7)
    ax1.text(line_stride * 1.05, latency.max() * 0.95,
             f"1 float64 per {cache_line_bytes}-byte line",
             fontsize=8, color="red", va="top")

    ax2.plot(strides, throughput, "o-", color=PATTERN_COLORS["strided"], linewidth=2, markersize=6)
    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Stride (elements)")
    ax2.set_ylabel("Useful payload throughput (GB/s)")
    ax2.set_title("Memory bandwidth vs stride")
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    out = out_dir / "01a_stride_sweep.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def _size_label(x, _):
    if x < 1024 ** 2:
        return f"{int(x / 1024)}KB"
    return f"{int(x / 1024 ** 2)}MB"


def plot_size_sweep(df_b, out_dir, cache_lines):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Sub-experiment B: Dataset Size Sweep — Latency per Element",
                 fontsize=13, fontweight="bold")

    for pattern in ["sequential", "strided_8", "strided_64", "gather", "pointer_chase"]:
        sub = df_b[df_b["pattern"] == pattern].sort_values("working_set_bytes")
        if sub.empty:
            continue
        ax.plot(sub["working_set_bytes"], sub["latency_per_element_ns"],
                "o-", color=PATTERN_COLORS.get(pattern, "black"),
                label=PATTERN_LABELS.get(pattern, pattern),
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Working set (array + access indices)")
    ax.set_ylabel("Latency per element (ns, log scale)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_size_label))

    ax.autoscale()
    add_cache_boundaries(ax, cache_lines, label_height=0.9)
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "01b_size_sweep_latency.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Sub-experiment B: Dataset Size Sweep — Throughput",
                 fontsize=13, fontweight="bold")

    for pattern in ["sequential", "strided_8", "strided_64", "gather", "pointer_chase"]:
        sub = df_b[df_b["pattern"] == pattern].sort_values("working_set_bytes")
        if sub.empty:
            continue
        ax.plot(sub["working_set_bytes"], sub["throughput_GB_s"],
                "o-", color=PATTERN_COLORS.get(pattern, "black"),
                label=PATTERN_LABELS.get(pattern, pattern),
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Working set (array + access indices)")
    ax.set_ylabel("Useful payload throughput (GB/s)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_size_label))

    ax.autoscale()
    add_cache_boundaries(ax, cache_lines, label_height=0.9)
    ax.legend(framealpha=0.9)
    fig.tight_layout()

    out = out_dir / "01b_size_sweep_throughput.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment 01 results")
    parser.add_argument("--data", type=str, default="data/01_cache_hierarchy.csv")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        parser.error(
            f"data file not found: {data_path}; "
            "run experiments/cache_hierarchy/experiment.py first"
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    df_a = df[df["subexp"] == "A"].copy()
    df_b = df[df["subexp"] == "B"].copy()

    df_a["stride"] = pd.to_numeric(df_a["stride"], errors="coerce")
    cache_lines = cache_sizes_from_frame(df)

    if not df_a.empty:
        plot_stride_sweep(df_a, out_dir)
    else:
        print("[warn] No sub-experiment A data found, skipping stride plot")

    if not df_b.empty:
        plot_size_sweep(df_b, out_dir, cache_lines)
    else:
        print("[warn] No sub-experiment B data found, skipping size sweep plots")

    print("\nAll figures saved to", out_dir)


if __name__ == "__main__":
    main()
