"""
Plot results from experiment 02: NumPy Memory Layouts.

Reads:  data/02_memory_layouts.csv
Writes: figures/02_latency.png
        figures/02_throughput.png
        figures/02_copy_breakeven.png

Usage: python plots/02_memory_layouts_plot.py [--data PATH] [--out-dir DIR]
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

VARIANT_COLORS = {
    "row_sum_c":             "#2196F3",
    "row_sum_f":             "#4CAF50",
    "col_sum_c":             "#F44336",
    "col_sum_f":             "#9C27B0",
    "transpose_view_sum":    "#FF9800",
    "slice_sum":             "#795548",
    "transpose_view_reuse":  "#FF9800",
    "transpose_copy_reuse":  "#4CAF50",
}

VARIANT_LABELS = {
    "row_sum_c":             "Row reduction (C-order)",
    "row_sum_f":             "Row reduction (Fortran-order)",
    "col_sum_c":             "Column reduction (C-order)",
    "col_sum_f":             "Column reduction (Fortran-order)",
    "transpose_view_sum":    "Transpose view reduction",
    "slice_sum":             "Strided slice reduction",
    "transpose_view_reuse":  "Transpose view reused",
    "transpose_copy_reuse":  "Copy once, then contiguous reductions",
}


def plot_latency(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 02: Latency per Element vs Matrix Size",
                 fontsize=13, fontweight="bold")

    for variant in [
        "row_sum_c", "row_sum_f", "col_sum_c", "col_sum_f",
        "transpose_view_sum", "slice_sum",
    ]:
        sub = df[df["variant"] == variant].sort_values("matrix_size_bytes")
        if sub.empty:
            continue
        ax.plot(sub["matrix_size_bytes"], sub["latency_per_element_ns"],
                "o-", color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Matrix size (bytes)")
    ax.set_ylabel("Latency per element (ns, log scale)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    ax.autoscale()
    add_cache_boundaries(ax, cache_sizes_from_frame(df))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = out_dir / "02_latency.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_throughput(df, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle("Experiment 02: Throughput vs Matrix Size",
                 fontsize=13, fontweight="bold")

    for variant in [
        "row_sum_c", "row_sum_f", "col_sum_c", "col_sum_f",
        "transpose_view_sum", "slice_sum",
    ]:
        sub = df[df["variant"] == variant].sort_values("matrix_size_bytes")
        if sub.empty:
            continue
        ax.plot(sub["matrix_size_bytes"], sub["throughput_GB_s"],
                "o-", color=VARIANT_COLORS[variant],
                label=VARIANT_LABELS[variant],
                linewidth=2, markersize=5)

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (bytes)")
    ax.set_ylabel("Effective logical input throughput (GB/s; not DRAM bandwidth)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    ax.autoscale()
    add_cache_boundaries(ax, cache_sizes_from_frame(df))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = out_dir / "02_throughput.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def plot_copy_breakeven(df, out_dir):
    """
    Show where paying for ascontiguousarray() upfront becomes cheaper than
    operating directly on the non-contiguous transposed view.
    """
    view = df[df["variant"] == "transpose_view_reuse"]
    contig = df[df["variant"] == "transpose_copy_reuse"]

    if view.empty or contig.empty:
        print("[warn] Not enough data for copy break-even plot")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Experiment 02: Copy Break-Even\n"
                 "When is ascontiguousarray() + fast op cheaper than slow op on view?",
                 fontsize=13, fontweight="bold")

    for operation_repeats, linestyle in ((1, "-"), (4, "--")):
        view_run = view[view["operation_repeats"] == operation_repeats].sort_values(
            "matrix_size_bytes"
        )
        copy_run = contig[contig["operation_repeats"] == operation_repeats].sort_values(
            "matrix_size_bytes"
        )
        if view_run.empty or copy_run.empty:
            continue
        sizes = view_run["matrix_size_bytes"].values
        view_t = view_run["time_s"].values * 1000
        copy_t = copy_run["time_s"].values * 1000
        ax.plot(sizes, view_t, marker="o", linestyle=linestyle,
                color=VARIANT_COLORS["transpose_view_reuse"],
                label=f"View, {operation_repeats} reduction(s)")
        ax.plot(sizes, copy_t, marker="o", linestyle=linestyle,
                color=VARIANT_COLORS["transpose_copy_reuse"],
                label=f"Copy once + reuse, {operation_repeats} reduction(s)")
        ax.fill_between(sizes, view_t, copy_t, where=(copy_t < view_t),
                        alpha=0.08, color=VARIANT_COLORS["transpose_copy_reuse"])

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Matrix size (bytes)")
    ax.set_ylabel("Total time including copy (ms)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(size_formatter))
    ax.autoscale()
    add_cache_boundaries(ax, cache_sizes_from_frame(df))
    ax.legend(framealpha=0.9, fontsize=9)
    fig.tight_layout()

    out = out_dir / "02_copy_breakeven.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot experiment 02 results")
    parser.add_argument("--data", type=str, default="data/02_memory_layouts.csv")
    parser.add_argument("--out-dir", type=str, default="figures")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        parser.error(
            f"data file not found: {data_path}; "
            "run experiments/memory_layouts/experiment.py first"
        )

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    plot_latency(df, out_dir)
    plot_throughput(df, out_dir)
    plot_copy_breakeven(df, out_dir)

    print("\nAll figures saved to", out_dir)


if __name__ == "__main__":
    main()
