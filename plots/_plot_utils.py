def add_cache_boundaries(ax, cache_lines=None, label_height=0.92):
    cache_lines = cache_lines or {}
    _, ymax = ax.get_ylim()
    for label, size_bytes in cache_lines.items():
        ax.axvline(
            x=size_bytes,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6,
        )
        ax.text(
            size_bytes * 1.05,
            ymax * label_height,
            label,
            fontsize=8,
            color="gray",
            va="top",
        )


def size_formatter(value, _):
    if value < 1024**2:
        return f"{int(value / 1024)}KB"
    return f"{int(value / 1024**2)}MB"


def cache_sizes_from_frame(frame):
    sizes = {}
    for level, column in (
        ("L1", "cache_l1_bytes"),
        ("L2", "cache_l2_bytes"),
        ("L3", "cache_l3_bytes"),
    ):
        if column in frame and frame[column].notna().any():
            size = int(frame[column].dropna().iloc[0])
            if size > 0:
                sizes[level] = size
    return sizes
