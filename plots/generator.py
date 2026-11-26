"""Generate all experiment plots from existing CSV files."""

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCRIPTS = (
    ("01_cache_hierarchy_plot.py", "01_cache_hierarchy.csv"),
    ("02_memory_layouts_plot.py", "02_memory_layouts.csv"),
    ("03_matrix_multiply_plot.py", "03_matrix_multiply.csv"),
)


def project_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data",
                        help="Directory containing experiment CSV files")
    parser.add_argument("--out-dir", default="plots/figures",
                        help="Directory to write figures")
    args = parser.parse_args()

    data_dir = project_path(args.data_dir)
    out_dir = project_path(args.out_dir)
    for script, csv_name in PLOT_SCRIPTS:
        csv_path = data_dir / csv_name
        if not csv_path.exists():
            parser.error(f"data file not found: {csv_path}")
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name(script)),
                "--data",
                str(csv_path),
                "--out-dir",
                str(out_dir),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()