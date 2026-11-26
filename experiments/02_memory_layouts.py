"""Command-line entry point for the NumPy memory-layout experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.memory_layouts.cli import main


if __name__ == "__main__":
    main()
