"""Command-line entry point for the matrix multiplication experiment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.matrix_multiply.cli import main


if __name__ == "__main__":
    main()
