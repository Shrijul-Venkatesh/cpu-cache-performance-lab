import csv
from collections.abc import Iterable, Sequence
from pathlib import Path


def save_csv(
    rows: Iterable[dict[str, object]],
    fieldnames: Sequence[str],
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV → {path}")
