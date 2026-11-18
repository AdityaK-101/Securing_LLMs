"""Deprecated shim — use ``python scripts/print_dataset_sizes.py``."""

from __future__ import annotations

import runpy
import sys
import warnings
from pathlib import Path

warnings.warn(
    "load_dataset.py moved to scripts/print_dataset_sizes.py; "
    "update callers to that path.",
    DeprecationWarning,
    stacklevel=1,
)

if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).resolve().parent / "scripts" / "print_dataset_sizes.py"),
        run_name="__main__",
    )
    sys.exit(0)
