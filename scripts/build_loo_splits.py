"""Deprecated alias → ``scripts/build_source_holdout.py``.

Prefer:
  python scripts/build_source_holdout.py
  # or run notebooks/build_source_holdout.ipynb
"""
from __future__ import annotations

import runpy
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    warnings.warn(
        "build_loo_splits.py is a deprecated alias; "
        "use scripts/build_source_holdout.py",
        DeprecationWarning,
        stacklevel=1,
    )
    runpy.run_path(str(ROOT / "scripts" / "build_source_holdout.py"), run_name="__main__")


if __name__ == "__main__":
    main()
