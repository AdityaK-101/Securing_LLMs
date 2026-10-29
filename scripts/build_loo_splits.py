"""Alias: rebuild source-holdout via notebooks/build_source_holdout.ipynb.

Usage:
  python scripts/build_loo_splits.py
"""
from __future__ import annotations

import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    runpy.run_path(str(ROOT / "scripts" / "build_source_holdout.py"), run_name="__main__")


if __name__ == "__main__":
    main()
