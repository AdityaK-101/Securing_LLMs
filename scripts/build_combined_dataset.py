"""Rebuild the grown mixed combined corpus + stratified train/val/test.

Archives the frozen v2.0 350-prompt files on first run, then writes:

  data/cleaned/combined_dataset.csv
  data/cleaned/combined_manifest.json
  data/splits/{train,val,test}.csv

Usage:
  python scripts/build_combined_dataset.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.splits.combined import build_combined_corpus


def main() -> None:
    manifest = build_combined_corpus()
    print(json.dumps(manifest["splits"], indent=2))
    print(
        f"\nDone. Total={manifest['total']} "
        f"(injection={manifest['injection']}, benign={manifest['benign']})"
    )


if __name__ == "__main__":
    main()
