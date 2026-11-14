"""
Generate AAAI paper figures into ``report/aaai/images/``.

Usage:
  python scripts/generate_paper_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.visualization.paper_figures import generate_all_paper_figures


def main() -> None:
    paths = generate_all_paper_figures()
    print(f"\nGenerated {len(paths)} figures:")
    for p in paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
