"""Rebuild paper tables for LOO matched-FPR (Wilson CIs, Neuralchemy, ID, dedup)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.loo_matched_fpr_report import build_paper_tables


def main() -> None:
    build_paper_tables()


if __name__ == "__main__":
    main()
