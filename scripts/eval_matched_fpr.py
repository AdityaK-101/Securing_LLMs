"""
Matched-FPR fair comparison (reviewer bFjj).

Tunes each score-based filter on val to FPR ≤ 5% (default), then reports
test Bypass / FPR. Includes MiniLM similarity-only, fold/split-trained
DeBERTa (ProtectAI checkpoint fine-tuned on our train), Methods A/B,
keyword, and off-the-shelf ProtectAI / Prompt Guard (threshold-only retune).

Usage:
  python scripts/eval_matched_fpr.py
  python scripts/eval_matched_fpr.py --target-fpr 0.05
  python scripts/eval_matched_fpr.py --skip-deberta --skip-portable
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, TEST_CSV, TRAIN_CSV, VAL_CSV
from src.evaluation.matched_fpr import DEFAULT_TARGET_FPR, run_matched_fpr_study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", default=TRAIN_CSV)
    parser.add_argument("--val-csv", default=VAL_CSV)
    parser.add_argument("--test-csv", default=TEST_CSV)
    parser.add_argument(
        "--target-fpr",
        type=float,
        default=DEFAULT_TARGET_FPR,
        help="Max validation FPR for the matched operating point (default 0.05)",
    )
    parser.add_argument(
        "--skip-deberta",
        action="store_true",
        help="Skip fine-tuning DeBERTa (faster)",
    )
    parser.add_argument(
        "--skip-portable",
        action="store_true",
        help="Skip ProtectAI / Prompt Guard",
    )
    parser.add_argument(
        "--out-dir",
        default=str(RESULTS_DIR / "matched_fpr"),
        help="Output directory (default results/matched_fpr)",
    )
    args = parser.parse_args()

    run_matched_fpr_study(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        target_fpr=float(args.target_fpr),
        include_portables=not args.skip_portable,
        include_deberta=not args.skip_deberta,
        out_dir=Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
