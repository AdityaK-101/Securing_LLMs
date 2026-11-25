"""Second-target True ASR for Table 1 (LOO-4 + Neuralchemy).

Default target: ``meta-llama/Llama-3.2-3B-Instruct`` (greedy, ≤512 tokens,
no system prompt). Replies are judged with the existing Qwen judge + rubric.

Gates / Bypass / FPR are unchanged (saved matched thresholds, no re-tuning).
Only True ASR is recomputed as a cache lookup on prompts each gate allows.

Does **not** run unless you invoke this script (or ``run_paper.py --second-target``).
Requires Hugging Face access to Llama-3.2-3B-Instruct when filling the cache.

  # Fill Llama replies + Qwen judge for ~1000 Table-1 test injections
  python scripts/eval_loo_second_target.py --fill-only

  # Re-score True ASR from an existing second-target cache
  python scripts/eval_loo_second_target.py --rescore-only

  # Both (default)
  python scripts/eval_loo_second_target.py

Artifacts:
  results/prompt_cache/loo/outcomes__meta-llama_Llama-3_2-3B-Instruct.jsonl
  results/loo_matched_fpr/metrics_second_target.csv
  results/loo_matched_fpr/true_asr_by_target.csv
  results/loo_matched_fpr/second_target_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LLM_CACHE_BATCH_SIZE, SECOND_TARGET_MODEL_NAME
from src.evaluation.loo_matched_fpr import LOO_FOLDS
from src.evaluation.second_target import run_second_target_suite


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model",
        default=SECOND_TARGET_MODEL_NAME,
        help=f"HF id (default: {SECOND_TARGET_MODEL_NAME})",
    )
    ap.add_argument(
        "--folds",
        nargs="+",
        default=list(LOO_FOLDS),
        help="Default: four LOO sources + neuralchemy",
    )
    ap.add_argument(
        "--fill-only",
        action="store_true",
        help="Only generate second-target replies + Qwen judge labels",
    )
    ap.add_argument(
        "--rescore-only",
        action="store_true",
        help="Only recompute True ASR from an existing second-target cache",
    )
    ap.add_argument(
        "--llm-batch-size",
        type=int,
        default=LLM_CACHE_BATCH_SIZE,
        help="Batch size for greedy generation into the outcome cache",
    )
    ap.add_argument(
        "--skip-portables",
        action="store_true",
        help="Skip ProtectAI / Prompt Guard when rescoring",
    )
    ap.add_argument(
        "--skip-deberta",
        action="store_true",
        help="Skip fine-tuned DeBERTa when rescoring",
    )
    args = ap.parse_args()
    if args.fill_only and args.rescore_only:
        raise SystemExit("Use only one of --fill-only / --rescore-only")

    fill = not args.rescore_only
    rescore = not args.fill_only
    if args.fill_only:
        fill, rescore = True, False
    if args.rescore_only:
        fill, rescore = False, True

    written = run_second_target_suite(
        model_name=args.model,
        folds=args.folds,
        fill=fill,
        rescore=rescore,
        batch_size=args.llm_batch_size,
        include_portables=not args.skip_portables,
        include_deberta=not args.skip_deberta,
    )
    print("\n[second_target] done")
    for key, path in written.items():
        print(f"  {key}: {path}")


if __name__ == "__main__":
    main()
