"""
One-time (or resumable) generation + judge pass for unique attack prompts.

Mentor speedup: run Qwen and the judge once per injection text, save to disk,
then every method/fold True ASR is a lookup. Safe to stop and re-run — appends
only missing rows.

Cache paths match ``src/run_experiments.py`` / ``evaluate.py``:

  --loo    → results/prompt_cache/loo/{outcomes.jsonl,features.pkl}
  --suite  → results/prompt_cache/{outcomes.jsonl,features.pkl}

Typical local use:
    python scripts/precompute_prompt_outcomes.py --loo --batch-size 4
    python scripts/precompute_prompt_outcomes.py --suite --batch-size 4

On a free Kaggle / Colab GPU:
    python scripts/precompute_prompt_outcomes.py --loo --batch-size 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PROMPT_CACHE_DIR,
    SOURCE_HOLDOUT_DIR,
    TEST_CSV,
)
from src.evaluation.prompt_cache import (  # noqa: E402
    FeatureCache,
    OutcomeCache,
    collect_prompts_from_csvs,
)
from src.models.judge import ComplianceJudge  # noqa: E402
from src.models.target_llm import LLMRunner  # noqa: E402
from src.utils.gpu import empty_cuda_cache  # noqa: E402

LOO_FOLDS = (
    "simsonsun",
    "mpdd",
    "prompt_injection",
    "bravansky",
    "neuralchemy",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loo",
        action="store_true",
        help="Collect unique injections from data/splits/source_holdout/*/*.csv",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Include mixed-suite test.csv injections (v3.0 ~60 injections)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Greedy generation batch size (default: 4)",
    )
    parser.add_argument(
        "--features-only",
        action="store_true",
        help="Only precompute MiniLM embeddings (no Qwen/judge)",
    )
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Generate LLM replies only (skip judge)",
    )
    args = parser.parse_args()

    if not args.loo and not args.suite:
        args.loo = True  # default: LOO holdout tests

    csvs: list[Path] = []
    if args.suite:
        csvs.append(Path(TEST_CSV))
    if args.loo:
        for fold in LOO_FOLDS:
            for split in ("train.csv", "val.csv", "test.csv"):
                path = SOURCE_HOLDOUT_DIR / fold / split
                if path.is_file():
                    csvs.append(path)

    if not csvs:
        raise SystemExit("No CSVs found. Build source_holdout first or pass --suite.")

    # Same layout as run_experiments / evaluate
    if args.loo:
        cache_dir = PROMPT_CACHE_DIR / "loo"
    else:
        cache_dir = PROMPT_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute] Cache dir: {cache_dir}")

    all_prompts = collect_prompts_from_csvs(csvs)
    inj_prompts = collect_prompts_from_csvs(csvs, labels={"injection"})
    print(f"[precompute] Unique prompts: {len(all_prompts)} (injections: {len(inj_prompts)})")

    features = FeatureCache(cache_dir / "features.pkl")
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        features.ensure_embeddings(all_prompts, model)
        del model
    except Exception as exc:
        print(f"[precompute] MiniLM skipped: {exc}")

    if args.features_only:
        print("[precompute] features-only done.")
        return

    outcomes = OutcomeCache(cache_dir / "outcomes.jsonl")
    runner = LLMRunner()
    try:
        outcomes.ensure_llm(inj_prompts, runner, batch_size=max(1, args.batch_size))
    finally:
        runner.release()
        empty_cuda_cache()

    if not args.no_judge:
        judge = ComplianceJudge()
        try:
            outcomes.ensure_judge(inj_prompts, judge)
        finally:
            judge.release()
            empty_cuda_cache()

    print(f"[OK] Outcomes: {outcomes.path}")
    print(f"[OK] Features: {features.path}")
    if args.loo and args.suite:
        print(
            "[note] --loo and --suite together write the LOO cache only. "
            "Run again with --suite alone to warm results/prompt_cache/ for the mixed suite."
        )
    print(
        "Re-run the same command after a crash to resume. "
        "Then: python -m src.run_experiments --loo "
        "or python -m src.run_experiments --suite a"
    )


if __name__ == "__main__":
    main()
