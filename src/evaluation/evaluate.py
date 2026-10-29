"""
src/evaluation/evaluate.py
==========================
Core evaluation logic for Milestone 3.

Two-layer evaluation design:

  Layer 1 — Sanitizer Evaluation (blocked / unblocked only):
    Detection Rate, Bypass Rate, FPR, TP, TN, FP, FN, confusion matrices.
    Bypass Rate = unblocked injection prompts / total injection prompts.
    This measures sanitizer bypass, NOT whether the target LLM actually complied.

  Layer 2 — Model Security Evaluation (judge on target LLM outputs):
    True ASR = successful attacks / total injection prompts, where a
    successful attack requires sanitizer bypass AND judge_label == COMPLIED.
    The judge model evaluates the relationship between original prompt
    and target LLM output via semantic reasoning (no keyword heuristics).

Speed path (default when use_llm=True):
  - One greedy Qwen reply + judge label per unique *injection* prompt (OutcomeCache).
  - Benign prompts: filters only (never Qwen / judge).
  - Generate-all → unload Qwen → judge-all; disk-append for crash resume.
  - Shared MiniLM / DistilGPT-2 FeatureCache across methods (and folds when shared).

Confusion matrices remain sanitizer-only — judge output does not affect
TP/TN/FP/FN computation.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import (
    LLM_CACHE_BATCH_SIZE,
    PROMPT_CACHE_DIR,
    RANDOM_SEED,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
)
from ..sanitizers import get_all_sanitizers
from .metrics import compute_metrics
from .pipeline import (
    release_all_sanitizer_models,
    run_two_phase_pipeline,
    should_skip_judge,
)
from .prompt_cache import (
    FeatureCache,
    OutcomeCache,
    attach_feature_cache,
    find_context_sanitizer,
    warm_minilm_embeddings,
    warm_raw_ppl_from_sanitizer,
)

__all__ = [
    "RANDOM_SEED",
    "TEST_CSV",
    "TRAIN_CSV",
    "VAL_CSV",
    "evaluate",
    "evaluate_sanitizer",
]


def evaluate(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    use_llm: bool = True,
    judge=None,
    include_method_a: bool = True,
    include_method_b: bool = True,
    include_similarity_only: bool = False,
    include_deberta_finetuned: bool = False,
    *,
    use_cached_lookup: bool | None = None,
    outcome_cache: OutcomeCache | None = None,
    feature_cache: FeatureCache | None = None,
    batch_size: int | None = None,
    warm_features: bool = True,
) -> tuple:
    """
    Run the full two-layer evaluation pipeline.

    Default LLM path uses the mentor speedup: one generate+judge pass per
    unique injection prompt, benign = filters only, resumable OutcomeCache.

    Returns:
        logs_df    — per-prompt-per-method DataFrame
        metrics_df — per-method summary (Bypass Rate, True ASR, FPR, TP/TN/FP/FN)
        cm_dict    — per-method confusion matrices
    """
    if use_cached_lookup is None:
        use_cached_lookup = bool(use_llm)
    bs = int(batch_size) if batch_size is not None else LLM_CACHE_BATCH_SIZE

    df_test = pd.read_csv(test_csv)
    feat = feature_cache
    if feat is None and warm_features and (
        include_method_a or include_method_b or include_similarity_only
    ):
        feat = FeatureCache(PROMPT_CACHE_DIR / "features.pkl")

    # Pre-encode MiniLM before Method A/B train so galleries hit the disk cache
    if feat is not None and warm_features:
        prompts = df_test["prompt"].dropna().astype(str).tolist()
        for path in (train_csv, val_csv):
            if path and Path(path).is_file():
                prompts.extend(
                    pd.read_csv(path)["prompt"].dropna().astype(str).tolist()
                )
        warm_minilm_embeddings(feat, prompts)

    sanitizers = get_all_sanitizers(
        train_csv=train_csv,
        val_csv=val_csv,
        include_method_a=include_method_a,
        include_method_b=include_method_b,
        include_similarity_only=include_similarity_only,
        include_deberta_finetuned=include_deberta_finetuned,
        feature_cache=feat,
    )

    if feat is not None and warm_features:
        prompts = df_test["prompt"].dropna().astype(str).tolist()
        for path in (train_csv, val_csv):
            if path and Path(path).is_file():
                prompts.extend(
                    pd.read_csv(path)["prompt"].dropna().astype(str).tolist()
                )
        ctx = find_context_sanitizer(sanitizers)
        warm_raw_ppl_from_sanitizer(feat, prompts, ctx)

    skip_judge = should_skip_judge(use_llm, None)
    cache = outcome_cache
    if use_cached_lookup and cache is None:
        cache = OutcomeCache(PROMPT_CACHE_DIR / "outcomes.jsonl")

    print(
        f"\n[Evaluate] Running experiment on {len(df_test)} prompts × "
        f"{len(sanitizers)} methods..."
    )
    if use_cached_lookup:
        print(
            "[Evaluate] Speed path: generate-all → unload → judge-all "
            f"(batch_size={bs}); benign = filters only; True ASR = lookup."
        )
    else:
        print("[Evaluate] Legacy path: LLM per unblocked (method, prompt).")
    print(
        f"[Evaluate] Judge (COMPLIED/REFUSED) — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

    # Cached path loads/releases its own runner; do not double-load here.
    runner = None
    if use_llm and not use_cached_lookup:
        from ..models.target_llm import LLMRunner

        runner = LLMRunner()

    records = run_two_phase_pipeline(
        df_test.iterrows(),
        sanitizers,
        runner,
        skip_judge,
        get_record_id=lambda item: item[1]["id"],
        get_prompt=lambda item: str(item[1]["prompt"]),
        get_label=lambda item: str(item[1]["label"]),
        defer_judge=lambda label, sj: label == "injection" and not sj,
        release_sanitizers=lambda: release_all_sanitizer_models(sanitizers),
        phase1_desc="Phase 1 (target LLM)",
        phase1_total=len(df_test),
        phase2_log_prefix="[Evaluate]",
        phase2_item_name="unblocked injections",
        use_cached_lookup=use_cached_lookup,
        outcome_cache=cache,
        cache_use_llm=use_llm,
        batch_size=bs,
    )

    logs_df = pd.DataFrame(records)
    metrics_df, cm_dict = compute_metrics(logs_df)

    return logs_df, metrics_df, cm_dict


def evaluate_sanitizer(
    sanitizer,
    test_csv: str = TEST_CSV,
    use_llm: bool = True,
    *,
    use_cached_lookup: bool | None = None,
    outcome_cache: OutcomeCache | None = None,
    feature_cache: FeatureCache | None = None,
    batch_size: int | None = None,
) -> tuple:
    """
    Run the full two-layer evaluation pipeline for a single sanitizer.

    Same logic as evaluate(), but accepts one sanitizer instance instead of
    calling get_all_sanitizers(). Used by ablation studies.
    """
    if use_cached_lookup is None:
        use_cached_lookup = bool(use_llm)
    bs = int(batch_size) if batch_size is not None else LLM_CACHE_BATCH_SIZE

    df_test = pd.read_csv(test_csv)
    if feature_cache is not None:
        attach_feature_cache([sanitizer], feature_cache)

    skip_judge = should_skip_judge(use_llm, None)
    cache = outcome_cache
    if use_cached_lookup and cache is None:
        cache = OutcomeCache(PROMPT_CACHE_DIR / "outcomes.jsonl")

    print(
        f"\n[Evaluate] Running experiment on {len(df_test)} prompts "
        f"× 1 method ({sanitizer.name})..."
    )
    if use_cached_lookup:
        print(
            "[Evaluate] Speed path: generate-all → unload → judge-all; "
            "benign = filters only."
        )
    print(
        f"[Evaluate] Judge (COMPLIED/REFUSED) — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

    runner = None
    if use_llm and not use_cached_lookup:
        from ..models.target_llm import LLMRunner

        runner = LLMRunner()

    records = run_two_phase_pipeline(
        df_test.iterrows(),
        [sanitizer],
        runner,
        skip_judge,
        get_record_id=lambda item: item[1]["id"],
        get_prompt=lambda item: str(item[1]["prompt"]),
        get_label=lambda item: str(item[1]["label"]),
        defer_judge=lambda label, sj: label == "injection" and not sj,
        release_sanitizers=lambda: release_all_sanitizer_models([sanitizer]),
        phase1_desc="Phase 1 (target LLM)",
        phase1_total=len(df_test),
        phase2_log_prefix="[Evaluate]",
        phase2_item_name="unblocked injections",
        use_cached_lookup=use_cached_lookup,
        outcome_cache=cache,
        cache_use_llm=use_llm,
        batch_size=bs,
    )

    logs_df = pd.DataFrame(records)
    metrics_df, cm_dict = compute_metrics(logs_df)

    return logs_df, metrics_df, cm_dict
