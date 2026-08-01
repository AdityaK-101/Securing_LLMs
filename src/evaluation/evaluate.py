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

Confusion matrices remain sanitizer-only — judge output does not affect
TP/TN/FP/FN computation.
"""

import pandas as pd

from ..config import RANDOM_SEED, TEST_CSV, TRAIN_CSV
from ..models.target_llm import LLMRunner
from ..sanitizers import get_all_sanitizers
from .metrics import compute_metrics
from .pipeline import (
    release_all_sanitizer_models,
    run_two_phase_pipeline,
    should_skip_judge,
)

__all__ = [
    "RANDOM_SEED",
    "TEST_CSV",
    "TRAIN_CSV",
    "evaluate",
    "evaluate_sanitizer",
]


def evaluate(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
    judge=None,
) -> tuple:
    """
    Run the full two-layer evaluation pipeline.

    Memory-safe two-phase flow (only one large model in RAM at a time):
      Phase 1: prompt → sanitizer → target LLM (if not blocked)
      Phase 2: release target LLM → load judge → COMPLIED/REFUSED on unblocked injections

    Returns:
        logs_df    — per-prompt-per-method DataFrame
        metrics_df — per-method summary (Bypass Rate, True ASR, FPR, TP/TN/FP/FN)
    """
    df_test = pd.read_csv(test_csv)
    sanitizers = get_all_sanitizers(train_csv=train_csv)
    runner = LLMRunner() if use_llm else None

    skip_judge = should_skip_judge(use_llm, runner)

    print(f"\n[Evaluate] Running experiment on {len(df_test)} prompts × {len(sanitizers)} methods...")
    print(f"[Evaluate] Phase 1: sanitizer + target LLM")
    print(f"[Evaluate] Phase 2: judge (COMPLIED/REFUSED) — {'enabled' if not skip_judge else 'disabled'}")

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
    )

    logs_df = pd.DataFrame(records)
    metrics_df, cm_dict = compute_metrics(logs_df)

    return logs_df, metrics_df, cm_dict


def evaluate_sanitizer(
    sanitizer,
    test_csv: str = TEST_CSV,
    use_llm: bool = True,
) -> tuple:
    """
    Run the full two-layer evaluation pipeline for a single sanitizer.

    Same logic as evaluate(), but accepts one sanitizer instance instead of
    calling get_all_sanitizers(). Used by ablation studies.
    """
    df_test = pd.read_csv(test_csv)
    runner = LLMRunner() if use_llm else None

    skip_judge = should_skip_judge(use_llm, runner)

    print(
        f"\n[Evaluate] Running experiment on {len(df_test)} prompts "
        f"× 1 method ({sanitizer.name})..."
    )
    print(f"[Evaluate] Phase 1: sanitizer + target LLM")
    print(
        f"[Evaluate] Phase 2: judge (COMPLIED/REFUSED) — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

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
    )

    logs_df = pd.DataFrame(records)
    metrics_df, cm_dict = compute_metrics(logs_df)

    return logs_df, metrics_df, cm_dict
