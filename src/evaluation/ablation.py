"""
src/ablation.py
===============
Context-Aware sanitizer ablation studies.

Two separate modes:
  1. Weighted Signal Ablation — hard triggers disabled; one signal removed at a time.
  2. Hard Trigger Ablation — full weighted model; individual hard triggers removed.

Models (MiniLM, distilgpt2, target LLM, judge) are loaded once per study and
shared across all variants within that study.
"""

import pandas as pd

from ..config import TEST_CSV, TRAIN_CSV
from ..models.target_llm import LLMRunner
from ..sanitizers import ContextAwareSanitizer
from ..utils.gpu import empty_cuda_cache
from .metrics import compute_metrics
from .pipeline import run_two_phase_pipeline, should_skip_judge
WEIGHTED_ABLATION_VARIANTS = [
    ("Full Weighted Model", {"disable_hard_triggers": True}),
    ("-No Regex", {"disable_hard_triggers": True, "disable_regex": True}),
    ("-No Keyword", {"disable_hard_triggers": True, "disable_keyword": True}),
    ("-No Semantic", {"disable_hard_triggers": True, "disable_semantic": True}),
    ("-No Intent", {"disable_hard_triggers": True, "disable_intent": True}),
    ("-No Roleplay", {"disable_hard_triggers": True, "disable_roleplay": True}),
    (
        "-No Instruction Shift",
        {"disable_hard_triggers": True, "disable_instruction_shift": True},
    ),
    (
        "-No Objective Conflict",
        {"disable_hard_triggers": True, "disable_objective_conflict": True},
    ),
    ("-No Perplexity", {"disable_hard_triggers": True, "disable_perplexity": True}),
]

HARD_TRIGGER_ABLATION_VARIANTS = [
    ("Full Model", {}),
    ("-No Semantic Hard Trigger", {"disable_semantic_hard_trigger": True}),
    ("-No Intent Hard Trigger", {"disable_intent_hard_trigger": True}),
    (
        "-No Roleplay+Keyword Hard Trigger",
        {"disable_roleplay_keyword_hard_trigger": True},
    ),
    ("-No Hard Triggers", {"disable_hard_triggers": True}),
]

ABLATION_EXPORT_COLUMNS = [
    "variant",
    "Bypass_Rate_%",
    "True_ASR_%",
    "Detection_Rate",
    "FPR_%",
    "TP",
    "FN",
    "FP",
    "TN",
    "successful_attack_count",
]


def _evaluate_ablation_variants(
    study_name: str,
    variants: list,
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Evaluate multiple Context-Aware variants with shared model loading.

    Loads MiniLM/distilgpt2 once, runs Phase 1 (sanitizer + target LLM) for every
    variant, then Phase 2 (judge) once for all pending records.
    """
    df_test = pd.read_csv(test_csv)

    print(f"\n[Ablation] {study_name} — {len(variants)} variants")
    print("[Ablation] Loading shared Context-Aware auxiliary models once...")
    base = ContextAwareSanitizer(train_csv=train_csv)
    sanitizers = [
        base.copy_with(name=variant_name, **kwargs)
        for variant_name, kwargs in variants
    ]

    runner = LLMRunner() if use_llm else None
    skip_judge = should_skip_judge(use_llm, runner)

    print(
        f"[Ablation] Phase 1: {len(df_test)} prompts × "
        f"{len(sanitizers)} variants"
    )
    print(
        f"[Ablation] Phase 2: judge — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

    def _release_base_models():
        if hasattr(base, "release_models"):
            base.release_models()

    def _cleanup_after_phase1():
        _release_base_models()
        if runner:
            runner.release()
        empty_cuda_cache()

    records = run_two_phase_pipeline(
        df_test.iterrows(),
        sanitizers,
        runner,
        skip_judge,
        get_record_id=lambda item: item[1]["id"],
        get_prompt=lambda item: str(item[1]["prompt"]),
        get_label=lambda item: str(item[1]["label"]),
        defer_judge=lambda label, sj: label == "injection" and not sj,
        release_sanitizers=_release_base_models,
        phase1_desc="Phase 1 (sanitizer + target LLM)",
        phase1_total=len(df_test),
        phase2_log_prefix="[Ablation]",
        phase2_item_name="unblocked injections",
        cleanup_after_phase1=_cleanup_after_phase1,
    )

    logs_df = pd.DataFrame(records)
    metrics_df, _ = compute_metrics(logs_df)

    rows = []
    for _, mrow in metrics_df.iterrows():
        row = mrow.to_dict()
        row["variant"] = row.pop("method")
        rows.append(row)

    return pd.DataFrame(rows)[ABLATION_EXPORT_COLUMNS]


def run_weighted_ablation_study(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Weighted Signal Ablation: all hard triggers disabled; one signal removed per run.
    """
    return _evaluate_ablation_variants(
        "Weighted Signal Ablation",
        WEIGHTED_ABLATION_VARIANTS,
        test_csv=test_csv,
        train_csv=train_csv,
        use_llm=use_llm,
    )


def run_hard_trigger_ablation_study(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Hard Trigger Ablation: full weighted model; individual hard triggers removed.
    """
    return _evaluate_ablation_variants(
        "Hard Trigger Ablation",
        HARD_TRIGGER_ABLATION_VARIANTS,
        test_csv=test_csv,
        train_csv=train_csv,
        use_llm=use_llm,
    )
