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

import os

import pandas as pd
from tqdm import tqdm

from .evaluate import (
    TEST_CSV,
    TRAIN_CSV,
    _attack_success,
    _compute_metrics,
)
from .gpu_utils import empty_cuda_cache
from .judge import ComplianceJudge
from .llm_runner import LLMRunner
from .sanitizers import ContextAwareSanitizer

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
    skip_judge = (
        os.environ.get("SKIP_JUDGE", "0").strip() == "1"
        or not use_llm
        or runner is None
        or runner.is_mock()
    )

    records = []
    pending_judge = []

    print(
        f"[Ablation] Phase 1: {len(df_test)} prompts × "
        f"{len(sanitizers)} variants"
    )
    print(
        f"[Ablation] Phase 2: Qwen judge — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

    for _, row in tqdm(
        df_test.iterrows(),
        total=len(df_test),
        desc="Phase 1 (sanitizer + target LLM)",
    ):
        prompt_id = row["id"]
        prompt = str(row["prompt"])
        label = str(row["label"])

        for san in sanitizers:
            sanitized, blocked = san.sanitize(prompt)

            if blocked:
                llm_output = "[BLOCKED]"
                judge_label = "BLOCKED"
                judge_reasoning = ""
            elif runner:
                llm_output = runner.run(sanitized)
                if label == "injection" and not skip_judge:
                    judge_label = None
                    judge_reasoning = ""
                elif label == "injection":
                    judge_label = "SKIPPED"
                    judge_reasoning = ""
                else:
                    judge_label = "N/A"
                    judge_reasoning = ""
            else:
                llm_output = "[LLM_SKIPPED]"
                judge_label = "SKIPPED"
                judge_reasoning = ""

            rec = {
                "id":               prompt_id,
                "label":            label,
                "method":           san.name,
                "blocked":          blocked,
                "original":         prompt,
                "sanitized":        sanitized,
                "llm_output":       llm_output,
                "judge_label":      judge_label,
                "judge_reasoning":  judge_reasoning,
                "attack_success":   False,
            }
            if judge_label is None:
                pending_judge.append(len(records))
            elif judge_label not in (None, "N/A", "SKIPPED"):
                rec["attack_success"] = _attack_success(label, blocked, judge_label)
            records.append(rec)

    if pending_judge and not skip_judge:
        print(f"\n[Ablation] Phase 2: judging {len(pending_judge)} unblocked injections...")
        if hasattr(base, "release_models"):
            base.release_models()
        if runner:
            runner.release()
        empty_cuda_cache()

        judge = ComplianceJudge()

        if judge.is_mock():
            for idx in pending_judge:
                records[idx]["judge_label"] = "SKIPPED"
        else:
            for idx in tqdm(pending_judge, desc="Phase 2 (Qwen judge)"):
                rec = records[idx]
                judge_label, reasoning = judge.evaluate(
                    rec["original"], rec["llm_output"]
                )
                rec["judge_label"] = judge_label
                rec["judge_reasoning"] = reasoning
                rec["attack_success"] = _attack_success(
                    rec["label"], rec["blocked"], judge_label
                )

        judge.release()
        empty_cuda_cache()
    else:
        if hasattr(base, "release_models"):
            base.release_models()
        if runner:
            runner.release()
        empty_cuda_cache()

    logs_df = pd.DataFrame(records)
    metrics_df, _ = _compute_metrics(logs_df)

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
