"""
Context-Aware sanitizer ablation studies.

Three modes + one comparison:
  1. Weighted Signal Ablation (Method A) — hard triggers off; one soft signal removed.
  2. Hard Trigger Ablation (Method A) — full model; individual hard triggers removed.
  3. Learned Feature Ablation (Method B) — leave-one-out on continuous features.
  4. A vs B comparison — Full A, Soft-only A, Full B on the same test set.

All ablation tables include Δ Bypass / Δ FPR vs the study's full variant.
"""

from __future__ import annotations

import pandas as pd

from ..config import TEST_CSV, TRAIN_CSV, VAL_CSV
from ..models.target_llm import LLMRunner
from ..sanitizers import ContextAwareLearnedSanitizer, ContextAwareSanitizer
from ..sanitizers.context_aware_learned import FEATURE_NAMES
from ..utils.gpu import empty_cuda_cache
from .metrics import compute_metrics
from .pipeline import run_two_phase_pipeline, should_skip_judge

# Soft-signal leave-one-out. Shift / conflict kept last (historically near-zero effect).
WEIGHTED_ABLATION_VARIANTS = [
    ("Full Weighted Model", {"disable_hard_triggers": True}),
    ("-No Regex", {"disable_hard_triggers": True, "disable_regex": True}),
    ("-No Keyword", {"disable_hard_triggers": True, "disable_keyword": True}),
    ("-No Semantic", {"disable_hard_triggers": True, "disable_semantic": True}),
    ("-No Intent", {"disable_hard_triggers": True, "disable_intent": True}),
    ("-No Roleplay", {"disable_hard_triggers": True, "disable_roleplay": True}),
    ("-No Perplexity", {"disable_hard_triggers": True, "disable_perplexity": True}),
    (
        "-No Instruction Shift",
        {"disable_hard_triggers": True, "disable_instruction_shift": True},
    ),
    (
        "-No Objective Conflict",
        {"disable_hard_triggers": True, "disable_objective_conflict": True},
    ),
]

# Full A (with hard triggers) vs soft-only is the main paper contrast.
HARD_TRIGGER_ABLATION_VARIANTS = [
    ("Full Model (A + Hard Triggers)", {}),
    (
        "Soft-Only A (No Hard Triggers)",
        {"disable_hard_triggers": True},
    ),
    ("-No Semantic Hard Trigger", {"disable_semantic_hard_trigger": True}),
    ("-No Intent Hard Trigger", {"disable_intent_hard_trigger": True}),
    (
        "-No Roleplay+Keyword Hard Trigger",
        {"disable_roleplay_keyword_hard_trigger": True},
    ),
]

# Method B: zero features at inference under the fixed trained LR.
LEARNED_ABLATION_VARIANTS = [
    ("Full Learned Model", []),
    ("-No Regex", ["regex"]),
    ("-No Keyword", ["keyword"]),
    ("-No Semantic Max", ["semantic_max"]),
    ("-No Semantic Top3", ["semantic_top3"]),
    ("-No Semantic Contrast", ["semantic_contrast"]),
    (
        "-No All Semantic",
        ["semantic_max", "semantic_top3", "semantic_contrast"],
    ),
    ("-No Intent", ["intent"]),
    ("-No Roleplay", ["roleplay"]),
    ("-No Shift", ["shift"]),
    ("-No Conflict", ["conflict"]),
    ("-No Perplexity", ["perplexity"]),
]

# Matched comparison on the same test fold.
AB_COMPARISON_VARIANTS = [
    ("Method A Full (Hard Triggers)", "a_full"),
    ("Method A Soft-Only", "a_soft"),
    ("Method B Learned Soft", "b_full"),
]

ABLATION_EXPORT_COLUMNS = [
    "variant",
    "Bypass_Rate_%",
    "True_ASR_%",
    "Detection_Rate",
    "FPR_%",
    "Delta_Bypass_pp",
    "Delta_FPR_pp",
    "TP",
    "FN",
    "FP",
    "TN",
    "successful_attack_count",
]


def _add_delta_columns(
    df: pd.DataFrame,
    full_variant_names: set[str],
) -> pd.DataFrame:
    """Add Δ Bypass / Δ FPR (percentage points) vs the full variant row."""
    out = df.copy()
    full_rows = out[out["variant"].isin(full_variant_names)]
    if full_rows.empty:
        # Fall back to first row
        base_bypass = float(out.iloc[0]["Bypass_Rate_%"])
        base_fpr = float(out.iloc[0]["FPR_%"])
    else:
        base_bypass = float(full_rows.iloc[0]["Bypass_Rate_%"])
        base_fpr = float(full_rows.iloc[0]["FPR_%"])

    out["Delta_Bypass_pp"] = (out["Bypass_Rate_%"] - base_bypass).round(2)
    out["Delta_FPR_pp"] = (out["FPR_%"] - base_fpr).round(2)
    return out


def _metrics_from_records(records: list, full_variant_names: set[str]) -> pd.DataFrame:
    logs_df = pd.DataFrame(records)
    metrics_df, _ = compute_metrics(logs_df)
    rows = []
    for _, mrow in metrics_df.iterrows():
        row = mrow.to_dict()
        row["variant"] = row.pop("method")
        rows.append(row)
    df = pd.DataFrame(rows)
    df = _add_delta_columns(df, full_variant_names)
    # Preserve variant order from the study as much as possible
    return df


def _run_pipeline_on_sanitizers(
    study_name: str,
    sanitizers: list,
    base_for_release,
    test_csv: str,
    use_llm: bool,
) -> list:
    df_test = pd.read_csv(test_csv)
    runner = LLMRunner() if use_llm else None
    skip_judge = should_skip_judge(use_llm, runner)

    print(f"\n[Ablation] {study_name} — {len(sanitizers)} variants")
    print(
        f"[Ablation] Phase 1: {len(df_test)} prompts × "
        f"{len(sanitizers)} variants"
    )
    print(
        f"[Ablation] Phase 2: judge — "
        f"{'enabled' if not skip_judge else 'disabled'}"
    )

    def _release_base_models():
        if hasattr(base_for_release, "release_models"):
            base_for_release.release_models()

    def _cleanup_after_phase1():
        _release_base_models()
        if runner:
            runner.release()
        empty_cuda_cache()

    return run_two_phase_pipeline(
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


def _evaluate_method_a_variants(
    study_name: str,
    variants: list,
    full_variant_names: set[str],
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    print(f"[Ablation] Loading shared Method A auxiliary models once...")
    base = ContextAwareSanitizer(train_csv=train_csv)
    sanitizers = [
        base.copy_with(name=variant_name, **kwargs)
        for variant_name, kwargs in variants
    ]
    records = _run_pipeline_on_sanitizers(
        study_name, sanitizers, base, test_csv, use_llm
    )
    df = _metrics_from_records(records, full_variant_names)
    # Keep declared variant order
    order = [name for name, _ in variants]
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)
    df["variant"] = df["variant"].astype(str)
    return df[ABLATION_EXPORT_COLUMNS]


def run_weighted_ablation_study(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Weighted Signal Ablation: all hard triggers disabled; one signal removed per run.
    """
    return _evaluate_method_a_variants(
        "Weighted Signal Ablation (Method A, soft-only)",
        WEIGHTED_ABLATION_VARIANTS,
        full_variant_names={"Full Weighted Model"},
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
    Hard Trigger Ablation: Full A vs Soft-only A, plus individual trigger removals.
    """
    return _evaluate_method_a_variants(
        "Hard Trigger Ablation (Method A)",
        HARD_TRIGGER_ABLATION_VARIANTS,
        full_variant_names={"Full Model (A + Hard Triggers)"},
        test_csv=test_csv,
        train_csv=train_csv,
        use_llm=use_llm,
    )


def run_learned_ablation_study(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Method B feature ablation: fixed trained LR; zero one (or more) features
    at inference time.
    """
    print("[Ablation] Loading / training Method B once for feature ablation...")
    # Prefer sharing MiniLM/PPL from a temporary Method A load to avoid double init cost
    donor = ContextAwareSanitizer(train_csv=train_csv)
    base = ContextAwareLearnedSanitizer(
        train_csv=train_csv,
        val_csv=val_csv,
        share_from=donor,
    )
    # Drop donor ownership of weights so release goes through learned/base carefully
    sanitizers = [
        base.copy_with(name=variant_name, disabled_features=feats)
        for variant_name, feats in LEARNED_ABLATION_VARIANTS
    ]

    # Validate feature names early
    known = set(FEATURE_NAMES)
    for variant_name, feats in LEARNED_ABLATION_VARIANTS:
        unknown = set(feats) - known
        if unknown:
            raise ValueError(f"{variant_name}: unknown features {unknown}")

    records = _run_pipeline_on_sanitizers(
        "Learned Feature Ablation (Method B)",
        sanitizers,
        base,
        test_csv,
        use_llm,
    )
    # Also release donor models if still held
    if hasattr(donor, "release_models"):
        donor.release_models()

    df = _metrics_from_records(records, {"Full Learned Model"})
    order = [name for name, _ in LEARNED_ABLATION_VARIANTS]
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)
    df["variant"] = df["variant"].astype(str)
    return df[ABLATION_EXPORT_COLUMNS]


def run_ab_comparison_study(
    test_csv: str = TEST_CSV,
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    use_llm: bool = True,
) -> pd.DataFrame:
    """
    Matched comparison on one test fold:
      Method A Full, Method A Soft-Only, Method B Learned Soft.
    """
    print("[Ablation] Building matched A vs B comparison sanitizers...")
    method_a = ContextAwareSanitizer(train_csv=train_csv)
    a_full = method_a.copy_with(name="Method A Full (Hard Triggers)")
    a_soft = method_a.copy_with(
        name="Method A Soft-Only",
        disable_hard_triggers=True,
    )
    method_b = ContextAwareLearnedSanitizer(
        train_csv=train_csv,
        val_csv=val_csv,
        share_from=method_a,
    )
    method_b.name = "Method B Learned Soft"

    sanitizers = [a_full, a_soft, method_b]
    records = _run_pipeline_on_sanitizers(
        "Method A vs Method B Comparison",
        sanitizers,
        method_a,
        test_csv,
        use_llm,
    )
    if hasattr(method_b, "release_models"):
        method_b.release_models()

    df = _metrics_from_records(
        records,
        full_variant_names={"Method A Full (Hard Triggers)"},
    )
    order = [name for name, _ in AB_COMPARISON_VARIANTS]
    df["variant"] = pd.Categorical(df["variant"], categories=order, ordered=True)
    df = df.sort_values("variant").reset_index(drop=True)
    df["variant"] = df["variant"].astype(str)
    return df[ABLATION_EXPORT_COLUMNS]
