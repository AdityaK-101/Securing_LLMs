"""
Second-target True ASR (Table 1 LOO + Neuralchemy).

Gates / thresholds / Bypass / FPR are target-independent. This module:

  1. Fills a separate OutcomeCache for a second target (default Llama-3.2-3B-
     Instruct): greedy replies, then the existing Qwen judge + rubric.
  2. Re-applies **saved** matched thresholds (no re-tuning) to recover each
     method's allow list, then looks up True ASR in that cache.

Primary Qwen True ASR stays in ``results/prompt_cache/loo/outcomes.jsonl``.
Second-target rows go to ``outcomes__meta-llama_Llama-3_2-3B-Instruct.jsonl``
(see ``outcome_cache_filename_for_target``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import (
    LLM_CACHE_BATCH_SIZE,
    PROMPT_CACHE_DIR,
    RESULTS_DIR,
    SECOND_TARGET_MODEL_NAME,
    SOURCE_HOLDOUT_DIR,
    TARGET_MODEL_NAME,
    outcome_cache_filename_for_target,
)
from ..models.judge import ComplianceJudge
from ..models.target_llm import LLMRunner
from ..sanitizers.prompt_guard import PromptGuardSanitizer
from ..sanitizers.protectai import ProtectAISanitizer
from ..utils.gpu import empty_cuda_cache
from .loo_matched_fpr import (
    LOO_FOLDS,
    _blocked_method_a,
    _build_fold_sanitizers,
    _fold_dir,
    _labels,
    _macro_micro_average,
    _metrics_from_blocked,
    _prompts,
    _score_method_a_signals,
)
from .prompt_cache import FeatureCache, OutcomeCache, unique_prompts

OUT = RESULTS_DIR / "loo_matched_fpr"
THRESHOLDS = OUT / "thresholds.csv"
PRIMARY_METRICS = OUT / "metrics.csv"


def second_target_cache_path(model_name: str = SECOND_TARGET_MODEL_NAME) -> Path:
    return PROMPT_CACHE_DIR / "loo" / outcome_cache_filename_for_target(model_name)


def collect_table1_injection_prompts(
    folds: tuple[str, ...] | list[str] = LOO_FOLDS,
) -> list[str]:
    """Unique injection prompts from LOO-4 + Neuralchemy test sets (n≈1000)."""
    prompts: list[str] = []
    for fold in folds:
        path = _fold_dir(fold) / "test.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        df = pd.read_csv(path)
        inj = df[df["label"].astype(str) == "injection"]
        prompts.extend(inj["prompt"].astype(str).tolist())
    return unique_prompts(prompts)


def fill_second_target_outcomes(
    *,
    model_name: str = SECOND_TARGET_MODEL_NAME,
    folds: tuple[str, ...] | list[str] = LOO_FOLDS,
    batch_size: int | None = None,
    skip_llm: bool = False,
    skip_judge: bool = False,
) -> OutcomeCache:
    """Generate second-target replies + Qwen-judge labels into a separate cache."""
    prompts = collect_table1_injection_prompts(folds)
    cache_path = second_target_cache_path(model_name)
    cache = OutcomeCache(cache_path)
    print(
        f"[second_target] model={model_name}\n"
        f"  prompts={len(prompts)} (unique Table-1 test injections)\n"
        f"  cache={cache_path}"
    )

    bs = int(batch_size) if batch_size is not None else LLM_CACHE_BATCH_SIZE
    if not skip_llm:
        runner = LLMRunner(model_name=model_name)
        try:
            cache.ensure_llm(
                prompts,
                runner,
                batch_size=bs,
                meta={"target_model": model_name},
            )
        finally:
            runner.release()
            empty_cuda_cache()

    if not skip_judge:
        judge = ComplianceJudge()
        try:
            cache.ensure_judge(prompts, judge)
        finally:
            judge.release()
            empty_cuda_cache()

    n_miss = len(cache.missing_judge(prompts))
    print(
        f"[second_target] cache rows={len(cache._rows)}  "
        f"still_missing_judge={n_miss}"
    )
    return cache


def _thr_row(thr_df: pd.DataFrame, fold: str, method: str) -> pd.Series | None:
    hit = thr_df[(thr_df["fold"] == fold) & (thr_df["method"] == method)]
    if hit.empty:
        return None
    return hit.iloc[0]


def _score_prompts(san, prompts: list[str]) -> np.ndarray:
    return np.asarray([float(san.injection_score(p)) for p in prompts], dtype=float)


def _blocked_fixed(san, prompts: list[str]) -> np.ndarray:
    return np.asarray([bool(san.sanitize(p)[1]) for p in prompts], dtype=bool)


def rescore_true_asr_at_saved_thresholds(
    *,
    outcome_cache: OutcomeCache,
    model_name: str = SECOND_TARGET_MODEL_NAME,
    folds: tuple[str, ...] | list[str] = LOO_FOLDS,
    include_portables: bool = True,
    include_deberta: bool = True,
) -> pd.DataFrame:
    """
    Re-apply ``thresholds.csv`` (no re-tuning); True ASR from ``outcome_cache``.

    Bypass / FPR recomputed from the same gates (must match primary metrics).
    """
    if not THRESHOLDS.is_file():
        raise FileNotFoundError(
            f"Missing {THRESHOLDS}. Run LOO matched-FPR first."
        )
    thr_df = pd.read_csv(THRESHOLDS)
    feature_cache = FeatureCache(PROMPT_CACHE_DIR / "loo" / "features.pkl")

    portable_sans: dict[str, object] = {}
    if include_portables:
        if (thr_df["method"] == "protectai").any():
            portable_sans["protectai"] = ProtectAISanitizer()
        if (thr_df["method"] == "prompt_guard").any():
            portable_sans["prompt_guard"] = PromptGuardSanitizer()

    rows: list[dict] = []
    for fold in folds:
        split = _fold_dir(fold)
        train_csv = str(split / "train.csv")
        val_csv = str(split / "val.csv")
        test_csv = str(split / "test.csv")
        df_test = pd.read_csv(test_csv)
        y_test = _labels(df_test)
        test_prompts = _prompts(df_test)

        print(f"\n[second_target] rescore fold={fold}")
        sans = _build_fold_sanitizers(
            train_csv,
            val_csv,
            include_deberta=include_deberta,
            feature_cache=feature_cache,
        )

        # Fixed binary gates
        for name in ("baseline", "regex"):
            blocked = _blocked_fixed(sans[name], test_prompts)
            m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
            rows.append(
                {
                    "fold": fold,
                    "method": name,
                    "mode": "fixed",
                    "threshold": np.nan,
                    "semantic_hard_sim": np.nan,
                    "target_model": model_name,
                    "note": "second-target True ASR; gates/thresholds unchanged",
                    **m,
                }
            )

        # Method A at saved soft thr × semantic_hard_sim
        a_row = _thr_row(thr_df, fold, "context_aware")
        if a_row is not None:
            method_a = sans["context_aware"]
            soft_t = float(a_row["threshold"])
            hard_sim = float(a_row["semantic_hard_sim"])
            method_a.threshold = soft_t
            method_a.semantic_hard_sim = hard_sim
            signals = _score_method_a_signals(method_a, test_prompts)
            blocked = _blocked_method_a(signals, method_a, soft_t, hard_sim)
            m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
            rows.append(
                {
                    "fold": fold,
                    "method": "context_aware",
                    "mode": "matched_fpr_5pct",
                    "threshold": soft_t,
                    "semantic_hard_sim": hard_sim,
                    "target_model": model_name,
                    "note": "second-target True ASR; saved matched thresholds",
                    **m,
                }
            )
            try:
                method_a.release_models()
            except Exception:
                pass

        scoreable = [
            "keyword",
            "context_aware_learned",
            "similarity_only",
        ]
        if include_deberta and "deberta_finetuned" in sans:
            scoreable.append("deberta_finetuned")

        for name in scoreable:
            trow = _thr_row(thr_df, fold, name)
            if trow is None:
                continue
            san = sans[name]
            thr = float(trow["threshold"])
            san.threshold = thr
            scores = _score_prompts(san, test_prompts)
            blocked = scores >= thr
            m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
            rows.append(
                {
                    "fold": fold,
                    "method": name,
                    "mode": "matched_fpr_5pct",
                    "threshold": thr,
                    "semantic_hard_sim": np.nan,
                    "target_model": model_name,
                    "note": "second-target True ASR; saved matched thresholds",
                    **m,
                }
            )
            if hasattr(san, "release_models"):
                try:
                    san.release_models()
                except Exception:
                    pass

        for name, san in portable_sans.items():
            trow = _thr_row(thr_df, fold, name)
            if trow is None:
                continue
            thr = float(trow["threshold"])
            scores = _score_prompts(san, test_prompts)
            blocked = scores >= thr
            m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
            rows.append(
                {
                    "fold": fold,
                    "method": name,
                    "mode": "matched_fpr_5pct",
                    "threshold": thr,
                    "semantic_hard_sim": np.nan,
                    "target_model": model_name,
                    "note": "second-target True ASR; saved matched thresholds",
                    **m,
                }
            )

        empty_cuda_cache()

    df = pd.DataFrame(rows)
    # Aggregates (LOO-4 macro / micro) like the primary suite
    df = _macro_micro_average(df)
    # Tag aggregates with target model
    if "target_model" in df.columns:
        df["target_model"] = df["target_model"].fillna(model_name)
    else:
        df["target_model"] = model_name
    return df


def build_true_asr_comparison(
    second_metrics: pd.DataFrame,
    *,
    primary_metrics_path: Path = PRIMARY_METRICS,
    primary_model: str = TARGET_MODEL_NAME,
    second_model: str = SECOND_TARGET_MODEL_NAME,
) -> pd.DataFrame:
    """Side-by-side True ASR (primary Qwen vs second target) at shared folds/methods."""
    if not primary_metrics_path.is_file():
        raise FileNotFoundError(primary_metrics_path)
    primary = pd.read_csv(primary_metrics_path)
    primary = primary.rename(columns={"True_ASR_%": "True_ASR_primary_%"})
    second = second_metrics.rename(columns={"True_ASR_%": "True_ASR_second_%"})
    keys = ["fold", "method"]
    cols_p = keys + ["Bypass_Rate_%", "FPR_%", "True_ASR_primary_%"]
    cols_s = keys + ["True_ASR_second_%", "true_asr_cache_missing"]
    merged = primary[cols_p].merge(second[cols_s], on=keys, how="inner")
    merged["primary_target"] = primary_model
    merged["second_target"] = second_model
    merged["Delta_True_ASR_pp"] = (
        merged["True_ASR_second_%"] - merged["True_ASR_primary_%"]
    ).round(2)
    return merged


def run_second_target_suite(
    *,
    model_name: str = SECOND_TARGET_MODEL_NAME,
    folds: tuple[str, ...] | list[str] = LOO_FOLDS,
    fill: bool = True,
    rescore: bool = True,
    batch_size: int | None = None,
    include_portables: bool = True,
    include_deberta: bool = True,
) -> dict[str, Path]:
    """Fill cache and/or rescore; write artifacts under ``results/loo_matched_fpr/``."""
    OUT.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    cache = OutcomeCache(second_target_cache_path(model_name))

    if fill:
        cache = fill_second_target_outcomes(
            model_name=model_name,
            folds=folds,
            batch_size=batch_size,
        )

    if rescore:
        metrics = rescore_true_asr_at_saved_thresholds(
            outcome_cache=cache,
            model_name=model_name,
            folds=folds,
            include_portables=include_portables,
            include_deberta=include_deberta,
        )
        out_metrics = OUT / "metrics_second_target.csv"
        metrics.to_csv(out_metrics, index=False)
        written["metrics"] = out_metrics

        try:
            cmp_df = build_true_asr_comparison(metrics, second_model=model_name)
            out_cmp = OUT / "true_asr_by_target.csv"
            cmp_df.to_csv(out_cmp, index=False)
            written["comparison"] = out_cmp
        except FileNotFoundError as exc:
            print(f"[second_target] skip comparison table: {exc}")

        # Short summary for LOO-4 macro
        macro = metrics[metrics["fold"].astype(str) == "four_source_macro_avg"]
        summary = OUT / "second_target_summary.txt"
        lines = [
            f"Second target: {model_name}",
            f"Judge: existing Qwen judge (same rubric as primary True ASR)",
            f"Folds: {list(folds)}",
            f"Cache: {second_target_cache_path(model_name)}",
            f"No re-tuning: thresholds from {THRESHOLDS}",
            "",
            "LOO-4 macro True ASR % (second target):",
        ]
        if not macro.empty:
            show = macro[["method", "Bypass_Rate_%", "True_ASR_%", "FPR_%"]].copy()
            lines.append(show.to_string(index=False))
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written["summary"] = summary
        print(f"[OK] Wrote {out_metrics}")
        print(f"[OK] Wrote {summary}")

    return written
