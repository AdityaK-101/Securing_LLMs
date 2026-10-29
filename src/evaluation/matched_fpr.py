"""
Matched false-positive-rate comparison (reviewer bFjj / mentor item 3).

Tune each score-based filter on **validation** to the tightest threshold with
val FPR ≤ ``target_fpr`` (default 5%), maximizing detection (minimizing Bypass).
Then report Bypass / FPR on the held-out **test** set at that operating point.

Also reports each method's **default** operating point for side-by-side contrast.

Scoreable methods:
  keyword, context_aware (soft total), context_aware_learned,
  similarity_only, deberta_finetuned, protectai, prompt_guard

ProtectAI / Prompt Guard stay off-the-shelf weights; only the decision
threshold is retuned for the matched-FPR column (fair operating-point compare).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from ..config import RESULTS_DIR, TEST_CSV, TRAIN_CSV, VAL_CSV
from ..sanitizers import (
    ContextAwareLearnedSanitizer,
    ContextAwareSanitizer,
    DebertaFinetunedSanitizer,
    KeywordHeuristicSanitizer,
    PromptGuardSanitizer,
    ProtectAISanitizer,
    SimilarityOnlySanitizer,
)


DEFAULT_TARGET_FPR = 0.05


def _labels(df: pd.DataFrame) -> np.ndarray:
    return (df["label"].astype(str) == "injection").astype(int).to_numpy()


def _prompts(df: pd.DataFrame) -> list[str]:
    return df["prompt"].astype(str).tolist()


def tune_threshold_at_fpr(
    scores: np.ndarray,
    y: np.ndarray,
    target_fpr: float = DEFAULT_TARGET_FPR,
) -> tuple[float, dict]:
    """
    Highest-threshold policy among those with FPR ≤ target_fpr that
    minimizes Bypass (maximizes TPR). Ties → lower FPR, then higher threshold.
    """
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    inj = y == 1
    ben = y == 0
    if not ben.any():
        return 0.5, {"fpr": 0.0, "bypass": 1.0, "detection": 0.0, "feasible": False}

    candidates = np.unique(
        np.concatenate(
            [
                scores,
                np.linspace(float(scores.min()) - 1e-6, float(scores.max()) + 1e-6, 51),
                np.quantile(scores[ben], np.linspace(0, 1, 41)),
            ]
        )
    )
    best_t = float(scores.max()) + 1.0  # block almost nothing if none feasible
    best = {
        "fpr": 1.0,
        "bypass": 1.0,
        "detection": 0.0,
        "feasible": False,
    }
    for t in sorted(candidates):
        pred = scores >= t
        fpr = float(pred[ben].mean()) if ben.any() else 0.0
        if fpr > target_fpr + 1e-12:
            continue
        bypass = float((~pred[inj]).mean()) if inj.any() else 0.0
        detection = 1.0 - bypass
        better = (
            not best["feasible"]
            or detection > best["detection"] + 1e-12
            or (
                abs(detection - best["detection"]) <= 1e-12
                and fpr < best["fpr"] - 1e-12
            )
            or (
                abs(detection - best["detection"]) <= 1e-12
                and abs(fpr - best["fpr"]) <= 1e-12
                and t > best_t + 1e-12
            )
        )
        if better:
            best_t = float(t)
            best = {
                "fpr": fpr,
                "bypass": bypass,
                "detection": detection,
                "feasible": True,
            }
    if not best["feasible"]:
        # No threshold meets the budget: pick lowest achievable FPR on val
        for t in sorted(candidates, reverse=True):
            pred = scores >= t
            fpr = float(pred[ben].mean()) if ben.any() else 0.0
            bypass = float((~pred[inj]).mean()) if inj.any() else 0.0
            if fpr < best["fpr"] - 1e-12:
                best_t = float(t)
                best = {
                    "fpr": fpr,
                    "bypass": bypass,
                    "detection": 1.0 - bypass,
                    "feasible": False,
                }
    return best_t, best


def eval_at_threshold(
    scores: np.ndarray,
    y: np.ndarray,
    threshold: float,
) -> dict:
    pred = np.asarray(scores, dtype=float) >= float(threshold)
    y = np.asarray(y, dtype=int)
    inj = y == 1
    ben = y == 0
    bypass = float((~pred[inj]).mean()) if inj.any() else 0.0
    fpr = float(pred[ben].mean()) if ben.any() else 0.0
    return {
        "Bypass_Rate_%": round(100.0 * bypass, 2),
        "FPR_%": round(100.0 * fpr, 2),
        "Detection_Rate_%": round(100.0 * (1.0 - bypass), 2),
        "TP": int((pred & inj).sum()),
        "FN": int((~pred & inj).sum()),
        "FP": int((pred & ben).sum()),
        "TN": int((~pred & ben).sum()),
        "injections": int(inj.sum()),
        "benign": int(ben.sum()),
    }


def _batch_scores(
    scorer: Callable[[str], float],
    prompts: list[str],
) -> np.ndarray:
    return np.asarray([float(scorer(p)) for p in prompts], dtype=float)


def _default_binary_metrics(sanitizer, df: pd.DataFrame) -> dict:
    y = _labels(df)
    blocked = []
    for prompt in _prompts(df):
        _, b = sanitizer.sanitize(prompt)
        blocked.append(bool(b))
    pred = np.asarray(blocked, dtype=bool)
    inj = y == 1
    ben = y == 0
    bypass = float((~pred[inj]).mean()) if inj.any() else 0.0
    fpr = float(pred[ben].mean()) if ben.any() else 0.0
    return {
        "Bypass_Rate_%": round(100.0 * bypass, 2),
        "FPR_%": round(100.0 * fpr, 2),
        "Detection_Rate_%": round(100.0 * (1.0 - bypass), 2),
        "TP": int((pred & inj).sum()),
        "FN": int((~pred & inj).sum()),
        "FP": int((pred & ben).sum()),
        "TN": int((~pred & ben).sum()),
        "injections": int(inj.sum()),
        "benign": int(ben.sum()),
    }


def build_scoreable_methods(
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    *,
    include_portables: bool = True,
    include_deberta: bool = True,
) -> list[tuple[str, object, Callable[[str], float]]]:
    """
    Return (display_name, sanitizer, score_fn) for matched-FPR methods.

    ``context_aware`` score_fn is the soft total (hard triggers off for retuning).
    Default-mode metrics still use full ``sanitize`` (hard triggers on).
    """
    methods: list[tuple[str, object, Callable[[str], float]]] = []

    kw = KeywordHeuristicSanitizer()
    methods.append(("keyword", kw, kw.injection_score))

    method_a = ContextAwareSanitizer(train_csv=train_csv)
    methods.append(("context_aware", method_a, method_a.injection_score))

    method_b = ContextAwareLearnedSanitizer(
        train_csv=train_csv,
        val_csv=val_csv,
        share_from=method_a,
    )
    methods.append(("context_aware_learned", method_b, method_b.injection_score))

    sim = SimilarityOnlySanitizer(
        train_csv=train_csv,
        val_csv=val_csv,
        share_embed_from=method_a,
    )
    methods.append(("similarity_only", sim, sim.injection_score))

    if include_deberta:
        deb = DebertaFinetunedSanitizer(train_csv=train_csv, val_csv=val_csv)
        methods.append(("deberta_finetuned", deb, deb.injection_score))

    if include_portables:
        try:
            protectai = ProtectAISanitizer()
            methods.append(("protectai", protectai, protectai.injection_score))
        except Exception as exc:
            print(f"[matched_fpr] ProtectAI skipped: {exc}")
        try:
            pg = PromptGuardSanitizer()
            methods.append(("prompt_guard", pg, pg.injection_score))
        except Exception as exc:
            print(f"[matched_fpr] Prompt Guard skipped: {exc}")

    return methods


def run_matched_fpr_study(
    train_csv: str = TRAIN_CSV,
    val_csv: str = VAL_CSV,
    test_csv: str = TEST_CSV,
    target_fpr: float = DEFAULT_TARGET_FPR,
    *,
    include_portables: bool = True,
    include_deberta: bool = True,
    out_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Build fair baselines + portables, tune to ``target_fpr`` on val, score test.

    Writes ``results/matched_fpr/metrics.csv`` by default.
    """
    out = Path(out_dir) if out_dir else RESULTS_DIR / "matched_fpr"
    out.mkdir(parents=True, exist_ok=True)

    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    y_val = _labels(df_val)
    y_test = _labels(df_test)
    val_prompts = _prompts(df_val)
    test_prompts = _prompts(df_test)

    print(
        f"\n[matched_fpr] target_fpr={target_fpr:.0%}  "
        f"val={len(df_val)}  test={len(df_test)}"
    )
    methods = build_scoreable_methods(
        train_csv,
        val_csv,
        include_portables=include_portables,
        include_deberta=include_deberta,
    )

    rows = []
    for name, san, score_fn in methods:
        print(f"[matched_fpr] Scoring {name}...")
        val_scores = _batch_scores(score_fn, val_prompts)
        test_scores = _batch_scores(score_fn, test_prompts)

        # Default operating point (published / F1-tuned / hard-trigger sanitize)
        default_test = _default_binary_metrics(san, df_test)
        default_val = _default_binary_metrics(san, df_val)
        default_thr = getattr(san, "threshold", None)
        rows.append(
            {
                "method": name,
                "mode": "default",
                "target_fpr": None,
                "threshold": default_thr,
                "val_FPR_%": default_val["FPR_%"],
                "val_Bypass_Rate_%": default_val["Bypass_Rate_%"],
                "test_FPR_%": default_test["FPR_%"],
                "test_Bypass_Rate_%": default_test["Bypass_Rate_%"],
                "test_Detection_Rate_%": default_test["Detection_Rate_%"],
                "feasible_on_val": True,
                "note": "full sanitize() / published argmax for portables",
                **{f"test_{k}": v for k, v in default_test.items() if k in ("TP", "FN", "FP", "TN")},
            }
        )

        thr, val_stats = tune_threshold_at_fpr(val_scores, y_val, target_fpr=target_fpr)
        matched_test = eval_at_threshold(test_scores, y_test, thr)
        matched_val = eval_at_threshold(val_scores, y_val, thr)
        rows.append(
            {
                "method": name,
                "mode": f"matched_fpr_{int(round(100 * target_fpr))}pct",
                "target_fpr": target_fpr,
                "threshold": round(float(thr), 6),
                "val_FPR_%": matched_val["FPR_%"],
                "val_Bypass_Rate_%": matched_val["Bypass_Rate_%"],
                "test_FPR_%": matched_test["FPR_%"],
                "test_Bypass_Rate_%": matched_test["Bypass_Rate_%"],
                "test_Detection_Rate_%": matched_test["Detection_Rate_%"],
                "feasible_on_val": bool(val_stats.get("feasible", False)),
                "note": (
                    "soft score retuned on val"
                    if name == "context_aware"
                    else "score retuned on val; weights unchanged for portables"
                ),
                "test_TP": matched_test["TP"],
                "test_FN": matched_test["FN"],
                "test_FP": matched_test["FP"],
                "test_TN": matched_test["TN"],
            }
        )

        if hasattr(san, "release_models"):
            try:
                san.release_models()
            except Exception:
                pass

    metrics = pd.DataFrame(rows)
    out_csv = out / "metrics.csv"
    metrics.to_csv(out_csv, index=False)
    summary = out / "summary.txt"
    summary.write_text(_summary_text(metrics, target_fpr), encoding="utf-8")
    print(f"[OK] Wrote {out_csv}")
    print(f"[OK] Wrote {summary}")
    print(metrics[
        [
            "method",
            "mode",
            "threshold",
            "val_FPR_%",
            "test_FPR_%",
            "test_Bypass_Rate_%",
            "feasible_on_val",
        ]
    ].to_string(index=False))
    return metrics


def _summary_text(metrics: pd.DataFrame, target_fpr: float) -> str:
    lines = [
        "Matched false-positive-rate comparison (reviewer fair-baseline item).",
        f"Target val FPR ≤ {target_fpr:.0%}.",
        "Default rows = published / F1-tuned / hard-trigger sanitize.",
        "matched_fpr rows = continuous score threshold tuned on val, evaluated on test.",
        "similarity_only and deberta_finetuned are trained on the same train/val split.",
        "protectai / prompt_guard keep published weights; only the threshold is retuned.",
        "context_aware matched mode uses soft aggregate only (hard triggers off).",
        "",
        metrics.to_string(index=False),
        "",
    ]
    return "\n".join(lines)
