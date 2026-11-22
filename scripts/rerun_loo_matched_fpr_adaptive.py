"""
Re-run LOO matched-FPR adaptive only, using thresholds from
results/loo_matched_fpr/thresholds.csv (same operating points as Table A).

Gate + OutcomeCache lookup: FLAN attack texts and baseline Qwen/judge labels
are reused; sanitizers are rebuilt/retuned at the final matched thresholds.

Verifies per-fold orig Bypass against metrics.csv before writing outputs.
Does not re-run paraphrase / edge / main metrics.

  python scripts/rerun_loo_matched_fpr_adaptive.py
  python scripts/rerun_loo_matched_fpr_adaptive.py --folds simsonsun bravansky
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROMPT_CACHE_DIR, RESULTS_DIR
from src.evaluation.loo_matched_fpr import (
    LOO_FOLDS,
    LOO_SOURCE_FOLDS,
    _build_fold_sanitizers,
    _fold_dir,
    _macro_micro_average,
    _sanitizer_list,
)
from src.evaluation.prompt_cache import FeatureCache, OutcomeCache
from src.evaluation.robustness import robustness_adaptive

OUT = RESULTS_DIR / "loo_matched_fpr"


def _apply_thresholds(sans: dict, thr_rows: pd.DataFrame) -> None:
    for _, row in thr_rows.iterrows():
        name = str(row["method"])
        if name not in sans:
            continue
        sans[name].threshold = float(row["threshold"])
        if name == "context_aware" and pd.notna(row.get("semantic_hard_sim")):
            sans[name].semantic_hard_sim = float(row["semantic_hard_sim"])


def _orig_bypass(san, prompts: list[str]) -> float:
    n = len(prompts)
    if n == 0:
        return 0.0
    blocked = sum(1 for p in prompts if san.sanitize(p)[1])
    return 1.0 - blocked / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folds",
        nargs="+",
        default=list(LOO_FOLDS),
        choices=list(LOO_FOLDS),
        help="Folds to re-run (default: all five).",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.015,
        help="Max allowed |sanitize Bypass − metrics Bypass| before abort (default 0.015).",
    )
    parser.add_argument(
        "--skip-deberta",
        action="store_true",
        help="Skip fine-tuned DeBERTa (faster).",
    )
    parser.add_argument(
        "--no-outcome-cache",
        action="store_true",
        help="Force fresh target LLM + judge (slow). Default: cache lookup.",
    )
    args = parser.parse_args()

    thr_path = OUT / "thresholds.csv"
    metrics_path = OUT / "metrics.csv"
    if not thr_path.is_file() or not metrics_path.is_file():
        raise SystemExit(
            f"Missing {thr_path} or {metrics_path}. "
            "Run: python -m src.run_experiments --loo-matched-fpr --loo-skip-robustness"
        )

    thr_all = pd.read_csv(thr_path)
    metrics_all = pd.read_csv(metrics_path)
    feature_cache = FeatureCache(PROMPT_CACHE_DIR / "loo" / "features.pkl")
    adaptive_outcome_cache = OutcomeCache(
        PROMPT_CACHE_DIR / "loo" / "adaptive_outcomes.jsonl"
    )
    use_cache = not args.no_outcome_cache
    print(
        f"[adaptive-rerun] outcome cache rows: {len(adaptive_outcome_cache._rows)}  "
        f"lookup={'on' if use_cache else 'off'}"
    )

    collected: list[pd.DataFrame] = []
    # Keep non-rerun folds (and portable rows) from existing aggregate if present.
    existing = OUT / "robustness_adaptive.csv"
    portable_methods = {"protectai", "prompt_guard"}
    if existing.is_file():
        old = pd.read_csv(existing)
        # Drop aggregate rows; rebuild macros at the end.
        old = old[~old["fold"].astype(str).isin(["four_source_macro_avg", "four_source_micro"])]
        keep = old[
            (~old["fold"].isin(args.folds))
            | (old["method"].astype(str).isin(portable_methods))
        ]
        if not keep.empty:
            collected.append(keep)

    for fold in args.folds:
        split = _fold_dir(fold)
        train_csv = str(split / "train.csv")
        val_csv = str(split / "val.csv")
        test_csv = str(split / "test.csv")
        for path in (train_csv, val_csv, test_csv):
            if not Path(path).is_file():
                raise SystemExit(f"Missing {path}")

        print(f"\n{'=' * 60}\n[adaptive-rerun] Fold: {fold}\n{'=' * 60}")
        sans = _build_fold_sanitizers(
            train_csv,
            val_csv,
            include_deberta=not args.skip_deberta,
            feature_cache=feature_cache,
        )
        _apply_thresholds(sans, thr_all[thr_all["fold"] == fold])

        inj = (
            pd.read_csv(test_csv)
            .loc[lambda d: d["label"].astype(str) == "injection", "prompt"]
            .astype(str)
            .tolist()
        )
        fold_metrics = metrics_all[metrics_all["fold"] == fold]
        print("  Threshold check (sanitize Bypass vs metrics Bypass):")
        for san in _sanitizer_list(sans):
            key = san.name
            row = fold_metrics[fold_metrics["method"] == key]
            if row.empty:
                continue
            got = _orig_bypass(san, inj)
            exp = float(row["Bypass_Rate_%"].iloc[0]) / 100.0
            delta = abs(got - exp)
            status = "OK" if delta <= args.atol else "MISMATCH"
            print(
                f"    {key:24s} sanitize={got:.4f}  metrics={exp:.4f}  "
                f"|diff|={delta:.4f}  [{status}]"
            )
            if delta > args.atol:
                raise SystemExit(
                    f"Abort: {fold}/{key} sanitize Bypass {got:.4f} != metrics {exp:.4f}. "
                    "Refusing to write adaptive at the wrong operating point."
                )

        fold_dir = OUT / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        attacks_path = fold_dir / "adaptive_attacks.jsonl"
        default_attacks = RESULTS_DIR / "loo" / fold / "adaptive_attacks.jsonl"
        if default_attacks.is_file() and not attacks_path.is_file():
            attacks_path.write_bytes(default_attacks.read_bytes())

        san_list = _sanitizer_list(sans)
        adaptive_logs_df, adaptive_metrics_df, adaptive_cm = robustness_adaptive(
            test_csv=test_csv,
            train_csv=train_csv,
            val_csv=val_csv,
            use_llm=True,
            include_similarity_evasion=True,
            use_cached_lookup=use_cache,
            outcome_cache=adaptive_outcome_cache,
            feature_cache=feature_cache,
            attacks_cache_path=attacks_path,
            sanitizers=san_list,
        )
        adaptive_metrics_df = adaptive_metrics_df.copy()
        adaptive_metrics_df.insert(0, "fold", fold)
        by_style = adaptive_metrics_df.attrs.pop("by_style", None)
        adaptive_metrics_df.attrs.clear()
        adaptive_metrics_df.to_csv(fold_dir / "robustness_adaptive.csv", index=False)
        if by_style is not None:
            by_style.to_csv(fold_dir / "robustness_adaptive_by_style.csv", index=False)
        with open(fold_dir / "robustness_adaptive_logs.jsonl", "w", encoding="utf-8") as handle:
            for rec in adaptive_logs_df.to_dict(orient="records"):
                handle.write(json.dumps(rec, default=str) + "\n")
        with open(
            fold_dir / "robustness_adaptive_confusion_matrices.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(adaptive_cm, handle, indent=2)

        # Drop any prior portable-kept rows for this fold's main methods (already
        # excluded above); append refreshed main-suite metrics.
        collected.append(adaptive_metrics_df)
        keep_cols = [
            c
            for c in (
                "method",
                "Bypass_Rate_%",
                "True_ASR_%",
                "successful_attack_count",
                "injections",
            )
            if c in adaptive_metrics_df.columns
        ]
        print(adaptive_metrics_df[keep_cols].to_string(index=False))

        for san in sans.values():
            if hasattr(san, "release_models"):
                try:
                    san.release_models()
                except Exception:
                    pass

    adaptive_all = pd.concat(collected, ignore_index=True)
    # Dedup: one row per (fold, method); prefer last write (rerun).
    adaptive_all = adaptive_all.drop_duplicates(subset=["fold", "method"], keep="last")
    fold_order = {f: i for i, f in enumerate(LOO_FOLDS)}
    adaptive_all["_ord"] = adaptive_all["fold"].map(lambda f: fold_order.get(str(f), 99))
    adaptive_all = adaptive_all.sort_values(["_ord", "method"]).drop(columns=["_ord"])

    # Drop any stale macros before recomputing (main suite only; portables macros
    # are refreshed by eval_loo_matched_fpr_portables.py).
    adaptive_all = adaptive_all[
        ~adaptive_all["fold"].astype(str).isin(["four_source_macro_avg", "four_source_micro"])
    ]
    main_only = adaptive_all[~adaptive_all["method"].astype(str).isin(portable_methods)]
    portables = adaptive_all[adaptive_all["method"].astype(str).isin(portable_methods)]
    summarized = _macro_micro_average(main_only)
    if not portables.empty:
        # Keep portable per-fold rows; portables script will refresh their macros.
        summarized = pd.concat([summarized, portables], ignore_index=True)

    summarized.to_csv(OUT / "robustness_adaptive.csv", index=False)
    summarized.to_csv(OUT / "robustness_adaptive_summary.csv", index=False)
    print(f"\n[OK] Wrote {OUT / 'robustness_adaptive.csv'}")
    print(f"[OK] Wrote {OUT / 'robustness_adaptive_summary.csv'}")

    macro = summarized[summarized["fold"].astype(str) == "four_source_macro_avg"]
    if not macro.empty:
        print("\nLOO-4 macro adaptive (refreshed main suite):")
        for _, r in macro.iterrows():
            asr = r["True_ASR_%"] if "True_ASR_%" in r and pd.notna(r["True_ASR_%"]) else float("nan")
            print(
                f"  {r['method']:24s}  Bypass={r['Bypass_Rate_%']:.2f}%  "
                f"TrueASR={asr:.2f}%"
            )


if __name__ == "__main__":
    main()
