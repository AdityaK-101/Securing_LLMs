"""
Re-run LOO matched-FPR paraphrase only, using thresholds from
results/loo_matched_fpr/thresholds.csv (same operating points as Table A).

Verifies per-fold orig Bypass against metrics.csv before writing outputs.
Does not re-run edge / adaptive / main metrics.

  python scripts/rerun_loo_matched_fpr_paraphrase.py
  python scripts/rerun_loo_matched_fpr_paraphrase.py --folds simsonsun bravansky
"""

from __future__ import annotations

import argparse
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
    _paraphrase_average,
    _sanitizer_list,
)
from src.evaluation.prompt_cache import FeatureCache
from src.evaluation.robustness import robustness_paraphrase

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

    collected: list[pd.DataFrame] = []
    # Keep non-rerun folds from existing aggregate if present.
    existing = OUT / "robustness_paraphrase.csv"
    if existing.is_file():
        old = pd.read_csv(existing)
        keep = old[~old["fold"].isin(args.folds)]
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

        print(f"\n{'=' * 60}\n[paraphrase-rerun] Fold: {fold}\n{'=' * 60}")
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
                    "Refusing to write paraphrase at the wrong operating point."
                )

        san_list = _sanitizer_list(sans)
        para_df = robustness_paraphrase(
            prompt_csv=test_csv,
            train_csv=train_csv,
            val_csv=val_csv,
            n=None,
            sanitizers=san_list,
        )
        para_df = para_df.copy()
        para_df.insert(0, "fold", fold)
        fold_dir = OUT / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        para_df.to_csv(fold_dir / "robustness_paraphrase.csv", index=False)
        collected.append(para_df)
        print(
            para_df[
                ["method", "orig_Bypass_Rate", "para_Bypass_Rate", "Bypass_Rate_delta"]
            ].to_string(index=False)
        )

        for san in sans.values():
            if hasattr(san, "release_models"):
                try:
                    san.release_models()
                except Exception:
                    pass

    para_all = pd.concat(collected, ignore_index=True)
    # Stable fold order
    fold_order = {f: i for i, f in enumerate(LOO_FOLDS)}
    para_all["_ord"] = para_all["fold"].map(lambda f: fold_order.get(f, 99))
    para_all = para_all.sort_values(["_ord", "method"]).drop(columns=["_ord"])
    para_all.to_csv(OUT / "robustness_paraphrase.csv", index=False)
    summary = _paraphrase_average(para_all)
    summary.to_csv(OUT / "robustness_paraphrase_summary.csv", index=False)

    print(f"\n[OK] Wrote {OUT / 'robustness_paraphrase.csv'}")
    print(f"[OK] Wrote {OUT / 'robustness_paraphrase_summary.csv'}")
    macro = summary[summary["fold"] == "four_source_macro_avg"]
    if not macro.empty:
        print("\nLOO-4 macro paraphrase deltas (pp):")
        for _, r in macro.iterrows():
            print(
                f"  {r['method']:24s}  "
                f"orig={100 * r['orig_Bypass_Rate']:.2f}%  "
                f"para={100 * r['para_Bypass_Rate']:.2f}%  "
                f"delta={100 * r['Bypass_Rate_delta']:+.2f} pp"
            )


if __name__ == "__main__":
    main()
