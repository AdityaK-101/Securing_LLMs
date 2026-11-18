"""
Re-run LOO matched-FPR edge-benign FPR only, using thresholds from
results/loo_matched_fpr/thresholds.csv (same operating points as Table A).

Verifies per-fold injection Bypass against metrics.csv before scoring edge.
Preserves ProtectAI / Prompt Guard edge rows already in the aggregate
(those come from eval_loo_matched_fpr_portables.py).

  python scripts/rerun_loo_matched_fpr_edge.py
  python scripts/rerun_loo_matched_fpr_edge.py --folds simsonsun
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
    _edge_average,
    _fold_dir,
    _sanitizer_list,
)
from src.evaluation.prompt_cache import FeatureCache
from src.evaluation.robustness import robustness_edge_benign

OUT = RESULTS_DIR / "loo_matched_fpr"
PORTABLE_METHODS = {"protectai", "prompt_guard"}


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


def _rebuild_aggregate(core_frames: list[pd.DataFrame]) -> None:
    """Write fold-synced edge aggregate; keep portable rows from prior aggregate."""
    core = pd.concat(core_frames, ignore_index=True)
    portable = pd.DataFrame()
    existing = OUT / "robustness_edge_benign.csv"
    if existing.is_file():
        old = pd.read_csv(existing)
        portable = old[old["method"].isin(PORTABLE_METHODS)].copy()
        # Drop macro rows from old portables; we re-average below
        if "fold" in portable.columns:
            portable = portable[~portable["fold"].isin(["four_source_macro_avg"])]

    parts = [core]
    if not portable.empty:
        # Align columns
        for c in core.columns:
            if c not in portable.columns:
                portable[c] = pd.NA
        portable = portable[core.columns]
        parts.append(portable)

    all_rows = pd.concat(parts, ignore_index=True)
    # Drop any prior macro before re-average
    all_rows = all_rows[~all_rows["fold"].isin(["four_source_macro_avg"])]
    summary = _edge_average(all_rows)
    # Ensure edge_FPR_% for report convenience
    if "edge_FPR" in summary.columns and "edge_FPR_%" not in summary.columns:
        summary["edge_FPR_%"] = (summary["edge_FPR"] * 100.0).round(2)
    elif "edge_FPR" in summary.columns:
        summary["edge_FPR_%"] = (summary["edge_FPR"] * 100.0).round(2)

    fold_order = {f: i for i, f in enumerate(list(LOO_FOLDS) + ["four_source_macro_avg"])}
    summary["_ord"] = summary["fold"].map(lambda f: fold_order.get(f, 99))
    summary = summary.sort_values(["_ord", "method"]).drop(columns=["_ord"])
    summary.to_csv(OUT / "robustness_edge_benign.csv", index=False)
    _edge_average(all_rows).to_csv(OUT / "robustness_edge_benign_summary.csv", index=False)


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
        help="Max allowed |sanitize Bypass − metrics Bypass| before abort.",
    )
    parser.add_argument(
        "--skip-deberta",
        action="store_true",
        help="Skip fine-tuned DeBERTa (faster).",
    )
    parser.add_argument(
        "--rebuild-tables",
        action="store_true",
        help="Rebuild PAPER_TABLES.md after writing edge CSVs.",
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

    # Start from existing non-rerun folds so aggregate stays complete.
    core_frames: list[pd.DataFrame] = []
    existing = OUT / "robustness_edge_benign.csv"
    if existing.is_file():
        old = pd.read_csv(existing)
        keep = old[
            (~old["fold"].isin(args.folds))
            & (~old["fold"].isin(["four_source_macro_avg"]))
            & (~old["method"].isin(PORTABLE_METHODS))
        ]
        if not keep.empty:
            core_frames.append(keep)

    for fold in args.folds:
        split = _fold_dir(fold)
        train_csv = str(split / "train.csv")
        val_csv = str(split / "val.csv")
        test_csv = str(split / "test.csv")
        for path in (train_csv, val_csv, test_csv):
            if not Path(path).is_file():
                raise SystemExit(f"Missing {path}")

        print(f"\n{'=' * 60}\n[edge-rerun] Fold: {fold}\n{'=' * 60}")
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
                    f"Abort: {fold}/{key} sanitize Bypass {got:.4f} != metrics {exp:.4f}."
                )

        edge_df = robustness_edge_benign(
            train_csv=train_csv,
            val_csv=val_csv,
            sanitizers=_sanitizer_list(sans),
        )
        edge_df = edge_df.copy()
        edge_df.insert(0, "fold", fold)
        if "edge_FPR_%" not in edge_df.columns:
            edge_df["edge_FPR_%"] = (edge_df["edge_FPR"] * 100.0).round(2)
        fold_dir = OUT / fold
        fold_dir.mkdir(parents=True, exist_ok=True)
        edge_df.to_csv(fold_dir / "robustness_edge_benign.csv", index=False)
        core_frames.append(edge_df)
        print(edge_df[["method", "edge_FPR", "blocked_count", "total"]].to_string(index=False))

        for san in sans.values():
            if hasattr(san, "release_models"):
                try:
                    san.release_models()
                except Exception:
                    pass

    _rebuild_aggregate(core_frames)
    print(f"\n[OK] Wrote {OUT / 'robustness_edge_benign.csv'}")
    print(f"[OK] Wrote {OUT / 'robustness_edge_benign_summary.csv'}")

    summary = pd.read_csv(OUT / "robustness_edge_benign.csv")
    macro = summary[summary["fold"] == "four_source_macro_avg"]
    print("\nLOO-4 macro edge FPR:")
    for _, r in macro.iterrows():
        pct = float(r["edge_FPR"]) * 100.0
        print(f"  {r['method']:24s}  {pct:.2f}%")

    if args.rebuild_tables:
        from src.evaluation.loo_matched_fpr_report import build_paper_tables

        build_paper_tables()
        print(f"[OK] Rebuilt {OUT / 'PAPER_TABLES.md'}")


if __name__ == "__main__":
    main()
