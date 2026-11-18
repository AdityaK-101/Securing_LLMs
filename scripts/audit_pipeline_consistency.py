"""
Consistency gate for paper LOO matched-FPR artifacts.

Checks:
  - paraphrase orig Bypass ≈ metrics Bypass (matched thresholds)
  - edge / paraphrase fold CSVs sync with aggregates
  - adaptive True ASR ≤ Bypass
  - reminds that results/loo and results/source_holdout are not paper main

Exit 1 on BAD/STALE/LOGIC. Used by ``scripts/run_paper.py``.

  python scripts/audit_pipeline_consistency.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR

ATOL = 0.015
FOLDS = ["simsonsun", "mpdd", "prompt_injection", "bravansky", "neuralchemy"]
SCOREABLE = [
    "context_aware",
    "context_aware_learned",
    "similarity_only",
    "keyword",
    "deberta_finetuned",
]
OUT = RESULTS_DIR / "loo_matched_fpr"


def _check_para_vs_metrics() -> list[str]:
    issues = []
    metrics = pd.read_csv(OUT / "metrics.csv")
    para = pd.read_csv(OUT / "robustness_paraphrase.csv")
    for method in SCOREABLE:
        for fold in FOLDS:
            m = metrics[(metrics["fold"] == fold) & (metrics["method"] == method)]
            p = para[(para["fold"] == fold) & (para["method"] == method)]
            if m.empty or p.empty:
                issues.append(f"[MISS] paraphrase {fold}/{method}")
                continue
            mb = float(m["Bypass_Rate_%"].iloc[0]) / 100.0
            pb = float(p["orig_Bypass_Rate"].iloc[0])
            if abs(mb - pb) > ATOL:
                issues.append(
                    f"[BAD] paraphrase {fold}/{method}: metrics={mb:.4f} "
                    f"para_orig={pb:.4f} |diff|={abs(mb - pb):.4f}"
                )
    return issues


def _check_fold_vs_agg(filename: str, key_cols: list[str]) -> list[str]:
    issues = []
    agg_path = OUT / filename
    if not agg_path.is_file():
        return [f"[MISS] {filename}"]
    agg = pd.read_csv(agg_path)
    for fold in FOLDS:
        fpath = OUT / fold / filename
        if not fpath.is_file():
            issues.append(f"[MISS] {fold}/{filename}")
            continue
        d = pd.read_csv(fpath)
        a = agg[agg["fold"] == fold] if "fold" in agg.columns else agg
        for method in d["method"].unique():
            if method in {"protectai", "prompt_guard"}:
                continue  # fold files often omit portables
            dd = d[d["method"] == method]
            aa = a[a["method"] == method] if "method" in a.columns else a
            if aa.empty:
                issues.append(f"[MISS] agg {fold}/{method} in {filename}")
                continue
            for c in key_cols:
                if c not in dd.columns or c not in aa.columns:
                    continue
                try:
                    dv = float(dd[c].iloc[0])
                    av = float(aa[c].iloc[0])
                except Exception:
                    continue
                if abs(dv - av) > 1e-6:
                    issues.append(
                        f"[STALE] {fold}/{method}.{c}: fold={dv} agg={av} ({filename})"
                    )
    return issues


def _check_adaptive_logic() -> list[str]:
    issues = []
    path = OUT / "robustness_adaptive.csv"
    if not path.is_file():
        return ["[MISS] robustness_adaptive.csv"]
    df = pd.read_csv(path)
    if "Bypass_Rate_%" not in df.columns or "True_ASR_%" not in df.columns:
        return ["[MISS] adaptive Bypass/TrueASR columns"]
    for _, row in df.iterrows():
        if float(row["True_ASR_%"]) - float(row["Bypass_Rate_%"]) > 0.05:
            issues.append(
                f"[LOGIC] adaptive TrueASR>Bypass "
                f"{row.get('fold')}/{row.get('method')}: "
                f"{row['True_ASR_%']} > {row['Bypass_Rate_%']}"
            )
    return issues


def _check_macro_para_matches_metrics() -> list[str]:
    issues = []
    summary = OUT / "robustness_paraphrase_summary.csv"
    if not summary.is_file():
        # derive from main para file
        summary = OUT / "robustness_paraphrase.csv"
    if not summary.is_file():
        return ["[MISS] paraphrase summary"]
    s = pd.read_csv(summary)
    macro = s[s["fold"] == "four_source_macro_avg"]
    metrics = pd.read_csv(OUT / "metrics.csv")
    for method, label in (
        ("context_aware", "A"),
        ("context_aware_learned", "B"),
    ):
        a = macro[macro["method"] == method]
        m = metrics[
            (metrics["fold"] == "four_source_macro_avg") & (metrics["method"] == method)
        ]
        if a.empty or m.empty:
            issues.append(f"[MISS] macro paraphrase/metrics for {label}")
            continue
        got = float(a["orig_Bypass_Rate"].iloc[0])
        exp = float(m["Bypass_Rate_%"].iloc[0]) / 100.0
        if abs(got - exp) > ATOL:
            issues.append(
                f"[BAD] macro para orig {label} {got:.4f} != metrics {exp:.4f}"
            )
        else:
            delta = 100.0 * float(a["Bypass_Rate_delta"].iloc[0])
            print(
                f"  paraphrase {label}: orig={100 * got:.2f}%  "
                f"delta={delta:+.2f} pp  [OK]"
            )
    return issues


def main() -> int:
    print("=== Paper tree: results/loo_matched_fpr ===")
    print(
        "Do NOT cite results/loo (default/F1), results/source_holdout, "
        "or suite_a portable-adaptive as the matched-FPR paper numbers.\n"
    )
    if not (OUT / "metrics.csv").is_file():
        print("[MISS] results/loo_matched_fpr/metrics.csv")
        return 1

    issues: list[str] = []
    issues += _check_para_vs_metrics()
    issues += _check_macro_para_matches_metrics()
    issues += _check_fold_vs_agg(
        "robustness_paraphrase.csv",
        ["orig_Bypass_Rate", "para_Bypass_Rate"],
    )
    issues += _check_fold_vs_agg(
        "robustness_edge_benign.csv",
        ["edge_FPR", "blocked_count"],
    )
    issues += _check_adaptive_logic()

    # Cross-tree reminders (INFO only)
    if (RESULTS_DIR / "loo" / "summary.csv").is_file():
        print("  [INFO] results/loo/ present — contrast only, not paper matched-FPR")
    if (RESULTS_DIR / "source_holdout").is_dir():
        print("  [INFO] results/source_holdout/ present — gate-only; may be stale")
    if (RESULTS_DIR / "suite_a" / "robustness_adaptive_logs.jsonl").is_file():
        print(
            "  [INFO] suite_a adaptive logs present — "
            "eval_portable_adaptive uses default portable thr, not LOO matched Table F"
        )

    print("\n=== SUMMARY ===")
    bad = [i for i in issues if i.startswith(("[BAD]", "[STALE]", "[LOGIC]"))]
    miss = [i for i in issues if i.startswith("[MISS]")]
    for i in issues:
        print(i)
    if not issues:
        print("No issues found.")
    print(f"\ncounts: BAD/STALE/LOGIC={len(bad)} MISS={len(miss)}")
    return 1 if bad or miss else 0


if __name__ == "__main__":
    raise SystemExit(main())
