"""Score human labels against the hidden judge key.

Usage:
  1. Fill human_label (COMPLIED/REFUSED) in data/judge_human_validation/blind_labeling_sheet.csv
  2. python scripts/score_judge_human_validation.py

Writes:
  results/judge_human_validation/agreement_report.csv  (per-row)
  results/judge_human_validation/metrics.csv           (summary)
  results/judge_human_validation/metrics_summary.txt   (paper-ready)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "judge_human_validation"
BLIND = ROOT / "data" / "judge_human_validation" / "blind_labeling_sheet.csv"
KEY = OUT_DIR / "judge_key.csv"
OUT_REPORT = OUT_DIR / "agreement_report.csv"
OUT_METRICS = OUT_DIR / "metrics.csv"
OUT_SUMMARY = OUT_DIR / "metrics_summary.txt"


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for binomial proportion k/n (returned as percentages)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (100.0 * lo, 100.0 * hi)


def main() -> None:
    with BLIND.open(encoding="utf-8") as f:
        human_rows = list(csv.DictReader(f))
    with KEY.open(encoding="utf-8") as f:
        key = {r["sample_id"]: r for r in csv.DictReader(f)}

    missing = [
        r["sample_id"] for r in human_rows if not (r.get("human_label") or "").strip()
    ]
    if missing:
        raise SystemExit(
            f"{len(missing)}/{len(human_rows)} rows still unlabeled "
            f"(sample_id e.g. {missing[:5]}). Fill human_label first."
        )

    a = b = c = d = 0
    report = []
    for r in human_rows:
        sid = r["sample_id"]
        h = r["human_label"].strip().upper()
        if h == "COMPILED":
            h = "COMPLIED"
        j = key[sid]["judge_label"].strip().upper()
        if h not in ("COMPLIED", "REFUSED"):
            raise SystemExit(f"Invalid human_label on sample_id={sid}: {h!r}")
        agree = h == j
        if h == "COMPLIED" and j == "COMPLIED":
            a += 1
        elif h == "COMPLIED" and j == "REFUSED":
            b += 1
        elif h == "REFUSED" and j == "COMPLIED":
            c += 1
        else:
            d += 1
        report.append(
            {
                "sample_id": sid,
                "cohort": r.get("cohort", ""),
                "fold": r.get("fold", r.get("source_suite", "")),
                "method": r["method"],
                "prompt_id": r["prompt_id"],
                "human_label": h,
                "judge_label": j,
                "agree": agree,
                "notes": r.get("notes", ""),
            }
        )

    n = a + b + c + d
    n_agree = a + d
    po = n_agree / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    w_lo, w_hi = wilson_interval(n_agree, n)

    paper = (
        f"To validate the LLM judge on the Table~1 no-defense replies, an author "
        f"independently labeled a stratified random sample of $N={n}$ "
        f"(prompt, response) pairs (20 per held-out source across the four LOO "
        f"folds and Neuralchemy), blind to the judge, with the same "
        f"COMPLIED/REFUSED rubric. Human--judge agreement was "
        f"${po * 100:.1f}\\%$ (Wilson $95\\%$ CI $[{w_lo:.1f}, {w_hi:.1f}]\\%$; "
        f"Cohen's $\\kappa={kappa:.2f}$)."
    )

    metrics = {
        "N": n,
        "n_agree": n_agree,
        "n_disagree": n - n_agree,
        "percent_agreement": round(po * 100, 1),
        "wilson95_lo": round(w_lo, 1),
        "wilson95_hi": round(w_hi, 1),
        "cohens_kappa": round(kappa, 3),
        "CC_both_COMPLIED": a,
        "CR_human_COMPLIED_judge_REFUSED": b,
        "RC_human_REFUSED_judge_COMPLIED": c,
        "RR_both_REFUSED": d,
        "cohort": "loo_table1_baseline_stratified",
        "source": "data/splits/source_holdout/*/test.csv + results/prompt_cache/loo/outcomes.jsonl",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_REPORT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report[0].keys()))
        w.writeheader()
        w.writerows(report)

    with OUT_METRICS.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)

    OUT_SUMMARY.write_text(
        "\n".join(
            [
                "LLM judge human spot-validation",
                "================================",
                "Cohort               = Table 1 LOO+Neuralchemy no-defense (stratified)",
                f"N                    = {n}",
                f"Agree                = {n_agree}/{n}",
                f"Percent agreement    = {po * 100:.1f}%",
                f"Wilson 95% CI        = [{w_lo:.1f}, {w_hi:.1f}]%",
                f"Cohen's kappa        = {kappa:.3f}",
                f"Confusion (human\\judge): CC={a} CR={b} RC={c} RR={d}",
                "",
                "Paper sentence (LaTeX draft):",
                paper,
                "",
                "Replace the N=30 sentences in Section 5 (Target, judge, and "
                "robustness) and Limitations after labeling is complete.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"N = {n}")
    print(f"Percent agreement = {po * 100:.1f}%  ({n_agree}/{n})")
    print(f"Wilson 95% CI     = [{w_lo:.1f}, {w_hi:.1f}]%")
    print(f"Cohen's kappa     = {kappa:.3f}")
    print(f"Confusion (human\\judge): CC={a} CR={b} RC={c} RR={d}")
    print(f"Wrote {OUT_METRICS}")
    print(f"Wrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_REPORT}")
    print()
    print("Paper sentence draft:")
    print(paper)


if __name__ == "__main__":
    main()
