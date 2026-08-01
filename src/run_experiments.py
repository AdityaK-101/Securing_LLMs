"""
src/run_experiments.py
======================
CLI entrypoint for Milestone 3 experiments.

Produces:
  results/metrics.csv          — Bypass Rate / True ASR / FPR per method
  results/logs.jsonl           — per-prompt record (incl. judge fields)
  results/confusion_matrices.json
  results/robustness_paraphrase.csv
  results/robustness_edge_benign.csv
  results/figures/attack_success_bar.png   ← Bypass Rate bar chart
  results/figures/true_asr_bar.png         ← True ASR bar chart
  results/figures/confusion_matrix_*.png
  results/figures/fpr_chart.png
  results/robustness_note.txt   ← required deliverable

Usage:
    # With target LLM + judge (downloads models on first run)
    python -m src.run_experiments

    # Mock mode (no download, sanitizer-based Bypass Rate still valid)
    $env:SKIP_LLM=1; python -m src.run_experiments

    # Optional one-time judge sanity check (not run by default)
    python -m src.run_experiments --calibrate

    # Include stretch-goal classifier
    python -m src.run_experiments --classifier
"""

import os
import json
import argparse
import pandas as pd

from .config import FIGURES_DIR, RESULTS_DIR, TEST_CSV, TRAIN_CSV, TARGET_MODEL_NAME, JUDGE_MODEL_NAME
from .evaluation.ablation import (
    run_hard_trigger_ablation_study,
    run_weighted_ablation_study,
)
from .evaluation.evaluate import evaluate
from .evaluation.robustness import (
    robustness_adaptive,
    robustness_edge_benign,
    robustness_paraphrase,
)
from .models.judge import ComplianceJudge, run_calibration
from .visualization import (
    _plot_ablation_bypass_rate,
    _plot_adaptive_robustness,
    _plot_bypass_rate_bar,
    _plot_confusion_matrices,
    _plot_fpr_bar,
    _plot_paraphrase_robustness,
    _plot_true_asr_bar,
    _write_robustness_note,
)


def _banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


def _metrics_export_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select required metrics.csv columns per spec."""
    cols = [
        "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%",
        "TP", "FN", "FP", "TN", "successful_attack_count",
    ]
    return metrics_df[cols]


def main():
    parser = argparse.ArgumentParser(description="Milestone 3 experiment runner")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip target LLM inference (sanitizer-only, fast mode)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge model (True ASR unavailable)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run optional judge calibration (~20 cases) before evaluation")
    parser.add_argument("--classifier", action="store_true",
                        help="Run stretch-goal LR+TF-IDF classifier")
    parser.add_argument("--weighted-ablation", action="store_true",
                        help="Run weighted signal ablation study only (separate outputs)")
    parser.add_argument("--hard-trigger-ablation", action="store_true",
                        help="Run hard-trigger ablation study only (separate outputs)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    use_llm = not args.no_llm
    if args.no_judge:
        os.environ["SKIP_JUDGE"] = "1"

    ablation_cols = [
        "variant", "Bypass_Rate_%", "True_ASR_%", "Detection_Rate", "FPR_%",
        "TP", "FN", "FP", "TN", "successful_attack_count",
    ]

    if args.weighted_ablation or args.hard_trigger_ablation:
        print(f"  Target : {TARGET_MODEL_NAME} ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
        print(f"  Judge  : {JUDGE_MODEL_NAME} ({'enabled' if use_llm and not args.no_judge else 'SKIP'})")
        print(f"  Seed   : 42 (fixed for reproducibility)")
        print(f"  Test   : {TEST_CSV}")
        print(f"  Train  : {TRAIN_CSV}")

    if args.weighted_ablation:
        _banner("Weighted Signal Ablation")
        weighted_df = run_weighted_ablation_study(
            test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
        )
        print("\n--- Weighted Signal Ablation Summary ---")
        print(weighted_df[ablation_cols].to_string(index=False))
        weighted_df.to_csv(RESULTS_DIR / "weighted_ablation_metrics.csv", index=False)
        _plot_ablation_bypass_rate(
            weighted_df,
            title="Weighted Signal Ablation — Bypass Rate by Signal\n(Hard Triggers Disabled; Lower is Better)",
            filename="weighted_ablation.png",
            full_variant_names={"Full Weighted Model"},
        )
        print("[OK] Saved: results/weighted_ablation_metrics.csv")

    if args.hard_trigger_ablation:
        _banner("Hard Trigger Ablation")
        hard_trigger_df = run_hard_trigger_ablation_study(
            test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
        )
        print("\n--- Hard Trigger Ablation Summary ---")
        print(hard_trigger_df[ablation_cols].to_string(index=False))
        hard_trigger_df.to_csv(RESULTS_DIR / "hard_trigger_ablation.csv", index=False)
        _plot_ablation_bypass_rate(
            hard_trigger_df,
            title="Hard Trigger Ablation — Bypass Rate by Variant\n(Lower is Better)",
            filename="hard_trigger_ablation.png",
            full_variant_names={"Full Model"},
        )
        print("[OK] Saved: results/hard_trigger_ablation.csv")

    if args.weighted_ablation or args.hard_trigger_ablation:
        _banner("Ablation Study Complete")
        if args.weighted_ablation:
            print("  Weighted: results/weighted_ablation_metrics.csv")
            print("            results/figures/weighted_ablation.png")
        if args.hard_trigger_ablation:
            print("  Triggers: results/hard_trigger_ablation.csv")
            print("            results/figures/hard_trigger_ablation.png")
        print()
        return

    _banner("Milestone 3 — Prompt Injection Defense Evaluation")
    print(f"  Target : {TARGET_MODEL_NAME} ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
    print(f"  Judge  : {JUDGE_MODEL_NAME} ({'enabled' if use_llm and not args.no_judge else 'SKIP'})")
    print(f"  Seed   : 42 (fixed for reproducibility)")
    print(f"  Test   : {TEST_CSV}")
    print(f"  Train  : {TRAIN_CSV}")

    # Optional calibration (--calibrate); judge is released before main eval
    if args.calibrate and use_llm and not args.no_judge:
        _banner("Judge Calibration (~20 obvious cases)")
        cal_judge = ComplianceJudge()
        cal_results = run_calibration(cal_judge)
        cal_judge.release()
        cal_path = RESULTS_DIR / "judge_calibration.json"
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_results, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: results/judge_calibration.json")

    # -----------------------------------------------------------------------
    # Main Evaluation (Phase 1: target LLM, Phase 2: judge — one model at a time)
    # -----------------------------------------------------------------------
    _banner("4.5  Main Experiment")
    logs_df, metrics_df, cm_dict = evaluate(
        test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
    )

    print("\n--- Bypass Rate / True ASR / FPR Summary ---")
    cols = ["method", "Bypass_Rate_%", "True_ASR_%", "FPR_%",
            "TP", "FN", "FP", "TN", "successful_attack_count"]
    print(metrics_df[cols].to_string(index=False))

    # Save
    _metrics_export_columns(metrics_df).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    with open(RESULTS_DIR / "logs.jsonl", "w", encoding="utf-8") as f:
        for rec in logs_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(RESULTS_DIR / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)
    print(f"\n[OK] Saved: results/metrics.csv")
    print(f"[OK] Saved: results/logs.jsonl  ({len(logs_df)} records)")
    print(f"[OK] Saved: results/confusion_matrices.json")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    _banner("Generating Figures")
    _plot_bypass_rate_bar(metrics_df)
    _plot_true_asr_bar(metrics_df)
    _plot_fpr_bar(metrics_df)
    _plot_confusion_matrices(cm_dict)

    # -----------------------------------------------------------------------
    # Robustness Testing
    # -----------------------------------------------------------------------
    _banner("4.6  Robustness — Paraphrase Test")
    para_df = robustness_paraphrase(train_csv=TRAIN_CSV, n=50)
    print("\n--- Paraphrase Bypass Rate Delta ---")
    print(para_df[["method", "orig_Bypass_Rate", "para_Bypass_Rate", "Bypass_Rate_delta"]].to_string(index=False))
    para_df.to_csv(RESULTS_DIR / "robustness_paraphrase.csv", index=False)
    _plot_paraphrase_robustness(para_df)
    print(f"[OK] Saved: results/robustness_paraphrase.csv")

    _banner("4.6  Robustness — Edge Benign Cases")
    edge_df = robustness_edge_benign()
    print("\n--- Edge Benign FPR ---")
    print(edge_df.to_string(index=False))
    edge_df.to_csv(RESULTS_DIR / "robustness_edge_benign.csv", index=False)
    print(f"[OK] Saved: results/robustness_edge_benign.csv")

    _banner("4.6  Robustness — Adaptive Attacks (FLAN-T5)")
    adaptive_logs_df, adaptive_metrics_df, adaptive_cm_dict = robustness_adaptive(
        test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
    )
    print("\n--- Adaptive Bypass Rate / True ASR ---")
    print(adaptive_metrics_df[cols].to_string(index=False))
    _metrics_export_columns(adaptive_metrics_df).to_csv(
        RESULTS_DIR / "robustness_adaptive.csv", index=False,
    )
    with open(RESULTS_DIR / "robustness_adaptive_logs.jsonl", "w", encoding="utf-8") as f:
        for rec in adaptive_logs_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(
        RESULTS_DIR / "robustness_adaptive_confusion_matrices.json",
        "w", encoding="utf-8",
    ) as f:
        json.dump(adaptive_cm_dict, f, indent=2)
    _plot_adaptive_robustness(adaptive_metrics_df)
    print(f"[OK] Saved: results/robustness_adaptive.csv")
    print(f"[OK] Saved: results/robustness_adaptive_logs.jsonl  ({len(adaptive_logs_df)} records)")
    print(f"[OK] Saved: results/robustness_adaptive_confusion_matrices.json")

    # -----------------------------------------------------------------------
    # Robustness Note
    # -----------------------------------------------------------------------
    _banner("Writing Robustness Note")
    _write_robustness_note(
        metrics_df, para_df, edge_df, adaptive_metrics_df,
        out_path=RESULTS_DIR / "robustness_note.txt"
    )

    # -----------------------------------------------------------------------
    # Stretch Goal: Classifier
    # -----------------------------------------------------------------------
    if args.classifier:
        _banner("Stretch Goal — LR + TF-IDF Classifier")
        from .classifier import run_classifier_experiment
        clf_results = run_classifier_experiment(
            train_csv=TRAIN_CSV, test_csv=TEST_CSV
        )
        clf_out = {
            "val_accuracy":       clf_results["val"]["accuracy"],
            "val_f1":             clf_results["val"]["f1"],
            "val_Bypass_Rate":    clf_results["val"]["Bypass_Rate"],
            "val_FPR":            clf_results["val"]["FPR"],
            "test_accuracy":      clf_results["test"]["accuracy"],
            "test_f1":            clf_results["test"]["f1"],
            "test_Bypass_Rate":   clf_results["test"]["Bypass_Rate"],
            "test_FPR":           clf_results["test"]["FPR"],
        }
        pd.DataFrame([clf_out]).to_csv(
            RESULTS_DIR / "classifier_metrics.csv", index=False
        )
        print(f"[OK] Saved: results/classifier_metrics.csv")

    # -----------------------------------------------------------------------
    _banner("Experiment Complete")
    print("  All results written to results/")
    print("  Figures:  results/figures/")
    print("  Key file: results/robustness_note.txt\n")


if __name__ == "__main__":
    main()
