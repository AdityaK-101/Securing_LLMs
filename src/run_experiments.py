"""
src/run_experiments.py
======================
CLI entrypoint for experiment suites and ablations.

Suites (context-aware method selection):
  --suite a      baseline + regex + keyword + Method A  → results/suite_a/
  --suite b      baseline + regex + keyword + Method B  → results/suite_b/
  --suite ab     run suite a, then suite b (separate folders)
  --suite both   all 5 methods in one run              → results/

Usage:
    python -m src.run_experiments --suite a
    python -m src.run_experiments --suite b
    python -m src.run_experiments --suite ab
    python -m src.run_experiments --latency-only
    $env:SKIP_LLM=1; python -m src.run_experiments --suite a --no-llm
"""

import os
import json
import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .config import (
    RESULTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    VAL_CSV,
    TARGET_MODEL_NAME,
    JUDGE_MODEL_NAME,
    PROJECT_ROOT,
)
from .evaluation.ablation import (
    run_ab_comparison_study,
    run_hard_trigger_ablation_study,
    run_learned_ablation_study,
    run_weighted_ablation_study,
)
from .evaluation.evaluate import evaluate
from .evaluation.latency import (
    run_runtime_overhead_benchmark,
    save_runtime_overhead_report,
)
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

SUITE_CONFIG = {
    "a": {
        "label": "Suite A — baseline + regex + keyword + Method A",
        "include_method_a": True,
        "include_method_b": False,
        "results_subdir": "suite_a",
    },
    "b": {
        "label": "Suite B — baseline + regex + keyword + Method B",
        "include_method_a": False,
        "include_method_b": True,
        "results_subdir": "suite_b",
    },
    "both": {
        "label": "All methods (A + B together)",
        "include_method_a": True,
        "include_method_b": True,
        "results_subdir": None,  # write directly under results/
    },
}


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


def _resolve_output_dirs(suite_key: str) -> tuple[Path, Path]:
    cfg = SUITE_CONFIG[suite_key]
    if cfg["results_subdir"] is None:
        results_dir = RESULTS_DIR
    else:
        results_dir = RESULTS_DIR / cfg["results_subdir"]
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, figures_dir


def _run_full_pipeline(
    suite_key: str,
    use_llm: bool,
    no_judge: bool,
    calibrate: bool,
    run_classifier: bool,
    overhead_report: dict | None = None,
    skip_latency: bool = False,
):
    cfg = SUITE_CONFIG[suite_key]
    include_a = cfg["include_method_a"]
    include_b = cfg["include_method_b"]
    results_dir, figures_dir = _resolve_output_dirs(suite_key)

    _banner(cfg["label"])
    print(f"  Target : {TARGET_MODEL_NAME} ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
    print(f"  Judge  : {JUDGE_MODEL_NAME} ({'enabled' if use_llm and not no_judge else 'SKIP'})")
    print(f"  Seed   : 42 (fixed for reproducibility)")
    print(f"  Test   : {TEST_CSV}")
    print(f"  Train  : {TRAIN_CSV}")
    print(f"  Val    : {VAL_CSV}  (Method B threshold tuning)")
    print(f"  Suite  : {suite_key}  (A={include_a}, B={include_b})")
    print(f"  Output : {results_dir}")

    # Per-suite copy of shared overhead report (measured once upstream)
    if overhead_report is not None:
        save_runtime_overhead_report(overhead_report, results_dir)
    elif not skip_latency:
        _banner("Runtime / Latency Overhead (MiniLM + DistilGPT-2)")
        overhead_report = run_runtime_overhead_benchmark(
            train_csv=TRAIN_CSV,
            prompt_csv=TEST_CSV,
            n_prompts=50,
            warmup=5,
            out_dir=results_dir,
        )

    if calibrate and use_llm and not no_judge:
        _banner("Judge Calibration (~20 obvious cases)")
        cal_judge = ComplianceJudge()
        cal_results = run_calibration(cal_judge)
        cal_judge.release()
        cal_path = results_dir / "judge_calibration.json"
        with open(cal_path, "w", encoding="utf-8") as f:
            json.dump(cal_results, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved: {cal_path}")

    _banner("4.5  Main Experiment")
    logs_df, metrics_df, cm_dict = evaluate(
        test_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        use_llm=use_llm,
        include_method_a=include_a,
        include_method_b=include_b,
    )

    print("\n--- Bypass Rate / True ASR / FPR Summary ---")
    cols = ["method", "Bypass_Rate_%", "True_ASR_%", "FPR_%",
            "TP", "FN", "FP", "TN", "successful_attack_count"]
    print(metrics_df[cols].to_string(index=False))

    _metrics_export_columns(metrics_df).to_csv(results_dir / "metrics.csv", index=False)
    with open(results_dir / "logs.jsonl", "w", encoding="utf-8") as f:
        for rec in logs_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(results_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)
    print(f"\n[OK] Saved: {results_dir / 'metrics.csv'}")
    print(f"[OK] Saved: {results_dir / 'logs.jsonl'}  ({len(logs_df)} records)")
    print(f"[OK] Saved: {results_dir / 'confusion_matrices.json'}")

    _banner("Generating Figures")
    _plot_bypass_rate_bar(metrics_df, figures_dir=figures_dir)
    _plot_true_asr_bar(metrics_df, figures_dir=figures_dir)
    _plot_fpr_bar(metrics_df, figures_dir=figures_dir)
    _plot_confusion_matrices(cm_dict, figures_dir=figures_dir)

    _banner("4.6  Robustness — Paraphrase Test (held-out test injections)")
    para_df = robustness_paraphrase(
        prompt_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        n=None,
        include_method_a=include_a,
        include_method_b=include_b,
    )
    print("\n--- Paraphrase Bypass Rate Delta ---")
    print(para_df[["method", "orig_Bypass_Rate", "para_Bypass_Rate", "Bypass_Rate_delta"]].to_string(index=False))
    para_df.to_csv(results_dir / "robustness_paraphrase.csv", index=False)
    _plot_paraphrase_robustness(para_df, figures_dir=figures_dir)
    print(f"[OK] Saved: {results_dir / 'robustness_paraphrase.csv'}")

    _banner("4.6  Robustness — Edge Benign Cases")
    edge_df = robustness_edge_benign(
        include_method_a=include_a,
        include_method_b=include_b,
    )
    print("\n--- Edge Benign FPR ---")
    print(edge_df.to_string(index=False))
    edge_df.to_csv(results_dir / "robustness_edge_benign.csv", index=False)
    print(f"[OK] Saved: {results_dir / 'robustness_edge_benign.csv'}")

    _banner("4.6  Robustness — Adaptive Attacks (FLAN-T5)")
    adaptive_logs_df, adaptive_metrics_df, adaptive_cm_dict = robustness_adaptive(
        test_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        use_llm=use_llm,
        include_method_a=include_a,
        include_method_b=include_b,
    )
    print("\n--- Adaptive Bypass Rate / True ASR ---")
    print(adaptive_metrics_df[cols].to_string(index=False))
    _metrics_export_columns(adaptive_metrics_df).to_csv(
        results_dir / "robustness_adaptive.csv", index=False,
    )
    with open(results_dir / "robustness_adaptive_logs.jsonl", "w", encoding="utf-8") as f:
        for rec in adaptive_logs_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with open(
        results_dir / "robustness_adaptive_confusion_matrices.json",
        "w", encoding="utf-8",
    ) as f:
        json.dump(adaptive_cm_dict, f, indent=2)
    _plot_adaptive_robustness(adaptive_metrics_df, figures_dir=figures_dir)
    print(f"[OK] Saved: {results_dir / 'robustness_adaptive.csv'}")
    print(f"[OK] Saved: {results_dir / 'robustness_adaptive_logs.jsonl'}  ({len(adaptive_logs_df)} records)")
    print(f"[OK] Saved: {results_dir / 'robustness_adaptive_confusion_matrices.json'}")

    _banner("Writing Robustness Note")
    _write_robustness_note(
        metrics_df, para_df, edge_df, adaptive_metrics_df,
        out_path=results_dir / "robustness_note.txt",
        overhead_report=overhead_report,
    )

    if run_classifier:
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
            results_dir / "classifier_metrics.csv", index=False
        )
        print(f"[OK] Saved: {results_dir / 'classifier_metrics.csv'}")

    _banner("Suite Complete")
    print(f"  All results written to {results_dir}/")
    print(f"  Figures:  {figures_dir}/")
    print(f"  Key file: {results_dir / 'robustness_note.txt'}\n")
    if overhead_report is not None:
        print(f"  Overhead: {results_dir / 'runtime_overhead.csv'}\n")


def main():
    parser = argparse.ArgumentParser(description="Experiment runner")
    parser.add_argument(
        "--suite",
        choices=["a", "b", "ab", "both"],
        default="ab",
        help=(
            "Which methods to run: "
            "a = full+Method A -> results/suite_a/; "
            "b = full+Method B -> results/suite_b/; "
            "ab = run a then b separately (default); "
            "both = all 5 methods in one run -> results/"
        ),
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip target LLM inference (sanitizer-only, fast mode)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip judge model (True ASR unavailable)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run optional judge calibration (~20 cases) before evaluation")
    parser.add_argument("--classifier", action="store_true",
                        help="Run stretch-goal LR+TF-IDF classifier")
    parser.add_argument("--weighted-ablation", action="store_true",
                        help="Run Method A weighted signal ablation (hard triggers off)")
    parser.add_argument("--hard-trigger-ablation", action="store_true",
                        help="Run Method A hard-trigger ablation (Full A vs Soft-only A)")
    parser.add_argument("--learned-ablation", action="store_true",
                        help="Run Method B learned feature leave-one-out ablation")
    parser.add_argument("--compare-ab", action="store_true",
                        help="Matched comparison: Method A Full vs Soft-only A vs Method B")
    parser.add_argument(
        "--latency-only",
        action="store_true",
        help="Only benchmark MiniLM / DistilGPT-2 overhead (no full eval)",
    )
    parser.add_argument(
        "--skip-latency",
        action="store_true",
        help="Skip runtime overhead benchmark during the full pipeline",
    )
    parser.add_argument(
        "--portable-baselines",
        action="store_true",
        help=(
            "Run ProtectAI + Prompt Guard static (+ adaptive) gate-only scripts. "
            "Not part of suite A/B (gated HF models / separate cost)."
        ),
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)

    if args.portable_baselines:
        _banner("Portable baselines (ProtectAI / Prompt Guard)")
        scripts = [
            "scripts/eval_protectai.py",
            "scripts/eval_prompt_guard.py",
            "scripts/eval_portable_adaptive.py",
        ]
        for rel in scripts:
            cmd = [sys.executable, str(PROJECT_ROOT / rel)]
            print(f"[portable] Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
        print("[OK] Portable baseline metrics under results/protectai/ and results/prompt_guard/")
        return

    use_llm = not args.no_llm
    if args.no_judge:
        os.environ["SKIP_JUDGE"] = "1"

    if args.latency_only:
        _banner("Runtime / Latency Overhead Only")
        run_runtime_overhead_benchmark(
            train_csv=TRAIN_CSV,
            prompt_csv=TEST_CSV,
            n_prompts=50,
            warmup=5,
            out_dir=RESULTS_DIR,
        )
        return

    ablation_cols = [
        "variant", "Bypass_Rate_%", "True_ASR_%", "Detection_Rate", "FPR_%",
        "Delta_Bypass_pp", "Delta_FPR_pp",
        "TP", "FN", "FP", "TN", "successful_attack_count",
    ]
    any_ablation = (
        args.weighted_ablation
        or args.hard_trigger_ablation
        or args.learned_ablation
        or args.compare_ab
    )

    if any_ablation:
        print(f"  Target : {TARGET_MODEL_NAME} ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
        print(f"  Judge  : {JUDGE_MODEL_NAME} ({'enabled' if use_llm and not args.no_judge else 'SKIP'})")
        print(f"  Seed   : 42 (fixed for reproducibility)")
        print(f"  Test   : {TEST_CSV}")
        print(f"  Train  : {TRAIN_CSV}")
        print(f"  Val    : {VAL_CSV}")

    if args.weighted_ablation:
        _banner("Weighted Signal Ablation (Method A, soft-only)")
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
            figures_dir=RESULTS_DIR / "figures",
        )
        print("[OK] Saved: results/weighted_ablation_metrics.csv")

    if args.hard_trigger_ablation:
        _banner("Hard Trigger Ablation (Method A)")
        hard_trigger_df = run_hard_trigger_ablation_study(
            test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
        )
        print("\n--- Hard Trigger Ablation Summary ---")
        print(hard_trigger_df[ablation_cols].to_string(index=False))
        hard_trigger_df.to_csv(RESULTS_DIR / "hard_trigger_ablation.csv", index=False)
        _plot_ablation_bypass_rate(
            hard_trigger_df,
            title="Hard Trigger Ablation — Full A vs Soft-Only / Trigger Removals\n(Lower Bypass is Better)",
            filename="hard_trigger_ablation.png",
            full_variant_names={"Full Model (A + Hard Triggers)"},
            figures_dir=RESULTS_DIR / "figures",
        )
        print("[OK] Saved: results/hard_trigger_ablation.csv")

    if args.learned_ablation:
        _banner("Learned Feature Ablation (Method B)")
        learned_df = run_learned_ablation_study(
            test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
        )
        print("\n--- Learned Feature Ablation Summary ---")
        print(learned_df[ablation_cols].to_string(index=False))
        learned_df.to_csv(RESULTS_DIR / "learned_ablation_metrics.csv", index=False)
        _plot_ablation_bypass_rate(
            learned_df,
            title="Method B Feature Ablation — Bypass Rate\n(Leave-one-out at Inference; Lower is Better)",
            filename="learned_ablation.png",
            full_variant_names={"Full Learned Model"},
            figures_dir=RESULTS_DIR / "figures",
        )
        print("[OK] Saved: results/learned_ablation_metrics.csv")

    if args.compare_ab:
        _banner("Method A vs Method B Comparison")
        compare_df = run_ab_comparison_study(
            test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
        )
        print("\n--- A vs B Comparison Summary ---")
        print(compare_df[ablation_cols].to_string(index=False))
        compare_df.to_csv(RESULTS_DIR / "ab_comparison_metrics.csv", index=False)
        _plot_ablation_bypass_rate(
            compare_df,
            title="Method A vs Method B — Bypass Rate\n(Matched Test Set; Lower is Better)",
            filename="ab_comparison.png",
            full_variant_names={"Method A Full (Hard Triggers)"},
            figures_dir=RESULTS_DIR / "figures",
        )
        print("[OK] Saved: results/ab_comparison_metrics.csv")

    if any_ablation:
        _banner("Ablation Study Complete")
        if args.weighted_ablation:
            print("  Weighted: results/weighted_ablation_metrics.csv")
            print("            results/figures/weighted_ablation.png")
        if args.hard_trigger_ablation:
            print("  Triggers: results/hard_trigger_ablation.csv")
            print("            results/figures/hard_trigger_ablation.png")
        if args.learned_ablation:
            print("  Learned:  results/learned_ablation_metrics.csv")
            print("            results/figures/learned_ablation.png")
        if args.compare_ab:
            print("  Compare:  results/ab_comparison_metrics.csv")
            print("            results/figures/ab_comparison.png")
        print()
        return

    overhead_report = None
    if not args.skip_latency:
        _banner("Runtime / Latency Overhead (MiniLM + DistilGPT-2)")
        overhead_report = run_runtime_overhead_benchmark(
            train_csv=TRAIN_CSV,
            prompt_csv=TEST_CSV,
            n_prompts=50,
            warmup=5,
            out_dir=RESULTS_DIR,
        )

    suite_keys = ["a", "b"] if args.suite == "ab" else [args.suite]
    for suite_key in suite_keys:
        _run_full_pipeline(
            suite_key=suite_key,
            use_llm=use_llm,
            no_judge=args.no_judge,
            calibrate=args.calibrate,
            run_classifier=args.classifier,
            overhead_report=overhead_report,
            skip_latency=args.skip_latency,
        )


if __name__ == "__main__":
    main()
