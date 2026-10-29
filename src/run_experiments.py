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
    SOURCE_HOLDOUT_DIR,
    PROMPT_CACHE_DIR,
    LLM_CACHE_BATCH_SIZE,
)
from .evaluation.prompt_cache import (
    FeatureCache,
    OutcomeCache,
    collect_prompts_from_csvs,
    warm_minilm_embeddings,
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
    use_cached_lookup: bool = True,
    llm_batch_size: int = LLM_CACHE_BATCH_SIZE,
    outcome_cache: OutcomeCache | None = None,
    feature_cache: FeatureCache | None = None,
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
    # Fair baselines (reviewer): MiniLM similarity-only + ProtectAI ckpt fine-tuned
    # on this split — same train/val as Methods A/B (LOO already includes them).
    include_sim = True
    include_deb = True
    print(f"  Suite  : {suite_key}  (A={include_a}, B={include_b}, sim={include_sim}, deberta={include_deb})")
    print(f"  Output : {results_dir}")
    if use_llm and use_cached_lookup:
        print(
            f"  Cache  : generate-once lookup (batch={llm_batch_size}); "
            f"benign=filters-only; resume under {PROMPT_CACHE_DIR}"
        )

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

    suite_outcome = outcome_cache
    suite_features = feature_cache
    if use_llm and use_cached_lookup:
        suite_outcome = suite_outcome or OutcomeCache(PROMPT_CACHE_DIR / "outcomes.jsonl")
        suite_features = suite_features or FeatureCache(PROMPT_CACHE_DIR / "features.pkl")

    _banner("4.5  Main Experiment")
    logs_df, metrics_df, cm_dict = evaluate(
        test_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        use_llm=use_llm,
        include_method_a=include_a,
        include_method_b=include_b,
        include_similarity_only=include_sim,
        include_deberta_finetuned=include_deb,
        use_cached_lookup=use_cached_lookup and use_llm,
        outcome_cache=suite_outcome,
        feature_cache=suite_features,
        batch_size=llm_batch_size,
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
        val_csv=VAL_CSV,
        n=None,
        include_method_a=include_a,
        include_method_b=include_b,
        include_similarity_only=include_sim,
        include_deberta_finetuned=include_deb,
    )
    print("\n--- Paraphrase Bypass Rate Delta ---")
    print(para_df[["method", "orig_Bypass_Rate", "para_Bypass_Rate", "Bypass_Rate_delta"]].to_string(index=False))
    para_df.to_csv(results_dir / "robustness_paraphrase.csv", index=False)
    _plot_paraphrase_robustness(para_df, figures_dir=figures_dir)
    print(f"[OK] Saved: {results_dir / 'robustness_paraphrase.csv'}")

    _banner("4.6  Robustness — Edge Benign Cases")
    edge_df = robustness_edge_benign(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        include_method_a=include_a,
        include_method_b=include_b,
        include_similarity_only=include_sim,
        include_deberta_finetuned=include_deb,
    )
    print("\n--- Edge Benign FPR ---")
    print(edge_df.to_string(index=False))
    edge_df.to_csv(results_dir / "robustness_edge_benign.csv", index=False)
    print(f"[OK] Saved: {results_dir / 'robustness_edge_benign.csv'}")

    _banner("4.6  Robustness — Adaptive Attacks (FLAN-T5)")
    adaptive_logs_df, adaptive_metrics_df, adaptive_cm_dict = robustness_adaptive(
        test_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        use_llm=use_llm,
        include_method_a=include_a,
        include_method_b=include_b,
        include_similarity_only=include_sim,
        include_deberta_finetuned=include_deb,
        include_similarity_evasion=True,
        use_cached_lookup=use_cached_lookup and use_llm,
        outcome_cache=(
            OutcomeCache(PROMPT_CACHE_DIR / "adaptive_outcomes.jsonl")
            if use_llm and use_cached_lookup
            else None
        ),
        feature_cache=suite_features,
        batch_size=llm_batch_size,
    )
    print("\n--- Adaptive Bypass Rate / True ASR ---")
    print(adaptive_metrics_df[cols].to_string(index=False))
    _metrics_export_columns(adaptive_metrics_df).to_csv(
        results_dir / "robustness_adaptive.csv", index=False,
    )
    by_style = adaptive_metrics_df.attrs.get("by_style")
    if by_style is not None:
        by_style.to_csv(results_dir / "robustness_adaptive_by_style.csv", index=False)
        print("\n--- Adaptive Bypass by attack style (incl. similarity_evasion) ---")
        print(by_style.to_string(index=False))
        print(f"[OK] Saved: {results_dir / 'robustness_adaptive_by_style.csv'}")
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


LOO_FOLDS = (
    "simsonsun",
    "mpdd",
    "prompt_injection",
    "bravansky",
    "neuralchemy",
)
LOO_SOURCE_FOLDS = LOO_FOLDS[:4]


def _loo_fold_dir(fold: str) -> Path:
    return SOURCE_HOLDOUT_DIR / fold


def _run_loo(
    folds: list[str],
    use_llm: bool,
    no_judge: bool,
    include_method_a: bool,
    include_method_b: bool,
    rerun: bool,
    skip_robustness: bool = False,
    adaptive_max: int | None = None,
    use_cached_lookup: bool = True,
    llm_batch_size: int = LLM_CACHE_BATCH_SIZE,
) -> None:
    """Main two-layer experiment on each held-out source, then after-main stages.

    Same scorer as the regular suite (sanitizer, then target LLM, then judge).
    Train, val, and test for a fold all come from that fold's directory, so the
    held-out attack source is absent from the gallery and from Method B's
    threshold tune. After the main metrics, each fold also runs paraphrase,
    edge-benign, and adaptive (FLAN-T5) robustness — same stages as the suite.

    Speed path (default with LLM): one greedy Qwen+judge pass per unique attack
    prompt (shared OutcomeCache across folds), benign = filters only, shared
    MiniLM/PPL FeatureCache, resumable under results/prompt_cache/.
    """
    mode = "llm" if use_llm and not no_judge else "gate"
    collected = []
    para_collected = []
    edge_collected = []
    adaptive_collected = []
    export_cols = [
        "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%",
        "TP", "FN", "FP", "TN", "successful_attack_count",
        "injections", "benign",
    ]
    loo_baselines = dict(
        include_similarity_only=True,
        include_deberta_finetuned=True,
    )

    cache_dir = PROMPT_CACHE_DIR / "loo"
    cache_dir.mkdir(parents=True, exist_ok=True)
    shared_outcome = (
        OutcomeCache(cache_dir / "outcomes.jsonl")
        if use_llm and use_cached_lookup
        else None
    )
    shared_features = FeatureCache(cache_dir / "features.pkl")
    shared_adaptive = (
        OutcomeCache(cache_dir / "adaptive_outcomes.jsonl")
        if use_llm and use_cached_lookup
        else None
    )

    # Pre-warm MiniLM for all fold prompts once (reuse across folds)
    fold_csv_paths = []
    for fold in folds:
        split_dir = _loo_fold_dir(fold)
        for name in ("train.csv", "val.csv", "test.csv"):
            path = split_dir / name
            if path.is_file():
                fold_csv_paths.append(path)
    if fold_csv_paths:
        _banner("LOO FeatureCache — MiniLM warm-up (all folds)")
        all_prompts = collect_prompts_from_csvs(fold_csv_paths)
        print(f"  Unique prompts across folds: {len(all_prompts)}")
        warm_minilm_embeddings(shared_features, all_prompts)

    for fold in folds:
        split_dir = _loo_fold_dir(fold)
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        test_csv = split_dir / "test.csv"
        for path in (train_csv, val_csv, test_csv):
            if not path.is_file():
                raise SystemExit(
                    f"Missing {path}. Build the folds with: python scripts/build_source_holdout.py"
                )

        results_dir = RESULTS_DIR / "loo" / fold
        figures_dir = results_dir / "figures"
        results_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        marker = results_dir / "run_mode.txt"
        metrics_path = results_dir / "metrics.csv"
        para_path = results_dir / "robustness_paraphrase.csv"
        edge_path = results_dir / "robustness_edge_benign.csv"
        adaptive_path = results_dir / "robustness_adaptive.csv"
        rob_marker = results_dir / "robustness_run_mode.txt"

        _banner(f"Leave-one-source-out — {fold}")
        print(f"  Target : {TARGET_MODEL_NAME} ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
        print(f"  Judge  : {JUDGE_MODEL_NAME} ({'enabled' if use_llm and not no_judge else 'SKIP'})")
        print(f"  Train  : {train_csv}")
        print(f"  Val    : {val_csv}")
        print(f"  Test   : {test_csv}")
        print(f"  Output : {results_dir}")
        if use_llm and use_cached_lookup:
            print(
                f"  Cache  : shared OutcomeCache + FeatureCache under {cache_dir} "
                f"(batch={llm_batch_size})"
            )

        if (
            not rerun
            and metrics_path.is_file()
            and marker.is_file()
            and marker.read_text(encoding="utf-8").strip() == mode
        ):
            print(f"[skip] {fold} main already has a {mode} run. Pass --rerun to recompute.")
            metrics_df = pd.read_csv(metrics_path)
        else:
            logs_df, metrics_df, cm_dict = evaluate(
                test_csv=str(test_csv),
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                use_llm=use_llm,
                include_method_a=include_method_a,
                include_method_b=include_method_b,
                use_cached_lookup=use_cached_lookup and use_llm,
                outcome_cache=shared_outcome,
                feature_cache=shared_features,
                batch_size=llm_batch_size,
                warm_features=True,
                **loo_baselines,
            )
            keep = [col for col in export_cols if col in metrics_df.columns]
            metrics_df[keep].to_csv(metrics_path, index=False)
            with open(results_dir / "logs.jsonl", "w", encoding="utf-8") as handle:
                for rec in logs_df.to_dict(orient="records"):
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(results_dir / "confusion_matrices.json", "w", encoding="utf-8") as handle:
                json.dump(cm_dict, handle, indent=2)
            marker.write_text(mode + "\n", encoding="utf-8")
            _plot_bypass_rate_bar(metrics_df, figures_dir=figures_dir)
            _plot_true_asr_bar(metrics_df, figures_dir=figures_dir)
            _plot_fpr_bar(metrics_df, figures_dir=figures_dir)
            _plot_confusion_matrices(cm_dict, figures_dir=figures_dir)
            print(f"[OK] Saved: {metrics_path}")

        metrics_df = metrics_df.copy()
        metrics_df.insert(0, "fold", fold)
        collected.append(metrics_df)
        print(metrics_df[["fold", "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%"]].to_string(index=False))

        if skip_robustness:
            continue

        rob_ready = (
            not rerun
            and para_path.is_file()
            and edge_path.is_file()
            and adaptive_path.is_file()
            and rob_marker.is_file()
            and rob_marker.read_text(encoding="utf-8").strip() == mode
        )
        if rob_ready:
            print(f"[skip] {fold} robustness already has a {mode} run. Pass --rerun to recompute.")
            para_df = pd.read_csv(para_path)
            edge_df = pd.read_csv(edge_path)
            adaptive_metrics_df = pd.read_csv(adaptive_path)
        else:
            _banner(f"LOO Robustness — Paraphrase ({fold})")
            para_df = robustness_paraphrase(
                prompt_csv=str(test_csv),
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                n=None,
                include_method_a=include_method_a,
                include_method_b=include_method_b,
                **loo_baselines,
            )
            print(para_df[["method", "orig_Bypass_Rate", "para_Bypass_Rate", "Bypass_Rate_delta"]].to_string(index=False))
            para_df.to_csv(para_path, index=False)
            _plot_paraphrase_robustness(para_df, figures_dir=figures_dir)
            print(f"[OK] Saved: {para_path}")

            _banner(f"LOO Robustness — Edge Benign ({fold})")
            edge_df = robustness_edge_benign(
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                include_method_a=include_method_a,
                include_method_b=include_method_b,
                **loo_baselines,
            )
            print(edge_df.to_string(index=False))
            edge_df.to_csv(edge_path, index=False)
            print(f"[OK] Saved: {edge_path}")

            _banner(f"LOO Robustness — Adaptive ({fold})")
            if adaptive_max is not None:
                print(f"  Cap: first {adaptive_max} test injections rewritten")
            adaptive_logs_df, adaptive_metrics_df, adaptive_cm_dict = robustness_adaptive(
                test_csv=str(test_csv),
                train_csv=str(train_csv),
                val_csv=str(val_csv),
                use_llm=use_llm,
                include_method_a=include_method_a,
                include_method_b=include_method_b,
                max_injections=adaptive_max,
                include_similarity_evasion=True,
                use_cached_lookup=use_cached_lookup and use_llm,
                outcome_cache=shared_adaptive,
                feature_cache=shared_features,
                batch_size=llm_batch_size,
                **loo_baselines,
            )
            keep_adapt = [c for c in export_cols if c in adaptive_metrics_df.columns]
            if not keep_adapt:
                keep_adapt = list(_metrics_export_columns(adaptive_metrics_df).columns)
            adaptive_metrics_df[keep_adapt].to_csv(adaptive_path, index=False)
            by_style = adaptive_metrics_df.attrs.get("by_style")
            if by_style is not None:
                by_style.to_csv(
                    results_dir / "robustness_adaptive_by_style.csv", index=False
                )
                print(by_style.to_string(index=False))
                print(f"[OK] Saved: {results_dir / 'robustness_adaptive_by_style.csv'}")
            with open(results_dir / "robustness_adaptive_logs.jsonl", "w", encoding="utf-8") as handle:
                for rec in adaptive_logs_df.to_dict(orient="records"):
                    handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
            with open(
                results_dir / "robustness_adaptive_confusion_matrices.json",
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(adaptive_cm_dict, handle, indent=2)
            _plot_adaptive_robustness(adaptive_metrics_df, figures_dir=figures_dir)
            print(f"[OK] Saved: {adaptive_path}")

            _write_robustness_note(
                metrics_df.drop(columns=["fold"], errors="ignore"),
                para_df,
                edge_df,
                adaptive_metrics_df,
                out_path=results_dir / "robustness_note.txt",
                overhead_report=None,
            )
            rob_marker.write_text(mode + "\n", encoding="utf-8")

        para_df = para_df.copy()
        para_df.insert(0, "fold", fold)
        para_collected.append(para_df)
        edge_df = edge_df.copy()
        edge_df.insert(0, "fold", fold)
        edge_collected.append(edge_df)
        adaptive_metrics_df = adaptive_metrics_df.copy()
        adaptive_metrics_df.insert(0, "fold", fold)
        adaptive_collected.append(adaptive_metrics_df)

    summary = _loo_average(pd.concat(collected, ignore_index=True))
    out_dir = RESULTS_DIR / "loo"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    (out_dir / "summary.txt").write_text(_loo_summary_text(summary, mode), encoding="utf-8")
    print(f"\n[OK] Saved: {out_dir / 'summary.csv'}")
    print(_loo_summary_text(summary, mode))

    if skip_robustness or not para_collected:
        return

    para_all = pd.concat(para_collected, ignore_index=True)
    edge_all = pd.concat(edge_collected, ignore_index=True)
    adaptive_all = pd.concat(adaptive_collected, ignore_index=True)
    para_all.to_csv(out_dir / "robustness_paraphrase.csv", index=False)
    edge_all.to_csv(out_dir / "robustness_edge_benign.csv", index=False)
    adaptive_all.to_csv(out_dir / "robustness_adaptive.csv", index=False)

    para_avg = _loo_paraphrase_average(para_all)
    edge_avg = _loo_edge_average(edge_all)
    adaptive_avg = _loo_average(adaptive_all)
    para_avg.to_csv(out_dir / "robustness_paraphrase_summary.csv", index=False)
    edge_avg.to_csv(out_dir / "robustness_edge_benign_summary.csv", index=False)
    adaptive_avg.to_csv(out_dir / "robustness_adaptive_summary.csv", index=False)
    print(f"[OK] Saved: {out_dir / 'robustness_paraphrase.csv'}")
    print(f"[OK] Saved: {out_dir / 'robustness_edge_benign.csv'}")
    print(f"[OK] Saved: {out_dir / 'robustness_adaptive.csv'}")
    print(f"[OK] Saved: {out_dir / 'robustness_*_summary.csv'}")


def _loo_paraphrase_average(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics]
    source = metrics[metrics["fold"].isin(LOO_SOURCE_FOLDS)]
    averaged = []
    for method, group in source.groupby("method", sort=False):
        row = {
            "fold": "four_source_macro_avg",
            "method": method,
            "orig_Bypass_Rate": round(float(group["orig_Bypass_Rate"].mean()), 4),
            "para_Bypass_Rate": round(float(group["para_Bypass_Rate"].mean()), 4),
            "Bypass_Rate_delta": round(float(group["Bypass_Rate_delta"].mean()), 4),
            "n_prompts": int(group["n_prompts"].sum()) if "n_prompts" in group else 0,
            "changed_n": int(group["changed_n"].sum()) if "changed_n" in group else 0,
        }
        if "changed_n_synonym" in group:
            row["changed_n_synonym"] = int(group["changed_n_synonym"].sum())
        if "changed_n_llm" in group:
            row["changed_n_llm"] = int(group["changed_n_llm"].sum())
        averaged.append(row)
    rows.append(pd.DataFrame(averaged))
    return pd.concat(rows, ignore_index=True)


def _loo_edge_average(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics]
    source = metrics[metrics["fold"].isin(LOO_SOURCE_FOLDS)]
    averaged = []
    for method, group in source.groupby("method", sort=False):
        averaged.append({
            "fold": "four_source_macro_avg",
            "method": method,
            "edge_FPR": round(float(group["edge_FPR"].mean()), 4),
            "blocked_count": int(group["blocked_count"].sum()) if "blocked_count" in group else 0,
            "total": int(group["total"].sum()) if "total" in group else 0,
        })
    rows.append(pd.DataFrame(averaged))
    return pd.concat(rows, ignore_index=True)


def _loo_average(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [metrics]
    source = metrics[metrics["fold"].isin(LOO_SOURCE_FOLDS)]
    averaged = []
    for method, group in source.groupby("method", sort=False):
        injections = int(group["injections"].sum()) if "injections" in group else 0
        benign = int(group["benign"].sum()) if "benign" in group else 0
        successes = (
            int(group["successful_attack_count"].sum())
            if "successful_attack_count" in group
            else 0
        )
        missed = int(group["FN"].sum()) if "FN" in group else 0
        false_pos = int(group["FP"].sum()) if "FP" in group else 0
        tp = int(group["TP"].sum()) if "TP" in group else 0
        tn = int(group["TN"].sum()) if "TN" in group else 0
        true_asr_mean = (
            round(float(group["True_ASR_%"].mean()), 2)
            if "True_ASR_%" in group
            else 0.0
        )
        fpr_mean = (
            round(float(group["FPR_%"].mean()), 2) if "FPR_%" in group else 0.0
        )
        averaged.append({
            "fold": "four_source_macro_avg",
            "method": method,
            "Bypass_Rate_%": round(float(group["Bypass_Rate_%"].mean()), 2),
            "True_ASR_%": true_asr_mean,
            "FPR_%": fpr_mean,
            "TP": tp,
            "FN": missed,
            "FP": false_pos,
            "TN": tn,
            "successful_attack_count": successes,
            "injections": injections,
            "benign": benign,
        })
        averaged.append({
            "fold": "four_source_micro",
            "method": method,
            "Bypass_Rate_%": round(100 * missed / injections, 2) if injections else 0.0,
            "True_ASR_%": round(100 * successes / injections, 2) if injections else 0.0,
            "FPR_%": round(100 * false_pos / benign, 2) if benign else 0.0,
            "TP": tp,
            "FN": missed,
            "FP": false_pos,
            "TN": tn,
            "successful_attack_count": successes,
            "injections": injections,
            "benign": benign,
        })
    rows.append(pd.DataFrame(averaged))
    return pd.concat(rows, ignore_index=True)


def _loo_summary_text(summary: pd.DataFrame, mode: str) -> str:
    lines = [
        "Leave-one-source-out, same two-layer scorer as the regular suite.",
        f"Run mode: {mode}.",
        "four_source_macro_avg weights each of the four attack sources equally.",
        "neuralchemy is the fifth dataset and is outside that average.",
        "The benign prompts repeat across the four source folds, so micro FPR is not an independent sample.",
        "",
        summary[["fold", "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%", "injections", "benign"]].to_string(index=False),
        "",
    ]
    return "\n".join(lines) + "\n"


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
    parser.add_argument(
        "--matched-fpr",
        action="store_true",
        help=(
            "Fair operating-point comparison: tune each score-based filter on val "
            "to FPR ≤ 5%%, then report test Bypass/FPR. Writes results/matched_fpr/. "
            "Includes similarity_only, fold-trained DeBERTa, Methods A/B, keyword, "
            "and threshold-retuned ProtectAI / Prompt Guard."
        ),
    )
    parser.add_argument(
        "--matched-fpr-target",
        type=float,
        default=0.05,
        help="With --matched-fpr, max validation FPR (default 0.05).",
    )
    parser.add_argument(
        "--matched-fpr-skip-deberta",
        action="store_true",
        help="With --matched-fpr, skip fine-tuning DeBERTa.",
    )
    parser.add_argument(
        "--matched-fpr-skip-portable",
        action="store_true",
        help="With --matched-fpr, skip ProtectAI / Prompt Guard.",
    )
    parser.add_argument(
        "--loo",
        action="store_true",
        help=(
            "Run the main two-layer experiment once per held-out source "
            "(simsonsun, mpdd, prompt_injection, bravansky, neuralchemy), "
            "then paraphrase / edge-benign / adaptive robustness on each fold. "
            "Uses that fold's train/val/test. Writes results/loo/."
        ),
    )
    parser.add_argument(
        "--loo-folds",
        nargs="+",
        default=None,
        choices=list(LOO_FOLDS),
        help="Subset of --loo folds. Default: all five.",
    )
    parser.add_argument(
        "--loo-skip-robustness",
        action="store_true",
        help="With --loo, run only the main metrics (skip paraphrase / edge / adaptive).",
    )
    parser.add_argument(
        "--loo-adaptive-max",
        type=int,
        default=None,
        help=(
            "With --loo, cap how many test injections are rewritten for adaptive "
            "robustness (default: all injections in the fold test set)."
        ),
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Recompute --loo folds even when a matching results/loo/<fold>/ run exists.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=LLM_CACHE_BATCH_SIZE,
        help=(
            "Batch size for greedy target-LLM generation into the OutcomeCache "
            f"(default {LLM_CACHE_BATCH_SIZE}; env LLM_CACHE_BATCH_SIZE)."
        ),
    )
    parser.add_argument(
        "--no-outcome-cache",
        action="store_true",
        help=(
            "Disable mentor speed path (one Qwen+judge per unique attack). "
            "Falls back to legacy per-(method, prompt) LLM calls."
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

    if args.matched_fpr:
        _banner("Matched-FPR fair comparison (val ≤ target, test Bypass)")
        from .evaluation.matched_fpr import run_matched_fpr_study

        run_matched_fpr_study(
            train_csv=TRAIN_CSV,
            val_csv=VAL_CSV,
            test_csv=TEST_CSV,
            target_fpr=float(args.matched_fpr_target),
            include_portables=not args.matched_fpr_skip_portable,
            include_deberta=not args.matched_fpr_skip_deberta,
            out_dir=RESULTS_DIR / "matched_fpr",
        )
        return

    use_llm = not args.no_llm
    if args.no_judge:
        os.environ["SKIP_JUDGE"] = "1"

    if args.loo:
        if args.suite in ("ab", "both"):
            include_a, include_b = True, True
        elif args.suite == "a":
            include_a, include_b = True, False
        else:
            include_a, include_b = False, True
        _run_loo(
            folds=args.loo_folds or list(LOO_FOLDS),
            use_llm=use_llm,
            no_judge=args.no_judge,
            include_method_a=include_a,
            include_method_b=include_b,
            rerun=args.rerun,
            skip_robustness=args.loo_skip_robustness,
            adaptive_max=args.loo_adaptive_max,
            use_cached_lookup=not args.no_outcome_cache,
            llm_batch_size=args.llm_batch_size,
        )
        return

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
            use_cached_lookup=not args.no_outcome_cache,
            llm_batch_size=args.llm_batch_size,
        )


if __name__ == "__main__":
    main()
