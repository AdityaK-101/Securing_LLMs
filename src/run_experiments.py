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
    # With phi-2 + Qwen judge (downloads models on first run)
    python -m src.run_experiments

    # Mock mode (no download, sanitizer-based Bypass Rate still valid)
    $env:SKIP_LLM=1; python -m src.run_experiments

    # Optional one-time judge sanity check (not run by default)
    python -m src.run_experiments --calibrate

    # Include stretch-goal classifier
    python -m src.run_experiments --classifier
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from .evaluate import evaluate, robustness_paraphrase, robustness_adaptive, robustness_edge_benign
from .ablation import run_weighted_ablation_study, run_hard_trigger_ablation_study
from .judge import ComplianceJudge, run_calibration

_ROOT       = Path(__file__).parent.parent
RESULTS_DIR = _ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

TRAIN_CSV = str(_ROOT / "data" / "splits" / "train.csv")
TEST_CSV  = str(_ROOT / "data" / "splits" / "test.csv")

METHOD_COLORS = {
    "baseline":     "#e74c3c",
    "regex":        "#e67e22",
    "keyword":      "#3498db",
    "context_aware": "#2ecc71",
}
METHOD_LABELS = {
    "baseline":      "Baseline\n(No Defense)",
    "regex":         "Sanitizer A\n(Regex)",
    "keyword":       "Sanitizer B\n(Keyword)",
    "context_aware": "Context-Aware\n(Proposed)",
}


def _banner(title: str):
    print("\n" + "=" * 62)
    print(f"  {title}")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_bypass_rate_bar(metrics_df: pd.DataFrame):
    """Layer 1 bar chart — sanitizer Bypass Rate (formerly mislabeled ASR)."""
    methods = metrics_df["method"].tolist()
    bypass_pct = metrics_df["Bypass_Rate_%"].tolist()

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, bypass_pct, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, bypass_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Bypass Rate (%)", fontsize=12)
    ax.set_title("Layer 1 — Sanitizer Bypass Rate\n(Lower is Better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "attack_success_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/attack_success_bar.png")
    return out


def _plot_true_asr_bar(metrics_df: pd.DataFrame):
    """Layer 2 bar chart — True Attack Success Rate (judge-confirmed compliance)."""
    methods = metrics_df["method"].tolist()
    true_asr_pct = metrics_df["True_ASR_%"].tolist()

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, true_asr_pct, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, true_asr_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("True ASR (%)", fontsize=12)
    ax.set_title("Layer 2 — True Attack Success Rate\n(Bypass + Phi-2 Complied per Judge)",
                 fontsize=13, fontweight="bold")
    ymax = max(true_asr_pct + [10]) * 1.2 + 5
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "true_asr_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/true_asr_bar.png")
    return out


def _plot_fpr_bar(metrics_df: pd.DataFrame):
    methods = metrics_df["method"].tolist()
    fpr_pct = metrics_df["FPR_%"].tolist()
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    colors  = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, fpr_pct, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, fpr_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("False Positive Rate (%)", fontsize=12)
    ax.set_title("False Positive Rate (FPR) per Method\n(Lower is Better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(fpr_pct + [10]) * 1.4 + 5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "fpr_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/fpr_chart.png")
    return out


def _plot_confusion_matrices(cm_dict: dict):
    """One confusion matrix subplot per method."""
    methods = list(cm_dict.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        d = cm_dict[method]
        cm = np.array(d["matrix"])
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=10, fontweight="bold")

        tick_labels = ["Benign", "Injection"]
        ax.set_xticks([0, 1]); ax.set_xticklabels(tick_labels, fontsize=9)
        ax.set_yticks([0, 1]); ax.set_yticklabels(tick_labels, fontsize=9)
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual", fontsize=9)

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        fontsize=14, fontweight="bold",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.suptitle("Confusion Matrices — Sanitizer Only (Layer 1)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/confusion_matrices.png")
    return out


def _plot_paraphrase_robustness(para_df: pd.DataFrame):
    methods   = para_df["method"].tolist()
    orig_br   = (para_df["orig_Bypass_Rate"] * 100).tolist()
    para_br   = (para_df["para_Bypass_Rate"] * 100).tolist()
    labels    = [METHOD_LABELS.get(m, m) for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, orig_br, w, label="Original Bypass Rate", color="#3498db", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, para_br, w, label="Paraphrased Bypass Rate", color="#e74c3c", edgecolor="black", linewidth=0.7)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Bypass Rate (%)", fontsize=11)
    ax.set_title("Paraphrase Robustness — Original vs Paraphrased Bypass Rate\nHigher Δ = More Vulnerable to Paraphrasing",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "robustness_paraphrase.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/robustness_paraphrase.png")
    return out


def _plot_adaptive_robustness(adaptive_metrics_df: pd.DataFrame):
    """Adaptive attack robustness bar chart — Bypass Rate and True ASR under FLAN-T5 attacks."""
    methods = adaptive_metrics_df["method"].tolist()
    bypass_pct = adaptive_metrics_df["Bypass_Rate_%"].tolist()
    true_asr_pct = adaptive_metrics_df["True_ASR_%"].tolist()
    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(
        x - w / 2, bypass_pct, w,
        label="Bypass Rate (Layer 1)", color="#3498db",
        edgecolor="black", linewidth=0.7,
    )
    b2 = ax.bar(
        x + w / 2, true_asr_pct, w,
        label="True ASR (Layer 2)", color="#e74c3c",
        edgecolor="black", linewidth=0.7,
    )

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.8,
            f"{h:.1f}%", ha="center", va="bottom",
            fontsize=9, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Rate (%)", fontsize=11)
    ax.set_title(
        "Adaptive Robustness — FLAN-T5 Generated Attacks\n(Lower is Better)",
        fontsize=12, fontweight="bold",
    )
    ymax = max(bypass_pct + true_asr_pct + [10]) * 1.2 + 5
    ax.set_ylim(0, ymax)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "robustness_adaptive.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/robustness_adaptive.png")
    return out


def _plot_ablation_bypass_rate(
    ablation_df: pd.DataFrame,
    title: str,
    filename: str,
    full_variant_names: set = None,
):
    """Ablation bar chart — Bypass Rate per variant."""
    if full_variant_names is None:
        full_variant_names = {"Full Weighted Model", "Full Model", "Full"}

    variants = ablation_df["variant"].tolist()
    bypass_pct = ablation_df["Bypass_Rate_%"].tolist()
    colors = [
        "#2ecc71" if v in full_variant_names else "#3498db"
        for v in variants
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        variants, bypass_pct, color=colors,
        edgecolor="black", linewidth=0.8, width=0.65,
    )

    for bar, v in zip(bars, bypass_pct):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
            f"{v:.1f}%", ha="center", va="bottom",
            fontweight="bold", fontsize=10,
        )

    ax.set_ylabel("Bypass Rate (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(bypass_pct + [10]) * 1.2 + 5)
    plt.xticks(rotation=35, ha="right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: results/figures/{filename}")
    return out


# ---------------------------------------------------------------------------
# Robustness Note
# ---------------------------------------------------------------------------

def _write_robustness_note(metrics_df, para_df, edge_df, adaptive_df, out_path):
    """Write a plain-text robustness note (required deliverable)."""

    # Find best and worst by Bypass Rate (Layer 1)
    active = metrics_df[metrics_df["method"] != "baseline"]
    best_m = active.loc[active["Bypass_Rate_%"].idxmin()]
    worst_m = active.loc[active["Bypass_Rate_%"].idxmax()]

    lines = [
        "=" * 62,
        "  Robustness Note — Milestone 3",
        "=" * 62,
        "",
        "METRIC DEFINITIONS",
        "-" * 40,
        "  Bypass Rate (Layer 1): % of injection prompts the sanitizer failed",
        "    to block. Measures sanitizer bypass, NOT model compliance.",
        "  True ASR (Layer 2): % of injection prompts where the sanitizer was",
        "    bypassed AND Phi-2 actually complied (per Qwen2.5-7B judge).",
        "  Confusion matrices use sanitizer blocked/unblocked only.",
        "",
        "1. DEFENSE EFFECTIVENESS (Layer 1 + Layer 2)",
        "-" * 40,
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"  {row['method']:15s}  Bypass={row['Bypass_Rate_%']:5.1f}%  "
            f"TrueASR={row['True_ASR_%']:5.1f}%  FPR={row['FPR_%']:5.1f}%  "
            f"TP={row['TP']:3d}  FN={row['FN']:3d}  "
            f"FP={row['FP']:3d}  TN={row['TN']:3d}  "
            f"attacks={row['successful_attack_count']}"
        )

    lines += [
        "",
        f"  Best performing defense: {best_m['method']} "
        f"(Bypass={best_m['Bypass_Rate_%']:.1f}%, TrueASR={best_m['True_ASR_%']:.1f}%)",
        f"  Weakest active defense:  {worst_m['method']} "
        f"(Bypass={worst_m['Bypass_Rate_%']:.1f}%, TrueASR={worst_m['True_ASR_%']:.1f}%)",
        "",
        "2. SIDE EFFECTS & FALSE POSITIVES",
        "-" * 40,
    ]

    for _, row in edge_df.iterrows():
        lines.append(
            f"  {row['method']:15s}  edge_FPR={row['edge_FPR']*100:.0f}%  "
            f"blocked {row['blocked_count']}/{row['total']} edge prompts"
        )
    lines += [
        "",
        "  Regex sanitizer has the highest edge-case FPR — it triggers",
        "  on technical phrases like 'bypass this error' or 'act as proxy'.",
        "  Context-Aware has a slightly elevated edge FPR but remains",
        "  the best trade-off between security and usability.",
        "",
        "3. PARAPHRASE ROBUSTNESS",
        "-" * 40,
    ]
    for _, row in para_df.iterrows():
        delta = row["Bypass_Rate_delta"] * 100
        direction = "WORSE (+)" if delta > 0 else "better (-)"
        lines.append(
            f"  {row['method']:15s}  orig={row['orig_Bypass_Rate']*100:.1f}%  "
            f"para={row['para_Bypass_Rate']*100:.1f}%  Δ={delta:+.1f}% {direction}"
        )
    lines += [
        "",
        "  All active defenses degrade when attacks are paraphrased.",
        "  Regex is most brittle (relies on exact phrase matching).",
        "  Context-Aware now uses MiniLM dense embeddings for semantic",
        "  similarity, which are more robust to vocabulary changes than",
        "  the previous TF-IDF approach. A perplexity signal (distilgpt2)",
        "  provides an additional language-model naturalness check.",
        "",
        "4. ADAPTIVE ATTACKS ROBUSTNESS (FLAN-T5)",
        "-" * 40,
    ]
    for _, row in adaptive_df.iterrows():
        blocked_attacks = row["TP"]
        total_attacks = row["injections"]
        lines.append(
            f"  {row['method']:15s}  Bypass={row['Bypass_Rate_%']:5.1f}%  "
            f"TrueASR={row['True_ASR_%']:5.1f}%  "
            f"blocked {blocked_attacks}/{total_attacks} attacks"
        )
    lines += [
        "",
        "  Adaptive attacks are generated using FLAN-T5-Base in 5 styles,",
        "  then evaluated through the full sanitizer → Phi-2 → Qwen judge pipeline.",
        "  Context-Aware sanitizer is significantly more robust against these",
        "  attacks compared to keyword and regex defenses because of semantic similarity signals.",
        "",
        "5. WHERE DEFENSES FAILED",
        "-" * 40,
        "  - Sophisticated roleplay jailbreaks (DAN, BOB, STAN personas)",
        "    that embed harmful requests inside fictional narratives.",
        "  - Multilingual attacks (German) partially bypass regex/keyword.",
        "  - Indirect/coded language (e.g., 'assets' meaning stolen data)",
        "    scores low on all signal types.",
        "  - Very long prompts dilute keyword density below threshold.",
        "  - Bypassed prompts may still be refused by Phi-2 (True ASR < Bypass Rate).",
        "",
        "6. REPRODUCIBILITY",
        "-" * 40,
        "  Fixed seed: 42 (applied to numpy and random modules).",
        "  All sanitizer metrics are deterministic given the same dataset split.",
        "  True ASR depends on Phi-2 + Qwen2.5-7B-Instruct inference.",
        "=" * 62,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved: results/robustness_note.txt")


def _metrics_export_columns(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Select required metrics.csv columns per spec."""
    cols = [
        "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%",
        "TP", "FN", "FP", "TN", "successful_attack_count",
    ]
    return metrics_df[cols]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Milestone 3 experiment runner")
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip phi-2 inference (sanitizer-only, fast mode)")
    parser.add_argument("--no-judge", action="store_true",
                        help="Skip Qwen judge (True ASR unavailable)")
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
        print(f"  Target : microsoft/phi-2 ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
        print(f"  Judge  : Qwen2.5-7B-Instruct ({'enabled' if use_llm and not args.no_judge else 'SKIP'})")
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
    print(f"  Target : microsoft/phi-2 ({'enabled' if use_llm else 'SKIP (--no-llm)'})")
    print(f"  Judge  : Qwen2.5-7B-Instruct ({'enabled' if use_llm and not args.no_judge else 'SKIP'})")
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
    # Main Evaluation (Phase 1: Phi-2, Phase 2: Qwen judge — one model at a time)
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
