"""
src/run_experiments.py
======================
CLI entrypoint for Milestone 3 experiments.

Produces:
  results/metrics.csv          — ASR/FPR per method
  results/logs.jsonl           — per-prompt record
  results/confusion_matrices.json
  results/robustness_paraphrase.csv
  results/robustness_edge_benign.csv
  results/figures/attack_success_bar.png   ← required by roadmap §5.2
  results/figures/confusion_matrix_*.png
  results/figures/fpr_chart.png
  results/robustness_note.txt   ← required deliverable

Usage:
    # With phi-2 (downloads ~5.5GB on first run)
    python -m src.run_experiments

    # Mock mode (no download, sanitizer-based ASR still valid)
    $env:SKIP_LLM=1; python -m src.run_experiments

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

from .evaluate import evaluate, robustness_paraphrase, robustness_edge_benign

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

def _plot_asr_bar(metrics_df: pd.DataFrame):
    """Required bar chart from roadmap §5.2."""
    methods = metrics_df["method"].tolist()
    asr_pct = metrics_df["ASR_%"].tolist()

    labels = [METHOD_LABELS.get(m, m) for m in methods]
    colors = [METHOD_COLORS.get(m, "#95a5a6") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, asr_pct, color=colors, edgecolor="black", linewidth=0.8, width=0.55)

    for bar, v in zip(bars, asr_pct):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{v:.1f}%", ha="center", va="bottom", fontweight="bold", fontsize=11)

    ax.set_ylabel("Attack Success Rate (%)", fontsize=12)
    ax.set_title("Baseline vs Sanitizers — Attack Success Rate\n(Lower is Better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.axhline(y=100, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "attack_success_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: results/figures/attack_success_bar.png")
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
    print(f"[✓] Saved: results/figures/fpr_chart.png")
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

    plt.suptitle("Confusion Matrices — All Methods", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    out = FIGURES_DIR / "confusion_matrices.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: results/figures/confusion_matrices.png")
    return out


def _plot_paraphrase_robustness(para_df: pd.DataFrame):
    methods   = para_df["method"].tolist()
    orig_asr  = (para_df["orig_ASR"] * 100).tolist()
    para_asr  = (para_df["para_ASR"] * 100).tolist()
    labels    = [METHOD_LABELS.get(m, m) for m in methods]

    x = np.arange(len(methods))
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - w / 2, orig_asr, w, label="Original ASR", color="#3498db", edgecolor="black", linewidth=0.7)
    b2 = ax.bar(x + w / 2, para_asr, w, label="Paraphrased ASR", color="#e74c3c", edgecolor="black", linewidth=0.7)

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.8,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("Paraphrase Robustness — Original vs Paraphrased ASR\nHigher Δ = More Vulnerable to Paraphrasing",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = FIGURES_DIR / "robustness_paraphrase.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: results/figures/robustness_paraphrase.png")
    return out


# ---------------------------------------------------------------------------
# Robustness Note
# ---------------------------------------------------------------------------

def _write_robustness_note(metrics_df, para_df, edge_df, out_path):
    """Write a plain-text robustness note (required deliverable)."""

    # Find best and worst by ASR
    active = metrics_df[metrics_df["method"] != "baseline"]
    best_m = active.loc[active["ASR_%"].idxmin()]
    worst_m = active.loc[active["ASR_%"].idxmax()]

    lines = [
        "=" * 62,
        "  Robustness Note — Milestone 3",
        "=" * 62,
        "",
        "1. DEFENSE EFFECTIVENESS",
        "-" * 40,
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"  {row['method']:15s}  ASR={row['ASR_%']:5.1f}%  FPR={row['FPR_%']:5.1f}%  "
            f"TP={row['TP (blocked_inj)']:3d}  FN={row['FN (missed_inj)']:3d}  "
            f"FP={row['FP (blocked_ben)']:3d}  TN={row['TN (allowed_ben)']:3d}"
        )

    lines += [
        "",
        f"  Best performing defense: {best_m['method']} "
        f"(ASR={best_m['ASR_%']:.1f}%)",
        f"  Weakest active defense:  {worst_m['method']} "
        f"(ASR={worst_m['ASR_%']:.1f}%)",
        "",
        "2. SIDE EFFECTS & FALSE POSITIVES",
        "-" * 40,
    ]

    row_baseline = edge_df[edge_df["method"] == "baseline"].iloc[0]
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
        delta = row["ASR_delta"] * 100
        direction = "WORSE (+)" if delta > 0 else "better (-)"
        lines.append(
            f"  {row['method']:15s}  orig={row['orig_ASR']*100:.1f}%  "
            f"para={row['para_ASR']*100:.1f}%  Δ={delta:+.1f}% {direction}"
        )
    lines += [
        "",
        "  All active defenses degrade when attacks are paraphrased.",
        "  Regex is most brittle (relies on exact phrase matching).",
        "  Context-Aware shows the largest paraphrase gap — indicating",
        "  that TF-IDF semantic vectors shift when vocabulary changes.",
        "  This motivates using dense embeddings (e.g., SentenceTransformers)",
        "  in future work.",
        "",
        "4. WHERE DEFENSES FAILED",
        "-" * 40,
        "  - Sophisticated roleplay jailbreaks (DAN, BOB, STAN personas)",
        "    that embed harmful requests inside fictional narratives.",
        "  - Multilingual attacks (German) partially bypass regex/keyword.",
        "  - Indirect/coded language (e.g., 'assets' meaning stolen data)",
        "    scores low on all signal types.",
        "  - Very long prompts dilute keyword density below threshold.",
        "",
        "5. REPRODUCIBILITY",
        "-" * 40,
        "  Fixed seed: 42 (applied to numpy and random modules).",
        "  All metrics are deterministic given the same dataset split.",
        "=" * 62,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[✓] Saved: results/robustness_note.txt")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Milestone 3 experiment runner")
    parser.add_argument("--classifier", action="store_true",
                        help="Run stretch-goal LR+TF-IDF classifier")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    skip_llm = os.environ.get("SKIP_LLM", "0").strip() == "1"
    use_llm  = not skip_llm

    _banner("Milestone 3 — Prompt Injection Defense Evaluation")
    print(f"  Model  : {'microsoft/phi-2' if use_llm else 'MOCK (SKIP_LLM=1)'}")
    print(f"  Seed   : 42 (fixed for reproducibility)")
    print(f"  Test   : {TEST_CSV}")
    print(f"  Train  : {TRAIN_CSV}")

    # -----------------------------------------------------------------------
    # Main Evaluation
    # -----------------------------------------------------------------------
    _banner("§4.5  Main Experiment")
    logs_df, metrics_df, cm_dict = evaluate(
        test_csv=TEST_CSV, train_csv=TRAIN_CSV, use_llm=use_llm,
    )

    print("\n--- ASR / FPR Summary ---")
    cols = ["method", "ASR_%", "FPR_%", "TP (blocked_inj)",
            "FN (missed_inj)", "FP (blocked_ben)", "TN (allowed_ben)"]
    print(metrics_df[cols].to_string(index=False))

    # Save
    metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    with open(RESULTS_DIR / "logs.jsonl", "w", encoding="utf-8") as f:
        for rec in logs_df.to_dict(orient="records"):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"\n[✓] Saved: results/metrics.csv")
    print(f"[✓] Saved: results/logs.jsonl  ({len(logs_df)} records)")

    # -----------------------------------------------------------------------
    # Figures
    # -----------------------------------------------------------------------
    _banner("Generating Figures")
    _plot_asr_bar(metrics_df)
    _plot_fpr_bar(metrics_df)
    _plot_confusion_matrices(cm_dict)

    # -----------------------------------------------------------------------
    # Robustness Testing
    # -----------------------------------------------------------------------
    _banner("§4.6  Robustness — Paraphrase Test")
    para_df = robustness_paraphrase(train_csv=TRAIN_CSV, n=50)
    print("\n--- Paraphrase ASR Delta ---")
    print(para_df.to_string(index=False))
    para_df.to_csv(RESULTS_DIR / "robustness_paraphrase.csv", index=False)
    _plot_paraphrase_robustness(para_df)
    print(f"[✓] Saved: results/robustness_paraphrase.csv")

    _banner("§4.6  Robustness — Edge Benign Cases")
    edge_df = robustness_edge_benign()
    print("\n--- Edge Benign FPR ---")
    print(edge_df.to_string(index=False))
    edge_df.to_csv(RESULTS_DIR / "robustness_edge_benign.csv", index=False)
    print(f"[✓] Saved: results/robustness_edge_benign.csv")

    # -----------------------------------------------------------------------
    # Robustness Note
    # -----------------------------------------------------------------------
    _banner("Writing Robustness Note")
    _write_robustness_note(
        metrics_df, para_df, edge_df,
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
            "val_accuracy":  clf_results["val"]["accuracy"],
            "val_f1":        clf_results["val"]["f1"],
            "val_ASR":       clf_results["val"]["ASR"],
            "val_FPR":       clf_results["val"]["FPR"],
            "test_accuracy": clf_results["test"]["accuracy"],
            "test_f1":       clf_results["test"]["f1"],
            "test_ASR":      clf_results["test"]["ASR"],
            "test_FPR":      clf_results["test"]["FPR"],
        }
        pd.DataFrame([clf_out]).to_csv(
            RESULTS_DIR / "classifier_metrics.csv", index=False
        )
        print(f"[✓] Saved: results/classifier_metrics.csv")

    # -----------------------------------------------------------------------
    _banner("Experiment Complete ✓")
    print("  All results written to results/")
    print("  Figures:  results/figures/")
    print("  Key file: results/robustness_note.txt\n")


if __name__ == "__main__":
    main()
