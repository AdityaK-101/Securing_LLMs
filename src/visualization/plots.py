"""Matplotlib figures for experiment results."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..config import FIGURES_DIR

METHOD_COLORS = {
    "baseline":               "#e74c3c",
    "regex":                  "#e67e22",
    "keyword":                "#3498db",
    "context_aware":          "#2ecc71",
    "context_aware_learned":  "#1abc9c",
}
METHOD_LABELS = {
    "baseline":              "Baseline\n(No Defense)",
    "regex":                 "Sanitizer A\n(Regex)",
    "keyword":               "Sanitizer B\n(Keyword)",
    "context_aware":         "Context-Aware\n(Hand + Triggers)",
    "context_aware_learned": "Context-Aware\n(Learned Soft)",
}


def _figures_dir(figures_dir=None) -> Path:
    d = Path(figures_dir) if figures_dir is not None else FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _plot_bypass_rate_bar(metrics_df: pd.DataFrame, figures_dir=None):
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

    out = _figures_dir(figures_dir) / "attack_success_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_true_asr_bar(metrics_df: pd.DataFrame, figures_dir=None):
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
    ax.set_title("Layer 2 — True Attack Success Rate\n(Bypass + Target LLM Complied per Judge)",
                 fontsize=13, fontweight="bold")
    ymax = max(true_asr_pct + [10]) * 1.2 + 5
    ax.set_ylim(0, ymax)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out = _figures_dir(figures_dir) / "true_asr_bar.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_fpr_bar(metrics_df: pd.DataFrame, figures_dir=None):
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

    out = _figures_dir(figures_dir) / "fpr_chart.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_confusion_matrices(cm_dict: dict, figures_dir=None):
    """One confusion matrix subplot per method."""
    methods = list(cm_dict.keys())
    fig, axes = plt.subplots(1, len(methods), figsize=(4 * len(methods), 4))
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        d = cm_dict[method]
        cm = np.array(d["matrix"])
        ax.imshow(cm, interpolation="nearest", cmap="Blues")
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

    out = _figures_dir(figures_dir) / "confusion_matrices.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_paraphrase_robustness(para_df: pd.DataFrame, figures_dir=None):
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

    out = _figures_dir(figures_dir) / "robustness_paraphrase.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_adaptive_robustness(adaptive_metrics_df: pd.DataFrame, figures_dir=None):
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

    out = _figures_dir(figures_dir) / "robustness_adaptive.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out


def _plot_ablation_bypass_rate(
    ablation_df: pd.DataFrame,
    title: str,
    filename: str,
    full_variant_names: set = None,
    figures_dir=None,
):
    """Ablation bar chart — Bypass Rate per variant."""
    if full_variant_names is None:
        full_variant_names = {
            "Full Weighted Model",
            "Full Model",
            "Full Model (A + Hard Triggers)",
            "Full Learned Model",
            "Method A Full (Hard Triggers)",
            "Full",
        }

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

    out = _figures_dir(figures_dir) / filename
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved: {out}")
    return out
