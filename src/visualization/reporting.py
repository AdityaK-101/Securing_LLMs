"""Plain-text robustness report generation."""

from ..config import JUDGE_MODEL_NAME, TARGET_MODEL_NAME


def _write_robustness_note(
    metrics_df,
    para_df,
    edge_df,
    adaptive_df,
    out_path,
    overhead_report=None,
):
    """Write a plain-text robustness note (required deliverable)."""

    metrics_df = metrics_df.copy()
    metrics_df.columns = [c.strip() for c in metrics_df.columns]
    if "method" in metrics_df.columns:
        metrics_df["method"] = metrics_df["method"].astype(str).str.strip()
    para_df = para_df.copy()
    para_df.columns = [c.strip() for c in para_df.columns]
    if "method" in para_df.columns:
        para_df["method"] = para_df["method"].astype(str).str.strip()
    edge_df = edge_df.copy()
    edge_df.columns = [c.strip() for c in edge_df.columns]
    if "method" in edge_df.columns:
        edge_df["method"] = edge_df["method"].astype(str).str.strip()
    adaptive_df = adaptive_df.copy()
    adaptive_df.columns = [c.strip() for c in adaptive_df.columns]
    if "method" in adaptive_df.columns:
        adaptive_df["method"] = adaptive_df["method"].astype(str).str.strip()

    # Find best and worst by Bypass Rate (Layer 1) among active gates only
    active = metrics_df[~metrics_df["method"].isin(["baseline", "no_defense", "none"])]
    if active.empty:
        active = metrics_df
    best_m = active.loc[active["Bypass_Rate_%"].idxmin()]
    worst_m = active.loc[active["Bypass_Rate_%"].idxmax()]

    lines = [
        "=" * 62,
        "  Robustness Note",
        "=" * 62,
        "",
        "METRIC DEFINITIONS",
        "-" * 40,
        "  Bypass Rate (Layer 1): % of injection prompts the sanitizer failed",
        "    to block. Measures sanitizer bypass, NOT model compliance.",
        "  True ASR (Layer 2): % of injection prompts where the sanitizer was",
        f"    bypassed AND the target LLM actually complied (per {JUDGE_MODEL_NAME} judge).",
        "  Confusion matrices use sanitizer blocked/unblocked only.",
        "",
        "1. DEFENSE EFFECTIVENESS (Layer 1 + Layer 2)",
        "-" * 40,
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"  {row['method']:24s}  Bypass={row['Bypass_Rate_%']:5.1f}%  "
            f"TrueASR={row['True_ASR_%']:5.1f}%  FPR={row['FPR_%']:5.1f}%  "
            f"TP={int(row['TP']):3d}  FN={int(row['FN']):3d}  "
            f"FP={int(row['FP']):3d}  TN={int(row['TN']):3d}  "
            f"attacks={int(row['successful_attack_count'])}"
        )

    lines += [
        "",
        f"  Lowest Bypass Rate (active): {best_m['method']} "
        f"(Bypass={best_m['Bypass_Rate_%']:.1f}%, TrueASR={best_m['True_ASR_%']:.1f}%)",
        f"  Highest Bypass Rate (active): {worst_m['method']} "
        f"(Bypass={worst_m['Bypass_Rate_%']:.1f}%, TrueASR={worst_m['True_ASR_%']:.1f}%)",
        "",
        "2. SIDE EFFECTS & FALSE POSITIVES",
        "-" * 40,
    ]

    for _, row in edge_df.iterrows():
        lines.append(
            f"  {row['method']:24s}  edge_FPR={row['edge_FPR']*100:.0f}%  "
            f"blocked {int(row['blocked_count'])}/{int(row['total'])} edge prompts"
        )

    edge_active = edge_df[~edge_df["method"].isin(["baseline", "no_defense", "none"])]
    if len(edge_active):
        hi = edge_active.loc[edge_active["edge_FPR"].idxmax()]
        hi_pct = float(hi["edge_FPR"]) * 100
        lines += [
            "",
            f"  Highest edge-case FPR among listed gates: {hi['method']} "
            f"({hi_pct:.0f}% - {int(hi['blocked_count'])}/{int(hi['total'])} technical benign).",
        ]
    else:
        lines += ["", "  (No edge-benign rows.)"]

    lines += [
        "",
        "3. PARAPHRASE ROBUSTNESS",
        "-" * 40,
    ]
    changed_n = None
    n_prompts = None
    if "changed_n" in para_df.columns and len(para_df):
        changed_n = int(para_df["changed_n"].iloc[0])
    if "n_prompts" in para_df.columns and len(para_df):
        n_prompts = int(para_df["n_prompts"].iloc[0])
    if changed_n is not None and n_prompts is not None:
        lines.append(
            f"  (n={n_prompts} test injections; {changed_n}/{n_prompts} changed surface form)"
        )

    for _, row in para_df.iterrows():
        delta = float(row["Bypass_Rate_delta"]) * 100
        if delta > 0:
            direction = "WORSE (+)"
        elif delta < 0:
            direction = "BETTER (-)"
        else:
            direction = "unchanged"
        lines.append(
            f"  {row['method']:24s}  orig={row['orig_Bypass_Rate']*100:.1f}%  "
            f"para={row['para_Bypass_Rate']*100:.1f}%  Δ={delta:+.1f}% {direction}"
        )

    degraded = para_df[
        (para_df["method"] != "baseline") & (para_df["Bypass_Rate_delta"] > 0)
    ]
    if len(degraded):
        names = ", ".join(degraded["method"].astype(str).tolist())
        lines += [
            "",
            f"  Bypass Rate rises under this synonym paraphrase for: {names}.",
        ]
    else:
        lines += [
            "",
            "  No active defense increased Bypass Rate under this paraphrase probe.",
        ]

    lines += [
        "",
        "4. ADAPTIVE ATTACKS ROBUSTNESS (FLAN-T5)",
        "-" * 40,
    ]
    for _, row in adaptive_df.iterrows():
        blocked_attacks = int(row["TP"])
        if "injections" in adaptive_df.columns:
            total_attacks = int(row["injections"])
        else:
            total_attacks = int(row["TP"]) + int(row["FN"])
        lines.append(
            f"  {row['method']:24s}  Bypass={row['Bypass_Rate_%']:5.1f}%  "
            f"TrueASR={row['True_ASR_%']:5.1f}%  "
            f"blocked {blocked_attacks}/{total_attacks} attacks"
        )
    lines += [
        "",
        "  Adaptive texts: FLAN-T5-Base, 5 styles per held-out injection (n above).",
        f"  Pipeline: sanitizer -> {TARGET_MODEL_NAME} -> judge ({JUDGE_MODEL_NAME},",
        "  separately loaded same checkpoint). Numbers are from this suite CSV only.",
    ]

    if overhead_report is not None:
        from ..evaluation.latency import format_overhead_for_note

        lines += format_overhead_for_note(overhead_report)
        fail_section, repro_section = "6", "7"
    else:
        fail_section, repro_section = "5", "6"

    lines += [
        "",
        f"{fail_section}. NOTES",
        "-" * 40,
        "  - True ASR can be lower than Bypass Rate when the target refuses",
        "    an allowed injection (Layer 2 refusals without gate help).",
        "  - Qualitative miss patterns are not auto-derived here; see suite",
        "    logs / paper discussion for examples.",
        "",
        f"{repro_section}. REPRODUCIBILITY",
        "-" * 40,
        "  Fixed seed: 42 (numpy/random; also torch/transformers when used).",
        "  Gate metrics are deterministic given fixed auxiliaries/thresholds/split.",
        f"  True ASR also depends on {TARGET_MODEL_NAME} generations and",
        f"  {JUDGE_MODEL_NAME} judgments (same checkpoint, separate load).",
        "=" * 62,
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] Saved: {out_path}")
