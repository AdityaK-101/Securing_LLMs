"""
Build paper-facing LOO matched-FPR tables:

- ProtectAI / Prompt Guard rows (when present in metrics.csv)
- Neuralchemy as its own headline column / block
- Mixed suite matched-FPR (combined-dataset random split; paper ID column)
- Per-fold Wilson 95% CIs + four-fold macro average + pooled micro
- Deduplication counts and train-benign-by-fold inventory
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import RESULTS_DIR, SOURCE_HOLDOUT_DIR, TEST_CSV
from .loo_matched_fpr import LOO_FOLDS, LOO_SOURCE_FOLDS
from .wilson import wilson_pct_str

OUT = RESULTS_DIR / "loo_matched_fpr"
MATCHED_FPR = RESULTS_DIR / "matched_fpr" / "metrics.csv"
MANIFEST = SOURCE_HOLDOUT_DIR / "manifest.json"

METHOD_ORDER = [
    "baseline",
    "regex",
    "keyword",
    "context_aware",
    "context_aware_learned",
    "similarity_only",
    "deberta_finetuned",
    "protectai",
    "prompt_guard",
]

METHOD_LABEL = {
    "baseline": "No sanitizer (baseline)",
    "regex": "Regex-only",
    "keyword": "Keyword-only",
    "context_aware": "Method A (hybrid)",
    "context_aware_learned": "Method B (learned)",
    "similarity_only": "Similarity-only",
    "deberta_finetuned": "DeBERTa fine-tuned",
    "protectai": "ProtectAI (DeBERTa-v3)",
    "prompt_guard": "Prompt Guard 2",
}


def _rate_ci(successes: int, total: int, rate_pct: float) -> str:
    if total <= 0:
        return "—"
    return f"{rate_pct:.2f} {wilson_pct_str(int(successes), int(total))}"


def _enrich_with_wilson(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        n_inj = int(r.get("injections") or 0)
        n_ben = int(r.get("benign") or 0)
        bypass_n = int(r.get("FN") or 0)
        fpr_n = int(r.get("FP") or 0)
        asr_n = int(r.get("successful_attack_count") or 0)
        out = dict(r)
        out["Bypass_Wilson95"] = (
            wilson_pct_str(bypass_n, n_inj) if n_inj else ""
        )
        out["True_ASR_Wilson95"] = (
            wilson_pct_str(asr_n, n_inj) if n_inj else ""
        )
        out["FPR_Wilson95"] = wilson_pct_str(fpr_n, n_ben) if n_ben else ""
        # Macro avg: no single binomial N — leave Wilson blank.
        if str(r.get("fold", "")).endswith("macro_avg"):
            out["Bypass_Wilson95"] = ""
            out["True_ASR_Wilson95"] = ""
            out["FPR_Wilson95"] = ""
        rows.append(out)
    return pd.DataFrame(rows)


def _in_distribution_matched() -> pd.DataFrame:
    """Mixed random-split matched-FPR (val FPR ≤ 5%) Bypass / True ASR / FPR."""
    if not MATCHED_FPR.is_file():
        return pd.DataFrame()
    m = pd.read_csv(MATCHED_FPR)
    m.columns = [c.strip() for c in m.columns]
    mode = m["mode"].astype(str).str.strip()
    matched = m[mode.str.startswith("matched_fpr")].copy()
    rows = []
    for _, r in matched.iterrows():
        method = str(r["method"]).strip()
        n_inj = int(r.get("test_FN", 0) or 0) + int(r.get("test_TP", 0) or 0)
        asr_n = r.get("test_successful_attack_count")
        if asr_n is None or (isinstance(asr_n, float) and pd.isna(asr_n)):
            # Prefer explicit True ASR % when count missing.
            asr_pct = r.get("test_True_ASR_%")
            asr_n = (
                int(round(float(asr_pct) / 100.0 * n_inj))
                if asr_pct is not None and not (isinstance(asr_pct, float) and pd.isna(asr_pct))
                else 0
            )
        else:
            asr_n = int(asr_n)
        asr_pct = r.get("test_True_ASR_%")
        rows.append(
            {
                "method": method,
                "id_Bypass_Rate_%": float(r["test_Bypass_Rate_%"]),
                "id_True_ASR_%": (
                    float(asr_pct)
                    if asr_pct is not None and not (isinstance(asr_pct, float) and pd.isna(asr_pct))
                    else None
                ),
                "id_FPR_%": float(r["test_FPR_%"]),
                "id_threshold": r.get("threshold"),
                "id_n_inj": n_inj,
                "id_n_ben": int(r.get("test_FP", 0) or 0) + int(r.get("test_TN", 0) or 0),
                "id_bypass_n": int(r.get("test_FN", 0) or 0),
                "id_asr_n": asr_n,
                "id_fpr_n": int(r.get("test_FP", 0) or 0),
            }
        )
    return pd.DataFrame(rows)


def _dedup_and_benign_tables(manifest: dict) -> tuple[str, str, str]:
    nd = manifest.get("near_duplicate_filter", {})
    exact = manifest.get("exact_overlaps", {})
    dedup_lines = [
        "| Filter | Count |",
        "|--------|------:|",
        f"| Cross-source MiniLM pairs compared | {nd.get('compared_prompts', '—')} |",
        f"| Near-duplicates removed (cosine ≥ {nd.get('threshold', 0.9)}) | {nd.get('removed', '—')} |",
    ]
    for src, n in (nd.get("removed_by_source") or {}).items():
        dedup_lines.append(f"| · removed from `{src}` | {n} |")
    dedup_lines.append("| Exact string overlaps (pre-filter inventory) |  |")
    for src, n in exact.items():
        dedup_lines.append(f"| · exact overlap involving `{src}` | {n} |")
    nc = (manifest.get("folds") or {}).get("neuralchemy", {})
    if "dropped_overlap_with_four_sources" in nc:
        dedup_lines.append(
            f"| Neuralchemy prompts dropped for overlap with four sources | "
            f"{nc['dropped_overlap_with_four_sources']} |"
        )

    benign_lines = [
        "| Fold | Train Alpaca | Val Alpaca | Fit total (train+val) | Train PI-benign | Val PI-benign |",
        "|------|-------------:|-----------:|----------------------:|----------------:|--------------:|",
    ]
    for fold in LOO_FOLDS:
        info = (manifest.get("folds") or {}).get(fold, {})
        tb = info.get("train_benign_by_source") or {}
        vb = info.get("val_benign_by_source") or {}
        t_alp = int(tb.get("alpaca_benign", 0) or 0)
        v_alp = int(vb.get("alpaca_benign", 0) or 0)
        t_pi = int(tb.get("prompt_injection_benign", 0) or 0)
        v_pi = int(vb.get("prompt_injection_benign", 0) or 0)
        # Legacy ChatGPT columns (pre held-out protocol) roll into Alpaca slot if present.
        if t_alp == 0 and v_alp == 0 and (
            tb.get("chatgpt_benign") is not None or vb.get("chatgpt_benign") is not None
        ):
            t_alp = int(tb.get("chatgpt_benign", 0) or 0)
            v_alp = int(vb.get("chatgpt_benign", 0) or 0)
        benign_lines.append(
            f"| `{fold}` | {t_alp} | {v_alp} | {t_alp + v_alp} | {t_pi} | {v_pi} |"
        )
    benign_lines.append("")
    alp = manifest.get("alpaca_fit") or {}
    fit_n = int(alp.get("rows") or (alp.get("train", 0) + alp.get("val", 0)) or 300)
    benign_lines.append(
        f"Each fold fits Method B / similarity / DeBERTa on **{fit_n}** Alpaca "
        "instruction prompts "
        f"(train={int(alp.get('train', 200))}, val={int(alp.get('val', 100))}). "
        "**PI-benign is test-only** (never in train/val). Shared test benign "
        f"= **{manifest.get('shared_benign_test', {}).get('rows', 200)}** "
        "PI-benign prompts, identical across **all** folds including Neuralchemy."
    )

    spot = nd.get("spot_check") or []
    spot_lines = [
        "| Removed source | Kept source | Cosine |",
        "|----------------|-------------|-------:|",
    ]
    for s in spot:
        spot_lines.append(
            f"| `{s.get('removed_source')}` | `{s.get('kept_source')}` | "
            f"{float(s.get('cosine', 0)):.3f} |"
        )
    return "\n".join(dedup_lines), "\n".join(benign_lines), "\n".join(spot_lines)


def build_paper_tables(out_dir: Path | None = None) -> Path:
    out = Path(out_dir) if out_dir else OUT
    metrics_path = out / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(metrics_path)

    raw = pd.read_csv(metrics_path)
    enriched = _enrich_with_wilson(raw)
    enriched.to_csv(out / "metrics_with_wilson.csv", index=False)

    id_df = _in_distribution_matched()
    id_map = (
        id_df.set_index("method").to_dict(orient="index") if not id_df.empty else {}
    )

    methods = [m for m in METHOD_ORDER if m in set(raw["method"].astype(str))]

    # --- Headline comparison table ---
    headline = []
    for method in methods:
        row = {"method": method, "label": METHOD_LABEL.get(method, method)}
        for fold_key, prefix in (
            ("four_source_macro_avg", "loo4_macro"),
            ("four_source_micro", "loo4_micro"),
            ("neuralchemy", "neuralchemy"),
        ):
            sub = raw[(raw["fold"] == fold_key) & (raw["method"] == method)]
            if sub.empty:
                row[f"{prefix}_Bypass_%"] = None
                row[f"{prefix}_True_ASR_%"] = None
                row[f"{prefix}_FPR_%"] = None
                row[f"{prefix}_Bypass_Wilson95"] = ""
                row[f"{prefix}_True_ASR_Wilson95"] = ""
                row[f"{prefix}_FPR_Wilson95"] = ""
                continue
            r = sub.iloc[0]
            row[f"{prefix}_Bypass_%"] = float(r["Bypass_Rate_%"])
            row[f"{prefix}_True_ASR_%"] = float(r["True_ASR_%"])
            row[f"{prefix}_FPR_%"] = float(r["FPR_%"])
            n_inj = int(r.get("injections") or 0)
            n_ben = int(r.get("benign") or 0)
            if fold_key == "four_source_macro_avg":
                row[f"{prefix}_Bypass_Wilson95"] = ""
                row[f"{prefix}_True_ASR_Wilson95"] = ""
                row[f"{prefix}_FPR_Wilson95"] = ""
            else:
                row[f"{prefix}_Bypass_Wilson95"] = wilson_pct_str(
                    int(r.get("FN") or 0), n_inj
                )
                row[f"{prefix}_True_ASR_Wilson95"] = wilson_pct_str(
                    int(r.get("successful_attack_count") or 0), n_inj
                )
                row[f"{prefix}_FPR_Wilson95"] = wilson_pct_str(
                    int(r.get("FP") or 0), n_ben
                )
        id_info = id_map.get(method, {})
        row["id_Bypass_%"] = id_info.get("id_Bypass_Rate_%")
        row["id_True_ASR_%"] = id_info.get("id_True_ASR_%")
        row["id_FPR_%"] = id_info.get("id_FPR_%")
        if id_info:
            n_inj = int(id_info.get("id_n_inj") or 0)
            row["id_Bypass_Wilson95"] = wilson_pct_str(
                int(id_info.get("id_bypass_n") or 0),
                n_inj,
            )
            row["id_True_ASR_Wilson95"] = wilson_pct_str(
                int(id_info.get("id_asr_n") or 0),
                n_inj,
            )
            row["id_FPR_Wilson95"] = wilson_pct_str(
                int(id_info.get("id_fpr_n") or 0),
                int(id_info.get("id_n_ben") or 0),
            )
        else:
            row["id_Bypass_Wilson95"] = ""
            row["id_True_ASR_Wilson95"] = ""
            row["id_FPR_Wilson95"] = ""
        headline.append(row)
    headline_df = pd.DataFrame(headline)
    headline_df.to_csv(out / "paper_headline.csv", index=False)

    # --- Per-fold with Wilson ---
    per_fold = enriched[enriched["fold"].isin(LOO_FOLDS)].copy()
    per_fold.to_csv(out / "paper_per_fold_wilson.csv", index=False)

    # --- Markdown ---
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    dedup_md, benign_md, spot_md = _dedup_and_benign_tables(manifest)

    def _fmt(v, wilson=""):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "—"
        if wilson:
            return f"{float(v):.2f} {wilson}"
        return f"{float(v):.2f}"

    lines: list[str] = []
    lines.append("# LOO matched-FPR ≈ 5% — paper tables")
    lines.append("")
    lines.append(
        "Operating point: per-fold validation threshold chosen to **hit val FPR ≈ 5%** "
        "(closest). ProtectAI / Prompt Guard 2 keep published weights; only the score "
        "threshold is retuned. Soft features / Method A–B architecture unchanged."
    )
    lines.append("")
    lines.append(
        f"**ID** column = **mixed suite matched-FPR** on the combined-dataset "
        f"random split (`{Path(TEST_CSV).name}` → `results/matched_fpr/`, "
        "val FPR ≤ 5%). LOO columns use leave-one-source-out test sets "
        "(200 inj / 200 ben each)."
    )
    lines.append("")

    lines.append(
        "## Table A — Main comparison "
        "(mixed suite matched-FPR / ID vs four-source LOO vs Neuralchemy)"
    )
    lines.append("")
    lines.append(
        "| Method | ID Bypass % | ID True ASR % | ID FPR % | LOO-4 macro Bypass % | LOO-4 macro True ASR % | LOO-4 macro FPR % | "
        "LOO-4 pooled Bypass % (Wilson) | LOO-4 pooled True ASR % (Wilson) | LOO-4 pooled FPR % (Wilson) | "
        "Neuralchemy Bypass % (Wilson) | Neuralchemy True ASR % (Wilson) | Neuralchemy FPR % (Wilson) |"
    )
    lines.append(
        "|--------|------------:|--------------:|---------:|---------------------:|-----------------------:|------------------:|"
        "--------------------------------:|----------------------------------:|------------------------------:|"
        "-------------------------------:|---------------------------------:|-----------------------------:|"
    )
    for _, r in headline_df.iterrows():
        lines.append(
            "| {lab} | {id_b} | {id_a} | {id_f} | {m_b} | {m_a} | {m_f} | {p_b} | {p_a} | {p_f} | {n_b} | {n_a} | {n_f} |".format(
                lab=r["label"],
                id_b=_fmt(r.get("id_Bypass_%"), r.get("id_Bypass_Wilson95") or ""),
                id_a=_fmt(r.get("id_True_ASR_%"), r.get("id_True_ASR_Wilson95") or ""),
                id_f=_fmt(r.get("id_FPR_%"), r.get("id_FPR_Wilson95") or ""),
                m_b=_fmt(r.get("loo4_macro_Bypass_%")),
                m_a=_fmt(r.get("loo4_macro_True_ASR_%")),
                m_f=_fmt(r.get("loo4_macro_FPR_%")),
                p_b=_fmt(
                    r.get("loo4_micro_Bypass_%"),
                    r.get("loo4_micro_Bypass_Wilson95") or "",
                ),
                p_a=_fmt(
                    r.get("loo4_micro_True_ASR_%"),
                    r.get("loo4_micro_True_ASR_Wilson95") or "",
                ),
                p_f=_fmt(
                    r.get("loo4_micro_FPR_%"),
                    r.get("loo4_micro_FPR_Wilson95") or "",
                ),
                n_b=_fmt(
                    r.get("neuralchemy_Bypass_%"),
                    r.get("neuralchemy_Bypass_Wilson95") or "",
                ),
                n_a=_fmt(
                    r.get("neuralchemy_True_ASR_%"),
                    r.get("neuralchemy_True_ASR_Wilson95") or "",
                ),
                n_f=_fmt(
                    r.get("neuralchemy_FPR_%"),
                    r.get("neuralchemy_FPR_Wilson95") or "",
                ),
            )
        )
    lines.append("")
    lines.append(
        "LOO-4 **macro** = equal weight per held-out source. LOO-4 **pooled** = micro "
        "(800 inj / 800 ben) with Wilson 95% CI. Neuralchemy is the cleanest unseen "
        "external holdout and is **not** inside the four-source average."
    )
    if "protectai" not in methods or "prompt_guard" not in methods:
        lines.append("")
        lines.append(
            "> ProtectAI / Prompt Guard rows appear after: "
            "`python scripts/eval_loo_matched_fpr_portables.py`"
        )
    lines.append("")

    lines.append("## Table B — Per-fold main metrics with Wilson 95% CIs")
    lines.append("")
    lines.append(
        "| Fold | Method | Bypass % (Wilson) | True ASR % (Wilson) | FPR % (Wilson) |"
    )
    lines.append(
        "|------|--------|------------------:|--------------------:|---------------:|"
    )
    for fold in LOO_FOLDS:
        for method in methods:
            sub = per_fold[
                (per_fold["fold"] == fold) & (per_fold["method"] == method)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            lines.append(
                f"| `{fold}` | {METHOD_LABEL.get(method, method)} | "
                f"{float(r['Bypass_Rate_%']):.2f} {r['Bypass_Wilson95']} | "
                f"{float(r['True_ASR_%']):.2f} {r['True_ASR_Wilson95']} | "
                f"{float(r['FPR_%']):.2f} {r['FPR_Wilson95']} |"
            )
    lines.append("")

    lines.append("## Table C — Four-source average and pooled")
    lines.append("")
    lines.append(
        "| Method | Macro Bypass % | Macro True ASR % | Macro FPR % | "
        "Pooled Bypass % (Wilson) | Pooled True ASR % (Wilson) | Pooled FPR % (Wilson) |"
    )
    lines.append(
        "|--------|---------------:|-----------------:|------------:|"
        "------------------------:|--------------------------:|----------------------:|"
    )
    for method in methods:
        macro = raw[
            (raw["fold"] == "four_source_macro_avg") & (raw["method"] == method)
        ]
        micro = enriched[
            (enriched["fold"] == "four_source_micro") & (enriched["method"] == method)
        ]
        if macro.empty or micro.empty:
            continue
        a, b = macro.iloc[0], micro.iloc[0]
        lines.append(
            f"| {METHOD_LABEL.get(method, method)} | "
            f"{float(a['Bypass_Rate_%']):.2f} | {float(a['True_ASR_%']):.2f} | "
            f"{float(a['FPR_%']):.2f} | "
            f"{float(b['Bypass_Rate_%']):.2f} {b['Bypass_Wilson95']} | "
            f"{float(b['True_ASR_%']):.2f} {b['True_ASR_Wilson95']} | "
            f"{float(b['FPR_%']):.2f} {b['FPR_Wilson95']} |"
        )
    lines.append("")

    lines.append("## Table D — Deduplication")
    lines.append("")
    lines.append(dedup_md)
    lines.append("")
    lines.append("### Near-duplicate spot checks")
    lines.append("")
    lines.append(spot_md)
    lines.append("")

    lines.append("## Table E — Train benign inventory by fold")
    lines.append("")
    lines.append(benign_md)
    lines.append("")

    # --- Table F: three-regime FPR + adaptive Bypass (incl. portables) ---
    edge_path = out / "robustness_edge_benign.csv"
    adaptive_path = out / "robustness_adaptive.csv"
    lines.append(
        "## Table F — Three-regime FPR + adaptive Bypass "
        "(matched LOO thresholds; edge = 80 technical prompts)"
    )
    lines.append("")
    lines.append(
        "| Method | LOO-4 PI-benign FPR % | Neuralchemy-side FPR % | "
        "LOO-4 Edge FPR % | LOO-4 Adaptive Bypass % |"
    )
    lines.append(
        "|--------|----------------------:|-----------------------:|"
        "-----------------:|-------------------------:|"
    )

    def _edge_macro_pct(method: str) -> float | None:
        if not edge_path.is_file():
            return None
        e = pd.read_csv(edge_path)
        e.columns = [str(c).strip() for c in e.columns]
        macro = e[
            (e["fold"].astype(str) == "four_source_macro_avg")
            & (e["method"].astype(str) == method)
        ]
        if not macro.empty:
            if "edge_FPR_%" in macro.columns and pd.notna(macro.iloc[0].get("edge_FPR_%")):
                return float(macro.iloc[0]["edge_FPR_%"])
            if "edge_FPR" in macro.columns and pd.notna(macro.iloc[0].get("edge_FPR")):
                return float(macro.iloc[0]["edge_FPR"]) * 100.0
        src = e[
            (e["fold"].isin(LOO_SOURCE_FOLDS)) & (e["method"].astype(str) == method)
        ]
        if src.empty:
            return None
        if "edge_FPR_%" in src.columns and src["edge_FPR_%"].notna().any():
            return float(src["edge_FPR_%"].mean())
        if "edge_FPR" in src.columns:
            return float(src["edge_FPR"].mean()) * 100.0
        return None

    def _adaptive_macro_pct(method: str) -> float | None:
        if not adaptive_path.is_file():
            return None
        a = pd.read_csv(adaptive_path)
        a.columns = [str(c).strip() for c in a.columns]
        macro = a[
            (a["fold"].astype(str) == "four_source_macro_avg")
            & (a["method"].astype(str) == method)
        ]
        if not macro.empty and "Bypass_Rate_%" in macro.columns:
            return float(macro.iloc[0]["Bypass_Rate_%"])
        src = a[
            (a["fold"].isin(LOO_SOURCE_FOLDS)) & (a["method"].astype(str) == method)
        ]
        if src.empty or "Bypass_Rate_%" not in src.columns:
            return None
        return float(src["Bypass_Rate_%"].mean())

    for method in methods:
        lab = METHOD_LABEL.get(method, method)
        macro = raw[
            (raw["fold"] == "four_source_macro_avg") & (raw["method"] == method)
        ]
        neur = raw[(raw["fold"] == "neuralchemy") & (raw["method"] == method)]
        pi_fpr = float(macro.iloc[0]["FPR_%"]) if not macro.empty else None
        neur_fpr = float(neur.iloc[0]["FPR_%"]) if not neur.empty else None
        edge_fpr = _edge_macro_pct(method)
        adapt_b = _adaptive_macro_pct(method)
        lines.append(
            f"| {lab} | {_fmt(pi_fpr)} | {_fmt(neur_fpr)} | "
            f"{_fmt(edge_fpr)} | {_fmt(adapt_b)} |"
        )
    lines.append("")
    lines.append(
        "Edge / adaptive use the **same** per-fold matched thresholds as Table A. "
        "ProtectAI / Prompt Guard adaptive Bypass is **gate-only** (True ASR not "
        "re-judged)."
    )
    lines.append("")
    lines.append(
        f"Sources: `{metrics_path.as_posix()}`, `{MATCHED_FPR.as_posix()}`, "
        f"`{MANIFEST.as_posix()}`, `{edge_path.as_posix()}`, "
        f"`{adaptive_path.as_posix()}`."
    )

    md_path = out / "PAPER_TABLES.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Also refresh train_benign export from manifest
    tb_rows = []
    for fold in LOO_FOLDS:
        info = (manifest.get("folds") or {}).get(fold, {})
        tb = info.get("train_benign_by_source") or {}
        vb = info.get("val_benign_by_source") or {}
        t_alp = int(tb.get("alpaca_benign", 0) or 0)
        v_alp = int(vb.get("alpaca_benign", 0) or 0)
        t_pi = int(tb.get("prompt_injection_benign", 0) or 0)
        v_pi = int(vb.get("prompt_injection_benign", 0) or 0)
        tb_rows.append(
            {
                "fold": fold,
                "train_alpaca_benign": t_alp,
                "val_alpaca_benign": v_alp,
                "fit_benign_total": t_alp + v_alp,
                "train_prompt_injection_benign": t_pi,
                "val_prompt_injection_benign": v_pi,
            }
        )
    pd.DataFrame(tb_rows).to_csv(out / "train_benign_by_fold.csv", index=False)
    pd.DataFrame(tb_rows).to_csv(
        SOURCE_HOLDOUT_DIR / "train_benign_by_fold.csv", index=False
    )

    print(f"[OK] Wrote {md_path}")
    print(f"[OK] Wrote {out / 'paper_headline.csv'}")
    print(f"[OK] Wrote {out / 'metrics_with_wilson.csv'}")
    return md_path


if __name__ == "__main__":
    build_paper_tables()
