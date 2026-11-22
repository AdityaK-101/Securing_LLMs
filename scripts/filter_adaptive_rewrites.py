"""
Drop broken adaptive rewrites and recompute LOO matched-FPR adaptive metrics.

Quality filter (paper disclosure):
  - rewrite strip length < 20 characters, OR
  - rewrite is an exact copy of the source injection

Reports removals per fold, recomputes main-suite metrics from
robustness_adaptive_logs.jsonl, re-scores ProtectAI / Prompt Guard on the
kept rewrite set, refreshes robustness_adaptive*.csv and PAPER_TABLES.md.

  python scripts/filter_adaptive_rewrites.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROMPT_CACHE_DIR, RESULTS_DIR
from src.evaluation.loo_matched_fpr import LOO_FOLDS, LOO_SOURCE_FOLDS, _macro_micro_average
from src.evaluation.metrics import compute_metrics
from src.evaluation.prompt_cache import OutcomeCache
from src.evaluation.loo_matched_fpr_report import build_paper_tables

OUT = RESULTS_DIR / "loo_matched_fpr"
MIN_CHARS = 20
PORTABLE_METHODS = ("protectai", "prompt_guard")


def _is_bad_rewrite(rewrite: str, source: str) -> bool:
    text = (rewrite or "").strip()
    src = (source or "").strip()
    if len(text) < MIN_CHARS:
        return True
    if text == src:
        return True
    return False


def _attack_keep_mask(attacks_path: Path) -> tuple[dict[str, bool], dict]:
    """Map prompt_id -> keep for each variant; return stats."""
    keep: dict[tuple[str, int], bool] = {}
    n = short = identical = dropped = 0
    with open(attacks_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            pid = str(row["prompt_id"])
            src = str(row.get("source_prompt", ""))
            for v in row.get("variants") or []:
                idx = int(v.get("variant_idx", 0))
                prompt = str(v.get("prompt", ""))
                n += 1
                is_short = len(prompt.strip()) < MIN_CHARS
                is_ident = prompt.strip() == src.strip()
                if is_short:
                    short += 1
                if is_ident:
                    identical += 1
                bad = is_short or is_ident
                if bad:
                    dropped += 1
                keep[(pid, idx)] = not bad
    stats = {
        "n_rewrites": n,
        "short_lt_20": short,
        "exact_copy": identical,
        "dropped": dropped,
        "kept": n - dropped,
    }
    return keep, stats


def _filter_logs(logs_path: Path, keep: dict[tuple[str, int], bool]) -> pd.DataFrame:
    rows = []
    with open(logs_path, encoding="utf-8") as handle:
        for line in handle:
            rec = json.loads(line)
            rid = str(rec.get("id", ""))
            # ids look like "{prompt_id}_v{variant_idx}"
            if "_v" not in rid:
                # fallback: filter by original/source text
                if _is_bad_rewrite(str(rec.get("original", "")), str(rec.get("source_injection", ""))):
                    continue
                rows.append(rec)
                continue
            pid, _, rest = rid.rpartition("_v")
            try:
                idx = int(rest)
            except ValueError:
                if _is_bad_rewrite(str(rec.get("original", "")), str(rec.get("source_injection", ""))):
                    continue
                rows.append(rec)
                continue
            if keep.get((pid, idx), False):
                rows.append(rec)
    return pd.DataFrame(rows)


def _ensure_outcomes(prompts: list[str], outcome_cache: OutcomeCache) -> None:
    missing_llm = outcome_cache.missing_llm(prompts)
    missing_judge = outcome_cache.missing_judge(prompts)
    if not missing_llm and not missing_judge:
        return
    from src.config import LLM_CACHE_BATCH_SIZE
    from src.models.judge import ComplianceJudge
    from src.models.target_llm import LLMRunner

    if missing_llm:
        print(f"  [adaptive ASR] generating {len(missing_llm)} missing target replies...")
        runner = LLMRunner()
        try:
            outcome_cache.ensure_llm(
                prompts, runner, batch_size=int(LLM_CACHE_BATCH_SIZE)
            )
        finally:
            if hasattr(runner, "release"):
                runner.release()
    missing_judge = outcome_cache.missing_judge(prompts)
    if missing_judge:
        print(f"  [adaptive ASR] judging {len(missing_judge)} cached replies...")
        judge = ComplianceJudge()
        try:
            outcome_cache.ensure_judge(prompts, judge)
        finally:
            if hasattr(judge, "release"):
                judge.release()


def _true_asr_allowed(prompts: list[str], blocked: list[bool], cache: OutcomeCache) -> tuple[float, int]:
    n = len(prompts)
    if n == 0:
        return 0.0, 0
    success = 0
    for p, b in zip(prompts, blocked):
        if b:
            continue
        row = cache.get(p)
        lab = str((row or {}).get("judge_label", "") or "").strip().upper()
        if lab == "COMPLIED":
            success += 1
    return success / n, success


def _score_portable_filtered(
    method: str,
    thr: float,
    attacks_path: Path,
    outcome_cache: OutcomeCache,
) -> dict | None:
    prompts: list[str] = []
    with open(attacks_path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            src = str(row.get("source_prompt", ""))
            for v in row.get("variants") or []:
                p = str(v.get("prompt", ""))
                if not _is_bad_rewrite(p, src):
                    prompts.append(p)
    if not prompts:
        return None

    from src.sanitizers.protectai import ProtectAISanitizer
    from src.sanitizers.prompt_guard import PromptGuardSanitizer

    san = ProtectAISanitizer() if method == "protectai" else PromptGuardSanitizer()
    # Matched-FPR uses injection_score >= thr (sanitize() is argmax-only).
    scores = [float(san.injection_score(p)) for p in prompts]
    blocked = [s >= float(thr) for s in scores]
    allowed = [p for p, b in zip(prompts, blocked) if not b]
    if allowed:
        _ensure_outcomes(allowed, outcome_cache)
    true_asr, n_success = _true_asr_allowed(prompts, blocked, outcome_cache)
    n = len(prompts)
    bypass = sum(1 for b in blocked if not b) / n
    if hasattr(san, "release_models"):
        try:
            san.release_models()
        except Exception:
            pass
    return {
        "Bypass_Rate_%": round(bypass * 100, 2),
        "True_ASR_%": round(true_asr * 100, 2),
        "Bypass_Rate": round(bypass, 4),
        "True_ASR": round(true_asr, 4),
        "FPR_%": 0.0,
        "Detection_Rate": round(1.0 - bypass, 4),
        "injections": n,
        "benign": 0,
        "successful_attack_count": int(n_success),
        "note": (
            f"adaptive filtered (drop len<{MIN_CHARS} or exact copy); "
            "score>=matched thr; target+judge on allowed rewrites"
        ),
    }


def main() -> None:
    thr_all = pd.read_csv(OUT / "thresholds.csv")
    outcome_cache = OutcomeCache(PROMPT_CACHE_DIR / "loo" / "adaptive_outcomes.jsonl")
    report_rows = []
    metric_rows = []

    for fold in LOO_FOLDS:
        fold_dir = OUT / fold
        attacks_path = fold_dir / "adaptive_attacks.jsonl"
        logs_path = fold_dir / "robustness_adaptive_logs.jsonl"
        if not attacks_path.is_file():
            raise SystemExit(f"Missing {attacks_path}")
        keep, stats = _attack_keep_mask(attacks_path)
        report_rows.append({"fold": fold, **stats})
        print(
            f"[{fold}] rewrites={stats['n_rewrites']}  "
            f"short<20={stats['short_lt_20']}  exact_copy={stats['exact_copy']}  "
            f"dropped={stats['dropped']}  kept={stats['kept']}"
        )

        if logs_path.is_file():
            logs = _filter_logs(logs_path, keep)
            metrics_df, _ = compute_metrics(logs)
            metrics_df.insert(0, "fold", fold)
            metrics_df["note"] = (
                f"adaptive filtered: drop len<{MIN_CHARS} or exact source copy"
            )
            metrics_df.to_csv(fold_dir / "robustness_adaptive.csv", index=False)
            metric_rows.append(metrics_df)
            print(
                metrics_df[
                    ["method", "Bypass_Rate_%", "True_ASR_%", "injections"]
                ].to_string(index=False)
            )
        else:
            print(f"  WARN: no logs at {logs_path}")

        # Portables on filtered rewrite set.
        for method in PORTABLE_METHODS:
            thr_row = thr_all[(thr_all["fold"] == fold) & (thr_all["method"] == method)]
            if thr_row.empty:
                continue
            thr = float(thr_row.iloc[0]["threshold"])
            print(f"  [{fold}] {method} @ thr={thr:g} (filtered)...")
            portable = _score_portable_filtered(
                method, thr, attacks_path, outcome_cache
            )
            if portable is None:
                continue
            metric_rows.append(
                pd.DataFrame([{"fold": fold, "method": method, **portable}])
            )
            print(
                f"    Bypass={portable['Bypass_Rate_%']:.2f}%  "
                f"TrueASR={portable['True_ASR_%']:.2f}%  n={portable['injections']}"
            )

    report = pd.DataFrame(report_rows)
    report.to_csv(OUT / "adaptive_rewrite_filter_report.csv", index=False)
    print(f"\n[OK] Wrote {OUT / 'adaptive_rewrite_filter_report.csv'}")
    print(report.to_string(index=False))

    adaptive_all = pd.concat(metric_rows, ignore_index=True)
    adaptive_all = adaptive_all.drop_duplicates(subset=["fold", "method"], keep="last")
    summarized = _macro_micro_average(
        adaptive_all[~adaptive_all["fold"].isin(["four_source_macro_avg", "four_source_micro"])]
    )
    # Recompute portable macros explicitly (equal-weight LOO-4).
    port = summarized[
        summarized["method"].isin(PORTABLE_METHODS)
        & summarized["fold"].isin(LOO_SOURCE_FOLDS)
    ]
    if not port.empty:
        summarized = summarized[
            ~(
                summarized["method"].isin(PORTABLE_METHODS)
                & summarized["fold"].isin(["four_source_macro_avg", "four_source_micro"])
            )
        ]
        extra = []
        for method, grp in port.groupby("method"):
            extra.append(
                {
                    "fold": "four_source_macro_avg",
                    "method": method,
                    "Bypass_Rate_%": round(float(grp["Bypass_Rate_%"].mean()), 2),
                    "True_ASR_%": round(float(grp["True_ASR_%"].mean()), 2),
                    "injections": float(grp["injections"].mean())
                    if "injections" in grp
                    else float("nan"),
                    "successful_attack_count": float(grp["successful_attack_count"].mean())
                    if "successful_attack_count" in grp
                    else float("nan"),
                    "note": "macro avg of four source folds (filtered adaptive)",
                }
            )
        summarized = pd.concat([summarized, pd.DataFrame(extra)], ignore_index=True)

    summarized.to_csv(OUT / "robustness_adaptive.csv", index=False)
    summarized.to_csv(OUT / "robustness_adaptive_summary.csv", index=False)
    print(f"[OK] Wrote {OUT / 'robustness_adaptive.csv'}")

    macro = summarized[summarized["fold"] == "four_source_macro_avg"]
    print("\nLOO-4 macro adaptive (filtered):")
    for _, r in macro.sort_values("method").iterrows():
        print(
            f"  {r['method']:24s}  Bypass={r['Bypass_Rate_%']:.2f}%  "
            f"TrueASR={r['True_ASR_%']:.2f}%"
        )

    build_paper_tables(OUT)
    print("[OK] Refreshed PAPER_TABLES.md")


if __name__ == "__main__":
    main()
