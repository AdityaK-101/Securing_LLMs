"""
Score ProtectAI + Prompt Guard 2 on LOO folds at matched val FPR ≈ 5%.

Weights stay published; only the decision threshold is retuned per fold
(same protocol as other score-based methods in ``--loo-matched-fpr``).
True ASR uses ``results/prompt_cache/loo/outcomes.jsonl``.

Writes / merges into ``results/loo_matched_fpr/`` then rebuilds paper tables.

Usage:
  python scripts/eval_loo_matched_fpr_portables.py
  python scripts/eval_loo_matched_fpr_portables.py --skip-prompt-guard
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROMPT_CACHE_DIR, RESULTS_DIR, SOURCE_HOLDOUT_DIR
from src.evaluation.loo_matched_fpr import (
    LOO_FOLDS,
    LOO_SOURCE_FOLDS,
    _labels,
    _macro_micro_average,
    _metrics_from_blocked,
    _prompts,
)
from src.evaluation.matched_fpr import DEFAULT_TARGET_FPR, tune_threshold_equal_fpr
from src.evaluation.prompt_cache import OutcomeCache
from src.evaluation.robustness import EDGE_BENIGN_PROMPTS
from src.sanitizers.prompt_guard import PromptGuardSanitizer
from src.sanitizers.protectai import ProtectAISanitizer

OUT = RESULTS_DIR / "loo_matched_fpr"
SCORE_CACHE = OUT / "portable_scores.jsonl"


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def _load_score_cache(path: Path) -> dict[tuple[str, str], float]:
    cache: dict[tuple[str, str], float] = {}
    if not path.is_file():
        return cache
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            cache[(str(row["method"]), str(row["key"]))] = float(row["score"])
    return cache


def _append_score_cache(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


@torch.inference_mode()
def _batch_injection_scores(
    san,
    prompts: list[str],
    *,
    label_id: int,
    batch_size: int = 16,
) -> np.ndarray:
    """Softmax P(injection/malicious) in batches."""
    out = np.zeros(len(prompts), dtype=float)
    for start in tqdm(
        range(0, len(prompts), batch_size),
        desc=f"{san.name} scores",
        leave=False,
    ):
        chunk = prompts[start : start + batch_size]
        texts = ["" if p is None else str(p) for p in chunk]
        inputs = san.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        inputs = {k: v.to(san.device) for k, v in inputs.items()}
        logits = san.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        out[start : start + len(chunk)] = probs[:, label_id].detach().cpu().numpy()
    return out


def _scores_for_prompts(
    san,
    method: str,
    prompts: list[str],
    cache: dict[tuple[str, str], float],
    *,
    label_id: int,
    batch_size: int,
) -> np.ndarray:
    keys = [_norm(p) for p in prompts]
    scores = np.full(len(prompts), np.nan, dtype=float)
    missing_idx: list[int] = []
    for i, key in enumerate(keys):
        hit = cache.get((method, key))
        if hit is None:
            missing_idx.append(i)
        else:
            scores[i] = hit
    if missing_idx:
        miss_prompts = [prompts[i] for i in missing_idx]
        fresh = _batch_injection_scores(
            san, miss_prompts, label_id=label_id, batch_size=batch_size
        )
        new_rows = []
        for j, i in enumerate(missing_idx):
            scores[i] = float(fresh[j])
            cache[(method, keys[i])] = float(fresh[j])
            new_rows.append(
                {"method": method, "key": keys[i], "score": float(fresh[j])}
            )
        _append_score_cache(SCORE_CACHE, new_rows)
    return scores


def _eval_method_on_folds(
    method: str,
    san,
    *,
    label_id: int,
    folds: list[str],
    target_fpr: float,
    outcome_cache: OutcomeCache,
    score_cache: dict[tuple[str, str], float],
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    metric_rows: list[dict] = []
    threshold_rows: list[dict] = []
    for fold in folds:
        split = SOURCE_HOLDOUT_DIR / fold
        df_val = pd.read_csv(split / "val.csv")
        df_test = pd.read_csv(split / "test.csv")
        y_val = _labels(df_val)
        y_test = _labels(df_test)
        val_prompts = _prompts(df_val)
        test_prompts = _prompts(df_test)

        print(f"[{fold}] {method}: scoring val ({len(val_prompts)}) + test ({len(test_prompts)})...")
        val_scores = _scores_for_prompts(
            san,
            method,
            val_prompts,
            score_cache,
            label_id=label_id,
            batch_size=batch_size,
        )
        test_scores = _scores_for_prompts(
            san,
            method,
            test_prompts,
            score_cache,
            label_id=label_id,
            batch_size=batch_size,
        )
        thr, val_stats = tune_threshold_equal_fpr(
            val_scores, y_val, target_fpr=target_fpr
        )
        test_blocked = test_scores >= float(thr)
        val_blocked = val_scores >= float(thr)
        test_m = _metrics_from_blocked(
            test_blocked, y_test, test_prompts, outcome_cache
        )
        val_m = _metrics_from_blocked(
            val_blocked, y_val, val_prompts, outcome_cache
        )
        metric_rows.append(
            {
                "fold": fold,
                "method": method,
                "mode": f"matched_fpr_{int(round(100 * target_fpr))}pct",
                "threshold": round(float(thr), 6),
                "semantic_hard_sim": np.nan,
                "val_FPR_%": val_m["FPR_%"],
                "val_Bypass_Rate_%": val_m["Bypass_Rate_%"],
                "feasible_on_val": bool(val_stats.get("feasible", False)),
                "note": "published weights; score threshold retuned to hit val FPR≈target",
                **test_m,
            }
        )
        threshold_rows.append(
            {
                "fold": fold,
                "method": method,
                "threshold": round(float(thr), 6),
                "semantic_hard_sim": np.nan,
                "val_FPR": val_stats.get("fpr"),
                "val_FPR_gap": val_stats.get("fpr_gap"),
                "val_bypass": val_stats.get("bypass"),
                "feasible_on_val": bool(val_stats.get("feasible", False)),
            }
        )
        print(
            f"  thr={thr:.6f}  test Bypass={test_m['Bypass_Rate_%']:.2f}  "
            f"TrueASR={test_m['True_ASR_%']:.2f}  FPR={test_m['FPR_%']:.2f}"
        )
    return metric_rows, threshold_rows


def _merge_metrics(existing: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    new_df = pd.DataFrame(new_rows)
    methods = sorted(new_df["method"].unique())
    keep = existing[~existing["method"].isin(methods)].copy()
    # Drop aggregate rows; rebuild after merge.
    keep = keep[~keep["fold"].astype(str).str.startswith("four_source_")].copy()
    merged = pd.concat([keep, new_df], ignore_index=True)
    return _macro_micro_average(merged)


def _load_adaptive_prompts(fold: str) -> list[str]:
    """Flatten adaptive attack variants for one fold (gate-only Bypass)."""
    paths = [
        OUT / fold / "adaptive_attacks.jsonl",
        RESULTS_DIR / "loo" / fold / "adaptive_attacks.jsonl",
    ]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        return []
    prompts: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for var in row.get("variants") or []:
                text = str(var.get("prompt") or "").strip()
                if text:
                    prompts.append(text)
    return prompts


def _edge_fpr_at_threshold(
    san,
    method: str,
    thr: float,
    *,
    label_id: int,
    score_cache: dict[tuple[str, str], float],
    batch_size: int,
) -> dict:
    prompts = list(EDGE_BENIGN_PROMPTS)
    scores = _scores_for_prompts(
        san,
        method,
        prompts,
        score_cache,
        label_id=label_id,
        batch_size=batch_size,
    )
    blocked = scores >= float(thr)
    n = len(prompts)
    n_block = int(blocked.sum())
    return {
        "edge_FPR": float(n_block / n) if n else 0.0,
        "edge_FPR_%": round(100.0 * n_block / n, 2) if n else 0.0,
        "blocked_count": n_block,
        "total": n,
    }


def _adaptive_bypass_at_threshold(
    san,
    method: str,
    thr: float,
    fold: str,
    *,
    label_id: int,
    score_cache: dict[tuple[str, str], float],
    batch_size: int,
) -> dict | None:
    prompts = _load_adaptive_prompts(fold)
    if not prompts:
        return None
    scores = _scores_for_prompts(
        san,
        method,
        prompts,
        score_cache,
        label_id=label_id,
        batch_size=batch_size,
    )
    blocked = scores >= float(thr)
    bypass = float((~blocked).mean())
    n = len(prompts)
    return {
        "Bypass_Rate": bypass,
        "Bypass_Rate_%": round(100.0 * bypass, 2),
        "FPR_%": 0.0,  # attack-only probe
        "True_ASR_%": None,  # gate-only
        "injections": n,
        "benign": 0,
        "successful_attack_count": None,
        "note": "gate-only adaptive Bypass at matched thr; True ASR not scored",
    }


def _score_portable_robustness(
    method: str,
    san,
    *,
    label_id: int,
    folds: list[str],
    thresholds: dict[str, float],
    score_cache: dict[tuple[str, str], float],
    batch_size: int,
) -> tuple[list[dict], list[dict]]:
    edge_rows: list[dict] = []
    adaptive_rows: list[dict] = []
    for fold in folds:
        thr = thresholds.get(fold)
        if thr is None or (isinstance(thr, float) and np.isnan(thr)):
            print(f"[{fold}] {method}: skip robustness (no matched threshold)")
            continue
        print(f"[{fold}] {method}: edge FPR @ thr={float(thr):.6g}...")
        edge = _edge_fpr_at_threshold(
            san,
            method,
            float(thr),
            label_id=label_id,
            score_cache=score_cache,
            batch_size=batch_size,
        )
        edge_rows.append({"fold": fold, "method": method, **edge})
        print(f"  edge FPR={edge['edge_FPR_%']:.2f}% ({edge['blocked_count']}/{edge['total']})")

        print(f"[{fold}] {method}: adaptive Bypass (gate-only)...")
        adaptive = _adaptive_bypass_at_threshold(
            san,
            method,
            float(thr),
            fold,
            label_id=label_id,
            score_cache=score_cache,
            batch_size=batch_size,
        )
        if adaptive is None:
            print(f"  no adaptive_attacks.jsonl for {fold}")
            continue
        adaptive_rows.append({"fold": fold, "method": method, **adaptive})
        print(
            f"  adaptive Bypass={adaptive['Bypass_Rate_%']:.2f}% "
            f"(n={adaptive['injections']})"
        )
    return edge_rows, adaptive_rows


def _merge_robustness_csv(
    path: Path,
    new_rows: list[dict],
    *,
    methods: list[str],
) -> None:
    if not new_rows:
        return
    new_df = pd.DataFrame(new_rows)
    if path.is_file():
        prev = pd.read_csv(path)
        prev.columns = [str(c).strip() for c in prev.columns]
        if "method" in prev.columns:
            prev = prev[~prev["method"].isin(methods)]
        merged = pd.concat([prev, new_df], ignore_index=True)
    else:
        merged = new_df
    # Drop old aggregates for these methods, then re-add four-source macro.
    if "fold" in merged.columns:
        merged = merged[
            ~merged["fold"].astype(str).str.startswith("four_source_")
        ].copy()
    source = merged[merged["fold"].isin(LOO_SOURCE_FOLDS)]
    if not source.empty and "method" in source.columns:
        macro_rows = []
        for method, grp in source.groupby("method"):
            row = {"fold": "four_source_macro_avg", "method": method}
            for col in grp.columns:
                if col in ("fold", "method", "note"):
                    continue
                if pd.api.types.is_numeric_dtype(grp[col]):
                    row[col] = float(grp[col].mean())
            macro_rows.append(row)
        merged = pd.concat([merged, pd.DataFrame(macro_rows)], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    print(f"[OK] Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-fpr", type=float, default=DEFAULT_TARGET_FPR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--skip-protectai", action="store_true")
    parser.add_argument("--skip-prompt-guard", action="store_true")
    parser.add_argument(
        "--robustness-only",
        action="store_true",
        help="Skip main val/test retune; only edge FPR + adaptive Bypass at "
        "thresholds already in metrics.csv / thresholds.csv.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip edge / adaptive gate-only scoring.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=list(LOO_FOLDS),
        help="Folds to score (default: all five).",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    metrics_path = OUT / "metrics.csv"
    if not metrics_path.is_file():
        raise SystemExit(
            f"Missing {metrics_path}. Run: python -m src.run_experiments --loo-matched-fpr"
        )

    outcome_cache = OutcomeCache(PROMPT_CACHE_DIR / "loo" / "outcomes.jsonl")
    score_cache = _load_score_cache(SCORE_CACHE)
    print(f"[portables] score cache entries: {len(score_cache)}")
    print(f"[portables] outcome cache rows: {len(outcome_cache._rows)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[portables] device={device}")

    all_metric_rows: list[dict] = []
    all_thr_rows: list[dict] = []
    all_edge_rows: list[dict] = []
    all_adaptive_rows: list[dict] = []
    methods_run: list[str] = []

    def _thresholds_for(method: str) -> dict[str, float]:
        # Prefer freshly computed thr rows; else metrics.csv.
        thr_map: dict[str, float] = {}
        for r in all_thr_rows:
            if r["method"] == method:
                thr_map[str(r["fold"])] = float(r["threshold"])
        if thr_map:
            return thr_map
        m = pd.read_csv(metrics_path)
        m.columns = [str(c).strip() for c in m.columns]
        sub = m[
            (m["method"] == method)
            & (~m["fold"].astype(str).str.startswith("four_source_"))
        ]
        for _, r in sub.iterrows():
            if pd.notna(r.get("threshold")):
                thr_map[str(r["fold"])] = float(r["threshold"])
        return thr_map

    if not args.skip_protectai:
        san = ProtectAISanitizer(device=device)
        label_id = san._injection_label_id()
        methods_run.append("protectai")
        if not args.robustness_only:
            m_rows, t_rows = _eval_method_on_folds(
                "protectai",
                san,
                label_id=label_id,
                folds=list(args.folds),
                target_fpr=float(args.target_fpr),
                outcome_cache=outcome_cache,
                score_cache=score_cache,
                batch_size=int(args.batch_size),
            )
            all_metric_rows.extend(m_rows)
            all_thr_rows.extend(t_rows)
        if not args.skip_robustness:
            e_rows, a_rows = _score_portable_robustness(
                "protectai",
                san,
                label_id=label_id,
                folds=list(args.folds),
                thresholds=_thresholds_for("protectai"),
                score_cache=score_cache,
                batch_size=int(args.batch_size),
            )
            all_edge_rows.extend(e_rows)
            all_adaptive_rows.extend(a_rows)
        san.release_models()

    if not args.skip_prompt_guard:
        san = PromptGuardSanitizer(device=device)
        label_id = san._malicious_label_id()
        methods_run.append("prompt_guard")
        if not args.robustness_only:
            m_rows, t_rows = _eval_method_on_folds(
                "prompt_guard",
                san,
                label_id=label_id,
                folds=list(args.folds),
                target_fpr=float(args.target_fpr),
                outcome_cache=outcome_cache,
                score_cache=score_cache,
                batch_size=int(args.batch_size),
            )
            all_metric_rows.extend(m_rows)
            all_thr_rows.extend(t_rows)
        if not args.skip_robustness:
            e_rows, a_rows = _score_portable_robustness(
                "prompt_guard",
                san,
                label_id=label_id,
                folds=list(args.folds),
                thresholds=_thresholds_for("prompt_guard"),
                score_cache=score_cache,
                batch_size=int(args.batch_size),
            )
            all_edge_rows.extend(e_rows)
            all_adaptive_rows.extend(a_rows)
        san.release_models()

    if not methods_run:
        raise SystemExit("Nothing to evaluate (both methods skipped).")

    if all_metric_rows:
        existing = pd.read_csv(metrics_path)
        merged = _merge_metrics(existing, all_metric_rows)
        merged.to_csv(OUT / "metrics.csv", index=False)
        merged.to_csv(OUT / "summary.csv", index=False)

        thr_path = OUT / "thresholds.csv"
        thr_existing = pd.read_csv(thr_path) if thr_path.is_file() else pd.DataFrame()
        thr_new = pd.DataFrame(all_thr_rows)
        if not thr_existing.empty:
            thr_existing = thr_existing[
                ~thr_existing["method"].isin(thr_new["method"].unique())
            ]
        thr_merged = pd.concat([thr_existing, thr_new], ignore_index=True)
        thr_merged.to_csv(thr_path, index=False)

        # Per-fold portable metrics
        for fold in args.folds:
            fold_dir = OUT / fold
            fold_dir.mkdir(parents=True, exist_ok=True)
            fold_rows = [r for r in all_metric_rows if r["fold"] == fold]
            if not fold_rows:
                continue
            fold_path = fold_dir / "metrics.csv"
            if fold_path.is_file():
                prev = pd.read_csv(fold_path)
                prev.columns = [str(c).strip() for c in prev.columns]
                if "method" in prev.columns:
                    prev = prev[~prev["method"].isin({r["method"] for r in fold_rows})]
                pd.concat([prev, pd.DataFrame(fold_rows)], ignore_index=True).to_csv(
                    fold_path, index=False
                )
            else:
                pd.DataFrame(fold_rows).to_csv(fold_path, index=False)

        print(f"\n[OK] Merged portable metrics into {OUT / 'metrics.csv'}")
        print(
            merged[merged["method"].isin(["protectai", "prompt_guard"])][
                ["fold", "method", "Bypass_Rate_%", "True_ASR_%", "FPR_%", "threshold"]
            ].to_string(index=False)
        )

    _merge_robustness_csv(
        OUT / "robustness_edge_benign.csv",
        all_edge_rows,
        methods=methods_run,
    )
    _merge_robustness_csv(
        OUT / "robustness_adaptive.csv",
        all_adaptive_rows,
        methods=methods_run,
    )
    # Also write summaries used by paper docs.
    for src, dest in (
        (OUT / "robustness_edge_benign.csv", OUT / "robustness_edge_benign_summary.csv"),
        (OUT / "robustness_adaptive.csv", OUT / "robustness_adaptive_summary.csv"),
    ):
        if not src.is_file():
            continue
        df = pd.read_csv(src)
        if "fold" not in df.columns:
            continue
        summary = df[df["fold"].isin(list(LOO_SOURCE_FOLDS) + ["four_source_macro_avg", "neuralchemy"])]
        summary.to_csv(dest, index=False)

    from src.evaluation.loo_matched_fpr_report import build_paper_tables

    build_paper_tables()


if __name__ == "__main__":
    main()
