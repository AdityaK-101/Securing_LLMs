"""Score source-holdout folds with the gate only.

Methods per fold:
  baseline, regex, keyword, Method A, Method B,
  similarity_only (fold-trained), deberta_finetuned (fold-trained)

ProtectAI and Prompt Guard 2 are published gates: each unique prompt is scored
once and the decision is reused across folds.

Reports Bypass and FPR with Wilson intervals. True ASR is not computed here.

Usage:
  python scripts/eval_loo_holdout.py
  python scripts/eval_loo_holdout.py --skip-portable
  python scripts/eval_loo_holdout.py --skip-deberta
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.sanitizers.base import BaselineSanitizer
from src.sanitizers.context_aware import ContextAwareSanitizer
from src.sanitizers.context_aware_learned import ContextAwareLearnedSanitizer
from src.sanitizers.deberta_finetuned import DebertaFinetunedSanitizer
from src.sanitizers.keyword import KeywordHeuristicSanitizer
from src.sanitizers.prompt_guard import PromptGuardSanitizer
from src.sanitizers.protectai import ProtectAISanitizer
from src.sanitizers.regex import RegexSanitizer
from src.sanitizers.similarity_only import SimilarityOnlySanitizer
from src.splits.source_holdout import ATTACK_SOURCES, EXTERNAL_SOURCE

DEFAULT_LOO = ROOT / "data" / "splits" / "source_holdout"
DEFAULT_OUT = ROOT / "results" / "source_holdout"
FOLDS = (*ATTACK_SOURCES, EXTERNAL_SOURCE)
SOURCE_FOLDS = ATTACK_SOURCES
# Set in main() before fold scoring.
LOO = DEFAULT_LOO
OUT = DEFAULT_OUT


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def _install_perplexity_cache() -> None:
    cache: dict[str, float | None] = {}

    def wrap(cls):
        original = cls._compute_perplexity

        def cached(self, text: str):
            if text not in cache:
                cache[text] = original(self, text)
            return cache[text]

        cls._compute_perplexity = cached

    wrap(ContextAwareSanitizer)
    wrap(ContextAwareLearnedSanitizer)


def _assert_no_leak(fold: str, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_keys = {_norm(p) for p in train["prompt"]} | {_norm(p) for p in val["prompt"]}
    leaked = sum(_norm(p) in train_keys for p in test["prompt"])
    if leaked:
        raise SystemExit(f"{fold}: {leaked} test prompts also appear in train or val")
    if fold == "prompt_injection":
        banned = {"prompt_injection", "prompt_injection_benign"}
        bad = train[train["source"].isin(banned)]
        bad_v = val[val["source"].isin(banned)]
        if len(bad) or len(bad_v):
            raise SystemExit(
                f"{fold}: train/val still contain held-out source rows "
                f"(train={len(bad)}, val={len(bad_v)})"
            )
    if fold in ATTACK_SOURCES:
        inj = test[test["label"] == "injection"]
        bad_src = inj[~inj["source"].astype(str).str.startswith(fold)]
        if len(bad_src):
            raise SystemExit(f"{fold}: test injections from another source: {len(bad_src)}")


def _score_portable_once(prompts: list[str]) -> dict[str, dict[str, bool]]:
    """Score each unique prompt once with ProtectAI and Prompt Guard."""
    unique = sorted({str(p) for p in prompts})
    print(f"\n=== portable gates on {len(unique)} unique prompts ===", flush=True)
    decisions: dict[str, dict[str, bool]] = {"protectai": {}, "prompt_guard": {}}

    protectai = ProtectAISanitizer()
    for i, text in enumerate(unique, start=1):
        if i == 1 or i % 100 == 0 or i == len(unique):
            print(f"  [protectai] {i}/{len(unique)}", flush=True)
        _, blocked = protectai.sanitize(text)
        decisions["protectai"][text] = bool(blocked)
    protectai.release_models()

    try:
        guard = PromptGuardSanitizer()
    except Exception as exc:
        print(f"[prompt_guard] skipped: {exc}", flush=True)
        return decisions

    for i, text in enumerate(unique, start=1):
        if i == 1 or i % 100 == 0 or i == len(unique):
            print(f"  [prompt_guard] {i}/{len(unique)}", flush=True)
        _, blocked = guard.sanitize(text)
        decisions["prompt_guard"][text] = bool(blocked)
    if hasattr(guard, "release_models"):
        guard.release_models()
    return decisions


def _score_fold(
    fold: str,
    portable: dict[str, dict[str, bool]] | None,
    skip_deberta: bool,
) -> pd.DataFrame:
    folder = LOO / fold
    train = pd.read_csv(folder / "train.csv")
    val = pd.read_csv(folder / "val.csv")
    test = pd.read_csv(folder / "test.csv")
    _assert_no_leak(fold, train, val, test)

    print(f"\n=== {fold}  train={len(train)} val={len(val)} test={len(test)} ===", flush=True)
    method_a = ContextAwareSanitizer(train_csv=str(folder / "train.csv"))
    method_b = ContextAwareLearnedSanitizer(
        train_csv=str(folder / "train.csv"),
        val_csv=str(folder / "val.csv"),
        share_from=method_a,
    )
    sim = SimilarityOnlySanitizer(
        train_csv=str(folder / "train.csv"),
        val_csv=str(folder / "val.csv"),
        share_embed_from=method_a,
    )
    sanitizers = [
        BaselineSanitizer(),
        RegexSanitizer(),
        KeywordHeuristicSanitizer(),
        method_a,
        method_b,
        sim,
    ]
    deberta = None
    if not skip_deberta:
        deberta = DebertaFinetunedSanitizer(
            train_csv=str(folder / "train.csv"),
            val_csv=str(folder / "val.csv"),
        )
        sanitizers.append(deberta)

    records = []
    total = len(test)
    for index, row in enumerate(test.itertuples(index=False), start=1):
        if index == 1 or index % 50 == 0 or index == total:
            print(f"  [{fold}] {index}/{total}", flush=True)
        prompt = str(row.prompt)
        for sanitizer in sanitizers:
            _, blocked = sanitizer.sanitize(prompt)
            records.append(
                {
                    "fold": fold,
                    "id": row.id,
                    "label": row.label,
                    "source": row.source,
                    "method": sanitizer.name,
                    "blocked": bool(blocked),
                }
            )
        if portable:
            for method, table in portable.items():
                if prompt not in table:
                    continue
                records.append(
                    {
                        "fold": fold,
                        "id": row.id,
                        "label": row.label,
                        "source": row.source,
                        "method": method,
                        "blocked": bool(table[prompt]),
                    }
                )

    method_a.release_models()
    if hasattr(method_b, "release_models"):
        method_b.release_models()
    sim.release_models()
    if deberta is not None:
        deberta.release_models()
    return pd.DataFrame(records)


def _metrics(logs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fold, method), group in logs.groupby(["fold", "method"], sort=False):
        injections = group[group["label"] == "injection"]
        benign = group[group["label"] == "benign"]
        bypass_n = int((~injections["blocked"]).sum())
        fpr_n = int(benign["blocked"].sum())
        n_inj = int(len(injections))
        n_ben = int(len(benign))
        bypass_lo, bypass_hi = _wilson(bypass_n, n_inj)
        fpr_lo, fpr_hi = _wilson(fpr_n, n_ben)
        rows.append(
            {
                "fold": fold,
                "method": method,
                "n_injection": n_inj,
                "n_benign": n_ben,
                "bypass_n": bypass_n,
                "Bypass_%": round(100 * bypass_n / n_inj, 2) if n_inj else 0.0,
                "Bypass_Wilson95": f"[{100 * bypass_lo:.1f}, {100 * bypass_hi:.1f}]",
                "fpr_n": fpr_n,
                "FPR_%": round(100 * fpr_n / n_ben, 2) if n_ben else 0.0,
                "FPR_Wilson95": f"[{100 * fpr_lo:.1f}, {100 * fpr_hi:.1f}]",
            }
        )
    return pd.DataFrame(rows)


def _average(metrics: pd.DataFrame) -> pd.DataFrame:
    source = metrics[metrics["fold"].isin(SOURCE_FOLDS)]
    rows = []
    for method, group in source.groupby("method", sort=False):
        bypass_n = int(group["bypass_n"].sum())
        fpr_n = int(group["fpr_n"].sum())
        n_inj = int(group["n_injection"].sum())
        n_ben = int(group["n_benign"].sum())
        # Macro: each fold equal weight. Wilson on the mean of fold rates.
        bypass_rates = (group["bypass_n"] / group["n_injection"]).to_numpy()
        fpr_rates = (group["fpr_n"] / group["n_benign"]).to_numpy()
        macro_bypass = float(np_mean(bypass_rates) * 100)
        macro_fpr = float(np_mean(fpr_rates) * 100)
        # Approximate interval on the four fold rates via Wilson on rounded counts.
        # Prefer reporting the macro mean; interval from pooled micro remains secondary.
        rows.append(
            {
                "fold": "four_source_macro_avg",
                "method": method,
                "n_injection": n_inj,
                "n_benign": n_ben,
                "bypass_n": bypass_n,
                "Bypass_%": round(macro_bypass, 2),
                "Bypass_Wilson95": "",
                "fpr_n": fpr_n,
                "FPR_%": round(macro_fpr, 2),
                "FPR_Wilson95": "",
            }
        )
        micro_lo, micro_hi = _wilson(bypass_n, n_inj)
        fpr_lo, fpr_hi = _wilson(fpr_n, n_ben)
        rows.append(
            {
                "fold": "four_source_micro",
                "method": method,
                "n_injection": n_inj,
                "n_benign": n_ben,
                "bypass_n": bypass_n,
                "Bypass_%": round(100 * bypass_n / n_inj, 2) if n_inj else 0.0,
                "Bypass_Wilson95": f"[{100 * micro_lo:.1f}, {100 * micro_hi:.1f}]",
                "fpr_n": fpr_n,
                "FPR_%": round(100 * fpr_n / n_ben, 2) if n_ben else 0.0,
                "FPR_Wilson95": f"[{100 * fpr_lo:.1f}, {100 * fpr_hi:.1f}]",
            }
        )
    return pd.DataFrame(rows)


def np_mean(arr) -> float:
    import numpy as np

    return float(np.mean(arr)) if len(arr) else 0.0


def _summary(metrics: pd.DataFrame) -> str:
    lines = [
        "Source-holdout gate scores.",
        "Bypass = injection prompts the gate did not block.",
        "FPR = benign prompts the gate blocked.",
        "True ASR is not in this file: the target LLM and judge were not run.",
        "similarity_only and deberta_finetuned are trained inside each fold.",
        "protectai and prompt_guard are published gates scored once per prompt.",
        "four_source_macro_avg gives each held-out source equal weight.",
        "neuralchemy is the fifth dataset and is not included in the average.",
        "",
    ]
    show = metrics.copy()
    for column in ("Bypass_%", "FPR_%"):
        show[column] = show[column].map(lambda value: f"{value:.2f}")
    lines.append(show.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    global LOO, OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-portable", action="store_true", help="Skip ProtectAI / Prompt Guard")
    parser.add_argument("--skip-deberta", action="store_true", help="Skip fold-trained DeBERTa")
    parser.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Override split root (default: data/splits/source_holdout)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Override results dir (default: results/source_holdout)",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        default=None,
        choices=list(FOLDS),
        help="Subset of folds (default: all five)",
    )
    args = parser.parse_args()
    folds = args.folds or list(FOLDS)
    LOO = args.splits or DEFAULT_LOO
    OUT = args.out or DEFAULT_OUT
    if not LOO.is_dir():
        raise SystemExit(f"split root missing: {LOO}")
    print(f"splits={LOO}\nout={OUT}", flush=True)

    _install_perplexity_cache()
    OUT.mkdir(parents=True, exist_ok=True)

    portable = None
    if not args.skip_portable:
        all_prompts = []
        for fold in folds:
            all_prompts.extend(pd.read_csv(LOO / fold / "test.csv")["prompt"].astype(str).tolist())
        portable = _score_portable_once(all_prompts)
        # Persist for reuse / inspection.
        rows = []
        for method, table in portable.items():
            for prompt, blocked in table.items():
                rows.append({"method": method, "prompt": prompt, "blocked": blocked})
        pd.DataFrame(rows).to_csv(OUT / "portable_prompt_decisions.csv", index=False)

    parts = []
    for fold in folds:
        logs = _score_fold(fold, portable=portable, skip_deberta=args.skip_deberta)
        logs.to_csv(OUT / f"logs_{fold}.csv", index=False)
        parts.append(logs)
    logs = pd.concat(parts, ignore_index=True)
    per_fold = _metrics(logs)
    metrics = pd.concat([per_fold, _average(per_fold)], ignore_index=True)
    metrics.to_csv(OUT / "metrics.csv", index=False)
    text = _summary(metrics)
    (OUT / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print("\n" + text, flush=True)


if __name__ == "__main__":
    main()
