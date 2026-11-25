"""LOO matched-FPR ablations.

Repeats the main mixed-split ablations under leave-one-source-out at the
**same matched thresholds** as Table 1:

  Method A  - semantic hard trigger removed
  Method A  soft-only (all hard triggers off)
  Method A  - all semantic (hard semantic off + soft semantic weight 0)
  Method B  - all three semantic features zeroed

Plus full Method A / B reference rows for Δ Bypass.

The ``Method A − all semantic`` row is the matched-LOO attribution check:
soft-only still carries semantic weight 0.20, so removing the hard trigger
alone does not test whether residual defense is semantic-backed.

Cheap path: gate decisions + OutcomeCache judge lookup (no new Qwen/judge).
Four source folds only (LOO-4); reports per-fold + macro average.

Resume-safe: after each fold, writes ``ablation_loo_matched.csv`` and skips
folds that already have a complete variant set. Adding a new variant marks
old folds incomplete so they recompute. Use ``--fresh`` to wipe the CSV.

Wired into ``scripts/run_paper.py --mode full`` with the other ablations.

  python scripts/run_loo_matched_ablations.py
  python scripts/run_loo_matched_ablations.py --fresh
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PROMPT_CACHE_DIR, RESULTS_DIR
from src.evaluation.loo_matched_fpr import (
    LOO_SOURCE_FOLDS,
    _blocked_method_a,
    _build_fold_sanitizers,
    _fold_dir,
    _labels,
    _metrics_from_blocked,
    _prompts,
    _score_method_a_signals,
)
from src.evaluation.prompt_cache import FeatureCache, OutcomeCache
from src.sanitizers.context_aware import ContextAwareSanitizer
from src.sanitizers.context_aware_learned import ContextAwareLearnedSanitizer
from src.utils.gpu import empty_cuda_cache

OUT = RESULTS_DIR / "loo_matched_fpr"
THRESHOLDS = OUT / "thresholds.csv"
PER_PATH = OUT / "ablation_loo_matched.csv"
MACRO_PATH = OUT / "ablation_loo_matched_macro.csv"
MACRO_FOLD = "four_source_macro_avg"

# Main ablations + full references for deltas.
METHOD_A_VARIANTS = [
    ("Method A full", {}),
    ("Method A − semantic trigger", {"disable_semantic_hard_trigger": True}),
    ("Method A soft-only", {"disable_hard_triggers": True}),
    # Semantic removed everywhere: no hard semantic veto, soft weight W3 = 0.
    # Other hard triggers (intent / role-play+keyword) stay enabled.
    (
        "Method A − all semantic",
        {
            "disable_semantic": True,
            "disable_semantic_hard_trigger": True,
        },
    ),
]
METHOD_B_VARIANTS = [
    ("Method B full", []),
    (
        "Method B − all semantic",
        ["semantic_max", "semantic_top3", "semantic_contrast"],
    ),
]
EXPECTED_VARIANTS = [name for name, _ in METHOD_A_VARIANTS] + [
    name for name, _ in METHOD_B_VARIANTS
]


def _thr_row(thr_df: pd.DataFrame, fold: str, method: str) -> pd.Series:
    hit = thr_df[(thr_df["fold"] == fold) & (thr_df["method"] == method)]
    if hit.empty:
        raise SystemExit(f"Missing threshold for {fold}/{method} in {THRESHOLDS}")
    return hit.iloc[0]


def _delta_vs_full(df: pd.DataFrame, full_name: str) -> pd.DataFrame:
    out = df.copy()
    base = out[out["variant"] == full_name]
    if base.empty:
        out["Delta_Bypass_pp"] = np.nan
        out["Delta_True_ASR_pp"] = np.nan
        return out
    b0 = float(base.iloc[0]["Bypass_Rate_%"])
    t0 = float(base.iloc[0]["True_ASR_%"])
    out["Delta_Bypass_pp"] = (out["Bypass_Rate_%"] - b0).round(2)
    out["Delta_True_ASR_pp"] = (out["True_ASR_%"] - t0).round(2)
    return out


def _fold_rows_only(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df["fold"].astype(str) != MACRO_FOLD].copy()


def _fold_complete(df: pd.DataFrame, fold: str) -> bool:
    """True if this fold has every expected variant exactly once."""
    if df.empty:
        return False
    sub = df[df["fold"].astype(str) == fold]
    if sub.empty:
        return False
    got = set(sub["variant"].astype(str))
    return got >= set(EXPECTED_VARIANTS)


def _load_checkpoint() -> pd.DataFrame:
    if not PER_PATH.is_file():
        return pd.DataFrame()
    df = pd.read_csv(PER_PATH)
    return _fold_rows_only(df)


def _save_checkpoint(fold_df: pd.DataFrame) -> None:
    """Atomic-ish write of per-fold rows (no macro)."""
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = PER_PATH.with_suffix(".csv.tmp")
    fold_df.to_csv(tmp, index=False)
    tmp.replace(PER_PATH)


def _build_macro(all_df: pd.DataFrame) -> pd.DataFrame:
    macro_rows = []
    for variant, grp in all_df.groupby("variant", sort=False):
        macro_rows.append(
            {
                "fold": MACRO_FOLD,
                "family": grp.iloc[0]["family"],
                "variant": variant,
                "threshold": float(grp["threshold"].mean()),
                "semantic_hard_sim": float(grp["semantic_hard_sim"].mean())
                if grp["semantic_hard_sim"].notna().any()
                else np.nan,
                "Bypass_Rate_%": round(float(grp["Bypass_Rate_%"].mean()), 2),
                "True_ASR_%": round(float(grp["True_ASR_%"].mean()), 2),
                "FPR_%": round(float(grp["FPR_%"].mean()), 2),
                "TP": float(grp["TP"].mean()),
                "FN": float(grp["FN"].mean()),
                "FP": float(grp["FP"].mean()),
                "TN": float(grp["TN"].mean()),
                "successful_attack_count": float(grp["successful_attack_count"].mean()),
                "injections": float(grp["injections"].mean()),
                "benign": float(grp["benign"].mean()),
                "true_asr_cache_missing": float(grp["true_asr_cache_missing"].sum())
                if "true_asr_cache_missing" in grp
                else 0.0,
                "note": "equal-weight macro of LOO source folds at matched thresholds",
            }
        )
    return pd.DataFrame(macro_rows)


def _with_deltas(all_df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for fam, full in (
        ("method_a", "Method A full"),
        ("method_b", "Method B full"),
    ):
        for frame in (all_df, macro):
            sub = frame[frame["family"] == fam].copy()
            if sub.empty:
                continue
            pieces.append(_delta_vs_full(sub, full))
    return pd.concat(pieces, ignore_index=True)


def run_fold(
    fold: str,
    thr_df: pd.DataFrame,
    feature_cache: FeatureCache,
    outcome_cache: OutcomeCache,
) -> pd.DataFrame:
    split = _fold_dir(fold)
    train_csv = str(split / "train.csv")
    val_csv = str(split / "val.csv")
    test_csv = str(split / "test.csv")
    df_test = pd.read_csv(test_csv)
    y_test = _labels(df_test)
    test_prompts = _prompts(df_test)

    a_thr = _thr_row(thr_df, fold, "context_aware")
    soft_t = float(a_thr["threshold"])
    hard_sim = float(a_thr["semantic_hard_sim"])
    b_thr = float(_thr_row(thr_df, fold, "context_aware_learned")["threshold"])

    print(f"\n{'=' * 60}\n[LOO ablation] {fold}")
    print(f"  Method A matched: soft={soft_t:g}  semantic_hard_sim={hard_sim:g}")
    print(f"  Method B matched: thr={b_thr:g}")

    sans = _build_fold_sanitizers(
        train_csv, val_csv, include_deberta=False, feature_cache=feature_cache
    )
    method_a: ContextAwareSanitizer = sans["context_aware"]
    method_b: ContextAwareLearnedSanitizer = sans["context_aware_learned"]
    method_a.threshold = soft_t
    method_a.semantic_hard_sim = hard_sim
    method_b.threshold = b_thr

    print(f"  [{fold}] scoring Method A signals once...")
    test_a_signals = _score_method_a_signals(method_a, test_prompts)

    rows: list[dict] = []

    for name, kwargs in METHOD_A_VARIANTS:
        clone = method_a.copy_with(name=name, **kwargs)
        blocked = _blocked_method_a(test_a_signals, clone, soft_t, hard_sim)
        m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
        rows.append(
            {
                "fold": fold,
                "family": "method_a",
                "variant": name,
                "threshold": soft_t,
                "semantic_hard_sim": hard_sim,
                **m,
            }
        )
        print(
            f"  {name}: Bypass={m['Bypass_Rate_%']:.2f}  "
            f"TrueASR={m['True_ASR_%']:.2f}  FPR={m['FPR_%']:.2f}"
        )

    for name, disabled in METHOD_B_VARIANTS:
        clone = method_b.copy_with(name=name, disabled_features=disabled)
        scores = np.asarray(
            [float(clone.injection_score(p)) for p in test_prompts], dtype=float
        )
        blocked = scores >= b_thr
        m = _metrics_from_blocked(blocked, y_test, test_prompts, outcome_cache)
        rows.append(
            {
                "fold": fold,
                "family": "method_b",
                "variant": name,
                "threshold": b_thr,
                "semantic_hard_sim": np.nan,
                **m,
            }
        )
        print(
            f"  {name}: Bypass={m['Bypass_Rate_%']:.2f}  "
            f"TrueASR={m['True_ASR_%']:.2f}  FPR={m['FPR_%']:.2f}"
        )

    try:
        method_a.release_models()
    except Exception:
        pass
    empty_cuda_cache()
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--folds",
        nargs="+",
        default=list(LOO_SOURCE_FOLDS),
        help="Default: four source folds (LOO-4)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore checkpoint and recompute all requested folds",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="Alias of --fresh",
    )
    args = ap.parse_args()
    fresh = bool(args.fresh or args.no_resume)

    if not THRESHOLDS.is_file():
        raise SystemExit(f"Missing {THRESHOLDS}. Run LOO matched-FPR first.")

    thr_df = pd.read_csv(THRESHOLDS)
    outcome_cache = OutcomeCache(PROMPT_CACHE_DIR / "loo" / "outcomes.jsonl")
    feature_cache = FeatureCache(PROMPT_CACHE_DIR / "loo" / "features.pkl")

    if fresh and PER_PATH.is_file():
        PER_PATH.unlink()
        print(f"[LOO ablation] --fresh: removed {PER_PATH}")

    done = _load_checkpoint()
    # Drop incomplete folds for requested set so they are recomputed.
    if not done.empty:
        incomplete = {
            f
            for f in args.folds
            if f in set(done["fold"].astype(str)) and not _fold_complete(done, f)
        }
        for f in sorted(incomplete):
            n = int((done["fold"].astype(str) == f).sum())
            print(f"[LOO ablation] incomplete fold {f} ({n} rows), will recompute")
        if incomplete:
            done = done[~done["fold"].astype(str).isin(incomplete)].copy()
            _save_checkpoint(done)

    print(
        f"[LOO ablation] folds={args.folds}  resume={not fresh}\n"
        f"  thresholds={THRESHOLDS}\n"
        f"  OutcomeCache rows={len(outcome_cache._rows)}\n"
        f"  FeatureCache={feature_cache.path}\n"
        f"  checkpoint={PER_PATH} "
        f"({0 if done.empty else done['fold'].nunique()} complete folds)"
    )

    for fold in args.folds:
        if not fresh and _fold_complete(done, fold):
            print(f"\n[LOO ablation] skip {fold} (already in checkpoint)")
            continue
        part = run_fold(fold, thr_df, feature_cache, outcome_cache)
        # Replace any stale rows for this fold, then checkpoint.
        if done.empty:
            done = part
        else:
            done = done[done["fold"].astype(str) != fold]
            done = pd.concat([done, part], ignore_index=True)
        _save_checkpoint(done)
        print(f"  [checkpoint] wrote {PER_PATH} after {fold}")

    # Restrict to requested folds for macro.
    all_df = done[done["fold"].astype(str).isin(args.folds)].copy()
    missing = [f for f in args.folds if not _fold_complete(all_df, f)]
    if missing:
        raise SystemExit(
            f"Still missing complete folds {missing}. Re-run to resume."
        )

    macro = _build_macro(all_df)
    out_df = _with_deltas(all_df, macro)
    out_df.to_csv(PER_PATH, index=False)
    out_df[out_df["fold"] == MACRO_FOLD].to_csv(MACRO_PATH, index=False)

    print("\nLOO-4 MACRO (matched thresholds)")
    show = out_df[out_df["fold"] == MACRO_FOLD][
        [
            "variant",
            "Bypass_Rate_%",
            "True_ASR_%",
            "FPR_%",
            "Delta_Bypass_pp",
            "Delta_True_ASR_pp",
        ]
    ]
    print(show.to_string(index=False))
    print(f"\n[OK] Wrote {PER_PATH}")
    print(f"[OK] Wrote {MACRO_PATH}")


if __name__ == "__main__":
    main()
