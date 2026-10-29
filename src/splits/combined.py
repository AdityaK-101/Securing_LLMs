"""Grow and rebuild the mixed in-distribution combined corpus.

v2.0 was 350 prompts (50 attacks × 4 sources + 150 benign) with a 53-row test.
v3.0 samples more from the same raw sources (hundreds scale, aligned with LOO):

  - 100 attacks × 4 sources = 400 injection
  - 300 benign (50 ChatGPT + 250 PI-benign)
  - Stratified 70 / 15 / 15 → ~490 / 105 / 105

The frozen v2.0 files are archived under ``data/cleaned/legacy/`` and
``data/splits/legacy_v2.0_350/`` before overwrite.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import PROJECT_ROOT, RANDOM_SEED
from .source_holdout import (
    load_bravansky_raw,
    load_chatgpt_benign,
    load_mpdd,
    load_prompt_injection,
    load_simsonsun,
)

CLEAN = PROJECT_ROOT / "data" / "cleaned"
SPLITS = PROJECT_ROOT / "data" / "splits"
LEGACY_CLEAN = CLEAN / "legacy"
LEGACY_SPLITS = SPLITS / "legacy_v2.0_350"

# Growth knobs (keep in the hundreds, not thousands)
N_ATTACK_PER_SOURCE = 100
N_CHATGPT_BENIGN = 50
N_PI_BENIGN = 250  # all available PI benign
VAL_TEST_FRACTION = 0.30  # then half → ~15% val / 15% test
COLUMNS = ["id", "prompt", "label"]
COLUMNS_WITH_SOURCE = ["id", "prompt", "label", "source"]


def _sample(frame: pd.DataFrame, n: int, *, seed: int) -> pd.DataFrame:
    if len(frame) < n:
        raise ValueError(
            f"Need {n} rows from source={frame['source'].iloc[0] if len(frame) else '?'} "
            f"but only {len(frame)} unique prompts available"
        )
    return frame.sample(n=n, random_state=seed).reset_index(drop=True)


def _archive_v2() -> None:
    LEGACY_CLEAN.mkdir(parents=True, exist_ok=True)
    LEGACY_SPLITS.mkdir(parents=True, exist_ok=True)
    combined = CLEAN / "combined_dataset.csv"
    if combined.is_file():
        dest = LEGACY_CLEAN / "combined_dataset_v2.0_350.csv"
        if not dest.is_file():
            shutil.copy2(combined, dest)
            print(f"[combined] Archived {combined.name} → {dest}")
    for name in ("train.csv", "val.csv", "test.csv"):
        src = SPLITS / name
        dest = LEGACY_SPLITS / name
        if src.is_file() and not dest.is_file():
            shutil.copy2(src, dest)
            print(f"[combined] Archived {src} → {dest}")


def build_combined_corpus(
    *,
    n_attack_per_source: int = N_ATTACK_PER_SOURCE,
    n_chatgpt_benign: int = N_CHATGPT_BENIGN,
    n_pi_benign: int = N_PI_BENIGN,
    seed: int = RANDOM_SEED,
    archive_legacy: bool = True,
) -> dict:
    """Sample, write ``combined_dataset.csv`` + stratified splits, return manifest."""
    if archive_legacy:
        _archive_v2()

    print("[combined] Loading raw sources...")
    simsonsun = load_simsonsun()
    mpdd = load_mpdd()
    pi_attacks, pi_benign = load_prompt_injection()
    chatgpt = load_chatgpt_benign()
    bravansky = load_bravansky_raw(max_unique=2500)

    attacks = pd.concat(
        [
            _sample(simsonsun, n_attack_per_source, seed=seed),
            _sample(mpdd, n_attack_per_source, seed=seed + 1),
            _sample(pi_attacks, n_attack_per_source, seed=seed + 2),
            _sample(bravansky, n_attack_per_source, seed=seed + 3),
        ],
        ignore_index=True,
    )
    benign = pd.concat(
        [
            _sample(chatgpt, min(n_chatgpt_benign, len(chatgpt)), seed=seed + 4),
            _sample(pi_benign, min(n_pi_benign, len(pi_benign)), seed=seed + 5),
        ],
        ignore_index=True,
    )

    # Global exact dedupe (keep first)
    combined = pd.concat([attacks, benign], ignore_index=True)
    before = len(combined)
    combined = combined.drop_duplicates(subset=["prompt_key"]).reset_index(drop=True)
    if len(combined) < before:
        print(f"[combined] Dropped {before - len(combined)} cross-source exact duplicates")

    combined = combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    combined["id"] = [f"id_{i}" for i in range(1, len(combined) + 1)]
    out_combined = combined[COLUMNS_WITH_SOURCE].copy()
    out_path = CLEAN / "combined_dataset.csv"
    out_combined.to_csv(out_path, index=False, encoding="utf-8")
    print(
        f"[combined] Wrote {out_path} "
        f"({len(out_combined)} rows; "
        f"injection={(out_combined.label == 'injection').sum()}, "
        f"benign={(out_combined.label == 'benign').sum()})"
    )

    # Stratified 70 / 15 / 15 (same protocol as split.ipynb)
    train_df, temp_df = train_test_split(
        out_combined,
        test_size=VAL_TEST_FRACTION,
        random_state=seed,
        stratify=out_combined["label"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=seed,
        stratify=temp_df["label"],
    )
    for name, frame in ("train", train_df), ("val", val_df), ("test", test_df):
        # Evaluation code only requires id/prompt/label; keep source for provenance
        frame[COLUMNS_WITH_SOURCE].to_csv(SPLITS / f"{name}.csv", index=False)
        print(
            f"[combined] {name}: {len(frame)} "
            f"(inj={(frame.label == 'injection').sum()}, "
            f"ben={(frame.label == 'benign').sum()})"
        )

    manifest = {
        "version": "3.0",
        "seed": seed,
        "n_attack_per_source": n_attack_per_source,
        "n_chatgpt_benign": int((benign["source"] == "chatgpt_benign").sum()),
        "n_pi_benign": int((benign["source"] == "prompt_injection_benign").sum()),
        "total": int(len(out_combined)),
        "injection": int((out_combined.label == "injection").sum()),
        "benign": int((out_combined.label == "benign").sum()),
        "splits": {
            "train": {
                "total": int(len(train_df)),
                "injection": int((train_df.label == "injection").sum()),
                "benign": int((train_df.label == "benign").sum()),
            },
            "val": {
                "total": int(len(val_df)),
                "injection": int((val_df.label == "injection").sum()),
                "benign": int((val_df.label == "benign").sum()),
            },
            "test": {
                "total": int(len(test_df)),
                "injection": int((test_df.label == "injection").sum()),
                "benign": int((test_df.label == "benign").sum()),
            },
        },
        "by_source": {
            str(k): int(v)
            for k, v in out_combined["source"].value_counts().sort_index().items()
        },
        "legacy_v2_0": {
            "combined": str(LEGACY_CLEAN / "combined_dataset_v2.0_350.csv"),
            "splits": str(LEGACY_SPLITS),
        },
        "note": (
            "Mixed in-distribution corpus grown from raw sources. "
            "Distinct from data/splits/source_holdout/ (leave-one-source-out)."
        ),
    }
    man_path = CLEAN / "combined_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[combined] Wrote {man_path}")
    return manifest


def build_combined_dataset() -> dict:
    """CLI / notebook entrypoint."""
    return build_combined_corpus()
