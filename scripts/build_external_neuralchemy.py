"""Build the held-out Neuralchemy test files.

Downloads the published ``core`` test split of
``neuralchemy/Prompt-injection-dataset`` and writes:

  data/external/neuralchemy/test_core.csv   every kept row
  data/external/neuralchemy/test_main.csv   direct, single-turn slice
  data/external/neuralchemy/manifest.json   counts and dropped overlaps

Train and validation are not modified. Rows whose prompt text already
appears in data/splits/{train,val,test}.csv are dropped. The Neuralchemy
train split and the augmented ``full`` config are not saved.

Usage:
  python scripts/build_external_neuralchemy.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ROOT / "data" / "splits"
OUT_DIR = ROOT / "data" / "external" / "neuralchemy"
DATASET_ID = "neuralchemy/Prompt-injection-dataset"
CONFIG = "core"
SPLIT = "test"

# Matches the paper scope: direct, single-turn injection or jailbreak,
# plus ordinary and edge benign prompts. Other categories stay in
# test_core.csv with eval_slice=appendix.
MAIN_CATEGORIES = frozenset(
    {"direct_injection", "jailbreak", "benign", "edge_case"}
)


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def _existing_prompts() -> dict[str, str]:
    """Map normalized prompt text to the split file that already contains it."""
    found: dict[str, str] = {}
    for name in ("train", "val", "test"):
        frame = pd.read_csv(SPLITS / f"{name}.csv")
        for prompt in frame["prompt"].astype(str):
            found.setdefault(_norm(prompt), name)
    return found


def _tags(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, str):
        return value
    return "|".join(str(tag) for tag in list(value))


def main() -> None:
    raw = load_dataset(DATASET_ID, CONFIG, split=SPLIT).to_pandas()
    existing = _existing_prompts()

    raw["prompt_key"] = raw["text"].map(_norm)
    raw["overlap_split"] = raw["prompt_key"].map(existing)
    dropped = raw[raw["overlap_split"].notna()].copy()
    kept = raw[raw["overlap_split"].isna() & raw["prompt_key"].ne("")].copy()

    mapped = kept["label"].map({1: "injection", 0: "benign"})
    if mapped.isna().any():
        unknown = sorted(set(kept.loc[mapped.isna(), "label"].astype(str)))
        raise SystemExit(f"Unmapped Neuralchemy labels: {unknown}")
    kept["label"] = mapped

    kept["eval_slice"] = kept["category"].map(
        lambda category: "main" if category in MAIN_CATEGORIES else "appendix"
    )
    kept = kept.sort_values(
        ["eval_slice", "category", "source", "text"], kind="mergesort"
    ).reset_index(drop=True)
    kept.insert(0, "id", [f"nc_{i:04d}" for i in range(1, len(kept) + 1)])

    columns = [
        "id",
        "prompt",
        "label",
        "category",
        "source",
        "severity",
        "group_id",
        "eval_slice",
        "tags",
    ]
    export = pd.DataFrame(
        {
            "id": kept["id"],
            "prompt": kept["text"].astype(str),
            "label": kept["label"],
            "category": kept["category"].fillna("").astype(str),
            "source": kept["source"].fillna("").astype(str),
            "severity": kept["severity"].fillna("").astype(str),
            "group_id": kept["group_id"].fillna("").astype(str),
            "eval_slice": kept["eval_slice"],
            "tags": kept["tags"].map(_tags),
        }
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUT_DIR / "test_core.csv", index=False)
    main_slice = export[export["eval_slice"] == "main"]
    main_slice.to_csv(OUT_DIR / "test_main.csv", index=False)

    def _counts(frame: pd.DataFrame) -> dict:
        return {
            "rows": int(len(frame)),
            "injection": int((frame["label"] == "injection").sum()),
            "benign": int((frame["label"] == "benign").sum()),
            "by_category": {
                str(k): int(v)
                for k, v in frame["category"].value_counts().sort_index().items()
            },
        }

    manifest = {
        "dataset": DATASET_ID,
        "config": CONFIG,
        "split": SPLIT,
        "license_card": "Apache-2.0",
        "constituent_sources": {
            "neuralchemy_v1": "Apache-2.0 (dataset card: NeurAlchemy original)",
            "hackaprompt": "CC BY 4.0",
            "harmbench": "MIT",
            "harmbench_benign": "MIT",
            "wildguard_judgecomp": "research use; see WildGuard / JudgeComparison",
            "original": "Apache-2.0 (dataset card)",
        },
        "downloaded_rows": int(len(raw)),
        "dropped_exact_overlaps": [
            {
                "group_id": str(row.group_id),
                "category": str(row.category),
                "source": str(row.source),
                "overlap_split": str(row.overlap_split),
            }
            for row in dropped.itertuples(index=False)
        ],
        "test_core": _counts(export),
        "test_main": _counts(main_slice),
        "main_categories": sorted(MAIN_CATEGORIES),
        "note": (
            "External test only. Do not add these rows to train.csv, val.csv, "
            "or the Method A embedding gallery."
        ),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test_core": manifest["test_core"], "test_main": manifest["test_main"], "dropped": len(dropped)}, indent=2))


if __name__ == "__main__":
    main()
