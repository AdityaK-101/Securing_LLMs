"""CLI utility to print mixed-split and source-holdout sizes."""

from pathlib import Path

import pandas as pd

from src.config import SOURCE_HOLDOUT_DIR
from src.splits.source_holdout import ATTACK_SOURCES, EXTERNAL_SOURCE
from src.utils.data import load_external_test, load_splits


if __name__ == "__main__":
    train, val, test = load_splits()
    print(
        f"Mixed corpus — Train: {len(train)}  Val: {len(val)}  Test: {len(test)}  "
        f"(inj test={int((test['label'] == 'injection').sum())}, "
        f"ben test={int((test['label'] == 'benign').sum())})"
    )
    external = load_external_test("main")
    print(
        "External Neuralchemy main: "
        f"{len(external)}  "
        f"injection={int((external['label'] == 'injection').sum())}  "
        f"benign={int((external['label'] == 'benign').sum())}"
    )
    print("Source holdout:")
    for fold in (*ATTACK_SOURCES, EXTERNAL_SOURCE):
        folder = SOURCE_HOLDOUT_DIR / fold
        if not folder.is_dir():
            print(f"  {fold}: missing (run scripts/build_source_holdout.py)")
            continue
        parts = {
            split: len(pd.read_csv(folder / f"{split}.csv"))
            for split in ("train", "val", "test")
        }
        print(f"  {fold}: train={parts['train']} val={parts['val']} test={parts['test']}")
    summary = SOURCE_HOLDOUT_DIR / "summary_table.csv"
    if summary.is_file():
        print(f"  details: {summary}")
