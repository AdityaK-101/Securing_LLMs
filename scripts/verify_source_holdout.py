"""Quick integrity checks for data/splits/source_holdout."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "splits" / "source_holdout"
FOLDS = ("simsonsun", "mpdd", "prompt_injection", "bravansky", "neuralchemy")
ATTACK = ("simsonsun", "mpdd", "prompt_injection", "bravansky")


def _key(p: object) -> str:
    return " ".join(str(p).split()).casefold()


def main() -> None:
    errors: list[str] = []
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    proto = str(m.get("protocol", ""))
    assert "mentor source-holdout" in proto, m.get("protocol")

    shared = None
    for fold in FOLDS:
        for split in ("train", "val", "test"):
            path = OUT / fold / f"{split}.csv"
            if not path.is_file():
                errors.append(f"missing {path}")
                continue
            df = pd.read_csv(path)
            if list(df.columns) != ["id", "prompt", "label", "source"]:
                errors.append(f"{path}: bad columns {list(df.columns)}")
            if not set(df["label"]).issubset({"injection", "benign"}):
                errors.append(f"{path}: bad labels")

        tr = pd.read_csv(OUT / fold / "train.csv")
        va = pd.read_csv(OUT / fold / "val.csv")
        te = pd.read_csv(OUT / fold / "test.csv")
        fit_keys = {_key(p) for p in tr["prompt"]} | {_key(p) for p in va["prompt"]}
        leak = sum(_key(p) in fit_keys for p in te["prompt"])
        if leak:
            errors.append(f"{fold}: {leak} test prompts in train/val")

        if fold in ATTACK:
            inj = te[te["label"] == "injection"]
            ben = te[te["label"] == "benign"]
            if len(inj) != 200:
                errors.append(f"{fold}: expected 200 test inj, got {len(inj)}")
            if len(ben) != 200:
                errors.append(f"{fold}: expected 200 test ben, got {len(ben)}")
            if len(te) != 400:
                errors.append(f"{fold}: expected test=400, got {len(te)}")
            bad = inj[~inj["source"].astype(str).str.startswith(fold)]
            if len(bad):
                errors.append(f"{fold}: foreign injections {len(bad)}")
            keys = {_key(p) for p in ben["prompt"]}
            if shared is None:
                shared = keys
            else:
                shared &= keys

        if fold == "prompt_injection":
            banned = {"prompt_injection", "prompt_injection_benign"}
            if tr["source"].isin(banned).any() or va["source"].isin(banned).any():
                errors.append("prompt_injection: source rows still in train/val")

    # Four-source folds must share identical sizes.
    sizes = []
    for fold in ATTACK:
        tr = pd.read_csv(OUT / fold / "train.csv")
        va = pd.read_csv(OUT / fold / "val.csv")
        te = pd.read_csv(OUT / fold / "test.csv")
        sizes.append((len(tr), len(va), len(te)))
    if len(set(sizes)) != 1:
        errors.append(f"four-source sizes not uniform: {sizes}")

    # shared benign size from manifest
    sb = m["shared_benign_test"]["rows"]
    if sb != 200:
        errors.append(f"shared benign rows={sb}")

    print("manifest protocol:", m["protocol"])
    print("summary:")
    print(pd.read_csv(OUT / "summary_table.csv").to_string(index=False))
    print("train benign:")
    print(pd.read_csv(OUT / "train_benign_by_fold.csv").to_string(index=False))
    if errors:
        print("ERRORS:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)
    print("OK: holdout integrity checks passed")


if __name__ == "__main__":
    main()
