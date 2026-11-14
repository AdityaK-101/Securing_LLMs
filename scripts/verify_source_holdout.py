"""Quick integrity checks for data/splits/source_holdout."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "splits" / "source_holdout"
FOLDS = ("simsonsun", "mpdd", "prompt_injection", "bravansky", "neuralchemy")
ATTACK = ("simsonsun", "mpdd", "prompt_injection", "bravansky")
ALPACA = "alpaca_benign"
PI_BENIGN = "prompt_injection_benign"


def _key(p: object) -> str:
    return " ".join(str(p).split()).casefold()


def main() -> None:
    errors: list[str] = []
    m = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    proto = str(m.get("protocol", ""))
    assert "source-holdout" in proto, m.get("protocol")

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

        # PI-benign must never appear in train/val.
        if (tr["source"] == PI_BENIGN).any() or (va["source"] == PI_BENIGN).any():
            errors.append(f"{fold}: prompt_injection_benign present in train/val")

        # Fit benign = Alpaca only, with enough val for ~5% FPR.
        tr_ben = tr[tr["label"] == "benign"]
        va_ben = va[va["label"] == "benign"]
        if len(va_ben) < 100:
            errors.append(f"{fold}: val benign={len(va_ben)} < 100")
        if set(tr_ben["source"]) - {ALPACA}:
            errors.append(f"{fold}: non-Alpaca train benign {set(tr_ben['source'])}")
        if set(va_ben["source"]) - {ALPACA}:
            errors.append(f"{fold}: non-Alpaca val benign {set(va_ben['source'])}")
        if len(tr_ben) != 200:
            errors.append(f"{fold}: expected 200 Alpaca train benign, got {len(tr_ben)}")
        if len(va_ben) != 100:
            errors.append(f"{fold}: expected 100 Alpaca val benign, got {len(va_ben)}")

        # Test benign = shared PI-benign on every fold.
        te_ben = te[te["label"] == "benign"]
        te_inj = te[te["label"] == "injection"]
        if len(te_ben) != 200:
            errors.append(f"{fold}: expected 200 test ben, got {len(te_ben)}")
        if set(te_ben["source"]) != {PI_BENIGN}:
            errors.append(f"{fold}: test benign sources={set(te_ben['source'])}")
        keys = {_key(p) for p in te_ben["prompt"]}
        if shared is None:
            shared = keys
        elif keys != shared:
            errors.append(f"{fold}: test benign set differs from other folds")

        if fold in ATTACK:
            if len(te_inj) != 200:
                errors.append(f"{fold}: expected 200 test inj, got {len(te_inj)}")
            if len(te) != 400:
                errors.append(f"{fold}: expected test=400, got {len(te)}")
            bad = te_inj[~te_inj["source"].astype(str).str.startswith(fold)]
            if len(bad):
                errors.append(f"{fold}: foreign injections {len(bad)}")

        if fold == "prompt_injection":
            banned = {"prompt_injection", PI_BENIGN}
            if tr["source"].isin(banned).any() or va["source"].isin(banned).any():
                errors.append("prompt_injection: source rows still in train/val")

        if fold == "neuralchemy":
            if len(te_inj) != 200:
                errors.append(f"neuralchemy: expected 200 test inj, got {len(te_inj)}")

    # Four-source folds must share identical sizes.
    sizes = []
    for fold in ATTACK:
        tr = pd.read_csv(OUT / fold / "train.csv")
        va = pd.read_csv(OUT / fold / "val.csv")
        te = pd.read_csv(OUT / fold / "test.csv")
        sizes.append((len(tr), len(va), len(te)))
    if len(set(sizes)) != 1:
        errors.append(f"four-source sizes not uniform: {sizes}")

    # Expected sizes after Alpaca fit: 480 inj train + 200 alpaca = 680;
    # 120 inj val + 100 alpaca = 220; test 400.
    if sizes and sizes[0] != (680, 220, 400):
        errors.append(f"four-source expected (680,220,400), got {sizes[0]}")

    sb = m["shared_benign_test"]["rows"]
    if sb != 200:
        errors.append(f"shared benign rows={sb}")

    knobs = m.get("knobs") or {}
    if int(knobs.get("n_benign_fit_val", 0) or 0) < 100:
        errors.append("manifest knobs: n_benign_fit_val < 100")

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
