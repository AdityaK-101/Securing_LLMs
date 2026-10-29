"""Source-holdout helpers and loaders.

Canonical **build** lives in ``notebooks/build_source_holdout.ipynb``
(fully uniform mentor protocol). ``scripts/build_source_holdout.py`` and
``build_source_holdout()`` here only execute that notebook's code cells.

This module owns loaders, near-dup filtering, and fold I/O helpers used by
the notebook and by ``eval_loo_holdout.py`` / ``run_experiments --loo``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import (
    EXTERNAL_TEST_MAIN_CSV,
    PROJECT_ROOT,
    RANDOM_SEED,
    SOURCE_HOLDOUT_DIR,
)

RAW = PROJECT_ROOT / "data" / "raw"
CLEAN = PROJECT_ROOT / "data" / "cleaned"

ATTACK_SOURCES = ("simsonsun", "mpdd", "prompt_injection", "bravansky")
EXTERNAL_SOURCE = "neuralchemy"
# Benign rows that belong to the prompt-injection attack source.
SOURCE_BENIGN = {
    "prompt_injection": ("prompt_injection_benign",),
}
BENIGN_TEST_SIZE = 200
BRAVANSKY_TEST_SIZE = 250
VAL_FRACTION = 0.2
NEAR_DUP_THRESHOLD = 0.9
SPOT_CHECK_N = 20
COLUMNS = ["id", "prompt", "label", "source"]


def _norm(text: object) -> str:
    return " ".join(str(text).split()).casefold()


def _as_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["id", "prompt", "label", "source", "prompt_key"])
    frame = frame.drop_duplicates(subset=["prompt_key"]).reset_index(drop=True)
    source = frame["source"].iloc[0]
    frame["id"] = [f"{source}_{i:04d}" for i in range(1, len(frame) + 1)]
    return frame[["id", "prompt", "label", "source", "prompt_key"]]


def _counts(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "injection": int((frame["label"] == "injection").sum()),
        "benign": int((frame["label"] == "benign").sum()),
        "by_source": {
            str(k): int(v) for k, v in frame["source"].value_counts().sort_index().items()
        },
    }


def _benign_inventory(frame: pd.DataFrame) -> dict:
    benign = frame[frame["label"] == "benign"]
    return {
        "rows": int(len(benign)),
        "by_source": {
            str(k): int(v) for k, v in benign["source"].value_counts().sort_index().items()
        },
    }


# --- stage 1: load full sources -------------------------------------------------

def load_simsonsun() -> pd.DataFrame:
    """Prefer Hugging Face Dataset_2 (~2.3k); fall back to local Dataset_1 (67)."""
    path = RAW / "SimsonsunJailbreakPrompts_Dataset2.csv"
    if not path.is_file():
        path = RAW / "SimsonsunJailbreakPrompts.csv"
    raw = pd.read_csv(path)
    rows = [
        {"prompt": text.strip(), "prompt_key": _norm(text), "label": "injection", "source": "simsonsun"}
        for text in raw["Prompt"].astype(str)
        if _norm(text)
    ]
    frame = _as_frame(rows)
    print(f"[source_holdout] Simsonsun from {path.name}: {len(frame)} unique prompts", flush=True)
    return frame


def load_mpdd() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "malicous_deepset.csv", usecols=["Prompt"])
    rows = [
        {"prompt": text.strip(), "prompt_key": _norm(text), "label": "injection", "source": "mpdd"}
        for text in raw["Prompt"].astype(str)
        if _norm(text) and _norm(text) != "nan"
    ]
    return _as_frame(rows)


def load_prompt_injection() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_json(RAW / "Prompt_INJECTION_And_Benign_DATASET.jsonl", lines=True)
    attacks, benign = [], []
    for row in raw.itertuples(index=False):
        key = _norm(row.prompt)
        if not key:
            continue
        item = {"prompt": str(row.prompt).strip(), "prompt_key": key}
        if str(row.label) == "malicious":
            attacks.append({**item, "label": "injection", "source": "prompt_injection"})
        elif str(row.label) == "benign":
            benign.append({**item, "label": "benign", "source": "prompt_injection_benign"})
    return _as_frame(attacks), _as_frame(benign)


def load_chatgpt_benign() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "benign_prompts_chatgpt.csv")
    rows = [
        {
            "prompt": text.strip(),
            "prompt_key": _norm(text),
            "label": "benign",
            "source": "chatgpt_benign",
        }
        for text in raw["Prompt"].astype(str)
        if _norm(text)
    ]
    return _as_frame(rows)


def load_bravansky_gallery() -> pd.DataFrame:
    """Cleaned 50-row Bravansky slice (legacy paper gallery). Prefer ``load_bravansky_raw`` for holdout."""
    rows = []
    with (CLEAN / "jailbreak_prompts_Bravansky.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            key = _norm(item["prompt"])
            if key:
                rows.append(
                    {
                        "prompt": str(item["prompt"]).strip(),
                        "prompt_key": key,
                        "label": "injection",
                        "source": "bravansky",
                    }
                )
    return _as_frame(rows)


def load_bravansky_raw(max_unique: int | None = 2500) -> pd.DataFrame:
    """Full raw Bravansky attacks (``Bravanskycompact-jailbreaks.csv``).

    Optionally subsample to ``max_unique`` after exact dedupe so near-duplicate
    filtering against other sources stays tractable (~57k unique in the raw file).
    """
    raw = pd.read_csv(RAW / "Bravanskycompact-jailbreaks.csv", usecols=["attack"])
    rows = [
        {
            "prompt": text.strip(),
            "prompt_key": _norm(text),
            "label": "injection",
            "source": "bravansky",
        }
        for text in raw["attack"].astype(str)
        if _norm(text) and _norm(text) != "nan"
    ]
    frame = _as_frame(rows)
    if max_unique is not None and len(frame) > max_unique:
        frame = frame.sample(n=max_unique, random_state=RANDOM_SEED).reset_index(drop=True)
        frame["id"] = [f"bravansky_{i:04d}" for i in range(1, len(frame) + 1)]
    print(
        f"[source_holdout] Bravansky raw: {len(frame)} unique prompts "
        f"(max_unique={max_unique})",
        flush=True,
    )
    return frame


def load_bravansky_test_only(gallery_keys: set[str], blocked: set[str], target_total: int) -> pd.DataFrame:
    """Extra Bravansky attacks beyond a gallery (legacy helper). Prefer ``load_bravansky_raw``."""
    raw = pd.read_csv(RAW / "Bravanskycompact-jailbreaks.csv", usecols=["vanilla", "attack"])
    grouped: dict[str, list[str]] = {}
    text_by_key: dict[str, str] = {}
    for vanilla, attack in zip(raw["vanilla"].astype(str), raw["attack"].astype(str)):
        key = _norm(attack)
        if not key or key in gallery_keys or key in blocked:
            continue
        text_by_key.setdefault(key, attack.strip())
        bucket = grouped.setdefault(vanilla, [])
        if key not in bucket:
            bucket.append(key)

    need = target_total - len(gallery_keys)
    order = pd.Series(sorted(grouped)).sample(frac=1, random_state=RANDOM_SEED).tolist()
    chosen: list[dict] = []
    seen = set(gallery_keys)
    depth = 0
    while len(chosen) < need and depth < 5:
        progressed = False
        for vanilla in order:
            bucket = grouped.get(vanilla, [])
            if depth >= len(bucket):
                continue
            key = bucket[depth]
            if key in seen or key in blocked:
                continue
            seen.add(key)
            chosen.append(
                {
                    "prompt": text_by_key[key],
                    "prompt_key": key,
                    "label": "injection",
                    "source": "bravansky",
                }
            )
            progressed = True
            if len(chosen) >= need:
                break
        if not progressed:
            break
        depth += 1
    return _as_frame(chosen)


# --- stage 2: exact then near-duplicate cleanup -------------------------------

def claim(frame: pd.DataFrame, taken: set[str]) -> tuple[pd.DataFrame, int]:
    overlap = int(frame["prompt_key"].isin(taken).sum())
    kept = frame[~frame["prompt_key"].isin(taken)].copy().reset_index(drop=True)
    taken.update(kept["prompt_key"].tolist())
    return kept, overlap


def assign_exclusive_source(
    frames: list[tuple[str, pd.DataFrame]],
) -> tuple[dict[str, pd.DataFrame], set[str], dict[str, int]]:
    """Earlier sources keep a duplicate. Later sources lose it."""
    taken: set[str] = set()
    owned = {}
    overlaps = {}
    for name, frame in frames:
        owned[name], overlaps[name] = claim(frame, taken)
    return owned, taken, overlaps


def drop_near_duplicates(
    frames: dict[str, pd.DataFrame],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> tuple[dict[str, pd.DataFrame], dict]:
    """Remove prompts that are too similar to an earlier source (MiniLM cosine)."""
    ordered = [name for name in ATTACK_SOURCES if name in frames]
    pieces = []
    for name in ordered:
        frame = frames[name].copy()
        frame["owner"] = name
        pieces.append(frame)
    pool = pd.concat(pieces, ignore_index=True)
    if pool.empty:
        return frames, {"removed": 0, "threshold": threshold, "spot_check": []}

    print(
        f"[source_holdout] Encoding {len(pool)} prompts for near-duplicate filter "
        f"(threshold={threshold})...",
        flush=True,
    )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = model.encode(
        pool["prompt"].astype(str).tolist(),
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)
    drop_idx: set[int] = set()
    pairs: list[dict] = []

    # Compare each later source against the union of earlier kept rows.
    for later_i, later_name in enumerate(ordered):
        later_mask = (pool["owner"] == later_name).to_numpy()
        later_idx = np.flatnonzero(later_mask)
        if later_i == 0 or later_idx.size == 0:
            continue
        earlier_names = ordered[:later_i]
        earlier_mask = pool["owner"].isin(earlier_names).to_numpy()
        earlier_idx = np.array(
            [i for i in np.flatnonzero(earlier_mask) if i not in drop_idx],
            dtype=int,
        )
        if earlier_idx.size == 0:
            continue
        sims = emb[later_idx] @ emb[earlier_idx].T
        for row_pos, global_i in enumerate(later_idx):
            if global_i in drop_idx:
                continue
            best = float(sims[row_pos].max())
            if best < threshold:
                continue
            match_pos = int(sims[row_pos].argmax())
            match_i = int(earlier_idx[match_pos])
            drop_idx.add(int(global_i))
            pairs.append(
                {
                    "removed_source": later_name,
                    "kept_source": str(pool.loc[match_i, "owner"]),
                    "cosine": round(best, 4),
                    "removed_prompt": str(pool.loc[global_i, "prompt"])[:240],
                    "kept_prompt": str(pool.loc[match_i, "prompt"])[:240],
                }
            )

    kept_pool = pool.drop(index=list(drop_idx)).reset_index(drop=True)
    cleaned = {
        name: kept_pool[kept_pool["owner"] == name]
        .drop(columns=["owner"])
        .reset_index(drop=True)
        for name in ordered
    }
    # Preserve frames that were not in the attack ordered list (none today).
    for name, frame in frames.items():
        if name not in cleaned:
            cleaned[name] = frame

    rng = np.random.default_rng(RANDOM_SEED)
    spot = pairs
    if len(pairs) > SPOT_CHECK_N:
        pick = rng.choice(len(pairs), size=SPOT_CHECK_N, replace=False)
        spot = [pairs[i] for i in sorted(pick.tolist())]

    report = {
        "threshold": threshold,
        "compared_prompts": int(len(pool)),
        "removed": int(len(drop_idx)),
        "removed_by_source": {
            name: int(sum(1 for p in pairs if p["removed_source"] == name))
            for name in ordered
        },
        "spot_check": spot,
        "note": (
            "Pairs with MiniLM cosine above the threshold across different "
            "sources. The later source in ATTACK_SOURCES loses the prompt."
        ),
    }
    print(
        f"[source_holdout] Near-duplicates removed: {report['removed']} "
        f"(by source={report['removed_by_source']})",
        flush=True,
    )
    return cleaned, report


# --- stage 5: benign false-positive holdout -----------------------------------

def partition_benign(benign: pd.DataFrame, blocked: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pool = benign.drop_duplicates(subset=["prompt_key"])
    pool = pool[~pool["prompt_key"].isin(blocked)].reset_index(drop=True)
    test_size = min(BENIGN_TEST_SIZE, len(pool) - 40)
    benign_test, benign_fit = train_test_split(
        pool,
        train_size=test_size,
        random_state=RANDOM_SEED,
        stratify=pool["source"] if pool["source"].nunique() > 1 else None,
    )
    return benign_test.reset_index(drop=True), benign_fit.reset_index(drop=True)


def _exclude_source_benign(benign_fit: pd.DataFrame, held_out: str) -> pd.DataFrame:
    """When a source is held out, its dataset-native benign rows leave train/val."""
    banned = SOURCE_BENIGN.get(held_out, ())
    if not banned:
        return benign_fit
    return benign_fit[~benign_fit["source"].isin(banned)].reset_index(drop=True)


def _held_out_source_benign(benign_all: pd.DataFrame, held_out: str, shared_test_keys: set[str]) -> pd.DataFrame:
    """Extra held-out-source benign prompts that are not already in the shared test."""
    banned = SOURCE_BENIGN.get(held_out, ())
    if not banned:
        return benign_all.iloc[0:0].copy()
    extra = benign_all[benign_all["source"].isin(banned)].copy()
    extra = extra[~extra["prompt_key"].isin(shared_test_keys)].reset_index(drop=True)
    return extra


# --- stage 6 and 7: write one fold --------------------------------------------

def _with_ids(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    out["id"] = [f"{prefix}_{i:04d}" for i in range(1, len(out) + 1)]
    return out


def _fit_split(pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if pool["label"].nunique() < 2 or len(pool) < 5:
        raise ValueError(
            f"Cannot stratify train/val on a pool of size {len(pool)} "
            f"with labels {pool['label'].value_counts().to_dict()}"
        )
    train, val = train_test_split(
        pool,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=pool["label"],
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def write_fold(folder: Path, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    name = folder.name
    _with_ids(train, f"{name}_train")[COLUMNS].to_csv(folder / "train.csv", index=False)
    _with_ids(val, f"{name}_val")[COLUMNS].to_csv(folder / "val.csv", index=False)
    test_out = test if not test["id"].duplicated().any() else _with_ids(test, f"{name}_test")
    test_out[COLUMNS].to_csv(folder / "test.csv", index=False)


def held_out_fold(
    held_out: str,
    attacks: dict[str, pd.DataFrame],
    bravansky_test: pd.DataFrame,
    benign_test: pd.DataFrame,
    benign_fit: pd.DataFrame,
    benign_all: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train on the other sources only. Held-out source (attacks + its benign) is test."""
    test_attacks = bravansky_test if held_out == "bravansky" else attacks[held_out]
    train_attacks = pd.concat(
        [attacks[name] for name in ATTACK_SOURCES if name != held_out],
        ignore_index=True,
    )
    fit_benign = _exclude_source_benign(benign_fit, held_out)
    train, val = _fit_split(pd.concat([train_attacks, fit_benign], ignore_index=True))
    shared_keys = set(benign_test["prompt_key"])
    extra_benign = _held_out_source_benign(benign_all, held_out, shared_keys)
    test = pd.concat([test_attacks, benign_test, extra_benign], ignore_index=True)
    return train, val, test


def external_fold(
    attacks: dict[str, pd.DataFrame],
    benign_fit: pd.DataFrame,
    blocked: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Train on all four sources. Test is the Neuralchemy main slice."""
    neuralchemy = pd.read_csv(EXTERNAL_TEST_MAIN_CSV)
    neuralchemy["prompt_key"] = neuralchemy["prompt"].map(_norm)
    dropped = int(neuralchemy["prompt_key"].isin(blocked).sum())
    neuralchemy = neuralchemy[~neuralchemy["prompt_key"].isin(blocked)].copy()
    neuralchemy["source"] = "neuralchemy:" + neuralchemy["source"].astype(str)
    pool = pd.concat(list(attacks.values()) + [benign_fit], ignore_index=True)
    train, val = _fit_split(pool)
    return train, val, neuralchemy, dropped


def load_source_holdout(source: str, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return ``(train, val, test)`` for one source-holdout fold."""
    if source not in (*ATTACK_SOURCES, EXTERNAL_SOURCE):
        raise ValueError(f"Unknown source-holdout fold: {source}")
    folder = (root or SOURCE_HOLDOUT_DIR) / source
    return (
        pd.read_csv(folder / "train.csv"),
        pd.read_csv(folder / "val.csv"),
        pd.read_csv(folder / "test.csv"),
    )


def build_source_holdout(out_dir: Path | None = None) -> dict:
    """Build the final mentor leave-one-source-out splits.

    Canonical implementation: ``notebooks/build_source_holdout.ipynb``.
    This wrapper runs ``scripts/build_source_holdout.py``, which executes
    the notebook's code cells.

    ``out_dir`` must be omitted or equal to ``SOURCE_HOLDOUT_DIR``.
    """
    import runpy

    if out_dir is not None and Path(out_dir).resolve() != SOURCE_HOLDOUT_DIR.resolve():
        raise ValueError(
            f"Mentor holdout only writes to {SOURCE_HOLDOUT_DIR}; got {out_dir}. "
            "Change OUT in notebooks/build_source_holdout.ipynb if you need another root."
        )
    script = PROJECT_ROOT / "scripts" / "build_source_holdout.py"
    runpy.run_path(str(script), run_name="__main__")
    return json.loads((SOURCE_HOLDOUT_DIR / "manifest.json").read_text(encoding="utf-8"))
