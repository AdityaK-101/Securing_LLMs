"""
Shared prompt caches for the mentor speedup path.

1. OutcomeCache — greedy target-LLM reply + judge label, once per attack prompt.
   True ASR for every method/fold is then a lookup (decoding does not depend on
   which filter let the prompt through).

2. FeatureCache — MiniLM embedding + raw DistilGPT-2 perplexity once per prompt,
   reused across folds (gallery / PPL calibration still fold-specific).

Both caches append to disk as they go so a crash can resume.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from ..config import PROMPT_CACHE_DIR


def prompt_key(prompt: str) -> str:
    """Stable key for a prompt string (UTF-8 SHA-256 hex)."""
    return hashlib.sha256(str(prompt).encode("utf-8")).hexdigest()


def unique_prompts(prompts: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for prompt in prompts:
        text = str(prompt)
        key = prompt_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


class OutcomeCache:
    """Per-attack-prompt greedy LLM output + judge label (JSONL, resumable)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else PROMPT_CACHE_DIR / "outcomes.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = row.get("prompt_key") or prompt_key(row.get("prompt", ""))
                self._rows[key] = row
        print(f"[OutcomeCache] Loaded {len(self._rows)} rows from {self.path}")

    def _append(self, row: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def get(self, prompt: str) -> dict | None:
        return self._rows.get(prompt_key(prompt))

    def missing_llm(self, prompts: Iterable[str]) -> list[str]:
        missing = []
        for prompt in unique_prompts(prompts):
            row = self.get(prompt)
            if row is None or not str(row.get("llm_output", "")).strip():
                missing.append(prompt)
        return missing

    def missing_judge(self, prompts: Iterable[str]) -> list[str]:
        missing = []
        for prompt in unique_prompts(prompts):
            row = self.get(prompt)
            if row is None:
                missing.append(prompt)
                continue
            if "llm_output" not in row or not str(row.get("llm_output", "")).strip():
                missing.append(prompt)
                continue
            label = str(row.get("judge_label", "") or "").strip().upper()
            # SKIPPED / empty → still need a real COMPLIED/REFUSED when judge runs
            if label in ("", "NONE", "NULL", "SKIPPED"):
                missing.append(prompt)
        return missing

    def ensure_llm(
        self,
        prompts: Iterable[str],
        runner,
        *,
        batch_size: int = 4,
    ) -> None:
        """Generate greedy replies for missing injection prompts; save as you go."""
        missing = self.missing_llm(prompts)
        if not missing:
            print("[OutcomeCache] LLM outcomes complete (nothing to generate).")
            return
        print(
            f"[OutcomeCache] Generating {len(missing)} missing LLM replies "
            f"(batch_size={batch_size})..."
        )
        if hasattr(runner, "run_batch"):
            for start in tqdm(range(0, len(missing), batch_size), desc="LLM cache"):
                batch = missing[start : start + batch_size]
                outputs = runner.run_batch(batch, batch_size=len(batch))
                for prompt, llm_output in zip(batch, outputs):
                    self._store_llm(prompt, llm_output)
        else:
            for prompt in tqdm(missing, desc="LLM cache"):
                self._store_llm(prompt, runner.run(prompt))

    def _store_llm(self, prompt: str, llm_output: str) -> None:
        key = prompt_key(prompt)
        row = dict(self._rows.get(key) or {})
        row.update({
            "prompt_key": key,
            "prompt": prompt,
            "llm_output": llm_output,
        })
        # Keep prior judge fields if regenerating LLM only
        self._rows[key] = row
        self._append({
            "prompt_key": key,
            "prompt": prompt,
            "llm_output": llm_output,
            "judge_label": row.get("judge_label"),
            "judge_reasoning": row.get("judge_reasoning", ""),
            "stage": "llm",
        })

    def ensure_judge(self, prompts: Iterable[str], judge) -> None:
        """Judge cached LLM outputs for missing labels; save as you go."""
        missing = self.missing_judge(prompts)
        # Only judge rows that already have an llm_output
        todo = []
        for prompt in missing:
            row = self.get(prompt)
            if row is None or not str(row.get("llm_output", "")).strip():
                continue
            todo.append(prompt)
        if not todo:
            print("[OutcomeCache] Judge outcomes complete (nothing to judge).")
            return
        print(f"[OutcomeCache] Judging {len(todo)} cached LLM replies...")
        if judge.is_mock():
            for prompt in todo:
                self._store_judge(prompt, "SKIPPED", "[JUDGE_MOCK]")
            return
        for prompt in tqdm(todo, desc="Judge cache"):
            row = self.get(prompt)
            label, reasoning = judge.evaluate(prompt, row["llm_output"])
            self._store_judge(prompt, label, reasoning)

    def _store_judge(self, prompt: str, judge_label: str, judge_reasoning: str) -> None:
        key = prompt_key(prompt)
        row = dict(self._rows.get(key) or {})
        row.update({
            "prompt_key": key,
            "prompt": prompt,
            "judge_label": judge_label,
            "judge_reasoning": judge_reasoning or "",
        })
        self._rows[key] = row
        self._append({
            "prompt_key": key,
            "prompt": prompt,
            "llm_output": row.get("llm_output", ""),
            "judge_label": judge_label,
            "judge_reasoning": judge_reasoning or "",
            "stage": "judge",
        })

    def lookup(self, prompt: str) -> tuple[str, str, str]:
        """Return (llm_output, judge_label, judge_reasoning)."""
        row = self.get(prompt)
        if row is None:
            return "[LLM_CACHE_MISS]", "SKIPPED", ""
        return (
            str(row.get("llm_output", "[LLM_CACHE_MISS]")),
            str(row.get("judge_label") or "SKIPPED"),
            str(row.get("judge_reasoning") or ""),
        )


class FeatureCache:
    """MiniLM embeddings + raw perplexity once per prompt (pickle, resumable)."""

    def __init__(self, path: Path | None = None):
        self.path = Path(path) if path else PROMPT_CACHE_DIR / "features.pkl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embeddings: dict[str, np.ndarray] = {}
        self.raw_ppl: dict[str, float] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        with open(self.path, "rb") as handle:
            blob = pickle.load(handle)
        self.embeddings = {
            k: np.asarray(v, dtype=np.float32) for k, v in blob.get("embeddings", {}).items()
        }
        self.raw_ppl = {k: float(v) for k, v in blob.get("raw_ppl", {}).items()}
        print(
            f"[FeatureCache] Loaded {len(self.embeddings)} embeddings, "
            f"{len(self.raw_ppl)} PPL values from {self.path}"
        )

    def save(self) -> None:
        payload = {
            "embeddings": self.embeddings,
            "raw_ppl": self.raw_ppl,
        }
        tmp = self.path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self.path)

    def get_embedding(self, prompt: str) -> np.ndarray | None:
        return self.embeddings.get(prompt_key(prompt))

    def get_raw_ppl(self, prompt: str) -> float | None:
        key = prompt_key(prompt)
        return self.raw_ppl[key] if key in self.raw_ppl else None

    def ensure_embeddings(
        self,
        prompts: Iterable[str],
        embed_model,
        *,
        batch_size: int = 64,
        save_every: int = 256,
    ) -> None:
        missing = [
            p for p in unique_prompts(prompts) if self.get_embedding(p) is None
        ]
        if not missing:
            return
        print(f"[FeatureCache] Encoding {len(missing)} MiniLM embeddings...")
        done = 0
        for start in tqdm(range(0, len(missing), batch_size), desc="MiniLM cache"):
            batch = missing[start : start + batch_size]
            emb = embed_model.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
                batch_size=min(batch_size, len(batch)),
            )
            for prompt, vec in zip(batch, emb):
                self.embeddings[prompt_key(prompt)] = np.asarray(vec, dtype=np.float32)
            done += len(batch)
            if done % save_every < batch_size:
                self.save()
        self.save()

    def ensure_raw_ppl(
        self,
        prompts: Iterable[str],
        compute_ppl,
        *,
        save_every: int = 64,
    ) -> None:
        missing = [
            p for p in unique_prompts(prompts) if self.get_raw_ppl(p) is None
        ]
        if not missing:
            return
        print(f"[FeatureCache] Computing {len(missing)} raw perplexities...")
        for i, prompt in enumerate(tqdm(missing, desc="PPL cache"), start=1):
            value = compute_ppl(prompt)
            if value is None:
                continue
            self.raw_ppl[prompt_key(prompt)] = float(value)
            if i % save_every == 0:
                self.save()
        self.save()


def attach_feature_cache(sanitizers: list, feature_cache: FeatureCache | None) -> None:
    """Attach a shared FeatureCache to sanitizers that support it."""
    if feature_cache is None:
        return
    for san in sanitizers:
        if hasattr(san, "set_feature_cache"):
            san.set_feature_cache(feature_cache)


def collect_prompts_from_csvs(csv_paths: Iterable[Path | str], *, labels=None) -> list[str]:
    """Load unique prompts from CSV paths; optional label filter (e.g. injection)."""
    import pandas as pd

    prompts: list[str] = []
    label_set = None if labels is None else set(labels)
    for path in csv_paths:
        df = pd.read_csv(path)
        if label_set is not None and "label" in df.columns:
            df = df[df["label"].isin(label_set)]
        prompts.extend(df["prompt"].dropna().astype(str).tolist())
    return unique_prompts(prompts)


def warm_minilm_embeddings(
    feature_cache: FeatureCache,
    prompts: Iterable[str],
    *,
    batch_size: int = 64,
) -> None:
    """Encode missing prompts with MiniLM into ``feature_cache`` (CPU)."""
    missing = [
        p for p in unique_prompts(prompts) if feature_cache.get_embedding(p) is None
    ]
    if not missing:
        print("[FeatureCache] MiniLM embeddings already complete.")
        return
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    )
    try:
        feature_cache.ensure_embeddings(missing, model, batch_size=batch_size)
    finally:
        del model
        try:
            import gc

            gc.collect()
        except Exception:
            pass


def warm_raw_ppl_from_sanitizer(
    feature_cache: FeatureCache,
    prompts: Iterable[str],
    sanitizer,
) -> None:
    """Fill missing raw DistilGPT-2 perplexities using a trained sanitizer."""
    if sanitizer is None or not hasattr(sanitizer, "_compute_perplexity"):
        return
    feature_cache.ensure_raw_ppl(prompts, sanitizer._compute_perplexity)


def find_context_sanitizer(sanitizers: list):
    """Return Method A (or Method B) instance that owns MiniLM / DistilGPT-2."""
    for san in sanitizers:
        if san.__class__.__name__ == "ContextAwareSanitizer":
            return san
    for san in sanitizers:
        if hasattr(san, "_compute_perplexity") and hasattr(san, "_embed_model"):
            return san
    return None
