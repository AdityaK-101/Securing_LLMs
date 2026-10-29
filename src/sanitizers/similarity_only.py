"""Similarity-only baseline: max MiniLM cosine to the training injection gallery.

Threshold is tuned on the validation split (F1, then lower Bypass, then lower FPR).
No hard triggers and no other signals. Used for the fair fold-trained comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.linalg import norm
from sklearn.metrics import f1_score

from ..config import RANDOM_SEED, TRAIN_CSV, VAL_CSV


class SimilarityOnlySanitizer:
    """Block if max cosine to train injections >= val-tuned threshold."""

    name = "similarity_only"

    def __init__(
        self,
        train_csv: str = TRAIN_CSV,
        val_csv: str = VAL_CSV,
        share_embed_from=None,
        feature_cache=None,
        _skip_train: bool = False,
    ):
        self.threshold = 0.5
        self.val_metrics: dict = {}
        self._embed_model = None
        self._injection_embeddings = None
        self._owns_embed = share_embed_from is None
        self._feature_cache = feature_cache

        if share_embed_from is not None:
            self._embed_model = share_embed_from._embed_model
            # Prefer the donor gallery only if it matches this train set; else rebuild.
            self._injection_embeddings = None
            if feature_cache is None and getattr(share_embed_from, "_feature_cache", None):
                self._feature_cache = share_embed_from._feature_cache

        if not _skip_train:
            self._train(train_csv, val_csv)

    def set_feature_cache(self, feature_cache) -> None:
        self._feature_cache = feature_cache

    def _encode_prompts(self, prompts: list[str]) -> np.ndarray:
        if self._feature_cache is not None:
            if self._embed_model is not None:
                self._feature_cache.ensure_embeddings(prompts, self._embed_model)
            vectors = []
            for prompt in prompts:
                cached = self._feature_cache.get_embedding(prompt)
                if cached is None:
                    vectors = []
                    break
                vectors.append(np.asarray(cached, dtype=np.float32))
            if len(vectors) == len(prompts):
                return np.stack(vectors)
        if self._embed_model is None:
            raise RuntimeError("similarity_only: no embed model and FeatureCache miss")
        return self._embed_model.encode(
            prompts,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def _train(self, train_csv: str, val_csv: str) -> None:
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        injections = df_train[df_train["label"] == "injection"]["prompt"].dropna().astype(str).tolist()
        if not injections:
            raise ValueError("similarity_only: no injection rows in train")

        if self._embed_model is None:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
        self._injection_embeddings = self._encode_prompts(injections)
        print(
            f"[SimilarityOnly] MiniLM gallery: {len(injections)} train injections.",
            flush=True,
        )

        train_scores = self._scores(df_train["prompt"].astype(str).tolist())
        val_scores = self._scores(df_val["prompt"].astype(str).tolist())
        y_val = (df_val["label"] == "injection").astype(int).to_numpy()
        self.threshold, self.val_metrics = self._tune_threshold(val_scores, y_val)
        # Silence unused warning; train_scores documents the fit distribution.
        _ = train_scores
        print(
            f"[SimilarityOnly] val-tuned threshold={self.threshold:.3f} "
            f"(F1={self.val_metrics.get('f1', 0):.3f}, "
            f"FPR={self.val_metrics.get('fpr', 0):.3f}, "
            f"Bypass={self.val_metrics.get('bypass', 0):.3f}).",
            flush=True,
        )

    def _max_cosine(self, prompt: str) -> float:
        if self._injection_embeddings is None:
            return 0.0
        if self._embed_model is None and self._feature_cache is None:
            return 0.0
        emb = self._encode_prompts([prompt])
        g = self._injection_embeddings
        g = g / (norm(g, axis=1, keepdims=True) + 1e-10)
        p = emb / (norm(emb, axis=1, keepdims=True) + 1e-10)
        return float(np.dot(g, p.T).flatten().max())

    def _scores(self, prompts: list[str]) -> np.ndarray:
        if self._injection_embeddings is None:
            return np.zeros(len(prompts), dtype=float)
        if self._embed_model is None and self._feature_cache is None:
            return np.zeros(len(prompts), dtype=float)
        emb = self._encode_prompts(prompts)
        g = self._injection_embeddings
        g = g / (norm(g, axis=1, keepdims=True) + 1e-10)
        p = emb / (norm(emb, axis=1, keepdims=True) + 1e-10)
        return np.dot(p, g.T).max(axis=1)

    def _tune_threshold(self, scores: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        candidates = np.unique(
            np.concatenate(
                [
                    np.linspace(0.05, 0.95, 37),
                    np.quantile(scores, np.linspace(0, 1, 21)),
                ]
            )
        )
        best_t = 0.5
        best = {"f1": -1.0, "bypass": 1.0, "fpr": 1.0}
        for t in candidates:
            pred = (scores >= t).astype(int)
            f1 = float(f1_score(y, pred, zero_division=0))
            inj = y == 1
            ben = y == 0
            bypass = float((pred[inj] == 0).mean()) if inj.any() else 0.0
            fpr = float((pred[ben] == 1).mean()) if ben.any() else 0.0
            better = (
                f1 > best["f1"] + 1e-12
                or (abs(f1 - best["f1"]) <= 1e-12 and bypass < best["bypass"] - 1e-12)
                or (
                    abs(f1 - best["f1"]) <= 1e-12
                    and abs(bypass - best["bypass"]) <= 1e-12
                    and fpr < best["fpr"] - 1e-12
                )
            )
            if better:
                best_t = float(t)
                best = {"f1": f1, "bypass": bypass, "fpr": fpr}
        return best_t, best

    def injection_score(self, prompt: str) -> float:
        """Max MiniLM cosine to the train injection gallery."""
        return float(self._max_cosine("" if prompt is None else str(prompt)))

    def sanitize(self, prompt: str):
        text = "" if prompt is None else str(prompt)
        score = self._max_cosine(text)
        return text, bool(score >= self.threshold)

    def release_models(self) -> None:
        if self._owns_embed and self._embed_model is not None:
            del self._embed_model
            self._embed_model = None
