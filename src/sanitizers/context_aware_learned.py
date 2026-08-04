"""
Context-Aware Learned sanitizer — soft-only Method B.

Same multi-signal direction as ContextAwareSanitizer, but:
  - continuous semantic features (raw cosine + top-3 mean + benign contrast)
  - logistic-regression fusion instead of hand weights W1…W8
  - decision threshold tuned on the validation split
  - no hard-trigger overrides

Method A (ContextAwareSanitizer) is left unchanged for comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED, TRAIN_CSV, VAL_CSV
from .context_aware import INTENT_PATTERNS, ContextAwareSanitizer
from .keyword import KeywordHeuristicSanitizer
from .regex import RegexSanitizer

FEATURE_NAMES = (
    "regex",
    "keyword",
    "semantic_max",
    "semantic_top3",
    "semantic_contrast",
    "intent",
    "roleplay",
    "shift",
    "conflict",
    "perplexity",
)

# How strongly benign similarity pulls semantic risk down
BENIGN_CONTRAST_LAMBDA = 0.5


class ContextAwareLearnedSanitizer:
    """
    Soft-only multi-signal sanitizer with learned fusion.

    Blocks if P(injection | features) >= threshold_val.
    No hard triggers.
    """

    name = "context_aware_learned"

    def __init__(
        self,
        train_csv: str = TRAIN_CSV,
        val_csv: str = VAL_CSV,
        share_from: ContextAwareSanitizer | None = None,
        target_fpr: float | None = None,
        _skip_train: bool = False,
    ):
        self.threshold = 0.5
        self.target_fpr = target_fpr
        self._owns_models = share_from is None
        self._regex_san = RegexSanitizer()
        self._kw_san = KeywordHeuristicSanitizer()

        self._embed_model = None
        self._injection_embeddings = None
        self._benign_embeddings = None
        self._ppl_model = None
        self._ppl_tokenizer = None
        self._ppl_low = None
        self._ppl_high = None
        self._device = "cpu"

        self._scaler = StandardScaler()
        self._model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_SEED,
            C=1.0,
            solver="lbfgs",
            class_weight="balanced",
        )
        self._is_trained = False
        self.val_metrics: dict = {}
        self._disabled_features: set[str] = set()

        if share_from is not None:
            self._adopt_shared_models(share_from)

        if not _skip_train:
            self._train(train_csv, val_csv)

    def _adopt_shared_models(self, donor: ContextAwareSanitizer) -> None:
        """Reuse MiniLM / distilgpt2 already loaded by Method A."""
        self._embed_model = donor._embed_model
        self._injection_embeddings = donor._injection_embeddings
        self._ppl_model = donor._ppl_model
        self._ppl_tokenizer = donor._ppl_tokenizer
        self._ppl_low = donor._ppl_low
        self._ppl_high = donor._ppl_high
        self._device = getattr(donor, "_device", "cpu")

    def _train(self, train_csv: str, val_csv: str) -> None:
        try:
            df_train = pd.read_csv(train_csv)
            df_val = pd.read_csv(val_csv)
        except Exception as e:
            print(f"[ContextAwareLearned] Warning: could not read splits — {e}")
            return

        injections = df_train[df_train["label"] == "injection"]["prompt"].dropna().tolist()
        benign = df_train[df_train["label"] == "benign"]["prompt"].dropna().tolist()
        if not injections:
            print("[ContextAwareLearned] Warning: no injection rows in train — skip.")
            return

        if self._embed_model is None:
            self._init_embedding_model(injections)
        else:
            print(
                f"[ContextAwareLearned] Reusing MiniLM "
                f"({len(injections)} train injections)."
            )

        self._encode_benign_embeddings(benign)

        if self._ppl_model is None:
            self._init_perplexity_model(injections, benign)
        else:
            print("[ContextAwareLearned] Reusing distilgpt2 perplexity calibration.")

        X_train = self._features_matrix(df_train["prompt"].astype(str).tolist())
        y_train = (df_train["label"] == "injection").astype(int).to_numpy()
        X_val = self._features_matrix(df_val["prompt"].astype(str).tolist())
        y_val = (df_val["label"] == "injection").astype(int).to_numpy()

        X_train_s = self._scaler.fit_transform(X_train)
        X_val_s = self._scaler.transform(X_val)

        self._model.fit(X_train_s, y_train)
        self._is_trained = True

        self.threshold, self.val_metrics = self._tune_threshold(X_val_s, y_val)
        coef = dict(zip(FEATURE_NAMES, self._model.coef_.ravel().tolist()))
        print(
            f"[ContextAwareLearned] Trained LR on {len(df_train)} prompts; "
            f"val-tuned threshold={self.threshold:.3f} "
            f"(F1={self.val_metrics.get('f1', 0):.3f}, "
            f"FPR={self.val_metrics.get('fpr', 0):.3f}, "
            f"Bypass={self.val_metrics.get('bypass', 0):.3f})."
        )
        top = sorted(coef.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        print(
            "[ContextAwareLearned] Top |coef|: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in top)
        )

    def _init_embedding_model(self, injections: list) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
            self._injection_embeddings = self._embed_model.encode(
                injections, convert_to_numpy=True, show_progress_bar=False
            )
            print(
                f"[ContextAwareLearned] MiniLM loaded — "
                f"encoded {len(injections)} injection embeddings."
            )
        except Exception as e:
            print(f"[ContextAwareLearned] Warning: MiniLM loading failed — {e}")
            self._embed_model = None
            self._injection_embeddings = None

    def _encode_benign_embeddings(self, benign: list) -> None:
        if self._embed_model is None or not benign:
            self._benign_embeddings = None
            return
        try:
            self._benign_embeddings = self._embed_model.encode(
                benign, convert_to_numpy=True, show_progress_bar=False
            )
            print(
                f"[ContextAwareLearned] Encoded {len(benign)} benign embeddings "
                f"for contrastive semantic score."
            )
        except Exception as e:
            print(f"[ContextAwareLearned] Warning: benign encode failed — {e}")
            self._benign_embeddings = None

    def _init_perplexity_model(self, injections: list, benign: list) -> None:
        """Same calibration approach as Method A (distilgpt2, p5–p95)."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "distilgpt2"
            self._ppl_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._device = "cpu"
            self._ppl_model = AutoModelForCausalLM.from_pretrained(model_name).to(
                self._device
            )
            self._ppl_model.eval()

            ppls = []
            for prompt in injections + benign:
                ppl = self._compute_perplexity(prompt)
                if ppl is not None:
                    ppls.append(ppl)

            if ppls:
                self._ppl_low = float(np.percentile(ppls, 5))
                self._ppl_high = float(np.percentile(ppls, 95))
                if self._ppl_high <= self._ppl_low:
                    self._ppl_high = self._ppl_low + 1.0
                print(
                    f"[ContextAwareLearned] Perplexity calibrated on {len(ppls)} "
                    f"prompts (p5={self._ppl_low:.1f}, p95={self._ppl_high:.1f})."
                )
            else:
                self._ppl_model = None
                self._ppl_tokenizer = None
        except Exception as e:
            print(f"[ContextAwareLearned] Warning: perplexity init failed — {e}")
            self._ppl_model = None
            self._ppl_tokenizer = None

    def _compute_perplexity(self, text: str):
        if self._ppl_model is None or self._ppl_tokenizer is None:
            return None
        try:
            import torch

            encodings = self._ppl_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self._device)
            input_ids = encodings["input_ids"]
            if input_ids.size(1) < 2:
                return None
            with torch.no_grad():
                outputs = self._ppl_model(input_ids, labels=input_ids)
                loss = outputs.loss
            return float(torch.exp(loss))
        except Exception:
            return None

    def _max_cosine(self, prompt_emb: np.ndarray, gallery: np.ndarray | None) -> float:
        if gallery is None or gallery.size == 0:
            return 0.0
        from numpy.linalg import norm

        p = prompt_emb / (norm(prompt_emb, axis=1, keepdims=True) + 1e-10)
        g = gallery / (norm(gallery, axis=1, keepdims=True) + 1e-10)
        sims = np.dot(g, p.T).flatten()
        return float(sims.max()) if sims.size else 0.0

    def _top_k_mean_cosine(
        self, prompt_emb: np.ndarray, gallery: np.ndarray | None, k: int = 3
    ) -> float:
        if gallery is None or gallery.size == 0:
            return 0.0
        from numpy.linalg import norm

        p = prompt_emb / (norm(prompt_emb, axis=1, keepdims=True) + 1e-10)
        g = gallery / (norm(gallery, axis=1, keepdims=True) + 1e-10)
        sims = np.dot(g, p.T).flatten()
        if sims.size == 0:
            return 0.0
        top = np.partition(sims, -min(k, sims.size))[-min(k, sims.size) :]
        return float(top.mean())

    def _semantic_features(self, prompt: str) -> tuple[float, float, float]:
        """Return (max_inj, top3_inj, contrast) as continuous scores."""
        if self._embed_model is None or self._injection_embeddings is None:
            return 0.0, 0.0, 0.0
        try:
            prompt_emb = self._embed_model.encode(
                [prompt], convert_to_numpy=True, show_progress_bar=False
            )
            inj_max = self._max_cosine(prompt_emb, self._injection_embeddings)
            inj_top3 = self._top_k_mean_cosine(
                prompt_emb, self._injection_embeddings, k=3
            )
            ben_max = self._max_cosine(prompt_emb, self._benign_embeddings)
            contrast = float(
                np.clip(inj_max - BENIGN_CONTRAST_LAMBDA * ben_max, 0.0, 1.0)
            )
            return inj_max, inj_top3, contrast
        except Exception:
            return 0.0, 0.0, 0.0

    def _perplexity_feature(self, prompt: str) -> float:
        if self._ppl_model is None or self._ppl_low is None:
            return 0.0
        ppl = self._compute_perplexity(prompt)
        if ppl is None:
            return 0.0
        score = (ppl - self._ppl_low) / (self._ppl_high - self._ppl_low)
        return float(np.clip(score, 0.0, 1.0))

    def _binary_any(self, prompt: str, patterns) -> float:
        return 1.0 if any(p.search(prompt) for p in patterns) else 0.0

    def extract_features(self, prompt: str) -> dict:
        """Continuous / lightly-overlapping feature dict (no hard triggers)."""
        _, regex_blocked = self._regex_san.sanitize(prompt)
        keyword = min(self._kw_san.score(prompt), 1.0)
        sem_max, sem_top3, sem_contrast = self._semantic_features(prompt)

        intent = self._binary_any(prompt, INTENT_PATTERNS)
        roleplay = self._binary_any(prompt, ContextAwareSanitizer._ROLEPLAY_PATTERNS)
        lowered = prompt.lower()
        shift = (
            1.0
            if any(p.search(lowered) for p in ContextAwareSanitizer._SHIFT_PATTERNS)
            else 0.0
        )
        conflict = (
            1.0
            if any(p.search(lowered) for p in ContextAwareSanitizer._CONFLICT_PATTERNS)
            else 0.0
        )
        ppl = self._perplexity_feature(prompt)

        feats = {
            "regex": 1.0 if regex_blocked else 0.0,
            "keyword": float(keyword),
            "semantic_max": float(sem_max),
            "semantic_top3": float(sem_top3),
            "semantic_contrast": float(sem_contrast),
            "intent": float(intent),
            "roleplay": float(roleplay),
            "shift": float(shift),
            "conflict": float(conflict),
            "perplexity": float(ppl),
        }
        for name in self._disabled_features:
            if name in feats:
                feats[name] = 0.0
        return feats

    def _features_matrix(self, prompts: list[str]) -> np.ndarray:
        rows = [self.extract_features(p) for p in prompts]
        return np.asarray(
            [[row[name] for name in FEATURE_NAMES] for row in rows],
            dtype=np.float64,
        )

    def _tune_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> tuple[float, dict]:
        """Pick threshold on val: best F1, optionally under a max FPR."""
        probs = self._model.predict_proba(X_val)[:, 1]
        candidates = np.unique(
            np.concatenate(
                [np.linspace(0.05, 0.95, 37), np.quantile(probs, np.linspace(0, 1, 21))]
            )
        )

        best_t = 0.5
        best_f1 = -1.0
        best_stats = {"f1": 0.0, "fpr": 1.0, "bypass": 1.0}

        for t in candidates:
            pred = (probs >= t).astype(int)
            tp = int(((pred == 1) & (y_val == 1)).sum())
            fp = int(((pred == 1) & (y_val == 0)).sum())
            fn = int(((pred == 0) & (y_val == 1)).sum())
            tn = int(((pred == 0) & (y_val == 0)).sum())
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            bypass = fn / (fn + tp) if (fn + tp) else 0.0
            f1 = float(f1_score(y_val, pred, zero_division=0))

            if self.target_fpr is not None and fpr > self.target_fpr:
                continue

            # Prefer higher F1; break ties with lower bypass, then lower FPR
            better = f1 > best_f1 + 1e-12 or (
                abs(f1 - best_f1) <= 1e-12
                and (
                    bypass < best_stats["bypass"] - 1e-12
                    or (
                        abs(bypass - best_stats["bypass"]) <= 1e-12
                        and fpr < best_stats["fpr"]
                    )
                )
            )
            if better:
                best_f1 = f1
                best_t = float(t)
                best_stats = {
                    "f1": round(f1, 4),
                    "fpr": round(fpr, 4),
                    "bypass": round(bypass, 4),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                }

        # If target_fpr filtered everything out, fall back to unconstrained F1
        if best_f1 < 0 and self.target_fpr is not None:
            print(
                "[ContextAwareLearned] Warning: no threshold met target_fpr; "
                "fell back to unconstrained F1."
            )
            saved = self.target_fpr
            self.target_fpr = None
            best_t, best_stats = self._tune_threshold(X_val, y_val)
            self.target_fpr = saved
            return best_t, best_stats

        return best_t, best_stats

    def score(self, prompt: str) -> dict:
        feats = self.extract_features(prompt)
        if not self._is_trained:
            feats["probability"] = 0.0
            feats["total"] = 0.0
            return feats
        x = np.asarray([[feats[n] for n in FEATURE_NAMES]], dtype=np.float64)
        x_s = self._scaler.transform(x)
        prob = float(self._model.predict_proba(x_s)[0, 1])
        feats["probability"] = prob
        feats["total"] = prob
        return feats

    def sanitize(self, prompt: str):
        signals = self.score(prompt)
        blocked = signals["probability"] >= self.threshold
        sanitized = (
            "[PROMPT BLOCKED — CONTEXT ANALYSIS (LEARNED)]" if blocked else prompt
        )
        return sanitized, blocked

    def copy_with(
        self,
        name: str = None,
        disabled_features: list[str] | tuple[str, ...] | set[str] | None = None,
    ) -> "ContextAwareLearnedSanitizer":
        """
        Lightweight ablation clone sharing trained LR / MiniLM / DistilGPT-2.

        disabled_features: zero these feature values at inference (leave-one-out).
        """
        clone = object.__new__(ContextAwareLearnedSanitizer)
        clone.name = name if name is not None else self.name
        clone.threshold = self.threshold
        clone.target_fpr = self.target_fpr
        clone._owns_models = False
        clone._regex_san = self._regex_san
        clone._kw_san = self._kw_san
        clone._embed_model = self._embed_model
        clone._injection_embeddings = self._injection_embeddings
        clone._benign_embeddings = self._benign_embeddings
        clone._ppl_model = self._ppl_model
        clone._ppl_tokenizer = self._ppl_tokenizer
        clone._ppl_low = self._ppl_low
        clone._ppl_high = self._ppl_high
        clone._device = self._device
        clone._scaler = self._scaler
        clone._model = self._model
        clone._is_trained = self._is_trained
        clone.val_metrics = dict(self.val_metrics)
        clone._disabled_features = set(disabled_features or [])
        return clone

    def release_models(self):
        """
        Drop auxiliary model refs.

        When models were shared from Method A, only clear local references;
        ContextAwareSanitizer.release_models() owns the actual free.
        """
        if self._owns_models:
            if self._embed_model is not None:
                del self._embed_model
            if self._ppl_model is not None:
                del self._ppl_model
                self._ppl_tokenizer = None
            from ..utils.gpu import empty_cuda_cache

            empty_cuda_cache()
            print("[ContextAwareLearned] Auxiliary models released from memory.")
        self._embed_model = None
        self._ppl_model = None
        self._ppl_tokenizer = None
