"""Fold-trained DeBERTa prompt-injection classifier.

Fine-tunes protectai/deberta-v3-base-prompt-injection-v2 on the fold train
split, then tunes a decision threshold on validation (F1, then lower Bypass,
then lower FPR). Distinct from the published ProtectAI gate, which is not
retrained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from ..config import RANDOM_SEED, TRAIN_CSV, VAL_CSV

DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_LENGTH = 256
EPOCHS = 2
BATCH_SIZE = 8
LR = 2e-5


class _PromptDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class DebertaFinetunedSanitizer:
    """DeBERTa fine-tuned inside the fold; threshold tuned on validation."""

    name = "deberta_finetuned"

    def __init__(
        self,
        train_csv: str = TRAIN_CSV,
        val_csv: str = VAL_CSV,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
        _skip_train: bool = False,
    ):
        self.model_id = model_id
        self.threshold = 0.5
        self.val_metrics: dict = {}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = None
        self.model = None
        self._injection_label_id = 1
        if not _skip_train:
            self._train(train_csv, val_csv)

    def _resolve_injection_id(self) -> int:
        id2label = {int(k): str(v).upper() for k, v in self.model.config.id2label.items()}
        for i, name in id2label.items():
            if name in {"INJECTION", "INJECT", "MALICIOUS", "LABEL_1"}:
                return i
        # Fall back to label 1 if present.
        return 1 if 1 in id2label else max(id2label)

    def _train(self, train_csv: str, val_csv: str) -> None:
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        train_texts = df_train["prompt"].astype(str).tolist()
        train_y = (df_train["label"] == "injection").astype(int).tolist()
        val_texts = df_val["prompt"].astype(str).tolist()
        val_y = (df_val["label"] == "injection").astype(int).tolist()

        print(f"[DebertaFT] Loading {self.model_id} on {self.device}...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self.model.to(self.device)
        self._injection_label_id = self._resolve_injection_id()

        torch.manual_seed(RANDOM_SEED)
        train_loader = DataLoader(
            _PromptDataset(train_texts, train_y, self.tokenizer),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR)
        total_steps = max(1, EPOCHS * len(train_loader))
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
        )

        self.model.train()
        for epoch in range(EPOCHS):
            losses = []
            for batch in train_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # Map binary y to the model's injection / safe ids.
                # Assume id 0 is SAFE/BENIGN and injection id is self._injection_label_id.
                labels = batch["labels"]
                if self._injection_label_id != 1 or 0 not in self.model.config.id2label:
                    mapped = torch.where(
                        labels == 1,
                        torch.tensor(self._injection_label_id, device=self.device),
                        torch.tensor(0, device=self.device),
                    )
                    batch["labels"] = mapped
                out = self.model(**batch)
                loss = out.loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.item()))
            print(
                f"[DebertaFT] epoch {epoch + 1}/{EPOCHS} mean_loss={np.mean(losses):.4f}",
                flush=True,
            )

        self.model.eval()
        val_probs = self._injection_probs(val_texts)
        self.threshold, self.val_metrics = self._tune_threshold(
            val_probs, np.asarray(val_y, dtype=int)
        )
        print(
            f"[DebertaFT] val-tuned threshold={self.threshold:.3f} "
            f"(F1={self.val_metrics.get('f1', 0):.3f}, "
            f"FPR={self.val_metrics.get('fpr', 0):.3f}, "
            f"Bypass={self.val_metrics.get('bypass', 0):.3f}).",
            flush=True,
        )

    def _injection_probs(self, texts: list[str]) -> np.ndarray:
        probs = []
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), BATCH_SIZE):
                chunk = texts[i : i + BATCH_SIZE]
                enc = self.tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_LENGTH,
                    padding=True,
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                logits = self.model(**enc).logits
                p = torch.softmax(logits, dim=-1)[:, self._injection_label_id]
                probs.extend(p.detach().cpu().tolist())
        return np.asarray(probs, dtype=float)

    def _tune_threshold(self, probs: np.ndarray, y: np.ndarray) -> tuple[float, dict]:
        candidates = np.unique(
            np.concatenate([np.linspace(0.05, 0.95, 37), np.quantile(probs, np.linspace(0, 1, 21))])
        )
        best_t = 0.5
        best = {"f1": -1.0, "bypass": 1.0, "fpr": 1.0}
        for t in candidates:
            pred = (probs >= t).astype(int)
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
        """Injection-class probability after fine-tuning."""
        text = "" if prompt is None else str(prompt)
        if self.model is None or self.tokenizer is None:
            return 0.0
        return float(self._injection_probs([text])[0])

    def sanitize(self, prompt: str):
        text = "" if prompt is None else str(prompt)
        if self.model is None or self.tokenizer is None:
            return text, False
        prob = float(self._injection_probs([text])[0])
        return text, bool(prob >= self.threshold)

    def release_models(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
