"""
ProtectAI DeBERTa prompt-injection classifier — off-the-shelf portable gate (CPU).

Default model: protectai/deberta-v3-base-prompt-injection-v2
Block when predicted label is INJECTION (no retuning).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "protectai/deberta-v3-base-prompt-injection-v2"
MAX_LENGTH = 512


class ProtectAISanitizer:
    """Off-the-shelf ProtectAI DeBERTa classifier used as allow/block gate."""

    name = "protectai"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ):
        self.model_id = model_id
        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        print(f"[ProtectAI] Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"[ProtectAI] Ready. labels={self.id2label}")

    def sanitize(self, prompt: str):
        text = "" if prompt is None else str(prompt)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = int(logits.argmax(dim=-1).item())
        label = str(self.id2label.get(pred_id, "")).upper()
        # v2 labels: SAFE / INJECTION (also tolerate legacy BENIGN / LABEL_1)
        blocked = label in {"INJECTION", "INJECT", "MALICIOUS", "LABEL_1", "1"}
        return text, blocked

    def release_models(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
