"""
Meta Llama Prompt Guard 2 — off-the-shelf portable gate (CPU).

Default model: meta-llama/Llama-Prompt-Guard-2-86M
Block when the predicted label is MALICIOUS (no retuning).
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"
# Prompt Guard context window; longer prompts are truncated (default HF path).
MAX_LENGTH = 512


class PromptGuardSanitizer:
    """Off-the-shelf Meta Prompt Guard classifier used as allow/block gate."""

    name = "prompt_guard"

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str | None = None,
    ):
        self.model_id = model_id
        if device is None:
            device = "cpu"
        self.device = torch.device(device)
        print(f"[PromptGuard] Loading {model_id} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(self.device)
        self.model.eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        print(f"[PromptGuard] Ready. labels={self.id2label}")

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
        raw = str(self.id2label.get(pred_id, "")).upper()
        # HF config ships LABEL_0/LABEL_1; Meta docs: 0=BENIGN, 1=MALICIOUS.
        # Also accept named labels from older Prompt Guard checkpoints.
        label = {
            "LABEL_0": "BENIGN",
            "LABEL_1": "MALICIOUS",
        }.get(raw, raw)
        blocked = label in {"MALICIOUS", "INJECTION", "JAILBREAK"}
        return text, blocked

    def release_models(self) -> None:
        del self.model
        del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
