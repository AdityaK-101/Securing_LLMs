"""
src/models/target_llm.py
========================
Local target LLM runner via HuggingFace Transformers.

Usage:
    runner = LLMRunner()  # default: TARGET_MODEL_NAME (Qwen2.5-3B-Instruct)
    runner = LLMRunner(model_name="meta-llama/Llama-3.2-3B-Instruct")
    output = runner.run("Your prompt here")

Same decoding for every target: greedy, max_new_tokens=512, chat template
with a single user message (no system prompt).

Set SKIP_LLM=1 ONLY if you want to skip model download.
"""

from __future__ import annotations

import os
import warnings

# Suppress noisy but harmless generation deprecation warnings that fire on every call
warnings.filterwarnings("ignore", message=".*Both.*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*Passing.*generation_config.*generation-related.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*temperature.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*")

from ..config import TARGET_MODEL_NAME
from ..utils.gpu import empty_cuda_cache

MODEL_NAME = TARGET_MODEL_NAME

# Legacy export for notebooks that still reference the old instruction template.
INSTRUCTION_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"

# One pipeline per model id (supports primary + second target without clash).
_pipelines: dict[str, object] = {}


def release_model(model_name: str | None = None):
    """Unload one target model, or all if ``model_name`` is None."""
    global _pipelines
    names = [model_name] if model_name else list(_pipelines.keys())
    for name in names:
        if not name or name not in _pipelines:
            continue
        pipe = _pipelines.pop(name)
        try:
            if hasattr(pipe, "model"):
                del pipe.model
            if hasattr(pipe, "tokenizer"):
                del pipe.tokenizer
        except Exception:
            pass
        del pipe
        print(f"[LLMRunner] {name} released from memory.")
    empty_cuda_cache()


def _load_model(model_name: str):
    if model_name in _pipelines:
        return _pipelines[model_name]

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import logging as hf_logging
    import torch

    hf_logging.set_verbosity_error()

    print(f"[LLMRunner] Loading {model_name}...")
    print("[LLMRunner] First run may download weights. Please wait...")
    import sys

    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 0 if torch.cuda.is_available() else -1

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_full_text=False,
    )
    _pipelines[model_name] = pipe
    print(f"[LLMRunner] {model_name} ready.")
    return pipe


class LLMRunner:
    """
    Wraps a target LLM for instruction-style inference.
    Falls back to mock ONLY if SKIP_LLM=1 is explicitly set.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or TARGET_MODEL_NAME
        self.mock = os.environ.get("SKIP_LLM", "0").strip() == "1"
        self._pipe = None

        if not self.mock:
            try:
                self._pipe = _load_model(self.model_name)
            except MemoryError as e:
                print(f"[LLMRunner] OUT OF MEMORY loading {self.model_name}: {e}")
                print(
                    "[LLMRunner] Close other apps or set SKIP_LLM=1 for sanitizer-only mode."
                )
                raise
            except Exception as e:
                print(f"[LLMRunner] ERROR loading {self.model_name}: {e}")
                print(
                    "[LLMRunner] Refusing silent mock fallback. "
                    "Fix the load error, or set SKIP_LLM=1 / pass --no-llm for sanitizer-only mode."
                )
                raise RuntimeError(
                    f"Failed to load target LLM {self.model_name}. "
                    "Set SKIP_LLM=1 or use --no-llm for sanitizer-only runs."
                ) from e

    def _format_chat(self, prompt: str) -> str:
        # Single user turn, no system prompt (same protocol for every target).
        messages = [{"role": "user", "content": prompt}]
        return self._pipe.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def run(self, prompt: str) -> str:
        """Run prompt through the configured target LLM (greedy, ≤512 new tokens)."""

        if self.mock:
            return (
                f"[LLM_MOCK] (set SKIP_LLM=0 or unset env var to use {self.model_name})"
            )

        try:
            result = self._pipe(
                self._format_chat(prompt),
                max_new_tokens=512,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
            )
            generated = result[0]["generated_text"].strip()
            return generated if generated else "[empty response]"
        except Exception as e:
            return f"[LLM_ERROR] {e}"

    def run_batch(self, prompts: list[str], batch_size: int | None = None) -> list[str]:
        """Greedy batched generation (same decoding as ``run``)."""
        if self.mock:
            return [
                f"[LLM_MOCK] (set SKIP_LLM=0 or unset env var to use {self.model_name})"
                for _ in prompts
            ]
        if not prompts:
            return []
        bs = int(batch_size) if batch_size else len(prompts)
        bs = max(1, bs)
        outs: list[str] = []
        try:
            for start in range(0, len(prompts), bs):
                chunk = prompts[start : start + bs]
                formatted = [self._format_chat(p) for p in chunk]
                results = self._pipe(
                    formatted,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    batch_size=len(formatted),
                    pad_token_id=self._pipe.tokenizer.eos_token_id,
                )
                for item in results:
                    row = item[0] if isinstance(item, list) else item
                    text = str(row.get("generated_text", "")).strip()
                    outs.append(text if text else "[empty response]")
        except Exception as e:
            print(f"[LLMRunner] batch failed ({e}); falling back to sequential.")
            outs = [self.run(p) for p in prompts]
        return outs

    def is_mock(self) -> bool:
        return self.mock

    def release(self):
        """Free this runner's model weights from memory."""
        self._pipe = None
        if not self.mock:
            release_model(self.model_name)
