"""
src/models/target_llm.py
========================
Local target LLM runner via HuggingFace Transformers.

Usage:
    runner = LLMRunner()
    output = runner.run("Your prompt here")

Set environment variable SKIP_LLM=1 ONLY if you want to skip model download.
By default, the target model is loaded and used for real inference.
"""

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

_pipeline = None


def release_model():
    """Unload target model to free memory before loading the judge."""
    global _pipeline
    if _pipeline is None:
        return
    try:
        if hasattr(_pipeline, "model"):
            del _pipeline.model
        if hasattr(_pipeline, "tokenizer"):
            del _pipeline.tokenizer
    except Exception:
        pass
    del _pipeline
    _pipeline = None
    empty_cuda_cache()
    print(f"[LLMRunner] {MODEL_NAME} released from memory.")


def _load_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence HF token nag
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import logging as hf_logging
    import torch

    # Supress transformers verbose generation warnings
    hf_logging.set_verbosity_error()

    print(f"[LLMRunner] Loading {MODEL_NAME}...")
    print("[LLMRunner] First run: downloading model (~5.5 GB). Please wait...")
    import sys
    sys.stdout.flush()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # Use float16 on GPU, float32 on CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 0 if torch.cuda.is_available() else -1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        return_full_text=False,
    )

    print(f"[LLMRunner] {MODEL_NAME} ready.")
    return _pipeline


class LLMRunner:
    """
    Wraps the target LLM for instruction-style inference.
    Falls back to mock ONLY if SKIP_LLM=1 is explicitly set.
    """

    def __init__(self):
        self.mock = os.environ.get("SKIP_LLM", "0").strip() == "1"
        self._pipe = None

        if not self.mock:
            try:
                self._pipe = _load_model()
            except MemoryError as e:
                print(f"[LLMRunner] OUT OF MEMORY loading {MODEL_NAME}: {e}")
                print("[LLMRunner] Close other apps or set SKIP_LLM=1 for sanitizer-only mode.")
                raise
            except Exception as e:
                print(f"[LLMRunner] ERROR loading {MODEL_NAME}: {e}")
                print("[LLMRunner] Falling back to mock. Set SKIP_LLM=1 to suppress this.")
                self.mock = True

    def run(self, prompt: str) -> str:
        """Run prompt through the configured target LLM."""

        if self.mock:
            return f"[LLM_MOCK] (set SKIP_LLM=0 or unset env var to use {MODEL_NAME})"

        try:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                }
            ]

            formatted = self._pipe.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            result = self._pipe(
                formatted,
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

    def is_mock(self) -> bool:
        return self.mock

    def release(self):
        """Free target model weights from memory."""
        self._pipe = None
        if not self.mock:
            release_model()
