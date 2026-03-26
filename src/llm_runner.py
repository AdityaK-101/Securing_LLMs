"""
src/llm_runner.py
=================
Local LLM runner using microsoft/phi-2 via HuggingFace Transformers.

Usage:
    runner = LLMRunner()
    output = runner.run("Your prompt here")

Set environment variable SKIP_LLM=1 ONLY if you want to skip model download.
By default, phi-2 is loaded and used for real inference.
"""

import os
import warnings

# Suppress noisy but harmless phi-2 deprecation warnings that fire on every call
warnings.filterwarnings("ignore", message=".*Both.*max_new_tokens.*max_length.*")
warnings.filterwarnings("ignore", message=".*Passing.*generation_config.*generation-related.*")
warnings.filterwarnings("ignore", message=".*generation flags are not valid.*temperature.*")
warnings.filterwarnings("ignore", message=".*torch_dtype.*deprecated.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*HF Hub.*")

import os
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")  # silence HF token nag
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "microsoft/phi-2"

# Instruction template per spec §4.2
INSTRUCTION_TEMPLATE = "### Instruction:\n{prompt}\n\n### Response:\n"

_pipeline = None


def _load_model():
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from transformers import logging as hf_logging
    import torch

    # Supress transformers verbose generation warnings
    hf_logging.set_verbosity_error()

    print(f"[LLMRunner] Loading {MODEL_NAME}...")
    print("[LLMRunner] First run: downloading model (~5.5 GB). Please wait...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    # Use float16 on GPU, float32 on CPU
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = 0 if torch.cuda.is_available() else -1

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    # phi-2 ships with max_length=20 in generation_config.json — clear it
    # to avoid the "Both max_new_tokens and max_length" warning on every call
    model.generation_config.max_length = None

    _pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        pad_token_id=tokenizer.eos_token_id,
        return_full_text=False,
    )

    print(f"[LLMRunner] {MODEL_NAME} ready.")
    return _pipeline


class LLMRunner:
    """
    Wraps microsoft/phi-2 for instruction-style inference.
    Falls back to mock ONLY if SKIP_LLM=1 is explicitly set.
    """

    def __init__(self):
        self.mock = os.environ.get("SKIP_LLM", "0").strip() == "1"
        self._pipe = None

        if not self.mock:
            try:
                self._pipe = _load_model()
            except Exception as e:
                print(f"[LLMRunner] ERROR loading phi-2: {e}")
                print("[LLMRunner] Falling back to mock. Set SKIP_LLM=1 to suppress this.")
                self.mock = True

    def run(self, prompt: str) -> str:
        """Format and run prompt through phi-2. Returns response text."""
        formatted = INSTRUCTION_TEMPLATE.format(prompt=prompt)

        if self.mock:
            return f"[LLM_MOCK] (set SKIP_LLM=0 or unset env var to use phi-2)"

        try:
            result = self._pipe(
                formatted,
                max_new_tokens=150,
                do_sample=False,
                pad_token_id=self._pipe.tokenizer.eos_token_id,
            )
            generated = result[0]["generated_text"].strip()
            # Return only the response part (stop at next ### if present)
            if "###" in generated:
                generated = generated.split("###")[0].strip()
            return generated if generated else "[empty response]"
        except Exception as e:
            return f"[LLM_ERROR] {e}"

    def is_mock(self) -> bool:
        return self.mock
