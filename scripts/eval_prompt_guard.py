"""
Evaluate Meta Llama Prompt Guard 2 on the frozen test split + edge-benign set.

Gate-only for Bypass / FPR / Edge FPR. True ASR reuses baseline COMPLIED labels
from suite_a logs (greedy target+judge), so we do not retune or re-split.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.portable_gates import evaluate_static_gate
from src.sanitizers.prompt_guard import PromptGuardSanitizer


def main() -> None:
    evaluate_static_gate("prompt_guard", PromptGuardSanitizer)


if __name__ == "__main__":
    main()
