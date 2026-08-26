"""
Evaluate ProtectAI DeBERTa prompt-injection classifier on the frozen test split
plus the edge-benign set.

Gate-only for Bypass / FPR / Edge FPR. True ASR reuses baseline COMPLIED labels
from suite_a logs (greedy target+judge). No retuning, no split changes.
"""

from __future__ import annotations

from src.evaluation.portable_gates import evaluate_static_gate
from src.sanitizers.protectai import ProtectAISanitizer


def main() -> None:
    evaluate_static_gate("protectai", ProtectAISanitizer)


if __name__ == "__main__":
    main()
