"""
src/sanitizers — Prompt injection defense strategies.

Four sanitizer classes, all implementing the common interface:
    sanitizer.sanitize(prompt: str) -> (sanitized: str, blocked: bool)

This __init__ re-exports every public name that was previously importable
from the monolithic ``src.sanitizers`` module, so existing code like
``from .sanitizers import ContextAwareSanitizer`` continues to work.
"""

from .base import BaselineSanitizer
from .context_aware import (
    CONTEXT_THRESHOLD,
    INTENT_PATTERNS,
    W1, W2, W3, W4, W5, W6, W7, W8,
    ContextAwareSanitizer,
)
from .keyword import (
    KEYWORD_THRESHOLD,
    KEYWORD_WEIGHTS,
    KeywordHeuristicSanitizer,
)
from .regex import (
    REGEX_PATTERNS,
    RegexSanitizer,
)


def get_all_sanitizers(train_csv: str = None):
    kwargs = {}
    if train_csv is not None:
        kwargs["train_csv"] = train_csv
    return [
        BaselineSanitizer(),
        RegexSanitizer(),
        KeywordHeuristicSanitizer(),
        ContextAwareSanitizer(**kwargs),
    ]


__all__ = [
    "BaselineSanitizer",
    "RegexSanitizer",
    "KeywordHeuristicSanitizer",
    "ContextAwareSanitizer",
    "get_all_sanitizers",
    "REGEX_PATTERNS",
    "KEYWORD_WEIGHTS",
    "KEYWORD_THRESHOLD",
    "INTENT_PATTERNS",
    "CONTEXT_THRESHOLD",
    "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8",
]
