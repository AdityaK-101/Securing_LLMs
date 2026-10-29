"""
src/sanitizers — Prompt injection defense strategies.

Sanitizer classes all implement the common interface:
    sanitizer.sanitize(prompt: str) -> (sanitized: str, blocked: bool)

Method A: ContextAwareSanitizer (hand weights + hard triggers)
Method B: ContextAwareLearnedSanitizer (continuous features + LR + val threshold)
"""

from .base import BaselineSanitizer
from .context_aware import (
    CONTEXT_THRESHOLD,
    INTENT_PATTERNS,
    W1, W2, W3, W4, W5, W6, W7, W8,
    ContextAwareSanitizer,
)
from .context_aware_learned import (
    BENIGN_CONTRAST_LAMBDA,
    FEATURE_NAMES,
    ContextAwareLearnedSanitizer,
)
from .keyword import (
    KEYWORD_THRESHOLD,
    KEYWORD_WEIGHTS,
    KeywordHeuristicSanitizer,
)
from .deberta_finetuned import DebertaFinetunedSanitizer
from .prompt_guard import PromptGuardSanitizer
from .protectai import ProtectAISanitizer
from .regex import (
    REGEX_PATTERNS,
    RegexSanitizer,
)
from .similarity_only import SimilarityOnlySanitizer


def get_all_sanitizers(
    train_csv: str = None,
    val_csv: str = None,
    include_method_a: bool = True,
    include_method_b: bool = True,
    include_similarity_only: bool = False,
    include_deberta_finetuned: bool = False,
    feature_cache=None,
):
    """
    Return baseline + regex + keyword + optional Method A / Method B.

    Method B reuses MiniLM / distilgpt2 from Method A when both are included;
    otherwise Method B loads its own auxiliary models.

    ``include_similarity_only`` / ``include_deberta_finetuned`` add the
    fold-trained fair baselines used by the source-holdout protocol.

    ``feature_cache`` (optional) shares MiniLM embeddings / raw PPL across
    methods and folds when the same FeatureCache instance is reused.
    """
    if (
        not include_method_a
        and not include_method_b
        and not include_similarity_only
        and not include_deberta_finetuned
    ):
        raise ValueError(
            "At least one of include_method_a / include_method_b / "
            "include_similarity_only / include_deberta_finetuned must be True"
        )

    kwargs = {}
    if train_csv is not None:
        kwargs["train_csv"] = train_csv
    if feature_cache is not None:
        kwargs["feature_cache"] = feature_cache

    sanitizers = [
        BaselineSanitizer(),
        RegexSanitizer(),
        KeywordHeuristicSanitizer(),
    ]

    context = None
    if include_method_a:
        context = ContextAwareSanitizer(**kwargs)
        sanitizers.append(context)

    if include_method_b:
        learned_kwargs = dict(kwargs)
        if val_csv is not None:
            learned_kwargs["val_csv"] = val_csv
        share_from = context if include_method_a else None
        sanitizers.append(
            ContextAwareLearnedSanitizer(share_from=share_from, **learned_kwargs)
        )

    if include_similarity_only:
        sim_kwargs = dict(kwargs)
        if val_csv is not None:
            sim_kwargs["val_csv"] = val_csv
        sanitizers.append(
            SimilarityOnlySanitizer(share_embed_from=context, **sim_kwargs)
        )

    if include_deberta_finetuned:
        deb_kwargs = {}
        if train_csv is not None:
            deb_kwargs["train_csv"] = train_csv
        if val_csv is not None:
            deb_kwargs["val_csv"] = val_csv
        sanitizers.append(DebertaFinetunedSanitizer(**deb_kwargs))

    return sanitizers


__all__ = [
    "BaselineSanitizer",
    "RegexSanitizer",
    "KeywordHeuristicSanitizer",
    "ContextAwareSanitizer",
    "ContextAwareLearnedSanitizer",
    "ProtectAISanitizer",
    "PromptGuardSanitizer",
    "SimilarityOnlySanitizer",
    "DebertaFinetunedSanitizer",
    "get_all_sanitizers",
    "REGEX_PATTERNS",
    "KEYWORD_WEIGHTS",
    "KEYWORD_THRESHOLD",
    "INTENT_PATTERNS",
    "CONTEXT_THRESHOLD",
    "FEATURE_NAMES",
    "BENIGN_CONTRAST_LAMBDA",
    "W1", "W2", "W3", "W4", "W5", "W6", "W7", "W8",
]
