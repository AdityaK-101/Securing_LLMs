"""
Model wrappers used by the experiment pipeline.

No shared abstract base class — each wrapper keeps its own load/release logic
to avoid behavioural changes.
"""

from .attack_generator import (
    EVASION_PROMPTS,
    STYLE_PROMPTS,
    generate_adaptive_variants,
    generate_attack,
    generate_paraphrase_rewrite,
    generate_similarity_evasion_attack,
    load_attack_gallery,
    max_gallery_similarity,
    release_generator,
)
from .judge import (
    CALIBRATION_CASES,
    JUDGE_SYSTEM_PROMPT,
    ComplianceJudge,
    parse_judge_output,
    release_judge,
    run_calibration,
)
from .target_llm import INSTRUCTION_TEMPLATE, LLMRunner, MODEL_NAME, release_model

__all__ = [
    "LLMRunner",
    "MODEL_NAME",
    "INSTRUCTION_TEMPLATE",
    "release_model",
    "ComplianceJudge",
    "JUDGE_SYSTEM_PROMPT",
    "CALIBRATION_CASES",
    "parse_judge_output",
    "release_judge",
    "run_calibration",
    "STYLE_PROMPTS",
    "EVASION_PROMPTS",
    "generate_attack",
    "generate_adaptive_variants",
    "generate_paraphrase_rewrite",
    "generate_similarity_evasion_attack",
    "load_attack_gallery",
    "max_gallery_similarity",
    "release_generator",
]
