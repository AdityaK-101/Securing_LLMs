"""
Evaluation pipeline package.

Groups the two-layer evaluation flow: pipeline, metrics, robustness, and ablation.
"""

from .ablation import (
    ABLATION_EXPORT_COLUMNS,
    AB_COMPARISON_VARIANTS,
    HARD_TRIGGER_ABLATION_VARIANTS,
    LEARNED_ABLATION_VARIANTS,
    WEIGHTED_ABLATION_VARIANTS,
    run_ab_comparison_study,
    run_hard_trigger_ablation_study,
    run_learned_ablation_study,
    run_weighted_ablation_study,
)
from .evaluate import evaluate, evaluate_sanitizer
from .metrics import attack_success, compute_metrics
from .pipeline import (
    release_all_sanitizer_models,
    run_two_phase_pipeline,
    should_skip_judge,
)
from .latency import (
    format_overhead_for_note,
    get_hardware_info,
    run_runtime_overhead_benchmark,
    save_runtime_overhead_report,
)
from .robustness import (
    EDGE_BENIGN_PROMPTS,
    PARAPHRASE_SUBS,
    robustness_adaptive,
    robustness_edge_benign,
    robustness_paraphrase,
)

__all__ = [
    "ABLATION_EXPORT_COLUMNS",
    "AB_COMPARISON_VARIANTS",
    "HARD_TRIGGER_ABLATION_VARIANTS",
    "LEARNED_ABLATION_VARIANTS",
    "WEIGHTED_ABLATION_VARIANTS",
    "EDGE_BENIGN_PROMPTS",
    "PARAPHRASE_SUBS",
    "evaluate",
    "evaluate_sanitizer",
    "attack_success",
    "compute_metrics",
    "release_all_sanitizer_models",
    "run_two_phase_pipeline",
    "should_skip_judge",
    "robustness_paraphrase",
    "robustness_adaptive",
    "robustness_edge_benign",
    "run_weighted_ablation_study",
    "run_hard_trigger_ablation_study",
    "run_learned_ablation_study",
    "run_ab_comparison_study",
    "run_runtime_overhead_benchmark",
    "save_runtime_overhead_report",
    "format_overhead_for_note",
    "get_hardware_info",
]
