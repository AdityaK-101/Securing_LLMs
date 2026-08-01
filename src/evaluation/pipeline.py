"""
Shared two-phase evaluation pipeline: Sanitizer → LLM → Judge.

Extracted from evaluate.py to eliminate duplicated Phase 1 + Phase 2 logic.
"""

import os
from typing import Callable, Iterable, Optional

from tqdm import tqdm

from ..models.judge import ComplianceJudge
from ..utils.gpu import empty_cuda_cache
from .metrics import attack_success


def should_skip_judge(use_llm: bool, runner) -> bool:
    return (
        os.environ.get("SKIP_JUDGE", "0").strip() == "1"
        or not use_llm
        or runner is None
        or runner.is_mock()
    )


def release_all_sanitizer_models(sanitizers) -> None:
    for san in sanitizers:
        if hasattr(san, "release_models"):
            san.release_models()


def run_two_phase_pipeline(
    items: Iterable,
    sanitizers: list,
    runner,
    skip_judge: bool,
    *,
    get_record_id: Callable,
    get_prompt: Callable,
    get_label: Callable,
    defer_judge: Callable[[str, bool], bool],
    extra_record_fields: Optional[Callable] = None,
    release_sanitizers: Optional[Callable] = None,
    phase1_desc: str = "Phase 1 (target LLM)",
    phase1_total: Optional[int] = None,
    phase2_log_prefix: str = "[Evaluate]",
    phase2_item_name: str = "unblocked injections",
    cleanup_after_phase1: Optional[Callable] = None,
) -> list:
    """
    Run Phase 1 (sanitize + target LLM) then Phase 2 (judge) on pending records.

    Returns a list of per-prompt-per-method record dicts.
    """
    if release_sanitizers is None:
        release_sanitizers = lambda: release_all_sanitizer_models(sanitizers)

    records = []
    pending_judge = []  # record indices needing judge evaluation

    for item in tqdm(items, total=phase1_total, desc=phase1_desc):
        prompt_id = get_record_id(item)
        prompt = get_prompt(item)
        label = get_label(item)

        for san in sanitizers:
            sanitized, blocked = san.sanitize(prompt)

            if blocked:
                llm_output = "[BLOCKED]"
                judge_label = "BLOCKED"
                judge_reasoning = ""
            elif runner:
                llm_output = runner.run(sanitized)
                if defer_judge(label, skip_judge):
                    judge_label = None  # filled in Phase 2
                    judge_reasoning = ""
                elif label == "injection":
                    judge_label = "SKIPPED"
                    judge_reasoning = ""
                else:
                    judge_label = "N/A"
                    judge_reasoning = ""
            else:
                llm_output = "[LLM_SKIPPED]"
                judge_label = "SKIPPED"
                judge_reasoning = ""

            rec = {
                "id":               prompt_id,
                "label":            label,
                "method":           san.name,
                "blocked":          blocked,
                "original":         prompt,
                "sanitized":        sanitized,
                "llm_output":       llm_output,
                "judge_label":      judge_label,
                "judge_reasoning":  judge_reasoning,
                "attack_success":   False,
            }
            if extra_record_fields is not None:
                rec.update(extra_record_fields(item))
            if judge_label is None:
                pending_judge.append(len(records))
            elif judge_label not in (None, "N/A", "SKIPPED"):
                rec["attack_success"] = attack_success(label, blocked, judge_label)
            records.append(rec)

    # Phase 2: free GPU, load judge (never alongside target LLM or sanitizer models)
    if pending_judge and not skip_judge:
        print(
            f"\n{phase2_log_prefix} Phase 2: judging "
            f"{len(pending_judge)} {phase2_item_name}..."
        )
        release_sanitizers()
        if runner:
            runner.release()
        empty_cuda_cache()

        judge = ComplianceJudge()

        if judge.is_mock():
            for idx in pending_judge:
                records[idx]["judge_label"] = "SKIPPED"
        else:
            for idx in tqdm(pending_judge, desc="Phase 2 (judge)"):
                rec = records[idx]
                judge_label, reasoning = judge.evaluate(rec["original"], rec["llm_output"])
                rec["judge_label"] = judge_label
                rec["judge_reasoning"] = reasoning
                rec["attack_success"] = attack_success(
                    rec["label"], rec["blocked"], judge_label
                )

        judge.release()
        empty_cuda_cache()
    elif cleanup_after_phase1 is not None:
        cleanup_after_phase1()

    return records
