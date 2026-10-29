"""
Shared two-phase evaluation pipeline: Sanitizer → LLM → Judge.

Default path still runs LLM per unblocked (method, prompt). Prefer
``run_cached_lookup_pipeline`` for LOO / large tests: one greedy Qwen+judge
pass per attack prompt, benign = filters only, resumable disk cache.
"""

import os
from typing import Callable, Iterable, Optional

from tqdm import tqdm

from ..models.judge import ComplianceJudge
from ..utils.gpu import empty_cuda_cache
from .metrics import attack_success
from .prompt_cache import OutcomeCache


def should_skip_judge(use_llm: bool, runner=None) -> bool:
    """Skip judge when disabled, no LLM, or an explicit mock runner."""
    if os.environ.get("SKIP_JUDGE", "0").strip() == "1":
        return True
    if not use_llm:
        return True
    if runner is not None and hasattr(runner, "is_mock") and runner.is_mock():
        return True
    return False


def release_all_sanitizer_models(sanitizers) -> None:
    for san in sanitizers:
        if hasattr(san, "release_models"):
            san.release_models()


def run_cached_lookup_pipeline(
    items: Iterable,
    sanitizers: list,
    *,
    outcome_cache: OutcomeCache,
    use_llm: bool,
    skip_judge: bool,
    get_record_id: Callable,
    get_prompt: Callable,
    get_label: Callable,
    extra_record_fields: Optional[Callable] = None,
    release_sanitizers: Optional[Callable] = None,
    batch_size: int = 4,
    phase_log_prefix: str = "[Evaluate]",
    items_total: Optional[int] = None,
) -> list:
    """
    Mentor speedup path:

    - Collect unique *injection* prompts; generate all greedy replies once;
      unload Qwen; judge all; save to ``outcome_cache`` as you go.
    - Benign prompts: filters only (never Qwen / judge).
    - Per method: sanitize, then True ASR = lookup of the shared judge label
      when the filter did not block (reply is filter-independent under greedy
      decoding and pass-through sanitizers).
    """
    if release_sanitizers is None:
        release_sanitizers = lambda: release_all_sanitizer_models(sanitizers)

    item_list = list(items)
    injection_prompts = [
        get_prompt(item)
        for item in item_list
        if str(get_label(item)).lower() == "injection"
    ]

    if use_llm and injection_prompts:
        from ..models.target_llm import LLMRunner

        print(
            f"\n{phase_log_prefix} Cache path: generate-all → unload → judge-all "
            f"({len(set(map(str, injection_prompts)))} unique injections)..."
        )
        runner = LLMRunner()
        try:
            outcome_cache.ensure_llm(
                injection_prompts, runner, batch_size=batch_size
            )
        finally:
            runner.release()
            empty_cuda_cache()

        if not skip_judge:
            judge = ComplianceJudge()
            try:
                outcome_cache.ensure_judge(injection_prompts, judge)
            finally:
                judge.release()
                empty_cuda_cache()

    print(
        f"\n{phase_log_prefix} Filter pass + outcome lookup "
        f"({len(item_list)} prompts × {len(sanitizers)} methods)..."
    )
    records = []
    for item in tqdm(item_list, total=items_total or len(item_list), desc="Filter+lookup"):
        prompt_id = get_record_id(item)
        prompt = get_prompt(item)
        label = str(get_label(item))
        is_injection = label.lower() == "injection"

        for san in sanitizers:
            sanitized, blocked = san.sanitize(prompt)

            if not is_injection:
                # Benign: filters only
                llm_output = "[BENIGN_NO_LLM]"
                judge_label = "N/A"
                judge_reasoning = ""
                success = False
            elif blocked:
                llm_output = "[BLOCKED]"
                judge_label = "BLOCKED"
                judge_reasoning = ""
                success = False
            elif not use_llm:
                llm_output = "[LLM_SKIPPED]"
                judge_label = "SKIPPED"
                judge_reasoning = ""
                success = False
            else:
                llm_output, judge_label, judge_reasoning = outcome_cache.lookup(prompt)
                if skip_judge and judge_label not in ("BLOCKED", "N/A"):
                    judge_label = "SKIPPED"
                    judge_reasoning = ""
                success = attack_success(label, False, judge_label)

            rec = {
                "id": prompt_id,
                "label": label,
                "method": san.name,
                "blocked": blocked,
                "original": prompt,
                "sanitized": sanitized,
                "llm_output": llm_output,
                "judge_label": judge_label,
                "judge_reasoning": judge_reasoning,
                "attack_success": success,
            }
            if extra_record_fields is not None:
                rec.update(extra_record_fields(item))
            records.append(rec)

    release_sanitizers()
    empty_cuda_cache()
    return records


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
    outcome_cache: Optional[OutcomeCache] = None,
    use_cached_lookup: bool = False,
    cache_use_llm: Optional[bool] = None,
    batch_size: int = 4,
) -> list:
    """
    Run Phase 1 (sanitize + target LLM) then Phase 2 (judge) on pending records.

    When ``use_cached_lookup=True``, delegates to ``run_cached_lookup_pipeline``
    (benign skip LLM; one generate+judge pass per unique injection).

    Returns a list of per-prompt-per-method record dicts.
    """
    if use_cached_lookup:
        cache = outcome_cache or OutcomeCache()
        # Prefer explicit flag; else infer from runner (legacy callers)
        if cache_use_llm is None:
            cache_use_llm = runner is not None and not (
                hasattr(runner, "is_mock") and runner.is_mock()
            )
        if runner is not None and hasattr(runner, "release"):
            runner.release()
            empty_cuda_cache()
        return run_cached_lookup_pipeline(
            items,
            sanitizers,
            outcome_cache=cache,
            use_llm=bool(cache_use_llm),
            skip_judge=skip_judge,
            get_record_id=get_record_id,
            get_prompt=get_prompt,
            get_label=get_label,
            extra_record_fields=extra_record_fields,
            release_sanitizers=release_sanitizers,
            batch_size=batch_size,
            phase_log_prefix=phase2_log_prefix,
            items_total=phase1_total,
        )

    if release_sanitizers is None:
        release_sanitizers = lambda: release_all_sanitizer_models(sanitizers)

    records = []
    pending_judge = []  # record indices needing judge evaluation

    for item in tqdm(items, total=phase1_total, desc=phase1_desc):
        prompt_id = get_record_id(item)
        prompt = get_prompt(item)
        label = get_label(item)
        is_injection = str(label).lower() == "injection"

        for san in sanitizers:
            sanitized, blocked = san.sanitize(prompt)

            if not is_injection:
                # Benign prompts only need filters (mentor speedup)
                llm_output = "[BENIGN_NO_LLM]"
                judge_label = "N/A"
                judge_reasoning = ""
            elif blocked:
                llm_output = "[BLOCKED]"
                judge_label = "BLOCKED"
                judge_reasoning = ""
            elif runner:
                llm_output = runner.run(sanitized)
                if defer_judge(label, skip_judge):
                    judge_label = None  # filled in Phase 2
                    judge_reasoning = ""
                else:
                    judge_label = "SKIPPED"
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
