"""
One-shot research-paper results pipeline.

Default (``core``) builds everything needed for the main holdout report:

  1. Verify leave-one-source-out splits
  2. Mixed suite matched-FPR (combined-dataset / ID column)
  3. LOO matched-FPR ≈ 5% (main suite + robustness + curves)
  4. ProtectAI + Prompt Guard 2 on LOO at matched thresholds
  5. Paper tables (Wilson CIs, Neuralchemy, ID, dedup, train-benign)

``full`` also runs: default-thr LOO contrast, mixed suites A/B, ablations,
portable static baselines, judge human-validation score.

Usage:
  python scripts/run_paper.py
  python scripts/run_paper.py --mode core
  python scripts/run_paper.py --mode full
  python scripts/run_paper.py --skip-loo-matched-fpr   # tables + portables only
  python -m src.run_experiments --paper

Resume policy (all steps): reuse only LLM/judge + feature (+ FLAN text) caches;
filters/metrics always rescore. ``--rerun`` / ``--no-outcome-cache`` regenerates
Qwen/judge replies.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS_DIR, SOURCE_HOLDOUT_DIR


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _run(cmd: list[str], *, step: str) -> None:
    _banner(step)
    print(">", " ".join(cmd))
    t0 = time.time()
    subprocess.run(cmd, check=True, cwd=str(ROOT))
    print(f"[OK] {step}  ({time.time() - t0:.0f}s)")


def _py(*args: str) -> list[str]:
    return [sys.executable, *args]


def run_paper(
    *,
    mode: str = "core",
    skip_loo_matched_fpr: bool = False,
    skip_portables: bool = False,
    skip_deberta: bool = False,
    skip_robustness: bool = False,
    rerun: bool = False,
    llm_batch_size: int | None = None,
    no_llm: bool = False,
) -> None:
    """Execute the paper pipeline. Raises on first failing step."""
    mode = mode.lower().strip()
    if mode not in {"core", "full"}:
        raise ValueError("mode must be 'core' or 'full'")

    py = sys.executable
    holdout_ok = (SOURCE_HOLDOUT_DIR / "manifest.json").is_file()
    if not holdout_ok:
        _run(
            _py("scripts/build_source_holdout.py"),
            step="Build source-holdout splits",
        )
    _run(_py("scripts/verify_source_holdout.py"), step="Verify source-holdout")

    # --- CORE ---
    matched_cmd = _py("-m", "src.run_experiments", "--matched-fpr")
    if skip_deberta:
        matched_cmd.append("--matched-fpr-skip-deberta")
    _run(matched_cmd, step="Mixed suite matched-FPR (combined split)")

    if not skip_loo_matched_fpr:
        loo_cmd = _py("-m", "src.run_experiments", "--loo-matched-fpr")
        if skip_deberta:
            loo_cmd.append("--matched-fpr-skip-deberta")
        if skip_robustness:
            loo_cmd.append("--loo-skip-robustness")
        if llm_batch_size is not None:
            loo_cmd.extend(["--llm-batch-size", str(llm_batch_size)])
        if no_llm:
            loo_cmd.append("--no-llm")
        if rerun:
            loo_cmd.append("--no-outcome-cache")
        _run(loo_cmd, step="LOO matched-FPR ≈ 5% (main report)")

    if not skip_portables:
        port_cmd = _py("scripts/eval_loo_matched_fpr_portables.py")
        if skip_deberta:
            # still score both portables; flag unused here
            pass
        _run(port_cmd, step="ProtectAI + Prompt Guard 2 on LOO matched-FPR")
    else:
        _run(
            _py("scripts/build_loo_matched_fpr_paper_tables.py"),
            step="Build paper tables (without re-scoring portables)",
        )

    # Portables script already rebuilds PAPER_TABLES; rebuild again for safety
    # if portables ran (idempotent).
    if not skip_portables:
        _run(
            _py("scripts/build_loo_matched_fpr_paper_tables.py"),
            step="Refresh PAPER_TABLES.md",
        )

    if mode == "full":
        loo_def = _py("-m", "src.run_experiments", "--loo")
        if skip_robustness:
            loo_def.append("--loo-skip-robustness")
        if llm_batch_size is not None:
            loo_def.extend(["--llm-batch-size", str(llm_batch_size)])
        if no_llm:
            loo_def.append("--no-llm")
        if rerun:
            loo_def.append("--no-outcome-cache")
        _run(loo_def, step="Default/F1 LOO contrast")

        suite = _py("-m", "src.run_experiments", "--suite", "ab")
        if no_llm:
            suite.append("--no-llm")
        if llm_batch_size is not None:
            suite.extend(["--llm-batch-size", str(llm_batch_size)])
        if rerun:
            suite.append("--no-outcome-cache")
        _run(suite, step="Mixed suites A + B")

        for flag, label in (
            ("--hard-trigger-ablation", "Hard-trigger ablation"),
            ("--weighted-ablation", "Weighted signal ablation"),
            ("--learned-ablation", "Learned feature ablation"),
            ("--compare-ab", "A vs B matched comparison"),
        ):
            abl = _py("-m", "src.run_experiments", flag)
            if no_llm:
                abl.append("--no-llm")
            _run(abl, step=label)

        _run(
            _py("-m", "src.run_experiments", "--portable-baselines"),
            step="Portable baselines (static + adaptive gate scripts)",
        )
        _run(
            _py("scripts/score_judge_human_validation.py"),
            step="Judge human-validation score",
        )

    paper = RESULTS_DIR / "loo_matched_fpr" / "PAPER_TABLES.md"
    _banner("Paper pipeline complete")
    print(f"Mode: {mode}")
    print(f"Main tables: {paper}")
    print(f"Folder map:  {RESULTS_DIR / 'README.md'}")
    print(f"Index:       {RESULTS_DIR / 'PAPER.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-shot research-paper results pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["core", "full"],
        default="core",
        help=(
            "core = main holdout report + paper tables (default); "
            "full = core + default LOO + suites + ablations + portables + judge HV"
        ),
    )
    parser.add_argument(
        "--skip-loo-matched-fpr",
        action="store_true",
        help="Skip the long LOO matched-FPR run (use existing results).",
    )
    parser.add_argument(
        "--skip-portables",
        action="store_true",
        help="Skip ProtectAI / Prompt Guard LOO scoring.",
    )
    parser.add_argument(
        "--matched-fpr-skip-deberta",
        action="store_true",
        help="Skip DeBERTa fine-tune in matched-FPR / LOO matched-FPR.",
    )
    parser.add_argument(
        "--loo-skip-robustness",
        action="store_true",
        help="Skip paraphrase / edge / adaptive on LOO runs.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help=(
            "Regenerate Qwen/judge replies (--no-outcome-cache). "
            "Filters/metrics always rescore anyway."
        ),
    )
    parser.add_argument("--llm-batch-size", type=int, default=None)
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Sanitizer-only (True ASR will be 0 / missing). Fast smoke.",
    )
    args = parser.parse_args()
    run_paper(
        mode=args.mode,
        skip_loo_matched_fpr=args.skip_loo_matched_fpr,
        skip_portables=args.skip_portables,
        skip_deberta=args.matched_fpr_skip_deberta,
        skip_robustness=args.loo_skip_robustness,
        rerun=args.rerun,
        llm_batch_size=args.llm_batch_size,
        no_llm=args.no_llm,
    )


if __name__ == "__main__":
    main()
