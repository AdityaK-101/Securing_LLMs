# Source-holdout gate-only results

**Not the paper main tree.**

Gate-only Bypass / FPR from `python scripts/eval_loo_holdout.py`.  
Committed CSVs here may be stale relative to the current uniform holdout splits.
Do **not** cite these numbers in the AAAI paper.

| Want | Path / command |
|------|----------------|
| **Main LOO (cite)** | `results/loo_matched_fpr/` ← `python scripts/run_paper.py` or `--loo-matched-fpr` |
| Default-thr LOO contrast | `results/loo/` ← `--loo` |
| Refresh this gate-only folder | `python scripts/eval_loo_holdout.py` |

Outputs here: `metrics.csv`, `summary.txt`, `logs_<fold>.csv`, `portable_prompt_decisions.csv`.
