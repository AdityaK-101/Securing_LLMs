# Source-holdout gate-only results

**Gate-only** Bypass / FPR from `python scripts/eval_loo_holdout.py`.  
Committed CSVs here may be stale relative to the current uniform holdout splits.

**Main LOO report** (FPR ≈ 5%, full suite):

```powershell
python -m src.run_experiments --loo-matched-fpr
```

→ `results/loo_matched_fpr/` (root README §6.0).

**Default-threshold LOO** (contrast; complete):

```powershell
python -m src.run_experiments --loo
```

→ `results/loo/` (root README §6.0b).

Rebuild gate-only scores after changing splits:

```powershell
python scripts/eval_loo_holdout.py
```

Outputs: `metrics.csv`, `summary.txt`, `logs_<fold>.csv`, `portable_prompt_decisions.csv`.
