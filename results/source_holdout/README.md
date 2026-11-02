# Source-holdout gate-only results

**Gate-only** Bypass / FPR from `python scripts/eval_loo_holdout.py`.  
Committed CSVs here may be stale relative to the current uniform holdout splits.

**Full LLM LOO (complete)** — Bypass, True ASR, FPR, paraphrase, edge-benign, adaptive:

```powershell
python -m src.run_experiments --loo
```

→ `results/loo/` (see `results/loo/README.md` and root `README.md` §6.0).

Rebuild gate-only scores after changing splits:

```powershell
python scripts/eval_loo_holdout.py
```

Outputs: `metrics.csv`, `summary.txt`, `logs_<fold>.csv`, `portable_prompt_decisions.csv`.
