# Source-holdout gate results

**Stale relative to current splits.** These CSVs predate the fully uniform
holdout (`520/130/400` four-source folds, raw Bravansky, notebook builder).
Do not cite them until re-scored.

Rebuild scores after the current splits:

```powershell
python scripts/eval_loo_holdout.py
```

Outputs: `metrics.csv`, `summary.txt`, `logs_<fold>.csv`, `portable_prompt_decisions.csv`.
Gate-only (Bypass / FPR + Wilson). True ASR + paraphrase / edge-benign / adaptive:
`python -m src.run_experiments --loo` → `results/loo/`.
