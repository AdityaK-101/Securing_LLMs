# Judge human validation

## Status

| Cohort | N | What | Scores |
|--------|--:|------|--------|
| **Current sheet** (this folder) | 100 | Stratified Table 1 LOO+Neuralchemy no-defense pairs (20/source) | **Done** — 77.0% agree, Wilson [67.8, 84.2]%, κ = 0.48 (paper) |
| **`legacy_v2.0_n30/`** | 30 | Archived mixed test v2.0 (not Table 1) | Historical only — 83.3%, κ = 0.67 |

## Re-score

```powershell
python scripts/score_judge_human_validation.py
```

Rubric: `RUBRIC.md`. Do not edit `results/judge_human_validation/judge_key.csv` after labeling.
