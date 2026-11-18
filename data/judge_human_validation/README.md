# Judge human validation

## Status

| Cohort | N | Overlap with current mixed test | Scores |
|--------|--:|--------------------------------:|--------|
| **`legacy_v2.0_n30/`** (paper-cited) | 30 | **30/30 on archived v2.0 test**; only ~3 ids in current v3.0 test | **Done** - 83.3% agree, κ = 0.67 |
| Current mixed suite (`results/suite_a/`, ~60 inj.) | 60 | Full current baseline set | **Not re-labeled yet** |

The mixed corpus grew from v2.0 (30 inj. / 23 ben. test) to v3.0 (~60 / 45). Suite logs were re-run on v3.0, but the human labels were collected on the **earlier full no-defense set (N = 30)**.

Paper numbers still refer to that completed legacy cohort (honest as a prior full-set check, not as a re-validation of every current mixed-test reply).

## Re-validate on current suite

```powershell
python scripts/build_judge_human_validation.py   # rebuilds blank sheet + key from suite_a (N=60)
# label data/judge_human_validation/blind_labeling_sheet.csv
python scripts/score_judge_human_validation.py
```

Do not open `results/judge_human_validation/judge_key.csv` until labeling is finished.
Rubric: `RUBRIC.md`.
