# Notebooks

Active notebooks for **data exploration and LOO split building**.  
Evaluation and paper tables are **not** notebook-driven: use `python scripts/run_paper.py` or `python -m src.run_experiments`.

## Active (keep)

| Notebook | Role |
|----------|------|
| **`build_source_holdout.ipynb`** | Canonical LOO fold builder → `data/splits/source_holdout/`. CLI wrapper: `python scripts/build_source_holdout.py`. |
| **`EDA.ipynb`** | Mixed-corpus EDA on `data/cleaned/combined_dataset.csv` + `data/splits/{train,val,test}.csv` (secondary ID contrast). |

After LOO splits exist, paper scoring:

```powershell
python scripts/run_paper.py
# → results/loo_matched_fpr/  (main)
# → results/matched_fpr/      (ID column)
```

## Archived (`archive/`)

Legacy / superseded notebooks (v2.0 350-row builders, early experiment scratchpads).  
**Do not run** them against current paths; they can overwrite canonical CSVs.

See `archive/README.md`.

## Mixed corpus rebuild (script, not notebook)

```powershell
python scripts/build_combined_dataset.py
```
