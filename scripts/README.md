# Scripts

Grouped by job. Prefer the **paper one-shot** unless you need a single piece.

## Paper (start here)

| Script | What it does |
|--------|----------------|
| **`run_paper.py`** | **One-shot** core (or `--mode full`) paper pipeline |
| `build_loo_matched_fpr_paper_tables.py` | Rebuild `PAPER_TABLES.md` from existing CSVs |
| `eval_loo_matched_fpr_portables.py` | ProtectAI + Prompt Guard 2 on LOO at matched FPR |

```powershell
python scripts/run_paper.py              # core
python scripts/run_paper.py --mode full  # everything
```

Equivalent: `python -m src.run_experiments --paper`.

---

## Data / splits

| Script | Output |
|--------|--------|
| `build_combined_dataset.py` | `data/cleaned/combined_dataset.csv` |
| `build_external_neuralchemy.py` | `data/external/neuralchemy/` |
| `build_source_holdout.py` | `data/splits/source_holdout/` (runs notebook) |
| `build_loo_splits.py` | Alias → `build_source_holdout.py` |
| `verify_source_holdout.py` | Integrity checks |
| `build_judge_human_validation.py` | Blind sheet under `data/judge_human_validation/` |

---

## Evaluation (individual)

| Script | Role |
|--------|------|
| `eval_matched_fpr.py` | Mixed suite matched-FPR (or use `--matched-fpr`) |
| `eval_loo_holdout.py` | Gate-only LOO Bypass/FPR → `results/source_holdout/` |
| `eval_protectai.py` / `eval_prompt_guard.py` | Static portable gates on mixed test |
| `eval_portable_adaptive.py` | Portable adaptive gate-only |
| `eval_paraphrase_test.py` | Mixed-suite paraphrase gate check |
| `precompute_prompt_outcomes.py` | Warm OutcomeCache |
| `score_judge_human_validation.py` | Offline judge agreement |
| `regen_robustness_notes.py` | Refresh fold note text |
| `reproduce.ps1` / `reproduce.sh` | Short gate-only smoke (not full paper) |

Main experiments live in **`python -m src.run_experiments`** (`--loo-matched-fpr`, `--loo`, `--suite`, ablations, …).
