# Results layout

| Folder / file | Role | How to produce |
|---------------|------|----------------|
| **[`PAPER.md`](PAPER.md)** | **Index of what to cite** | — |
| **`loo_matched_fpr/`** | **Main LOO report** (val FPR ≈ 5%) + [`PAPER_TABLES.md`](loo_matched_fpr/PAPER_TABLES.md) | `python scripts/run_paper.py` or `--loo-matched-fpr` |
| **`matched_fpr/`** | **Mixed suite matched-FPR** (combined-dataset random split, val FPR ≤ 5%; paper **ID** column) | `--matched-fpr` (also in `run_paper.py`) |
| **`loo/`** | Default / F1 LOO (**contrast**) | `--loo` or `run_paper.py --mode full` |
| **`prompt_cache/`** | Shared LLM/judge + MiniLM caches | used by LOO / suites |
| **`suite_a/` `suite_b/`** | Mixed suites (archived v2.0 figures in root README) | `--suite a` / `b` |
| **`protectai/` `prompt_guard/`** | Static portable baselines (mixed test) | `--portable-baselines` |
| **`source_holdout/`** | Gate-only LOO (may be stale) | `scripts/eval_loo_holdout.py` |
| **`judge_human_validation/`** | Judge agreement | `scripts/score_judge_human_validation.py` |
| **`figures/`** | Shared suite figures | suite runs |
| Root CSVs | Ablations / overhead | `--hard-trigger-ablation`, … |

### Do not confuse

| Path | Is |
|------|----|
| `matched_fpr/` | **Mixed suite matched-FPR** (combined random split), FPR **≤** 5% |
| `loo_matched_fpr/` | **Leave-one-source-out**, FPR **≈** 5% — **main paper holdout** |
| `loo/` | Same LOO folds, **default/F1** thresholds — contrast only |

### One-shot

```powershell
python scripts/run_paper.py           # core paper holdout + tables
python scripts/run_paper.py --mode full
```
