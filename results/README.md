# Results layout

**Cite only `loo_matched_fpr/` (+ `matched_fpr/` for the ID column) in the AAAI paper.**  
Other folders are contrast, smoke, or legacy. Mixing them silently is a common failure mode.

| Folder / file | Role | How to produce |
|---------------|------|----------------|
| **[`PAPER.md`](PAPER.md)** | **Index of what to cite** | - |
| **`loo_matched_fpr/`** | **Main LOO report** (val FPR ≈ 5%) + [`PAPER_TABLES.md`](loo_matched_fpr/PAPER_TABLES.md) | `python scripts/run_paper.py` or `--loo-matched-fpr` |
| **`matched_fpr/`** | **Mixed suite matched-FPR** (combined-dataset random split, val FPR ≤ 5%; paper **ID** column) | `--matched-fpr` (also in `run_paper.py`) |
| **`loo/`** | Default / F1 LOO (**contrast only**) | `--loo` or `run_paper.py --mode full` |
| **`prompt_cache/`** | Shared LLM/judge + MiniLM caches | used by LOO / suites |
| **`suite_a/` `suite_b/`** | Mixed suites (archived v2.0 figures in root README) | `--suite a` / `b` |
| **`protectai/` `prompt_guard/`** | Static portable baselines (mixed test) | `--portable-baselines` |
| **`source_holdout/`** | Gate-only LOO (**may be stale; not paper main**) | `scripts/eval_loo_holdout.py` |
| **`judge_human_validation/`** | Judge agreement | `scripts/score_judge_human_validation.py` |
| **`figures/`** | Shared suite / ablation figures | suite / ablation runs |
| Root `*ablation*.csv`, `ab_comparison_metrics.csv` | Mixed-split ablations (paper Table 2–style) | ablation flags / `run_paper.py --mode full` |
| Root `robustness_paraphrase_test.csv` | Mixed-suite paraphrase **smoke** (≠ LOO) | `eval_paraphrase_test.py` |
| Root `runtime_overhead.*` | Mirror of last latency run; prefer `suite_b/` | `--latency-only` |

### Do not confuse

| Path | Is |
|------|----|
| `matched_fpr/` | **Mixed suite matched-FPR** (combined random split), FPR **≤** 5% |
| `loo_matched_fpr/` | **Leave-one-source-out**, FPR **≈** 5% - **main paper holdout** |
| `loo/` | Same LOO folds, **default/F1** thresholds - contrast only |
| `source_holdout/` | Gate-only LOO; **not** the matched-FPR paper tree |
| `suite_a` + `eval_portable_adaptive.py` | Default portable thresholds on suite adaptive texts - **≠** Table F LOO matched adaptive |
| Root paraphrase CSV | Mixed gate-only smoke - **≠** `loo_matched_fpr/robustness_paraphrase*.csv` |

### Consistency

After any partial re-run of robustness under `loo_matched_fpr/`:

```powershell
python scripts/audit_pipeline_consistency.py
python scripts/rerun_loo_matched_fpr_paraphrase.py   # if paraphrase fold≠agg or ≠metrics
python scripts/rerun_loo_matched_fpr_edge.py --rebuild-tables
```

`run_paper.py` runs the audit gate at the end.

### One-shot

```powershell
python scripts/run_paper.py           # core paper holdout + tables
python scripts/run_paper.py --mode full
```
