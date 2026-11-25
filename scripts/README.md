# Scripts

Grouped by job. Prefer the **paper one-shot** unless you need a single piece.

## Paper (start here)

| Script | What it does |
|--------|----------------|
| **`run_paper.py`** | **One-shot** core (or `--mode full`) paper pipeline + consistency gate |
| `build_loo_matched_fpr_paper_tables.py` | Rebuild `PAPER_TABLES.md` from existing CSVs |
| `eval_loo_matched_fpr_portables.py` | ProtectAI + Prompt Guard 2 on LOO at matched FPR |
| `run_loo_matched_ablations.py` | Matched-FPR LOO-4 Method A/B semantic ablations (incl. A − all semantic); also via `run_paper.py --mode full` |
| `eval_loo_second_target.py` | Second-target True ASR (Llama-3.2-3B-Instruct + Qwen judge) at saved thresholds; optional via `run_paper.py --second-target` |
| `rerun_loo_matched_fpr_paraphrase.py` | Re-score paraphrase at saved matched thresholds |
| `rerun_loo_matched_fpr_edge.py` | Re-score edge FPR at saved matched thresholds |
| `audit_pipeline_consistency.py` | Consistency gate (fold↔agg, paraphrase↔metrics) |
| `generate_paper_figures.py` | Write AAAI figures → `report/aaai/images/` |

```powershell
python scripts/run_paper.py              # core
python scripts/run_paper.py --mode full  # everything
```

Equivalent: `python -m src.run_experiments --paper`.

**Cite:** `results/loo_matched_fpr/` - not `results/loo/`, not `results/source_holdout/`.

---

## Data / splits

| Script | Output |
|--------|--------|
| `build_combined_dataset.py` | `data/cleaned/combined_dataset.csv` |
| `build_external_neuralchemy.py` | `data/external/neuralchemy/` |
| `build_source_holdout.py` | `data/splits/source_holdout/` (runs notebook) |
| `verify_source_holdout.py` | Integrity checks |
| `build_judge_human_validation.py` | Blind sheet under `data/judge_human_validation/` |
| `print_dataset_sizes.py` | Print mixed + LOO fold sizes |
| `build_loo_splits.py` | **Deprecated alias** → `build_source_holdout.py` |

---

## Evaluation (individual)

| Script | Role | Writes |
|--------|------|--------|
| `eval_matched_fpr.py` | Mixed suite matched-FPR (or `--matched-fpr`) | `results/matched_fpr/` |
| `eval_loo_holdout.py` | Gate-only LOO Bypass/FPR | `results/source_holdout/` (**not paper**) |
| `eval_protectai.py` / `eval_prompt_guard.py` | Static portable gates on mixed test | `results/protectai/` / `prompt_guard/` |
| `eval_portable_adaptive.py` | Portable adaptive **default thr** on suite texts | suite adaptive logs (**≠** Table F) |
| `eval_paraphrase_test.py` | Mixed-suite paraphrase gate check | `results/robustness_paraphrase_test.csv` (**≠** LOO) |
| `precompute_prompt_outcomes.py` | Warm OutcomeCache | `results/prompt_cache/` |
| `score_judge_human_validation.py` | Offline judge agreement | `results/judge_human_validation/` |
| `regen_robustness_notes.py` | Refresh fold note text | `results/loo/**` |
| `reproduce.ps1` / `reproduce.sh` | Short gate-only smoke (**not** full paper) | - |

Main experiments: **`python -m src.run_experiments`** (`--loo-matched-fpr`, `--loo`, `--suite`, ablations, …).

Root shim `load_dataset.py` is deprecated; use `print_dataset_sizes.py`.
