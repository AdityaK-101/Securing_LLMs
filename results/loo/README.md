# Leave-one-source-out results

**Status: complete** (full LLM + judge; all five folds; main + paraphrase + edge-benign + adaptive).

Produced by:

```powershell
python -m src.run_experiments --loo
```

Protocol and headline tables: root `README.md` §6.0. Splits: `data/splits/source_holdout/`.

## Folds

| Fold | Role | Test size |
|------|------|-----------|
| `simsonsun` | Four-source holdout | 200 inj + 200 ben |
| `mpdd` | Four-source holdout | 200 inj + 200 ben |
| `prompt_injection` | Four-source holdout | 200 inj + 200 ben |
| `bravansky` | Four-source holdout | 200 inj + 200 ben |
| `neuralchemy` | Fifth external holdout (outside four-source avg) | 200 inj + 200 ben |

Methods per fold: `baseline`, `regex`, `keyword`, `context_aware` (A), `context_aware_learned` (B), `similarity_only`, `deberta_finetuned`.

## Aggregate files (this directory)

| File | Contents |
|------|----------|
| `summary.csv` / `summary.txt` | Per-fold main metrics + `four_source_macro_avg` / `four_source_micro` |
| `robustness_paraphrase.csv` | Concatenated fold paraphrase rows |
| `robustness_paraphrase_summary.csv` | Incl. four-source macro avg |
| `robustness_edge_benign.csv` | Concatenated edge FPR |
| `robustness_edge_benign_summary.csv` | Incl. four-source macro avg |
| `robustness_adaptive.csv` | Concatenated adaptive metrics |
| `robustness_adaptive_summary.csv` | Incl. four-source macro avg |

`four_source_macro_avg` weights each of the four attack sources equally. Benign prompts repeat across those folds, so micro FPR is not an independent sample. Neuralchemy is excluded from that average.

## Per-fold directory (`results/loo/<fold>/`)

| File | Contents |
|------|----------|
| `metrics.csv` | Main Bypass / True ASR / FPR |
| `logs.jsonl` | Per-prompt main pipeline log |
| `run_mode.txt` | `llm` / sanitizer-only |
| `confusion_matrices.json` | Gate confusion |
| `robustness_paraphrase.csv` | Orig vs para Bypass |
| `robustness_edge_benign.csv` | Edge FPR (80 prompts) |
| `robustness_adaptive.csv` | Adaptive Bypass / True ASR |
| `robustness_adaptive_by_style.csv` | Breakdown by rewrite style |
| `adaptive_attacks.jsonl` | Generated adaptive texts (resumable) |
| `robustness_adaptive_logs.jsonl` | Adaptive sanitize → LLM → judge logs |
| `robustness_adaptive_confusion_matrices.json` | Adaptive gate CM |
| `robustness_note.txt` | Fold write-up |
| `robustness_run_mode.txt` | Robustness run mode |
| `figures/` | Charts |

## Shared caches

| Path | Role |
|------|------|
| `results/prompt_cache/loo/outcomes.jsonl` | Greedy target reply + judge label (main) |
| `results/prompt_cache/loo/adaptive_outcomes.jsonl` | Same for adaptive texts |
| `results/prompt_cache/loo/features.pkl` | MiniLM embeddings + raw PPL |

See `results/cache/README.md`. Re-run is resume-safe; pass `--rerun` to force recomputation.

## Gate-only (no True ASR)

`python scripts/eval_loo_holdout.py` → `results/source_holdout/` (may be stale vs current splits). Not a substitute for this folder.
