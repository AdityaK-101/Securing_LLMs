# Leave-one-source-out results (default / F1 thresholds)

**Status: complete** (full LLM + judge; all five folds; main + paraphrase + edge-benign + adaptive).

**Role: contrast.** Thresholds are each method’s default / F1 setting (not FPR-matched).  
**Main LOO report** (val FPR ≈ 5%): `results/loo_matched_fpr/` via `--loo-matched-fpr`.

```powershell
python -m src.run_experiments --loo
```

Protocol tables: root `README.md` **§6.0b**. Splits: `data/splits/source_holdout/`.

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
| `adaptive_attacks.jsonl` | Generated adaptive texts (resumable; reused by `--loo-matched-fpr`) |
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

See `results/cache/README.md`. Filters/metrics always rescore; LLM/judge reuse
`results/prompt_cache/loo/`. Pass `--no-outcome-cache` / `--rerun` to regenerate replies.

## Gate-only (no True ASR)

`python scripts/eval_loo_holdout.py` → `results/source_holdout/` (may be stale vs current splits).
