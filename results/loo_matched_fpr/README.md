# LOO matched-FPR ≈ 5% — main LOO report

**Status: complete** (all five folds; main + paraphrase + edge-benign + adaptive + True ASR–FPR curves; ProtectAI + Prompt Guard 2 at matched thresholds).

**Paper tables (canonical):** [`PAPER_TABLES.md`](PAPER_TABLES.md) — ID vs LOO-4 vs Neuralchemy, Wilson CIs, dedup, train-benign inventory.

```powershell
# Preferred one-shot (also runs mixed suite matched-FPR + portables + tables):
python scripts/run_paper.py

# Or piece-wise:
python -m src.run_experiments --loo-matched-fpr
python scripts/eval_loo_matched_fpr_portables.py   # ProtectAI + Prompt Guard 2
python scripts/build_loo_matched_fpr_paper_tables.py
```

Do **not** confuse with `results/matched_fpr/` (mixed-split, FPR ≤ 5%, no LOO) or `results/loo/` (default / F1 thresholds; contrast).

| Fold | Role | Test size |
|------|------|-----------|
| `simsonsun` | Four-source holdout | 200 inj + 200 ben |
| `mpdd` | Four-source holdout | 200 inj + 200 ben |
| `prompt_injection` | Four-source holdout | 200 inj + 200 ben |
| `bravansky` | Four-source holdout | 200 inj + 200 ben |
| `neuralchemy` | Fifth external holdout (**outside** four-source avg; cleanest unseen test) | 200 inj + 200 ben |

Methods: `baseline`, `regex`, `keyword`, `context_aware` (A), `context_aware_learned` (B), `similarity_only`, `deberta_finetuned`, `protectai`, `prompt_guard`.

Per fold, score-based methods (including ProtectAI / Prompt Guard 2) are tuned on **that fold’s validation** set to **hit val FPR ≈ 5%** (closest). Soft weights / published portable weights are unchanged — threshold selection only. Method A also sweeps `semantic_hard_sim`.

---

## Quick headline (see PAPER_TABLES.md for full Wilson CIs)

| Method | ID Bypass % | ID FPR % | LOO-4 macro Bypass % | LOO-4 macro True ASR % | LOO-4 macro FPR % | Neuralchemy Bypass % | Neuralchemy True ASR % | Neuralchemy FPR % |
|--------|------------:|---------:|---------------------:|-----------------------:|------------------:|---------------------:|-----------------------:|------------------:|
| baseline | — | — | 100.00 | 58.75 | 0.00 | 100.00 | 64.50 | 0.00 |
| regex | — | — | 84.12 | 48.12 | 0.00 | 85.00 | 52.50 | 1.00 |
| keyword | 58.33 | 0.00 | 59.75 | 33.50 | 0.00 | 70.50 | 40.50 | 1.50 |
| **context_aware (A)** | **11.67** | 6.67 | **34.62** | **13.75** | 16.62 | **20.00** | **11.00** | 87.00 |
| context_aware_learned (B) | 13.33 | 0.00 | 47.62 | 19.00 | 11.38 | 26.00 | 14.50 | 56.50 |
| similarity_only | 30.00 | 2.22 | 53.75 | 23.00 | 11.75 | 57.00 | 35.00 | 37.00 |
| deberta_finetuned | 3.33 | 0.00 | 1.38 | 0.62 | 13.50 | 1.50 | 0.50 | 60.50 |
| protectai | 10.00 | 6.67 | 5.62 | 2.75 | 35.25 | 4.50 | 2.00 | 12.50 |
| prompt_guard | 3.33 | 6.67 | 1.38 | 0.50 | 13.62 | 0.00 | 0.00 | 85.00 |

**ID** = **mixed suite matched-FPR** on the combined-dataset random split (`results/matched_fpr/`; val FPR ≤ 5%). LOO-4 macro = equal weight over the four attack-source holdouts. Neuralchemy is reported as its own column (not inside that average). Full Wilson / pooled / dedup / train-benign: [`PAPER_TABLES.md`](PAPER_TABLES.md).

---

## Outputs

| Path | Contents |
|------|----------|
| **`PAPER_TABLES.md`** | Paper Tables A–E (ID / LOO / Neuralchemy, Wilson, dedup, train benign) |
| `paper_headline.csv` | Machine-readable headline |
| `metrics_with_wilson.csv` | All folds + Wilson 95% CIs |
| `summary.csv` / `metrics.csv` | Main matched-FPR metrics (+ macro / micro) |
| `thresholds.csv` | Chosen thr / `semantic_hard_sim` / val FPR |
| `train_benign_by_fold.csv` | Benign inventory used for fitting |
| `robustness_*{,_summary}.csv` | Paraphrase / edge / adaptive |
| `figures/` | True ASR vs FPR curves |
| `<fold>/` | Per-fold metrics + robustness |

Contrast (default / F1 thresholds): `results/loo/` — root README §6.0b. Protocol overview: root README §6.0.
