# Mixed suite matched-FPR (combined-dataset split; not LOO)

Same **combined-dataset** train/val/test as the mixed suites (`--suite`), but
with thresholds retuned so val **FPR ≤ 5%** (default), then reports **test**
Bypass / FPR. Paper tables call this the **ID** column.

This is **not** the main LOO report. For leave-one-source-out at
FPR ≈ 5% (full suite + True ASR + curves), use:

```powershell
python -m src.run_experiments --loo-matched-fpr
```

→ `results/loo_matched_fpr/` (see that folder’s README).

## This folder

```powershell
python scripts/eval_matched_fpr.py
# or
python -m src.run_experiments --matched-fpr
```

Outputs: `metrics.csv`, `summary.txt`.

Methods: keyword, context_aware (soft score), context_aware_learned,
similarity_only, deberta_finetuned, protectai, prompt_guard.

ProtectAI / Prompt Guard keep published weights; only the decision threshold
is retuned. DeBERTa fine-tune uses the combined train/val split (skip with
`--skip-deberta` / `--matched-fpr-skip-deberta`).
