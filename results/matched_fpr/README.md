# Matched-FPR fair comparison

Tunes each score-based filter on **val** to FPR ≤ 5% (default), maximizing
detection, then reports **test** Bypass / FPR at that operating point.

Also writes each method's **default** operating point for contrast.

```powershell
python scripts/eval_matched_fpr.py
# or
python -m src.run_experiments --matched-fpr
```

Outputs: `metrics.csv`, `summary.txt`.

Methods: keyword, context_aware (soft score), context_aware_learned,
similarity_only, deberta_finetuned, protectai, prompt_guard.

ProtectAI / Prompt Guard keep published weights; only the decision threshold
is retuned. DeBERTa fine-tune uses the mixed train/val split (skip with
`--skip-deberta` / `--matched-fpr-skip-deberta`).
