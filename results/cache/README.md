# Prompt caches (speed path)

**Canonical location:** `results/prompt_cache/` (see `src/config.py` `PROMPT_CACHE_DIR`).

| Path | Used by | Notes |
|------|---------|-------|
| `results/prompt_cache/outcomes.jsonl` | Mixed suite (`--suite a/b`) | |
| `results/prompt_cache/features.pkl` | Mixed suite FeatureCache | |
| `results/prompt_cache/adaptive_outcomes.jsonl` | Mixed suite adaptive rewrites | |
| `results/prompt_cache/loo/outcomes.jsonl` | `--loo` and `--loo-matched-fpr` | Shared main True ASR |
| `results/prompt_cache/loo/features.pkl` | `--loo` / `--loo-matched-fpr` FeatureCache | Windows-safe replace |
| `results/prompt_cache/loo/adaptive_outcomes.jsonl` | `--loo` / `--loo-matched-fpr` adaptive | Shared adaptive True ASR |

Files are append-only / resumable for **generation only**.

**Resume policy (suite / LOO / LOO matched-FPR):** filters and metrics always
recompute. Only Qwen/judge replies (and MiniLM/PPL / FLAN texts) are reused
when present. Never skip because a `metrics.csv` already exists.

- Default-threshold metrics: `results/loo/`  
- Main matched-FPR LOO: `results/loo_matched_fpr/`  

After changing the corpus (e.g. v2→v3), delete this folder or pass
`--no-outcome-cache` / `--rerun` once.

Precompute (optional, good on Colab/Kaggle GPU):

```powershell
python scripts/precompute_prompt_outcomes.py --loo --batch-size 8
python scripts/precompute_prompt_outcomes.py --suite --batch-size 8
```
