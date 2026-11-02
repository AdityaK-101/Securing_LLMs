# Prompt caches (mentor speed path)

**Canonical location:** `results/prompt_cache/` (see `src/config.py` `PROMPT_CACHE_DIR`).

| Path | Used by | Notes |
|------|---------|-------|
| `results/prompt_cache/outcomes.jsonl` | Mixed suite (`--suite a/b`) | |
| `results/prompt_cache/features.pkl` | Mixed suite FeatureCache | |
| `results/prompt_cache/adaptive_outcomes.jsonl` | Mixed suite adaptive rewrites | |
| `results/prompt_cache/loo/outcomes.jsonl` | `--loo` folds (shared) | Used by completed LOO run |
| `results/prompt_cache/loo/features.pkl` | `--loo` FeatureCache | Windows-safe replace + periodic save |
| `results/prompt_cache/loo/adaptive_outcomes.jsonl` | `--loo` adaptive | Used by completed LOO run |

Files are append-only / resumable. Full LOO metrics live in `results/loo/` (complete). After changing the corpus (e.g. v2→v3), delete this folder or pass `--no-outcome-cache` once.

Precompute (optional, good on Colab/Kaggle GPU):

```powershell
python scripts/precompute_prompt_outcomes.py --loo --batch-size 8
python scripts/precompute_prompt_outcomes.py --suite --batch-size 8
```
