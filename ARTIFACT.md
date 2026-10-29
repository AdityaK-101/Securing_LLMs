# Artifact notes (code companion)

This repository is the code / data / results companion for the paper
*Where Does the Security Come From? A Controlled Evaluation of Context-Aware Input Sanitization Against Direct Prompt Injection*.

## Licenses

- **Code:** MIT (`LICENSE`)
- **Compiled dataset:** CC BY-NC 4.0 (`data_cards/Data_Card.md`, `data_cards/license_log.txt`)
- **Mixed corpus:** v3.0 ~700 prompts (`data/cleaned/combined_dataset.csv`); v2.0 350 archived under `data/cleaned/legacy/`. Rebuild: `python scripts/build_combined_dataset.py`.
- **External Neuralchemy test:** Apache-2.0 dataset card, with HackAPrompt (CC BY 4.0), HarmBench (MIT), and WildGuard constituents. Files under `data/external/neuralchemy/`. Not part of the mixed corpus and not used for training.

## Models

| Role | Hugging Face ID | Notes |
|------|-----------------|-------|
| Target LLM | `Qwen/Qwen2.5-3B-Instruct` | Greedy decode, ≤512 new tokens, fp16 on GPU |
| Compliance judge | `Qwen/Qwen2.5-3B-Instruct` | Separate load; `FINAL_LABEL` rubric |
| Adaptive paraphrases | `google/flan-t5-base` | Frozen adaptive texts in suite logs |
| Semantic aux | `sentence-transformers/all-MiniLM-L6-v2` | **CPU by design** |
| Perplexity aux | `distilgpt2` | **CPU by design** |
| ProtectAI gate | `protectai/deberta-v3-base-prompt-injection-v2` | Portable baseline (published weights) |
| DeBERTa fine-tuned | same ProtectAI checkpoint + our train | Fair baseline (`deberta_finetuned`) |
| Prompt Guard 2 | `meta-llama/Llama-Prompt-Guard-2-86M` | Gated HF access may require `HF_TOKEN` |

## Environment (paper §Experimental setup)

| Item | Value |
|------|-------|
| OS | Windows 11 |
| CPU | 22-thread Intel (Family 6 Model 170) |
| RAM | 15.4 GB |
| GPU | NVIDIA GeForce RTX 4070 Laptop (8 GB) |
| Python | 3.13 |
| PyTorch | 2.6 / CUDA 12.4 |
| Seed | 42 (random, numpy, torch, transformers) |

Auxiliaries and portable gates run on **CPU**; target and judge on **GPU** when available.

## Runtime overhead (paper §Runtime overhead)

Mean ms/prompt on CPU, 50 prompts (`results/suite_b/runtime_overhead.csv`):

| Component | Mean ms |
|-----------|--------:|
| Keyword gate | 0.02 |
| Regex gate (45 patterns) | 0.22 |
| MiniLM semantic | 20.98 |
| DistilGPT-2 perplexity | 45.03 |
| Method A full sanitize | 64.79 |
| MiniLM + DistilGPT-2 (Method B aux) | 66.01 |

## Wall-clock guidance

| Mode | What | Approx. |
|------|------|---------|
| `--no-llm` suite / ablations | Sanitizer-only Bypass / FPR | minutes |
| Full suite A or B | Target + judge on test + robustness | hours (GPU helps) |
| `--loo` (5 folds) | Main + paraphrase / edge / adaptive per fold | many hours (GPU; default OutcomeCache / FeatureCache under `results/prompt_cache/` — resume-safe; `--llm-batch-size`, `--loo-adaptive-max`, `--loo-skip-robustness` to trim) |
| Portable gates | Gate-only; reuse suite logs for True ASR | minutes–tens of minutes |
| `--matched-fpr` | Val-tuned FPR≤5% score compare (+ optional DeBERTa FT) | tens of minutes–hours |
| Judge human score | Offline CSV compare | seconds |

Committed `results/` tables match the paper. Re-run full LLM suites only if you change methods or data.

`--no-llm` is sanitizer-only (Bypass/FPR). True ASR requires a target+judge run and is not copied from previous CSVs.

## Paper table → artifact map

| Paper table / section | Artifact path(s) |
|-----------------------|------------------|
| Main comparison (Table 1) | `results/suite_a/metrics.csv`, `results/suite_b/metrics.csv`, `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv` (suites include `similarity_only` + `deberta_finetuned`) |
| Matched-FPR fair baselines | `results/matched_fpr/metrics.csv` (`python -m src.run_experiments --matched-fpr` or `scripts/eval_matched_fpr.py`) |
| Ablation (Table 2) | `results/hard_trigger_ablation.csv`, `results/learned_ablation_metrics.csv`, `results/weighted_ablation_metrics.csv` |
| Adaptive rewriting (Table 3) | `results/suite_a/robustness_adaptive.csv`, `results/suite_a/robustness_adaptive_by_style.csv` (incl. `similarity_evasion`), `results/protectai/adaptive_metrics.csv`, `results/prompt_guard/adaptive_metrics.csv` |
| Paraphrase robustness | `results/robustness_paraphrase_test.csv` (expanded synonyms + LLM fallback; re-run after probe change) |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` |
| A vs B matched comparison | `results/ab_comparison_metrics.csv` |
| Source-holdout (mentor LOO) | `data/splits/source_holdout/`, gate scores in `results/source_holdout/` (re-run `eval_loo_holdout.py` after rebuilding splits); full LLM+robustness under `results/loo/` via `python -m src.run_experiments --loo` |

## Reproduce entry points

See `README.md` § “Reproduce paper tables” and `scripts/reproduce.ps1` / `scripts/reproduce.sh`.
