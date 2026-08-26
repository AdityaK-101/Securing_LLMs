# Artifact notes (code companion)

This repository is the code / data / results companion for the paper
*Where Does the Security Come From? A Controlled Evaluation of Context-Aware Input Sanitization Against Direct Prompt Injection*.

## Licenses

- **Code:** MIT (`LICENSE`)
- **Compiled dataset:** CC BY-NC 4.0 (`data_cards/Data_Card.md`, `data_cards/license_log.txt`)

## Models

| Role | Hugging Face ID | Notes |
|------|-----------------|-------|
| Target LLM | `Qwen/Qwen2.5-3B-Instruct` | Greedy decode, ≤512 new tokens, fp16 on GPU |
| Compliance judge | `Qwen/Qwen2.5-3B-Instruct` | Separate load; `FINAL_LABEL` rubric |
| Adaptive paraphrases | `google/flan-t5-base` | Frozen adaptive texts in suite logs |
| Semantic aux | `sentence-transformers/all-MiniLM-L6-v2` | **CPU by design** |
| Perplexity aux | `distilgpt2` | **CPU by design** |
| ProtectAI gate | `protectai/deberta-v3-base-prompt-injection-v2` | Portable baseline |
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
| Portable gates | Gate-only; reuse suite logs for True ASR | minutes–tens of minutes |
| Judge human score | Offline CSV compare | seconds |

Committed `results/` tables match the paper. Re-run full LLM suites only if you change methods or data.

`--no-llm` is sanitizer-only (Bypass/FPR). True ASR requires a target+judge run and is not copied from previous CSVs.

## Paper table → artifact map

| Paper table / section | Artifact path(s) |
|-----------------------|------------------|
| Main comparison (Table 1) | `results/suite_a/metrics.csv`, `results/suite_b/metrics.csv`, `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv` |
| Ablation (Table 2) | `results/hard_trigger_ablation.csv`, `results/learned_ablation_metrics.csv`, `results/weighted_ablation_metrics.csv` |
| Adaptive rewriting (Table 3) | `results/suite_a/robustness_adaptive.csv`, `results/protectai/adaptive_metrics.csv`, `results/prompt_guard/adaptive_metrics.csv` |
| Paraphrase robustness | `results/robustness_paraphrase_test.csv` |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` |
| A vs B matched comparison | `results/ab_comparison_metrics.csv` |

## Reproduce entry points

See `README.md` § “Reproduce paper tables” and `scripts/reproduce.ps1` / `scripts/reproduce.sh`.
