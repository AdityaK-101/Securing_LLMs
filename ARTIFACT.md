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
| `--loo-matched-fpr` | Main LOO: val FPR≈5% + full suite + curves | hours (GPU; writes `results/loo_matched_fpr/`; reuses `results/prompt_cache/loo/` + `results/loo/*/adaptive_attacks.jsonl`) |
| `--loo` (5 folds) | Default/F1 LOO + paraphrase / edge / adaptive | **complete** in `results/loo/` (contrast); resume-safe; `--rerun` to redo |
| Portable gates | Gate-only; reuse suite logs for True ASR | minutes–tens of minutes |
| `--matched-fpr` | Mixed suite matched-FPR (combined split, FPR≤5%; not LOO) | tens of minutes–hours |
| Judge human score | Offline CSV compare | seconds |

Committed `results/` tables match the paper. Re-run full LLM suites only if you change methods or data.

`--no-llm` is sanitizer-only (Bypass/FPR). True ASR requires a target+judge run and is not copied from previous CSVs.

## Paper table → artifact map

| Paper table / section | Artifact path(s) |
|-----------------------|------------------|
| **Main holdout (cite first)** | `results/loo_matched_fpr/PAPER_TABLES.md` (mixed suite matched-FPR / ID vs LOO-4 vs Neuralchemy, Wilson, ProtectAI / Prompt Guard 2, dedup, train-benign) |
| LOO matched-FPR suite | `results/loo_matched_fpr/` — `python scripts/run_paper.py` or `--loo-matched-fpr` |
| Mixed suite matched-FPR (**ID** column) | `results/matched_fpr/metrics.csv` |
| Default-thr LOO (contrast) | `results/loo/` |
| Mixed suite (archived v2.0) | `results/suite_a/`, `results/suite_b/`, `results/protectai/`, `results/prompt_guard/` |
| Ablation (Table 2) | `results/hard_trigger_ablation.csv`, `learned_ablation_metrics.csv`, `weighted_ablation_metrics.csv` |
| Adaptive (LOO) | `results/loo_matched_fpr/robustness_adaptive_summary.csv` (main); `results/loo/` (contrast) |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` |
| Index / folder map | `results/PAPER.md`, `results/README.md` |

## Reproduce entry points

```powershell
python scripts/run_paper.py              # core paper holdout
python scripts/run_paper.py --mode full  # + suites / ablations / judge HV
```

Also: `README.md` §9–§10, `scripts/README.md`. Gate-only smoke: `scripts/reproduce.ps1`.
