# Where Does the Security Come From?
### A Controlled Evaluation of Context-Aware Input Sanitization Against Direct Prompt Injection

> **Scope note:** Controlled evaluation of **direct, single-turn** prompt injection on a research-scale mixed corpus (~700 prompts; ~105-prompt test) plus leave-one-source-out holdouts. Not production-grade security guarantees.
>
> **Licenses:** Code = MIT (`LICENSE`); compiled dataset = CC BY-NC 4.0 (`data_cards/`). Cite via `CITATION.cff`. See also `ARTIFACT.md`.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Defense Methods](#4-defense-methods)
5. [Evaluation Pipeline](#5-evaluation-pipeline)
6. [Results](#6-results)
7. [Ablation Studies](#7-ablation-studies)
8. [Judge Human Validation](#8-judge-human-validation)
9. [How to Run](#9-how-to-run)
10. [Reproduce Paper Tables](#10-reproduce-paper-tables)
11. [Output Files](#11-output-files)
12. [Hardware & GPU Notes](#12-hardware--gpu-notes)
13. [Reproducibility](#13-reproducibility)
14. [Limitations](#14-limitations)
15. [Dependencies](#15-dependencies)

---

## 1. Project Overview

This repository supports a **controlled evaluation** of pre-inference sanitizers for **direct prompt injection** — attribution over a shared signal family, not a new detector.

| Tier | Methods |
|------|---------|
| No gate / lexical | `baseline`, `regex` (45 patterns), `keyword` (37 weighted phrases) |
| **Method A** | `context_aware` — hybrid hard triggers + weighted soft score |
| **Method B** | `context_aware_learned` — learned soft fusion (no hard triggers) |
| Portable (scripts) | ProtectAI DeBERTa-v3, Meta Prompt Guard 2 — off-the-shelf baselines |

Research questions (paper §Framework): end-to-end risk without a gate; whether lexical/portable gates reduce **True ASR** or mainly Bypass; how Methods A/B compare; which components carry the defense; usability cost (ordinary + edge-benign FPR).

### Key metrics (two-layer evaluation)

| Metric | Layer | Definition |
|--------|-------|------------|
| **Bypass Rate** | Layer 1 (sanitizer) | % of injection prompts the sanitizer fails to block — lower is better |
| **True ASR** | Layer 2 (model) | % of injection prompts where sanitizer was bypassed **and** the judge labels the target LLM reply as `COMPLIED` — lower is better |
| **FPR** | Layer 1 | % of benign prompts incorrectly blocked — lower is better |

### Enforcement mode

All defenses use **Block**: `blocked=True` → `llm_output = "[BLOCKED]"` (no target LLM call).

---

## 2. Repository Structure

```
Securing_LLMs/
├── .github/workflows/smoke.yml      ← CI smoke (offline judge score + compare-ab)
├── data/
│   ├── raw/                         ← Original downloads (Bravansky raw gitignored)
│   ├── cleaned/                     ← combined_dataset.csv (~700) + JSONL slices
│   ├── cleaned/legacy/              ← v1.0 250 + v2.0 350 snapshots
│   ├── splits/                      ← mixed train/val/test + source_holdout/ + legacy_v2.0_350/
│   ├── external/neuralchemy/        ← held-out core/test (not in mixed corpus)
│   └── judge_human_validation/      ← blind sheet, rubric, meta
├── data_cards/                      ← Data_Card.md + license_log.txt
├── notebooks/                       ← preprocess + EDA + build_source_holdout.ipynb
│   └── archive/                     ← superseded exploratory notebooks
├── scripts/
│   ├── build_source_holdout.py      ← runs notebooks/build_source_holdout.ipynb
│   ├── build_loo_splits.py          ← alias → build_source_holdout.py
│   ├── eval_loo_holdout.py          ← gate Bypass/FPR on source_holdout
│   ├── eval_matched_fpr.py          ← val-tuned FPR≤5% fair compare
│   ├── verify_source_holdout.py
│   ├── build_external_neuralchemy.py
│   ├── eval_protectai.py / eval_prompt_guard.py / …
│   ├── reproduce.sh / reproduce.ps1
├── src/
│   ├── config.py                    ← seed=42, model names, paths
│   ├── run_experiments.py           ← Main CLI (--loo / --matched-fpr)
│   ├── splits/                      ← holdout loaders + helpers
│   ├── sanitizers/                  ← baseline, regex, keyword, A, B, fair + portables
│   ├── models/ / evaluation/ / visualization/ / utils/
├── results/
│   ├── suite_a/ / suite_b/          ← paper mixed-split suites
│   ├── matched_fpr/                 ← matched operating-point table
│   ├── source_holdout/              ← LOO gate scores (re-run after split rebuild)
│   ├── protectai/ / prompt_guard/
│   ├── judge_human_validation/
│   └── figures/
├── load_dataset.py                  ← Print split sizes
├── ARTIFACT.md
├── CITATION.cff
├── LICENSE
├── environment.yml
├── requirements.txt
└── README.md
```

---

## 3. Dataset

Schema: `{id, prompt, label}` with `label ∈ {injection, benign}`.  
Canonical: `data/cleaned/combined_dataset.csv` (v3.0; see `data_cards/Data_Card.md`).  
Rebuild: `python scripts/build_combined_dataset.py`. Exact counts in `combined_manifest.json`.

| Split | Total | Injection | Benign |
|-------|------:|----------:|-------:|
| train | ~489 | ~280 | ~209 |
| val | ~105 | ~60 | ~45 |
| test | ~105 | ~60 | ~45 |
| **Total** | **~699** | **400** | **~299** |

v2.0 (350 / 53-test) is archived under `data/cleaned/legacy/` and `data/splits/legacy_v2.0_350/`. Suite CSVs under `results/suite_*` need a re-run against the grown test.

Sources and licensing: `data_cards/`. Bravansky raw CSV is **not shipped**; cleaned 50-sample JSONL is enough to evaluate.

### External held-out test

`neuralchemy/Prompt-injection-dataset`, config `core`, published `test` split only. These rows are not in train, validation, or the embedding gallery. Exact overlaps with the mixed corpus are dropped at build time (see `manifest.json`).

| File | Total | Injection | Benign | What it keeps |
|------|------:|----------:|-------:|---------------|
| `data/external/neuralchemy/test_main.csv` | 753 | 363 | 390 | `direct_injection` (314), `jailbreak` (49), `benign` (381), `edge_case` (9) |
| `data/external/neuralchemy/test_core.csv` | 936 | 546 | 390 | full kept test split; other categories are `eval_slice=appendix` |

Rebuild with `python scripts/build_external_neuralchemy.py`. Counts: `data/external/neuralchemy/manifest.json`.

### Leave-one-source-out (final mentor holdout)

Canonical splits: `data/splits/source_holdout/`. **Build in the notebook** `notebooks/build_source_holdout.ipynb` (run all cells). Optional CLI wrapper that executes those cells: `python scripts/build_source_holdout.py`.

| Rule | Behavior |
|------|----------|
| Hold out | Entire source: attacks **and** its native benign (PI fold trains on ChatGPT benign only) |
| Shared benign | Fixed **200** PI-benign on every four-source test; never in train/val |
| Four-source sizes | Identical **520 / 130 / 400** (incl. `prompt_injection`) |
| Attacks | Test **200** held-out; train pool **200 from each** other raw source; val 20% within each source |
| Fit benign | **50** every fold |
| Fifth set | Neuralchemy test 200+200; fit uses 200 from each of four sources → 680/170/400 |
| Inventory | `train_benign_by_fold.csv` + `manifest.json` |

Near-dup filter (MiniLM > 0.9) in `manifest.json`. Gate scores: `python scripts/eval_loo_holdout.py` → `results/source_holdout/` (**re-run** — committed scores may be stale). Bypass/FPR only; True ASR needs `--loo` with LLM.

```powershell
# preferred: run notebooks/build_source_holdout.ipynb
python scripts/build_source_holdout.py
python scripts/verify_source_holdout.py
python scripts/eval_loo_holdout.py
```

---

## 4. Defense Methods

Every sanitizer implements `sanitize(prompt) -> (sanitized_text, blocked)` (binary **Block**; blocked prompts never reach the target LLM).

- **Regex** — 45 case-insensitive patterns (overrides, personas, DAN-style names, developer mode, German overrides, hypothetical framing); any match blocks.
- **Keyword** — 37 weighted phrases; block if sum ≥ 0.5.
- **Method A** — eight shared signals (regex, keyword, semantic, intent, roleplay, instruction shift, objective conflict, perplexity). Hard triggers: semantic stepped score > 0.6, intent = 1.0, or roleplay = 1.0 with keyword > 0.3. Else soft aggregate (weights in paper §Method A) blocks at ≥ 0.40.
- **Method B** — same evidence; three continuous semantic features + LR on train, threshold tuned on val (F1, tie-break lower Bypass/FPR); **no hard triggers**.
- **ProtectAI / Prompt Guard 2** — published classifiers on CPU; no fine-tuning on our split (`src/sanitizers/protectai.py`, `prompt_guard.py`). Run via scripts / `--portable-baselines`.
- **Similarity-only** — MiniLM max-cosine to the train injection gallery; threshold F1-tuned on val (`similarity_only.py`). Fair baseline on mixed suites and LOO.
- **DeBERTa (fine-tuned)** — ProtectAI DeBERTa checkpoint fine-tuned on our train split; threshold on val (`deberta_finetuned.py`). Fair baseline distinct from the off-the-shelf ProtectAI gate.
- **Matched FPR (~5%)** — continuous scores retuned on val to FPR ≤ 5%, then test Bypass/FPR (`src/evaluation/matched_fpr.py`). Off-the-shelf ProtectAI / Prompt Guard keep published weights; only the decision threshold is retuned. Run: `python scripts/eval_matched_fpr.py` or `python -m src.run_experiments --matched-fpr`.

---

## 5. Evaluation Pipeline

Target and judge: **`Qwen/Qwen2.5-3B-Instruct`**. Adaptive paraphrases: **`google/flan-t5-base`**.

```
Phase 1 — sanitize (+ target LLM if not blocked)
Phase 2 — judge COMPLIED / REFUSED on unblocked injections
Robustness — expanded paraphrase, edge-benign, FLAN-T5 adaptive
             (+ similarity-evasion rewrite minimizing max cosine to train gallery)
```

**Speed path (default with LLM):** one greedy Qwen reply + judge label per unique *injection* prompt (`results/prompt_cache/`), then True ASR is a per-method lookup; benign prompts never call Qwen/judge; MiniLM embeddings + DistilGPT-2 PPL are cached and reused across methods/folds; generate-all → unload → judge-all with append-only resume. Batch with `--llm-batch-size N` (or `LLM_CACHE_BATCH_SIZE`). Opt out with `--no-outcome-cache`.

Same phases on `--loo` (per `data/splits/source_holdout/<fold>/`), writing under `results/loo/`. Adaptive logs include `attack_style` (`similarity_evasion` is the filter-targeting variant) and gallery similarity fields.

| `--suite` | Methods | Output |
|-----------|---------|--------|
| `a` | baseline, regex, keyword, Method A + similarity_only + deberta_finetuned | `results/suite_a/` |
| `b` | baseline, regex, keyword, Method B + similarity_only + deberta_finetuned | `results/suite_b/` |
| `ab` (default) | A then B (each with fair baselines) | both folders |
| `both` | A + B + fair baselines in one run | `results/` |
| `--loo` | fold fair baselines + A/B + after-main | `results/loo/` |
| `--matched-fpr` | score retune at val FPR ≤ 5% | `results/matched_fpr/` |

---

## 6. Results

Numbers below are from the **archived v2.0** held-out test (30 injection / 23 benign) unless noted. Re-run suites on the grown v3.0 test (~60 / 45) before citing new figures.

### 6.0 Primary comparison (paper Table 1)

| Method | Bypass % (n/30) | True ASR % (n/30) | FPR % (n/23) | Edge FPR % (n/80) |
|--------|----------------:|------------------:|-------------:|------------------:|
| No defense | 100.00 (30) | 46.67 (14) | 0.00 (0) | 0.00 (0) |
| Regex (45 patterns) | 86.67 (26) | 46.67 (14) | 0.00 (0) | 2.50 (2) |
| Keyword (37 phrases) | 96.67 (29) | 46.67 (14) | 0.00 (0) | 0.00 (0) |
| ProtectAI (DeBERTa) | 40.00 (12) | 33.33 (10) | 0.00 (0) | 25.00 (20) |
| Prompt Guard 2 (86M) | 70.00 (21) | 36.67 (11) | 0.00 (0) | 30.00 (24) |
| **Method A (hybrid)** | **6.67 (2)** | **0.00 (0)** | **13.04 (3)** | **36.25 (29)** |
| **Method B (learned)** | **20.00 (6)** | **13.33 (4)** | **8.70 (2)** | **7.50 (6)** |

### 6.1 Suite A — Method A

| Method | Bypass % ↓ | True ASR % ↓ | FPR % ↓ | Successful attacks |
|--------|-----------:|-------------:|--------:|-------------------:|
| baseline | 100.0 | 46.67 | 0.0 | 14 |
| regex | 86.67 | 46.67 | 0.0 | 14 |
| keyword | 96.67 | 46.67 | 0.0 | 14 |
| **context_aware (A)** | **6.67** | **0.0** | 13.04 | **0** |

### 6.2 Suite B — Method B

| Method | Bypass % ↓ | True ASR % ↓ | FPR % ↓ | Successful attacks |
|--------|-----------:|-------------:|--------:|-------------------:|
| baseline | 100.0 | 46.67 | 0.0 | 14 |
| regex | 86.67 | 46.67 | 0.0 | 14 |
| keyword | 96.67 | 46.67 | 0.0 | 14 |
| **context_aware_learned (B)** | **20.0** | **13.33** | **8.7** | **4** |

### 6.3 Portable baselines (static test)

| Method | Bypass % | True ASR % | FPR % | Edge FPR % |
|--------|---------:|-----------:|------:|-----------:|
| ProtectAI | 40.0 | 33.33 | 0.0 | 25.0 |
| Prompt Guard 2 | 70.0 | 36.67 | 0.0 | 30.0 |

Source: `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv`.

### 6.4 Paraphrase robustness

Canonical: `results/robustness_paraphrase_test.csv` (n=30).

Probe uses an **expanded synonym list** plus **FLAN-T5 paraphrase fallback** for any prompt synonyms leave unchanged (addresses the prior 6/30 change rate). Re-run `python scripts/eval_paraphrase_test.py` after this change — committed numbers below are from the old nine-pair list and are stale until refreshed.

| Method | Orig Bypass | Para Bypass | Δ |
|--------|------------:|------------:|--:|
| baseline | 1.00 | 1.00 | 0.00 |
| regex | 0.8667 | 0.8667 | 0.00 |
| keyword | 0.9667 | 1.00 | +0.0333 |
| Method A | 0.0667 | 0.0667 | 0.00 |
| Method B | 0.2000 | 0.2333 | +0.0333 |

### 6.5 Edge benign FPR

| Method | Edge FPR | Blocked / Total |
|--------|---------:|-----------------|
| baseline | 0% | 0 / 80 |
| regex | 2.5% | 2 / 80 |
| keyword | 0% | 0 / 80 |
| Method A | **36.25%** | 29 / 80 |
| Method B | **7.5%** | 6 / 80 |

### 6.6 Adaptive attacks (FLAN-T5 variants, n=150)

| Method | Bypass % | True ASR % | Successful attacks | Δ True ASR vs static (pp) |
|--------|---------:|-----------:|-------------------:|--------------------------:|
| baseline | 100.0 | 45.33 | 68 | −1.34 |
| regex | 93.33 | 42.0 | 63 | −4.67 |
| keyword | 96.67 | 45.33 | 68 | −1.34 |
| ProtectAI | 62.67 | 36.00 | 54 | +2.67 |
| Prompt Guard 2 | 82.67 | 40.67 | 61 | +4.00 |
| Method A | **16.67** | **8.67** | **13** | +8.67 |
| Method B | 40.0 | 15.33 | 23 | +2.00 |

### 6.7 Runtime overhead (aux models, CPU)

Paper-reported means on 50 prompts (`results/suite_b/runtime_overhead.csv`):

| Component | Mean ms / prompt | Median ms |
|-----------|-----------------:|----------:|
| Keyword gate | 0.02 | 0.01 |
| Regex gate | 0.22 | 0.07 |
| MiniLM semantic | 20.98 | 17.42 |
| DistilGPT-2 perplexity | 45.03 | 29.10 |
| Method A full `sanitize()` | 64.79 | 46.68 |
| Method B aux (MiniLM + DistilGPT-2) | 66.01 | 45.98 |

Suite A copy: `results/suite_a/runtime_overhead.csv`. See `results/RUNTIME_NOTE.md`.

---

## 7. Ablation Studies

Paper tables need target + judge (True ASR). Omit `--no-llm` for those:

```powershell
python -m src.run_experiments --weighted-ablation
python -m src.run_experiments --hard-trigger-ablation
python -m src.run_experiments --learned-ablation
python -m src.run_experiments --compare-ab
```

Sanitizer-only check (Bypass/FPR only; **True ASR will be 0**):

```powershell
python -m src.run_experiments --hard-trigger-ablation --no-llm
```

→ `results/weighted_ablation_metrics.csv`, `hard_trigger_ablation.csv`, `learned_ablation_metrics.csv`, `ab_comparison_metrics.csv`.

Hard triggers drive most of Method A’s blocking; removing the semantic hard trigger raises Bypass by **+56.66 pp** (paper Table 2). Semantic features dominate Method B (removing all three semantic features → Bypass 100%, True ASR 46.67%).

---

## 8. Judge Human Validation

Author-labeled all **N = 30** no-defense injection (prompt, response) pairs, blind to the judge, using the same COMPLIED/REFUSED rubric.

| Metric | Value |
|--------|------:|
| Agreement | 25/30 = **83.3%** |
| Cohen’s κ | **0.67** |

- Rubric / blind sheet: `data/judge_human_validation/`
- Metrics: `results/judge_human_validation/metrics_summary.txt`

```powershell
python scripts/score_judge_human_validation.py
```

---

## 9. How to Run

### Setup

```powershell
cd Securing_LLMs
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
```

Optional: `$env:HF_TOKEN = "hf_..."` (needed for gated Prompt Guard).

### Main experiments

```powershell
python -m src.run_experiments                  # Suite A then B (full LLM + judge)
python -m src.run_experiments --suite a --no-llm
python -m src.run_experiments --suite b --no-llm
python -m src.run_experiments --latency-only
python -m src.run_experiments --portable-baselines
python -m src.run_experiments --matched-fpr                 # val FPR≤5% fair Bypass table
python -m src.run_experiments --matched-fpr --matched-fpr-skip-deberta
python -m src.run_experiments --loo
python -m src.run_experiments --loo --no-llm
python -m src.run_experiments --loo --loo-skip-robustness   # main metrics only
python -m src.run_experiments --loo --loo-adaptive-max 50   # cap adaptive rewrites
python -m src.run_experiments --loo --llm-batch-size 8      # faster GPU batched generate
# Crash-safe: re-run the same command; OutcomeCache / FeatureCache resume under results/prompt_cache/
```

`--loo` keeps the regular scorer (sanitizer, target LLM, judge) and points it at `data/splits/source_holdout/<fold>/`. Each fold trains only on the other attack sources. After the main metrics, each fold also runs the same after-main stages as the suite: paraphrase, edge-benign, and FLAN-T5 adaptive (fit on that fold's train/val). Per-fold outputs live under `results/loo/<fold>/`; cross-fold tables and four-source averages are in `results/loo/summary.csv` and `results/loo/robustness_*{,_summary}.csv`. Use `--loo-skip-robustness` for main-only, or `--loo-adaptive-max N` to limit adaptive cost on ~200-injection tests. The mixed suite uses the grown combined corpus (`data/splits/test.csv`, ~105 rows).

### Scripts

```powershell
python scripts/eval_protectai.py
python scripts/eval_prompt_guard.py
python scripts/eval_portable_adaptive.py --gate all
python scripts/eval_matched_fpr.py
python scripts/eval_paraphrase_test.py
python scripts/build_judge_human_validation.py
python scripts/build_external_neuralchemy.py
python scripts/build_combined_dataset.py
python scripts/build_source_holdout.py
python scripts/eval_loo_holdout.py
python scripts/verify_source_holdout.py
python scripts/score_judge_human_validation.py
```

### One-shot gate checks (`reproduce.*`)

Runs paraphrase gate-only scoring, offline judge HV scoring, and portable baselines.
Steps 1–2 need no model download beyond what you already use for scoring; step 3
downloads ProtectAI / Prompt Guard from Hugging Face if not cached (`HF_TOKEN` may
be required for Prompt Guard). Does **not** rebuild suite/ablation paper tables.

```powershell
.\scripts\reproduce.ps1
# or: bash scripts/reproduce.sh
```

---

## 10. Reproduce Paper Tables

| Paper table / section | Artifact | Regenerate |
|-----------------------|----------|------------|
| Main comparison (Table 1) | `results/suite_a/metrics.csv`, `results/suite_b/metrics.csv`, `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv` | Suites + `--portable-baselines` |
| Matched-FPR fair baselines | `results/matched_fpr/metrics.csv` | `--matched-fpr` or `python scripts/eval_matched_fpr.py` |
| Ablation (Table 2) | `results/hard_trigger_ablation.csv`, `results/learned_ablation_metrics.csv`, `results/weighted_ablation_metrics.csv` | `--hard-trigger-ablation`, `--learned-ablation`, `--weighted-ablation` (full LLM+judge; `--no-llm` is sanitizer-only and zeros True ASR) |
| Adaptive rewriting (Table 3) | `results/suite_a/robustness_adaptive.csv`, `results/*/adaptive_metrics.csv` | Suite run + `--portable-baselines` |
| Paraphrase robustness | `results/robustness_paraphrase_test.csv` | `python scripts/eval_paraphrase_test.py` |
| Edge FPR | `results/suite_*/robustness_edge_benign.csv` | (with suite run) |
| A vs B matched comparison | `results/ab_comparison_metrics.csv` | `--compare-ab` (full LLM+judge) |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` | `python scripts/score_judge_human_validation.py` |
| Runtime overhead | `results/suite_b/runtime_overhead.csv` | `--latency-only` or suite without `--skip-latency` |

Committed `results/suite_*` and `results/source_holdout/` may still reflect the archived v2.0 / pre-uniform protocols — re-run after v3.0 / current holdout before citing. `--no-llm` is sanitizer-only: Bypass/FPR are live; True ASR is 0 because there is no judge.

---

## 11. Output Files

See suite folders under `results/suite_a/` and `results/suite_b/` for `metrics.csv`, `logs.jsonl`, robustness CSVs, figures, and `robustness_note.txt`. Shared ablations and portable-gate outputs live under `results/` as listed above.

---

## 12. Hardware & GPU Notes

Paper environment (§Experimental setup): Windows 11, 22-thread CPU, 15.4 GB RAM, RTX 4070 Laptop 8 GB, Python 3.13, PyTorch 2.6 / CUDA 12.4.

| Component | Device |
|-----------|--------|
| MiniLM + DistilGPT-2 + portable gates | **CPU** (intentional) |
| Target / judge (Qwen2.5-3B) | CUDA if available |
| `--no-llm` / `SKIP_LLM=1` | No Qwen load |

Models load sequentially to reduce OOM risk. See `ARTIFACT.md`.

---

## 13. Reproducibility

- Seed **42** (`random`, `numpy`, `torch`, `transformers.set_seed`)
- Frozen train / val / test CSVs
- Greedy decoding for target, judge, and attack generator
- Load failures for target/judge **raise** (no silent mock) unless `SKIP_LLM` / `SKIP_JUDGE` / `--no-llm` / `--no-judge`

---

## 14. Limitations

Aligned with paper §Limitations:

- Mixed test is larger in v3.0 (~60 injection / ~45 benign) but still research-scale; Wilson intervals remain wide — we do not rank Method A vs Method B statistically. Archived v2.0 was 30 / 23.
- Train/test injections share four public sources (distribution overlap); adaptive FLAN-T5 rewriting is the main shift probe
- **Direct, single-turn, English, text-only** injection only; indirect / agentic / multimodal out of scope
- True ASR depends on an LLM judge (same checkpoint as target); blind author labeling: 83.3% agreement, κ = 0.67
- Method A edge-benign FPR is high (36.25%); semantic hard trigger drives both security and usability cost
- Prompt Guard may require Hugging Face gated access

---

## 15. Dependencies

Pinned in `requirements.txt` / `environment.yml`:

```
pandas, numpy, scikit-learn, matplotlib, scipy, tqdm
torch, transformers, huggingface_hub, sentence-transformers
```

```powershell
pip install -r requirements.txt
```

### Related work (short)

- OWASP Top 10 for LLMs — LLM01 Prompt Injection
- Jailbreak / GCG-style attack literature
- Structured / alignment defenses (e.g. StruQ, SecAlign)
- Industry regex / heuristic firewalls — related as **baselines**

---

*For metric definitions, start with `results/suite_*/robustness_note.txt` and `ARTIFACT.md`.*
