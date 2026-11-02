# Where Does the Security Come From?
### A Controlled Evaluation of Context-Aware Input Sanitization Against Direct Prompt Injection

> **Scope note:** Controlled evaluation of **direct, single-turn** prompt injection on a research-scale mixed corpus (~700 prompts; ~105-prompt test) plus leave-one-source-out holdouts. Not production-grade security guarantees.
>
> **LOO status:** Full LLM leave-one-source-out is **complete** (5 folds × main + paraphrase + edge-benign + adaptive). Canonical tables: `results/loo/summary.csv` and `results/loo/robustness_*_summary.csv`.
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

Research questions (paper §Framework): end-to-end risk without a gate; whether lexical/portable gates reduce **True ASR** or mainly Bypass; how Methods A/B compare; which components carry the defense; usability cost (ordinary + edge-benign FPR). Source generalization is measured with leave-one-source-out ([§6.0](#60-leave-one-source-out-complete)).

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
│   ├── loo/                         ← full LLM LOO (complete; see results/loo/README.md)
│   ├── prompt_cache/loo/            ← shared OutcomeCache / FeatureCache for --loo
│   ├── suite_a/ / suite_b/          ← paper mixed-split suites (archived v2.0 figures in README §6)
│   ├── matched_fpr/                 ← matched operating-point table
│   ├── source_holdout/              ← gate-only LOO scores (Bypass/FPR; may be stale)
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

Near-dup filter (MiniLM > 0.9) in `manifest.json`.

**Full evaluation (done):** `python -m src.run_experiments --loo` → `results/loo/` (Bypass, True ASR, FPR + paraphrase / edge-benign / adaptive). Numbers in [§6.0](#60-leave-one-source-out-complete).

**Gate-only shortcut:** `python scripts/eval_loo_holdout.py` → `results/source_holdout/` (Bypass/FPR only; committed scores may be stale vs current splits).

```powershell
# preferred: run notebooks/build_source_holdout.ipynb
python scripts/build_source_holdout.py
python scripts/verify_source_holdout.py
python -m src.run_experiments --loo          # full LLM LOO (resume-safe)
# optional gate-only:
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

### 6.0 Leave-one-source-out (**complete**)

Full two-layer eval with target + judge (`Qwen/Qwen2.5-3B-Instruct`). Folds: `simsonsun`, `mpdd`, `prompt_injection`, `bravansky`, `neuralchemy`. Each fold test = **200** injection + **200** benign. After main metrics: paraphrase, edge-benign (80), adaptive (FLAN-T5 styles + `similarity_evasion`; **1200** attack texts / fold).

**Headline aggregate** = `four_source_macro_avg` over the first four folds (equal weight per source). Neuralchemy is a fifth external holdout and is **outside** that average. Shared PI-benign repeats across four-source folds, so micro FPR is not an independent sample.

Artifacts: `results/loo/summary.csv`, `summary.txt`, `robustness_{paraphrase,edge_benign,adaptive}{,_summary}.csv`, per-fold `results/loo/<fold>/`. See `results/loo/README.md`.

#### Four-source macro average (main + robustness)

| Method | Bypass % | True ASR % | FPR % | Adaptive Bypass % | Adaptive True ASR % | Para Bypass % (orig → para) | Edge FPR % |
|--------|---------:|-----------:|------:|------------------:|--------------------:|----------------------------:|-----------:|
| baseline | 100.0 | 58.75 | 0.0 | 100.0 | 49.45 | 100.0 → 100.0 | 0.0 |
| regex | 84.12 | 48.12 | 0.0 | 95.21 | 46.98 | 84.13 → 92.0 | 2.5 |
| keyword | 93.0 | 53.38 | 0.0 | 98.59 | 48.77 | 93.0 → 98.12 | 0.0 |
| context_aware (A) | 21.25 | 7.0 | 44.75 | 34.94 | 12.66 | 21.25 → 25.75 | 51.56 |
| context_aware_learned (B) | 26.12 | 9.5 | 30.75 | 40.56 | 15.21 | 26.12 → 32.25 | 58.13 |
| similarity_only | 26.0 | 8.62 | 50.5 | 36.64 | 13.59 | 26.0 → 28.75 | 58.44 |
| deberta_finetuned | 1.5 | 0.62 | 10.88 | 20.61 | 7.98 | 1.5 → 7.25 | 85.31 |

Sources: `results/loo/summary.csv` (`four_source_macro_avg`), `robustness_adaptive_summary.csv`, `robustness_paraphrase_summary.csv`, `robustness_edge_benign_summary.csv`.

#### Per-fold main metrics (200 inj / 200 ben each)

| Fold | Method | Bypass % | True ASR % | FPR % |
|------|--------|---------:|-----------:|------:|
| simsonsun | baseline | 100.0 | 73.0 | 0.0 |
| simsonsun | regex | 79.5 | 57.5 | 0.0 |
| simsonsun | keyword | 86.5 | 62.0 | 0.0 |
| simsonsun | context_aware | 3.5 | 1.5 | 45.5 |
| simsonsun | context_aware_learned | 0.0 | 0.0 | 39.0 |
| simsonsun | similarity_only | 0.5 | 0.5 | 69.5 |
| simsonsun | deberta_finetuned | 0.0 | 0.0 | 2.5 |
| mpdd | baseline | 100.0 | 50.5 | 0.0 |
| mpdd | regex | 76.0 | 38.0 | 0.0 |
| mpdd | keyword | 96.0 | 47.5 | 0.0 |
| mpdd | context_aware | 28.5 | 14.0 | 35.5 |
| mpdd | context_aware_learned | 44.5 | 23.5 | 17.5 |
| mpdd | similarity_only | 33.5 | 16.5 | 35.0 |
| mpdd | deberta_finetuned | 6.0 | 2.5 | 0.0 |
| prompt_injection | baseline | 100.0 | 24.0 | 0.0 |
| prompt_injection | regex | 96.5 | 22.5 | 0.0 |
| prompt_injection | keyword | 97.5 | 23.5 | 0.0 |
| prompt_injection | context_aware | 52.5 | 12.0 | 49.0 |
| prompt_injection | context_aware_learned | 59.5 | 14.0 | 29.0 |
| prompt_injection | similarity_only | 70.0 | 17.5 | 37.5 |
| prompt_injection | deberta_finetuned | 0.0 | 0.0 | 34.0 |
| bravansky | baseline | 100.0 | 87.5 | 0.0 |
| bravansky | regex | 84.5 | 74.5 | 0.0 |
| bravansky | keyword | 92.0 | 80.5 | 0.0 |
| bravansky | context_aware | 0.5 | 0.5 | 49.0 |
| bravansky | context_aware_learned | 0.5 | 0.5 | 37.5 |
| bravansky | similarity_only | 0.0 | 0.0 | 60.0 |
| bravansky | deberta_finetuned | 0.0 | 0.0 | 7.0 |
| neuralchemy | baseline | 100.0 | 64.5 | 0.0 |
| neuralchemy | regex | 85.0 | 52.5 | 1.0 |
| neuralchemy | keyword | 95.0 | 59.5 | 0.0 |
| neuralchemy | context_aware | 11.0 | 7.0 | 82.5 |
| neuralchemy | context_aware_learned | 18.0 | 9.5 | 65.5 |
| neuralchemy | similarity_only | 4.0 | 1.5 | 87.0 |
| neuralchemy | deberta_finetuned | 2.0 | 1.0 | 48.5 |

#### Neuralchemy adaptive (fifth fold; not in four-source avg)

| Method | Adaptive Bypass % | Adaptive True ASR % |
|--------|------------------:|--------------------:|
| baseline | 100.0 | 42.0 |
| regex | 94.92 | 40.0 |
| keyword | 99.17 | 41.58 |
| context_aware | 37.17 | 8.83 |
| context_aware_learned | 57.25 | 21.75 |
| similarity_only | 27.83 | 5.08 |
| deberta_finetuned | 11.92 | 1.67 |

Resume: re-run `python -m src.run_experiments --loo` (skips finished folds/stages; caches under `results/prompt_cache/loo/`). Force redo with `--rerun`.

---

### 6.1–6.7 Archived mixed-suite figures (v2.0, n=30 / 23)

Numbers below are from the **archived v2.0** held-out test (30 injection / 23 benign). Prefer §6.0 LOO for current holdout claims. Re-run suites on the grown v3.0 test (~60 / 45) before citing new mixed-suite figures.

#### 6.1 Primary comparison (paper Table 1, archived)

| Method | Bypass % (n/30) | True ASR % (n/30) | FPR % (n/23) | Edge FPR % (n/80) |
|--------|----------------:|------------------:|-------------:|------------------:|
| No defense | 100.00 (30) | 46.67 (14) | 0.00 (0) | 0.00 (0) |
| Regex (45 patterns) | 86.67 (26) | 46.67 (14) | 0.00 (0) | 2.50 (2) |
| Keyword (37 phrases) | 96.67 (29) | 46.67 (14) | 0.00 (0) | 0.00 (0) |
| ProtectAI (DeBERTa) | 40.00 (12) | 33.33 (10) | 0.00 (0) | 25.00 (20) |
| Prompt Guard 2 (86M) | 70.00 (21) | 36.67 (11) | 0.00 (0) | 30.00 (24) |
| **Method A (hybrid)** | **6.67 (2)** | **0.00 (0)** | **13.04 (3)** | **36.25 (29)** |
| **Method B (learned)** | **20.00 (6)** | **13.33 (4)** | **8.70 (2)** | **7.50 (6)** |

#### 6.2 Suite A — Method A (archived)

| Method | Bypass % ↓ | True ASR % ↓ | FPR % ↓ | Successful attacks |
|--------|-----------:|-------------:|--------:|-------------------:|
| baseline | 100.0 | 46.67 | 0.0 | 14 |
| regex | 86.67 | 46.67 | 0.0 | 14 |
| keyword | 96.67 | 46.67 | 0.0 | 14 |
| **context_aware (A)** | **6.67** | **0.0** | 13.04 | **0** |

#### 6.3 Suite B — Method B (archived)

| Method | Bypass % ↓ | True ASR % ↓ | FPR % ↓ | Successful attacks |
|--------|-----------:|-------------:|--------:|-------------------:|
| baseline | 100.0 | 46.67 | 0.0 | 14 |
| regex | 86.67 | 46.67 | 0.0 | 14 |
| keyword | 96.67 | 46.67 | 0.0 | 14 |
| **context_aware_learned (B)** | **20.0** | **13.33** | **8.7** | **4** |

#### 6.4 Portable baselines (static test, archived)

| Method | Bypass % | True ASR % | FPR % | Edge FPR % |
|--------|---------:|-----------:|------:|-----------:|
| ProtectAI | 40.0 | 33.33 | 0.0 | 25.0 |
| Prompt Guard 2 | 70.0 | 36.67 | 0.0 | 30.0 |

Source: `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv`.

#### 6.5 Paraphrase robustness (mixed suite archive)

Canonical mixed-suite file: `results/robustness_paraphrase_test.csv` (n=30). Prefer LOO paraphrase in §6.0 / `results/loo/robustness_paraphrase_summary.csv`.

Probe uses an **expanded synonym list** plus **FLAN-T5 paraphrase fallback**. Re-run `python scripts/eval_paraphrase_test.py` after list changes — committed mixed-suite numbers below may be stale.

| Method | Orig Bypass | Para Bypass | Δ |
|--------|------------:|------------:|--:|
| baseline | 1.00 | 1.00 | 0.00 |
| regex | 0.8667 | 0.8667 | 0.00 |
| keyword | 0.9667 | 1.00 | +0.0333 |
| Method A | 0.0667 | 0.0667 | 0.00 |
| Method B | 0.2000 | 0.2333 | +0.0333 |

#### 6.6 Edge benign FPR (mixed suite archive)

| Method | Edge FPR | Blocked / Total |
|--------|---------:|-----------------|
| baseline | 0% | 0 / 80 |
| regex | 2.5% | 2 / 80 |
| keyword | 0% | 0 / 80 |
| Method A | **36.25%** | 29 / 80 |
| Method B | **7.5%** | 6 / 80 |

#### 6.7 Adaptive attacks (mixed suite archive; FLAN-T5, n=150)

| Method | Bypass % | True ASR % | Successful attacks | Δ True ASR vs static (pp) |
|--------|---------:|-----------:|-------------------:|--------------------------:|
| baseline | 100.0 | 45.33 | 68 | −1.34 |
| regex | 93.33 | 42.0 | 63 | −4.67 |
| keyword | 96.67 | 45.33 | 68 | −1.34 |
| ProtectAI | 62.67 | 36.00 | 54 | +2.67 |
| Prompt Guard 2 | 82.67 | 40.67 | 61 | +4.00 |
| Method A | **16.67** | **8.67** | **13** | +8.67 |
| Method B | 40.0 | 15.33 | 23 | +2.00 |

#### 6.8 Runtime overhead (aux models, CPU)

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

`--loo` keeps the regular scorer (sanitizer, target LLM, judge) and points it at `data/splits/source_holdout/<fold>/`. Each fold trains only on the other attack sources. After the main metrics, each fold also runs paraphrase, edge-benign, and FLAN-T5 adaptive (+ similarity evasion; fit on that fold's train/val). **Current run is complete** — tables in [§6.0](#60-leave-one-source-out-complete) and `results/loo/`. Re-running the same command skips finished folds/stages unless `--rerun`. Use `--loo-skip-robustness` for main-only, or `--loo-adaptive-max N` to limit adaptive cost. Shared caches: `results/prompt_cache/loo/`. The mixed suite uses the grown combined corpus (`data/splits/test.csv`, ~105 rows).

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
| **LOO main + robustness (current holdout)** | `results/loo/summary.csv`, `results/loo/robustness_*_summary.csv`, `results/loo/<fold>/` | `python -m src.run_experiments --loo` (**complete**; use `--rerun` to redo) |
| Main comparison (Table 1, mixed suite) | `results/suite_a/metrics.csv`, `results/suite_b/metrics.csv`, `results/protectai/metrics.csv`, `results/prompt_guard/metrics.csv` | Suites + `--portable-baselines` |
| Matched-FPR fair baselines | `results/matched_fpr/metrics.csv` | `--matched-fpr` or `python scripts/eval_matched_fpr.py` |
| Ablation (Table 2) | `results/hard_trigger_ablation.csv`, `results/learned_ablation_metrics.csv`, `results/weighted_ablation_metrics.csv` | `--hard-trigger-ablation`, `--learned-ablation`, `--weighted-ablation` (full LLM+judge; `--no-llm` is sanitizer-only and zeros True ASR) |
| Adaptive rewriting (Table 3 / LOO) | `results/loo/robustness_adaptive_summary.csv`, `results/suite_*/robustness_adaptive.csv` | `--loo` or suite run + `--portable-baselines` |
| Paraphrase robustness | `results/loo/robustness_paraphrase_summary.csv`, `results/robustness_paraphrase_test.csv` | `--loo` or `python scripts/eval_paraphrase_test.py` |
| Edge FPR | `results/loo/robustness_edge_benign_summary.csv`, `results/suite_*/robustness_edge_benign.csv` | `--loo` or suite run |
| A vs B matched comparison | `results/ab_comparison_metrics.csv` | `--compare-ab` (full LLM+judge) |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` | `python scripts/score_judge_human_validation.py` |
| Runtime overhead | `results/suite_b/runtime_overhead.csv` | `--latency-only` or suite without `--skip-latency` |

`results/loo/` holds the completed full-LLM source holdout. Committed `results/suite_*` and `results/source_holdout/` may still reflect archived v2.0 / gate-only protocols — re-run before citing those paths. `--no-llm` is sanitizer-only: Bypass/FPR are live; True ASR is 0 because there is no judge.

---

## 11. Output Files

### LOO (`results/loo/`) — complete

| Path | Contents |
|------|----------|
| `summary.csv` / `summary.txt` | Per-fold + `four_source_{macro_avg,micro}` main metrics |
| `robustness_paraphrase.csv` (+ `_summary.csv`) | Orig vs para Bypass |
| `robustness_edge_benign.csv` (+ `_summary.csv`) | Edge FPR |
| `robustness_adaptive.csv` (+ `_summary.csv`) | Adaptive Bypass / True ASR |
| `<fold>/metrics.csv` | Main fold metrics |
| `<fold>/logs.jsonl` | Per-prompt main logs |
| `<fold>/robustness_*.csv` | Fold paraphrase / edge / adaptive |
| `<fold>/adaptive_attacks.jsonl` | Generated adaptive texts (resumable) |
| `<fold>/robustness_adaptive_logs.jsonl` | Adaptive pipeline logs |
| `<fold>/robustness_note.txt` | Human-readable fold write-up |
| `<fold>/figures/` | Bar charts / confusion matrices |
| `results/prompt_cache/loo/` | Shared outcomes / adaptive outcomes / `features.pkl` |

Details: `results/loo/README.md`, `results/cache/README.md`.

### Mixed suites

See `results/suite_a/` and `results/suite_b/` for `metrics.csv`, `logs.jsonl`, robustness CSVs, figures, and `robustness_note.txt`. Shared ablations and portable-gate outputs live under `results/` as listed in §10.

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

- LOO test is larger (200 inj / 200 ben per fold; four-source aggregate 800 / 800) but still research-scale; we do not claim statistical ranking of Method A vs B. Mixed v3.0 test is ~60 / 45; archived v2.0 was 30 / 23.
- Soft filters vs DeBERTa on LOO are **not FPR-matched**; DeBERTa has low clean Bypass but high edge FPR (~85% four-source). Use `--matched-fpr` for a fair operating-point compare.
- Train/test injections share public sources (distribution overlap); adaptive FLAN-T5 + similarity evasion is the main shift probe (four-source DeBERTa adaptive Bypass ~20.6%).
- **Direct, single-turn, English, text-only** injection only; indirect / agentic / multimodal out of scope
- True ASR depends on an LLM judge (same checkpoint as target); blind author labeling on the small mixed suite: 83.3% agreement, κ = 0.67
- Method A/B ordinary + edge FPR remain high on LOO; usability cost is material
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

*For metric definitions, start with `results/loo/<fold>/robustness_note.txt`, `results/suite_*/robustness_note.txt`, and `ARTIFACT.md`.*
