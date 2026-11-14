# Where Does the Security Come From?
### A Controlled Evaluation of Context-Aware Input Sanitization Against Direct Prompt Injection

> **Scope note:** Controlled evaluation of **direct, single-turn** prompt injection on a research-scale mixed corpus (~700 prompts; ~105-prompt test) plus leave-one-source-out holdouts. Not production-grade security guarantees.
>
> **Paper reporting (canonical):** main holdout = leave-one-source-out at **val FPR ≈ 5%** (closest), Method A with `semantic_hard_sim` sweep, full suite + True ASR–FPR curves; ProtectAI / Prompt Guard 2 in the **same** tables; **Neuralchemy** as its own column (not in the four-source average); **ID** = mixed suite matched-FPR on the combined-dataset random split; Wilson 95% CIs; near-dup + train-benign inventory. Cite [`results/loo_matched_fpr/PAPER_TABLES.md`](results/loo_matched_fpr/PAPER_TABLES.md).
>
> **One-shot paper run:** `python scripts/run_paper.py` (or `python -m src.run_experiments --paper`) → rebuilds that package. Index: [`results/PAPER.md`](results/PAPER.md).
>
> **LOO status:** Default/F1 LOO is **complete** under `results/loo/` (contrast only). **Main LOO report** = `results/loo_matched_fpr/`. **Mixed suite matched-FPR** (ID column; FPR ≤ 5%): `results/matched_fpr/`. Folder map: `results/README.md`.
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

Research questions (paper §Framework): end-to-end risk without a gate; whether lexical/portable gates reduce **True ASR** or mainly Bypass; how Methods A/B compare; which components carry the defense; usability cost (ordinary + edge-benign FPR). Source generalization is measured with leave-one-source-out ([§6.0](#60-leave-one-source-out--matched-fpr--5-main-loo-report)).

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
├── data/                  ← raw / cleaned / mixed splits / source_holdout / Neuralchemy
├── data_cards/            ← dataset license + card
├── notebooks/             ← preprocess, EDA, build_source_holdout.ipynb
├── scripts/               ← see scripts/README.md  ★ run_paper.py = one-shot paper
├── src/                   ← sanitizers, evaluation, run_experiments.py CLI
├── results/
│   ├── PAPER.md           ← ★ what to cite (start here for paper numbers)
│   ├── README.md          ← folder roles
│   ├── loo_matched_fpr/   ← ★ MAIN holdout (FPR≈5%) + PAPER_TABLES.md
│   ├── matched_fpr/       ← mixed suite matched-FPR (combined split, FPR≤5%)
│   ├── loo/               ← default/F1 LOO contrast
│   ├── suite_a/ suite_b/  ← mixed suites (default thr; same combined split)
│   ├── protectai/ prompt_guard/
│   ├── prompt_cache/      ← shared LLM/judge caches
│   └── …
├── ARTIFACT.md / CITATION.cff / LICENSE / requirements.txt
└── README.md
```

| Want… | Open / run |
|-------|------------|
| **All paper holdout tables** | `results/loo_matched_fpr/PAPER_TABLES.md` |
| **One command to regenerate them** | `python scripts/run_paper.py` |
| **What each results/ folder means** | `results/PAPER.md` + `results/README.md` |
| **What each script does** | `scripts/README.md` |

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

### Leave-one-source-out (canonical holdout)

Canonical splits: `data/splits/source_holdout/`. **Build in the notebook** `notebooks/build_source_holdout.ipynb` (run all cells). Optional CLI wrapper that executes those cells: `python scripts/build_source_holdout.py`.

| Rule | Behavior |
|------|----------|
| Hold out | Entire attack source (held-out injections never in train/val) |
| Shared test benign | Fixed **200** PI-benign on **every** fold test (incl. Neuralchemy); **never** in train/val |
| Fit benign | **Alpaca** instructions only: **200 train + 100 val** (same pool every fold; no PI / ChatGPT) |
| Four-source sizes | Identical **680 / 220 / 400** |
| Attacks | Test **200** held-out; train pool **200 from each** other raw source; val 20% within each attack source |
| Fifth set | Neuralchemy: 200 Neuralchemy attacks + shared 200 PI-benign; fit attacks from all four sources |
| Inventory | `train_benign_by_fold.csv` + `manifest.json` |

Near-dup filter (MiniLM > 0.9) in `manifest.json`.

**Main LOO report:** `python -m src.run_experiments --loo-matched-fpr` → `results/loo_matched_fpr/` (val FPR ≈ 5%; see [§6.0](#60-leave-one-source-out--matched-fpr--5-main-loo-report)).

**Default-threshold LOO (done):** `python -m src.run_experiments --loo` → `results/loo/` (contrast + paraphrase / edge / adaptive; [§6.0b](#60b-default--f1-loo-operating-point-contrast-complete)).

**Gate-only shortcut:** `python scripts/eval_loo_holdout.py` → `results/source_holdout/` (Bypass/FPR only; committed scores may be stale vs current splits).

```powershell
# preferred: run notebooks/build_source_holdout.ipynb
python scripts/build_source_holdout.py
python scripts/verify_source_holdout.py
python -m src.run_experiments --loo-matched-fpr   # MAIN LOO report
python -m src.run_experiments --loo               # default thr + robustness (LLM cache only)
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
- **Matched FPR (~5%)** — two paths: (1) **mixed suite matched-FPR** `--matched-fpr` = combined-dataset random split, val **FPR ≤ 5%** (`results/matched_fpr/`); (2) **LOO main** `--loo-matched-fpr` = val **FPR ≈ 5%** (closest) + full suite + curves (`results/loo_matched_fpr/`; Method A also sweeps `semantic_hard_sim`). Portables keep published weights; only the decision threshold is retuned.

---

## 5. Evaluation Pipeline

Target and judge: **`Qwen/Qwen2.5-3B-Instruct`**. Adaptive paraphrases: **`google/flan-t5-base`**.

```
Phase 1 — sanitize (+ target LLM if not blocked)
Phase 2 — judge COMPLIED / REFUSED on unblocked injections
Robustness — expanded paraphrase, edge-benign, FLAN-T5 adaptive
             (+ similarity-evasion rewrite minimizing max cosine to train gallery)
```

**Speed path (default with LLM):** one greedy Qwen reply + judge label per unique *injection* prompt (`results/prompt_cache/`), then True ASR is a per-method lookup; benign prompts never call Qwen/judge; MiniLM embeddings + DistilGPT-2 PPL are cached and reused across methods/folds; generate-all → unload → judge-all with append-only **generation** resume. Filters/metrics always rescore (no `metrics.csv` skip). Batch with `--llm-batch-size N` (or `LLM_CACHE_BATCH_SIZE`). Regenerate replies with `--no-outcome-cache` / `--rerun`.

Same phases on `--loo` (per `data/splits/source_holdout/<fold>/`), writing under `results/loo/`. Adaptive logs include `attack_style` (`similarity_evasion` is the filter-targeting variant) and gallery similarity fields.

| `--suite` | Methods | Output |
|-----------|---------|--------|
| `a` | baseline, regex, keyword, Method A + similarity_only + deberta_finetuned | `results/suite_a/` |
| `b` | baseline, regex, keyword, Method B + similarity_only + deberta_finetuned | `results/suite_b/` |
| `ab` (default) | A then B (each with fair baselines) | both folders |
| `both` | A + B + fair baselines in one run | `results/` |
| `--loo` | fold fair baselines + A/B + after-main (default thr) | `results/loo/` |
| `--loo-matched-fpr` | **main LOO:** val FPR≈5% + full suite + ASR–FPR curves | `results/loo_matched_fpr/` |
| `--matched-fpr` | **mixed suite matched-FPR:** combined split, val FPR ≤ 5% (not LOO) | `results/matched_fpr/` |

---

## 6. Results

### 6.0 Leave-one-source-out — matched FPR ≈ 5% (**main LOO report**)

<a id="loo-matched-fpr-main"></a>

For each fold, choose each method’s threshold on **that fold’s validation** set to **hit val FPR ≈ 5%** (closest achievable; `--matched-fpr-target`), then run the **full LOO suite** at those thresholds — test Bypass / True ASR / FPR, paraphrase, edge-benign, adaptive — plus True-ASR-vs-FPR curves. Method A also sweeps `semantic_hard_sim` (default **0.3**). Soft weights / features unchanged (no A/B redesign).

```powershell
python -m src.run_experiments --loo-matched-fpr
python -m src.run_experiments --loo-matched-fpr --loo-skip-robustness   # main + curves only
python -m src.run_experiments --loo-matched-fpr --matched-fpr-skip-deberta
```

Artifacts: `results/loo_matched_fpr/summary.csv`, `thresholds.csv`, `robustness_*{,_summary}.csv`, `true_asr_vs_fpr_curves.csv`, `figures/`, per-fold `results/loo_matched_fpr/<fold>/`.

**Canonical paper tables** (ProtectAI + Prompt Guard 2, Neuralchemy column, mixed-suite vs LOO gap, Wilson CIs, dedup, train-benign inventory): [`results/loo_matched_fpr/PAPER_TABLES.md`](results/loo_matched_fpr/PAPER_TABLES.md). Rebuild with `python scripts/build_loo_matched_fpr_paper_tables.py` (after `python scripts/eval_loo_matched_fpr_portables.py` for portables).

**Status: complete** (rebuild after held-out benign fix). Headline aggregate = `four_source_macro_avg` (equal weight per source). **Neuralchemy** is reported as its own column/block (outside that average). Each fold test = **200** inj + **200** held-out PI-benign; fit uses **300** Alpaca (100 val) so a 5% FPR threshold is identifiable; adaptive = **1200** attacks / fold; edge-benign = **80**.

#### Main comparison (mixed suite matched-FPR / ID vs LOO-4 vs Neuralchemy; ≈ 5%)

| Method | ID Bypass % | ID FPR % | LOO-4 macro Bypass % | LOO-4 macro True ASR % | LOO-4 macro FPR % | Neuralchemy Bypass % | Neuralchemy True ASR % | Neuralchemy FPR % |
|--------|------------:|---------:|---------------------:|-----------------------:|------------------:|---------------------:|-----------------------:|------------------:|
| baseline | — | — | 100.00 | 58.12 | 0.00 | 100.00 | 64.50 | 0.00 |
| regex | — | — | 84.12 | 47.62 | 0.00 | 85.00 | 52.50 | 0.00 |
| keyword | 58.33 | 0.00 | 59.75 | 33.00 | 0.00 | 70.50 | 40.50 | 0.00 |
| **context_aware (A)** | **11.67** | 6.67 | **41.62** | **16.50** | **4.50** | **49.50** | **30.00** | **3.50** |
| context_aware_learned (B) | 13.33 | 0.00 | 58.25 | 24.50 | 4.00 | 81.00 | 50.50 | 1.50 |
| similarity_only | 30.00 | 2.22 | 76.38 | 38.12 | 2.50 | 91.00 | 59.50 | 0.50 |
| deberta_finetuned | 3.33 | 0.00 | 2.25 | 0.75 | 0.38 | 2.50 | 1.00 | 0.00 |
| protectai | 10.00 | 6.67 | 11.62 | 7.12 | 2.25 | 4.50 | 2.00 | 4.50 |
| prompt_guard | 3.33 | 6.67 | 3.62 | 1.88 | 1.00 | 1.00 | 0.00 | 1.00 |

**ID** = **mixed suite matched-FPR** on the combined-dataset random split (`results/matched_fpr/`; val FPR ≤ 5%). LOO-4 **macro** = equal weight per held-out source. Neuralchemy is outside that average. Fit benign = Alpaca 200/100; test benign = held-out PI-benign on every fold. Full Wilson / pooled / per-fold / dedup / Table E: [`PAPER_TABLES.md`](results/loo_matched_fpr/PAPER_TABLES.md).

#### Four-source macro average (matched FPR ≈ 5%; main)

| Method | Bypass % | True ASR % | FPR % |
|--------|---------:|-----------:|------:|
| baseline | 100.00 | 58.12 | 0.00 |
| regex | 84.12 | 47.62 | 0.00 |
| keyword | 59.75 | 33.00 | 0.00 |
| **context_aware (A)** | **41.62** | **16.50** | **4.50** |
| context_aware_learned (B) | 58.25 | 24.50 | 4.00 |
| similarity_only | 76.38 | 38.12 | 2.50 |
| deberta_finetuned | 2.25 | 0.75 | 0.38 |
| protectai | 11.62 | 7.12 | 2.25 |
| prompt_guard | 3.62 | 1.88 | 1.00 |

Paraphrase / edge / adaptive at the new matched thresholds: re-run without `--loo-skip-robustness` (committed robustness CSVs may predate the held-out benign rebuild).

Sources: `results/loo_matched_fpr/summary.csv` (`four_source_macro_avg`).

#### Per-fold + Wilson + pooled

Full per-fold Bypass / True ASR / FPR with **Wilson 95% CIs**, four-source **macro** and **pooled (micro)**, ProtectAI / Prompt Guard 2, Neuralchemy, dedup counts, and Alpaca train-benign inventory: [`results/loo_matched_fpr/PAPER_TABLES.md`](results/loo_matched_fpr/PAPER_TABLES.md) (Tables B–E).

Resume: re-run `python -m src.run_experiments --loo-matched-fpr` (LLM/judge + feature caches under `results/prompt_cache/loo/`; filters always rescore). Force regenerating Qwen/judge with `--no-outcome-cache`.

### 6.0b Default / F1 LOO operating point (**contrast**; complete)

Full two-layer eval with target + judge (`Qwen/Qwen2.5-3B-Instruct`) at each method’s **default / F1-tuned** threshold (not FPR-matched). Folds: `simsonsun`, `mpdd`, `prompt_injection`, `bravansky`, `neuralchemy`. Each fold test = **200** injection + **200** benign. After main metrics: paraphrase, edge-benign (80), adaptive (FLAN-T5 styles + `similarity_evasion`; **1200** attack texts / fold).

**Headline aggregate** = `four_source_macro_avg` over the first four folds (equal weight per source). Neuralchemy is a fifth external holdout and is **outside** that average. Shared PI-benign repeats across four-source folds, so micro FPR is not an independent sample.

Artifacts: `results/loo/summary.csv`, `summary.txt`, `robustness_{paraphrase,edge_benign,adaptive}{,_summary}.csv`, per-fold `results/loo/<fold>/`. See `results/loo/README.md`.

#### Four-source macro average (default thr; main + robustness)

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

Resume: re-run `python -m src.run_experiments --loo` (always rescores filters; reuses Qwen/judge + features under `results/prompt_cache/loo/`). Force regenerating Qwen/judge with `--no-outcome-cache`.

---

### 6.1–6.7 Archived mixed-suite figures (v2.0, n=30 / 23)

Numbers below are from the **archived v2.0** held-out test (30 injection / 23 benign). Prefer §6.0 matched-FPR LOO for current holdout claims; §6.0b is the default-threshold contrast. Re-run suites on the grown v3.0 test (~60 / 45) before citing new mixed-suite figures.

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

Canonical mixed-suite file: `results/robustness_paraphrase_test.csv` (n=30). Prefer matched-FPR LOO paraphrase in `results/loo_matched_fpr/`; default-thr contrast: `results/loo/robustness_paraphrase_summary.csv`.

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

### ★ One-shot paper pipeline (recommended)

Builds / refreshes everything for the **main holdout report**: verify splits → **mixed suite matched-FPR** → LOO matched-FPR ≈ 5% → ProtectAI + Prompt Guard 2 → [`PAPER_TABLES.md`](results/loo_matched_fpr/PAPER_TABLES.md).

```powershell
python scripts/run_paper.py
# or:
python -m src.run_experiments --paper

# Also suites + default LOO + ablations + judge HV:
python scripts/run_paper.py --mode full

# Tables only (no long re-run):
python scripts/run_paper.py --skip-loo-matched-fpr --skip-portables
python scripts/build_loo_matched_fpr_paper_tables.py
```

Resume-safe LLM/judge via `results/prompt_cache/` (filters always rescore). Force regenerating replies with `--no-outcome-cache`. Index: [`results/PAPER.md`](results/PAPER.md). Script catalog: [`scripts/README.md`](scripts/README.md).

### Individual experiments

```powershell
python -m src.run_experiments --loo-matched-fpr   # main LOO only
python -m src.run_experiments --loo               # default/F1 LOO contrast
python -m src.run_experiments --matched-fpr       # mixed suite matched-FPR (ID column)
python -m src.run_experiments --suite ab          # mixed suites A then B
python -m src.run_experiments --portable-baselines
python -m src.run_experiments --hard-trigger-ablation
python -m src.run_experiments --help
```

### Data / utility scripts

See [`scripts/README.md`](scripts/README.md). Short gate-only smoke (not full paper):

```powershell
.\scripts\reproduce.ps1
```

---

## 10. Reproduce Paper Tables

**Preferred:** one command regenerates the main holdout package:

```powershell
python scripts/run_paper.py
# → results/loo_matched_fpr/PAPER_TABLES.md
```

| Paper table / section | Artifact | Regenerate |
|-----------------------|----------|------------|
| **Main holdout tables** (ID / LOO-4 / Neuralchemy, Wilson, ProtectAI / Prompt Guard) | `results/loo_matched_fpr/PAPER_TABLES.md` | `python scripts/run_paper.py` |
| LOO matched-FPR suite (full + ASR–FPR curves) | `results/loo_matched_fpr/` | `--loo-matched-fpr` (included in `run_paper.py`) |
| Mixed suite matched-FPR (**ID** column) | `results/matched_fpr/metrics.csv` | `--matched-fpr` (included in `run_paper.py`) |
| LOO default/F1 (contrast) | `results/loo/` | `--loo` or `run_paper.py --mode full` |
| Mixed suite (archived) | `results/suite_a/`, `suite_b/`, `protectai/`, `prompt_guard/` | `--suite ab` + `--portable-baselines` |
| Ablation (Table 2) | `results/*ablation*.csv`, `ab_comparison_metrics.csv` | ablation flags / `run_paper.py --mode full` |
| Adaptive (LOO) | `results/loo_matched_fpr/robustness_adaptive_summary.csv` | included in LOO matched-FPR |
| Judge human validation | `results/judge_human_validation/metrics_summary.txt` | `score_judge_human_validation.py` / `--mode full` |
| Runtime overhead | `results/suite_b/runtime_overhead.csv` | `--latency-only` |

Index: [`results/PAPER.md`](results/PAPER.md). `--no-llm` is sanitizer-only (True ASR = 0).

---

## 11. Output Files

### LOO matched-FPR (`results/loo_matched_fpr/`) — main report

| Path | Contents |
|------|----------|
| **`PAPER_TABLES.md`** | Paper Tables A–E (cite this) |
| `summary.csv` / `metrics.csv` / `summary.txt` | Per-fold + four-source avg at val FPR ≈ 5% |
| `metrics_with_wilson.csv` / `paper_headline.csv` | Wilson CIs + headline export |
| `thresholds.csv` | Chosen thr / `semantic_hard_sim` |
| `train_benign_by_fold.csv` | Fit benign inventory |
| `robustness_*{,_summary}.csv` | Paraphrase / edge / adaptive at matched thr |
| `true_asr_vs_fpr_curves.csv` + `figures/` | ASR–FPR curves |
| `<fold>/` | Per-fold metrics + robustness |

Details: `results/loo_matched_fpr/README.md`. Index: `results/PAPER.md`.

### LOO default thr (`results/loo/`) — contrast; complete

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
- Default-threshold LOO (§6.0b) is **not** FPR-matched; use `--loo-matched-fpr` (§6.0) for val **FPR ≈ 5%** (full suite + True-ASR-vs-FPR curves). Mixed `--matched-fpr` is a separate small-split pilot.
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
