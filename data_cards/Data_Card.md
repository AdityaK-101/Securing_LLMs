### Dataset Name

**Securing LLMs Combined Dataset v3.0** (canonical mixed evaluation corpus)

Canonical file: `data/cleaned/combined_dataset.csv`  
Frozen splits: `data/splits/{train,val,test}.csv`  
Manifest: `data/cleaned/combined_manifest.json`  
Rebuild: `python scripts/build_combined_dataset.py`

---

### Version & Date

* **Version:** 3.0 (~700 prompts; grown from raw sources)
* **Supersedes:** v2.0 350-prompt corpus (archived: `data/cleaned/legacy/combined_dataset_v2.0_350.csv`, splits in `data/splits/legacy_v2.0_350/`)
* **Legacy v1.0:** 250-prompt snapshot — **not** used by the evaluation CLI

---

### Source / Acquisition

**Injection prompts (400 total; 100 per source):**

* **100** from *Simsonsun/JailbreakPrompts* Dataset_2 (MIT)
* **100** from *Bravansky/compact-jailbreaks* raw sample (MIT)
* **100** from *Prompt Injection & Benign Prompt Dataset* labeled malicious (MIT)
* **100** from *Malicious Prompt Detection Dataset (MPDD)* / `malicous_deepset.csv` (CC0; filename typo preserved)

**Benign prompts (~300 total):**

* **50** ChatGPT / self-authored (`benign_prompts_chatgpt.csv`, CC0)
* **~250** from *Prompt Injection & Benign Prompt Dataset* benign rows (MIT)

Source license details: `data_cards/license_log.txt`.

---

### License

* **Compiled dataset:** Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)
* **Code / software in this repository:** MIT (see root `LICENSE`)
* Individual source licenses are preserved in `license_log.txt`.

---

### Dataset Fields

| Field  | Type   | Description                                       |
| ------ | ------ | ------------------------------------------------- |
| id     | string | Unique identifier (e.g., `id_1` …)                |
| prompt | string | The textual input prompt                          |
| label  | string | `"injection"` or `"benign"`                       |
| source | string | Provenance tag (splits include this column)       |

---

### Statistics & Splits

Exact counts: see `data/cleaned/combined_manifest.json` (seed 42). Typical v3.0:

| Split | Total | Injection | Benign | Role |
|-------|------:|----------:|-------:|------|
| train | ~489 | ~280 | ~209 | Embeddings / Method B training |
| val | ~105 | ~60 | ~45 | Method B threshold tuning |
| test | ~105 | ~60 | ~45 | Held-out evaluation |
| **Total** | **~699** | **400** | **~299** | |

* Split protocol: stratified `train_test_split` 70 / 15 / 15 with `random_state=42` (same as legacy `data/splits/split.ipynb`).
* Zero id / prompt overlap across train / val / test.
* Union of split ids equals `combined_dataset.csv`.

### External held-out test (not part of the mixed corpus)

* **Source:** [neuralchemy/Prompt-injection-dataset](https://huggingface.co/datasets/neuralchemy/Prompt-injection-dataset), config `core`, split `test`.
* **Files:** `data/external/neuralchemy/test_main.csv` and `test_core.csv`.
* **Role:** evaluation only. Do not add these rows to the Method A gallery or Method B training.
* **Rebuild:** `python scripts/build_external_neuralchemy.py`.

### Leave-one-source-out splits (mentor protocol)

* **Files:** `data/splits/source_holdout/{simsonsun,mpdd,prompt_injection,bravansky,neuralchemy}/{train,val,test}.csv`
* **Build:** `notebooks/build_source_holdout.ipynb` (canonical). CLI: `python scripts/build_source_holdout.py`.
* **Protocol:** leave-one-source-out on raw sources. Four-source folds are size-identical (**520 train / 130 val / 400 test**). Distinct from this mixed corpus.
* **Verify:** `python scripts/verify_source_holdout.py`
* **Score (gates):** `python scripts/eval_loo_holdout.py` → `results/source_holdout/`

**Known biases:** Injection prompts tend to be longer and more adversarial; models may overfit to length. Research-scale only — not a production security guarantee.

---

### Preprocessing Steps

1. Sample 100 attacks per source (+ ChatGPT / PI benign) via `scripts/build_combined_dataset.py`
2. Drop empty / exact-duplicate prompts; map labels to `injection` / `benign`
3. Write `combined_dataset.csv` + stratified splits

---

### Intended Use

**Appropriate:** research on prompt-injection detection / input sanitization; educational demos.

**Not intended:** production deployment without further validation; pretraining generative models on this corpus.

---

### Limitations & Ethical Considerations

* **Harmful content disclaimer:** This dataset contains explicit jailbreak, injection, and adversarial prompts. It is intended **strictly for research and educational use**. Do not use it to attack third-party systems.
* Some benign prompts are synthetic and may lack real-world noise.
* Upstream filename `malicous_deepset.csv` retains the original typo.
* Committed `results/suite_*` tables may still reflect the archived v2.0 53-prompt test until re-run.

---

### Contact & Attribution

* **Creator:** Aditya Kulkarni
* **Project:** Securing LLMs Research
* **Citation:** see root `CITATION.cff`

If you use this dataset, please cite the project CITATION.cff entry and preserve source attributions in `license_log.txt`.
