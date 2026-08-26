### Dataset Name

**Securing LLMs Combined Dataset v2.0** (canonical evaluation corpus)

Canonical file: `data/cleaned/combined_dataset.csv`  
Frozen splits: `data/splits/{train,val,test}.csv`

---

### Version & Date

* **Version:** 2.0 (350 prompts; supersedes legacy v1.0 250-prompt snapshot)
* **Date Created:** 10 January 2026 (v1.0); evaluation corpus finalized for 350-row splits
* **Legacy:** `data/cleaned/legacy/Securing_LLMs_Combined_Dataset_v1.0_250.csv` (150 injection / 100 benign) — **not** used by the evaluation CLI

---

### Source / Acquisition

**Injection prompts (200 total):**

* **50** from *Simsonsun/JailbreakPrompts* (MIT) — community jailbreaks
* **50** from *Bravansky/compact-jailbreaks* (MIT) — jailbreak variants  
  *Raw Bravansky CSV is **not shipped** (~431 MB, gitignored). The cleaned 50-sample slice `data/cleaned/jailbreak_prompts_Bravansky.jsonl` is sufficient to evaluate. Re-download the raw source only if you need to re-run sampling provenance.*
* **50** from *Prompt Injection & Benign Prompt Dataset* labeled as injection/jailbreak (MIT)
* **50** from *Malicious Prompt Detection Dataset (MPDD)* / `malicous_deepset.csv` (CC0; filename typo preserved from upstream)

**Benign prompts (150 total):**

* **50** self-authored / ChatGPT-generated (CC0)
* **100** from *Prompt Injection & Benign Prompt Dataset* (MIT)

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
| id     | string | Unique identifier (e.g., `id_1` … `id_350`)       |
| prompt | string | The textual input prompt                          |
| label  | string | `"injection"` or `"benign"`                       |

---

### Statistics & Splits

| Split | Total | Injection | Benign | Role |
|-------|------:|----------:|-------:|------|
| train | 245 | 140 | 105 | Embeddings / Method B training |
| val | 52 | 30 | 22 | Method B threshold tuning |
| test | 53 | 30 | 23 | Held-out evaluation |
| **Total** | **350** | **200** | **150** | |

* Split protocol: stratified `train_test_split` with `random_state=42` (`data/splits/split.ipynb`).
* Zero id / prompt overlap across train / val / test.
* Union of split ids equals `combined_dataset.csv`.

**Known biases:** Injection prompts tend to be longer and more adversarial; models may overfit to length. Research-scale only (~350 prompts) — not a production security guarantee.

---

### Preprocessing Steps

1. Sample 50 (or 100 for one benign source) items per source via preprocess notebooks
2. Drop empty prompts; map labels to `injection` / `benign`
3. Combine slices in `notebooks/combined_create.ipynb` → `combined_dataset.csv`
4. Stratified split → `data/splits/*.csv`

---

### Intended Use

**Appropriate:** research on prompt-injection detection / input sanitization; educational demos.

**Not intended:** production deployment without further validation; pretraining generative models on this corpus.

---

### Limitations & Ethical Considerations

* **Harmful content disclaimer:** This dataset contains explicit jailbreak, injection, and adversarial prompts. It is intended **strictly for research and educational use**. Do not use it to attack third-party systems.
* Some benign prompts are synthetic and may lack real-world noise.
* Upstream filename `malicous_deepset.csv` retains the original typo.

---

### Contact & Attribution

* **Creator:** Aditya Kulkarni
* **Project:** Securing LLMs Research
* **Citation:** see root `CITATION.cff`

If you use this dataset, please cite the project CITATION.cff entry and preserve source attributions in `license_log.txt`.

---
