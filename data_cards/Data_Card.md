
### Dataset Name

**Securing LLMs Combined Dataset v1.0**

---

### Version & Date

* **Version:** 1.0
* **Date Created:** 10 January 2026

---

### Source / Acquisition

**Malicious / Injection Prompts (150 total):**

* **50 samples** from *Simsonsun/JailbreakPrompts* (MIT License)

  * Curated community jailbreak prompts
* **50 samples** from *Bravansky/compact-jailbreaks* (MIT License)

  * Large-scale jailbreak variants
* **50 samples** from *Malicious Prompt Detection Dataset (MPDD)* (CC0)

  * Explicit prompt-injection and malicious attack prompts

**Benign Prompts (100 total):**

* **50 samples** self-authored / ChatGPT-generated (CC0)

  * Standard benign tasks (e.g., creative writing, factual queries)
* **50 samples** from *Prompt Injection & Benign Prompt Dataset* (MIT License)

  * Anonymized benign user queries

---

### License

* **Compiled Dataset License:**
  Creative Commons Attribution–NonCommercial 4.0 International (CC BY-NC 4.0)
* **Note:**
  Individual source licenses are preserved and documented in `license_log.txt`.

---

### Dataset Fields

| Field  | Type   | Description                                       |
| ------ | ------ | ------------------------------------------------- |
| id     | string | Unique identifier (e.g., `id_1`, `id_2`)          |
| prompt | string | The textual input prompt                          |
| label  | string | Classification label: `"injection"` or `"benign"` |

---

### Statistics & Known Biases

**Label Balance:**

* 150 injection samples (60%)
* 100 benign samples (40%)

*Note:* The dataset intentionally favors injection prompts to stress-test defense mechanisms.

**Length Distribution:**

* Injection prompts: generally longer, containing adversarial framing and instruction overrides
* Benign prompts: typically shorter, direct questions or tasks

*Potential Bias Warning:*
Models trained on this dataset may overfit to prompt length rather than semantic intent.

---

### Preprocessing Steps

1. **Sampling:**
   Random sampling of 50 items from each source dataset
2. **Cleaning:**
   Removal of rows with missing or empty prompt fields (`dropna`)
3. **Standardization:**
   * All malicious labels mapped to `"injection"`
   * All non-malicious labels mapped to `"benign"`
4. **Formatting:**
   Unified into CSV format with `id`, `prompt`, and `label` columns

---

### Intended Use

**Appropriate Use:**

* Research on prompt injection detection and defenses
* Evaluation of input sanitization and filtering mechanisms
* Educational demonstrations of LLM security vulnerabilities

**Not Intended For:**

* Production deployment without further validation
* Training generative language models (contains harmful content)

---

### Limitations & Ethical Considerations

* **Synthetic Data:**
  Some benign prompts are synthetic and may lack real-world noise
* **Harmful Content:**
  Dataset includes explicit jailbreak and adversarial prompts

  * Intended strictly for research and educational use
  * User discretion is advised

---

### Contact & Attribution

* **Creator:** Aditya Kulkarni
* **Project:** Securing LLMs Research
* **Contact:** kulkarni.aditya.m@gmail.com
* **Citation:**
If you use this dataset, please cite:

Kulkarni, A. (2026). *Securing LLMs Combined Dataset v1.0*.
Research dataset for prompt injection and LLM security.

---
