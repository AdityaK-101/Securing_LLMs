# Judge human-validation results

**Paper-cited scores** (`metrics_summary.txt`, `metrics.csv`): stratified Table 1 LOO+Neuralchemy no-defense pairs, **N = 100** (20 per held-out source) — **77.0%** agreement, Wilson **[67.8, 84.2]%**, Cohen's **κ = 0.48**.

Second-family judge (Mistral-7B-Instruct-v0.2) on the same pairs: human–Qwen 77.0% (κ=0.48), human–second 72.0% (κ=0.42), Qwen–second 75.0% (κ=0.49). See `second_judge_summary.txt`.

Canonical labeled files: `agreement_report.csv`, `judge_key.csv`, `metrics.csv`.

Archived mixed-test v2.0 spot-check (**N = 30**, 83.3%, κ = 0.67): `legacy_v2.0_n30/` (not cited in the paper).

Labeling sheet / rubric: `data/judge_human_validation/`.
