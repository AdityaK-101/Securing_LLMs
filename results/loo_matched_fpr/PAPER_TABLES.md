# LOO matched-FPR ≈ 5% — paper tables

Operating point: per-fold validation threshold chosen to **hit val FPR ≈ 5%** (closest). ProtectAI / Prompt Guard 2 keep published weights; only the score threshold is retuned. Soft features / Method A–B architecture unchanged.

**ID** column = **mixed suite matched-FPR** on the combined-dataset random split (`test.csv` → `results/matched_fpr/`, val FPR ≤ 5%). LOO columns use leave-one-source-out test sets (200 inj / 200 ben each).

## Table A — Main comparison (mixed suite matched-FPR / ID vs four-source LOO vs Neuralchemy)

| Method | ID Bypass % | ID True ASR % | ID FPR % | LOO-4 macro Bypass % | LOO-4 macro True ASR % | LOO-4 macro FPR % | LOO-4 pooled Bypass % (Wilson) | LOO-4 pooled True ASR % (Wilson) | LOO-4 pooled FPR % (Wilson) | Neuralchemy Bypass % (Wilson) | Neuralchemy True ASR % (Wilson) | Neuralchemy FPR % (Wilson) |
|--------|------------:|--------------:|---------:|---------------------:|-----------------------:|------------------:|--------------------------------:|----------------------------------:|------------------------------:|-------------------------------:|---------------------------------:|-----------------------------:|
| No sanitizer (baseline) | 100.00 [94.0, 100.0] | 58.33 [45.7, 69.9] | 0.00 [0.0, 7.9] | 100.00 | 58.12 | 0.00 | 100.00 [99.5, 100.0] | 58.13 [54.7, 61.5] | 0.00 [0.0, 0.5] | 100.00 [98.1, 100.0] | 64.50 [57.7, 70.8] | 0.00 [0.0, 1.9] |
| Regex-only | 83.33 [72.0, 90.7] | 48.33 [36.2, 60.7] | 0.00 [0.0, 7.9] | 84.12 | 47.62 | 0.00 | 84.12 [81.4, 86.5] | 47.62 [44.2, 51.1] | 0.00 [0.0, 0.5] | 85.00 [79.4, 89.3] | 52.50 [45.6, 59.3] | 0.00 [0.0, 1.9] |
| Keyword-only | 58.33 [45.7, 69.9] | 28.33 [18.5, 40.8] | 0.00 [0.0, 7.9] | 59.75 | 33.00 | 0.00 | 59.75 [56.3, 63.1] | 33.00 [29.8, 36.3] | 0.00 [0.0, 0.5] | 70.50 [63.8, 76.4] | 40.50 [33.9, 47.4] | 0.00 [0.0, 1.9] |
| Method A (hybrid) | 11.67 [5.8, 22.2] | 6.67 [2.6, 15.9] | 6.67 [2.3, 17.9] | 41.62 | 16.50 | 4.50 | 41.62 [38.3, 45.1] | 16.50 [14.1, 19.2] | 4.50 [3.3, 6.2] | 49.50 [42.6, 56.4] | 30.00 [24.1, 36.7] | 3.50 [1.7, 7.0] |
| Method B (learned) | 13.33 [6.9, 24.2] | 6.67 [2.6, 15.9] | 0.00 [0.0, 7.9] | 58.25 | 24.50 | 4.00 | 58.25 [54.8, 61.6] | 24.50 [21.6, 27.6] | 4.00 [2.8, 5.6] | 81.00 [75.0, 85.8] | 50.50 [43.6, 57.4] | 1.50 [0.5, 4.3] |
| Similarity-only | 30.00 [19.9, 42.5] | 18.33 [10.6, 29.9] | 2.22 [0.4, 11.6] | 76.38 | 38.12 | 2.50 | 76.38 [73.3, 79.2] | 38.12 [34.8, 41.5] | 2.50 [1.6, 3.8] | 91.00 [86.2, 94.2] | 59.50 [52.6, 66.1] | 0.50 [0.1, 2.8] |
| DeBERTa fine-tuned | 3.33 [0.9, 11.4] | 3.33 [0.9, 11.4] | 0.00 [0.0, 7.9] | 2.25 | 0.75 | 0.38 | 2.25 [1.4, 3.5] | 0.75 [0.3, 1.6] | 0.38 [0.1, 1.1] | 2.50 [1.1, 5.7] | 1.00 [0.3, 3.6] | 0.00 [0.0, 1.9] |
| ProtectAI (DeBERTa-v3) | 10.00 [4.7, 20.1] | 6.67 [2.6, 15.9] | 6.67 [2.3, 17.9] | 11.62 | 7.12 | 2.25 | 11.62 [9.6, 14.0] | 7.12 [5.5, 9.1] | 2.25 [1.4, 3.5] | 4.50 [2.4, 8.3] | 2.00 [0.8, 5.0] | 4.50 [2.4, 8.3] |
| Prompt Guard 2 | 3.33 [0.9, 11.4] | 1.67 [0.3, 8.9] | 6.67 [2.3, 17.9] | 3.62 | 1.88 | 1.00 | 3.62 [2.5, 5.2] | 1.88 [1.1, 3.1] | 1.00 [0.5, 2.0] | 1.00 [0.3, 3.6] | 0.00 [0.0, 1.9] | 1.00 [0.3, 3.6] |

LOO-4 **macro** = equal weight per held-out source. LOO-4 **pooled** = micro (800 inj / 800 ben) with Wilson 95% CI. Neuralchemy is the cleanest unseen external holdout and is **not** inside the four-source average.

## Table B — Per-fold main metrics with Wilson 95% CIs

| Fold | Method | Bypass % (Wilson) | True ASR % (Wilson) | FPR % (Wilson) |
|------|--------|------------------:|--------------------:|---------------:|
| `simsonsun` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 73.00 [66.5, 78.7] | 0.00 [0.0, 1.9] |
| `simsonsun` | Regex-only | 79.50 [73.4, 84.5] | 57.50 [50.6, 64.1] | 0.00 [0.0, 1.9] |
| `simsonsun` | Keyword-only | 50.50 [43.6, 57.4] | 39.50 [33.0, 46.4] | 0.00 [0.0, 1.9] |
| `simsonsun` | Method A (hybrid) | 17.00 [12.4, 22.8] | 9.50 [6.2, 14.4] | 4.50 [2.4, 8.3] |
| `simsonsun` | Method B (learned) | 28.00 [22.2, 34.6] | 15.00 [10.7, 20.6] | 6.00 [3.5, 10.2] |
| `simsonsun` | Similarity-only | 58.50 [51.6, 65.1] | 39.00 [32.5, 45.9] | 2.50 [1.1, 5.7] |
| `simsonsun` | DeBERTa fine-tuned | 0.50 [0.1, 2.8] | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] |
| `simsonsun` | ProtectAI (DeBERTa-v3) | 4.00 [2.0, 7.7] | 2.50 [1.1, 5.7] | 2.50 [1.1, 5.7] |
| `simsonsun` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 1.00 [0.3, 3.6] |
| `mpdd` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 48.00 [41.2, 54.9] | 0.00 [0.0, 1.9] |
| `mpdd` | Regex-only | 76.00 [69.6, 81.4] | 36.00 [29.7, 42.9] | 0.00 [0.0, 1.9] |
| `mpdd` | Keyword-only | 65.00 [58.2, 71.3] | 30.00 [24.1, 36.7] | 0.00 [0.0, 1.9] |
| `mpdd` | Method A (hybrid) | 51.00 [44.1, 57.8] | 26.00 [20.4, 32.5] | 6.00 [3.5, 10.2] |
| `mpdd` | Method B (learned) | 84.50 [78.8, 88.9] | 39.00 [32.5, 45.9] | 4.50 [2.4, 8.3] |
| `mpdd` | Similarity-only | 96.50 [93.0, 98.3] | 45.00 [38.3, 51.9] | 3.00 [1.4, 6.4] |
| `mpdd` | DeBERTa fine-tuned | 7.00 [4.2, 11.4] | 3.00 [1.4, 6.4] | 0.50 [0.1, 2.8] |
| `mpdd` | ProtectAI (DeBERTa-v3) | 31.50 [25.5, 38.2] | 16.50 [12.0, 22.3] | 1.50 [0.5, 4.3] |
| `mpdd` | Prompt Guard 2 | 11.00 [7.4, 16.1] | 6.00 [3.5, 10.2] | 1.00 [0.3, 3.6] |
| `prompt_injection` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 24.00 [18.6, 30.4] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Regex-only | 96.50 [93.0, 98.3] | 22.50 [17.3, 28.8] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Keyword-only | 75.00 [68.6, 80.5] | 19.00 [14.2, 25.0] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Method A (hybrid) | 89.00 [83.9, 92.6] | 22.00 [16.8, 28.2] | 4.00 [2.0, 7.7] |
| `prompt_injection` | Method B (learned) | 97.00 [93.6, 98.6] | 24.00 [18.6, 30.4] | 2.50 [1.1, 5.7] |
| `prompt_injection` | Similarity-only | 98.50 [95.7, 99.5] | 24.00 [18.6, 30.4] | 2.00 [0.8, 5.0] |
| `prompt_injection` | DeBERTa fine-tuned | 1.50 [0.5, 4.3] | 0.00 [0.0, 1.9] | 1.00 [0.3, 3.6] |
| `prompt_injection` | ProtectAI (DeBERTa-v3) | 1.50 [0.5, 4.3] | 0.00 [0.0, 1.9] | 2.50 [1.1, 5.7] |
| `prompt_injection` | Prompt Guard 2 | 3.50 [1.7, 7.0] | 1.50 [0.5, 4.3] | 1.00 [0.3, 3.6] |
| `bravansky` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 87.50 [82.2, 91.4] | 0.00 [0.0, 1.9] |
| `bravansky` | Regex-only | 84.50 [78.8, 88.9] | 74.50 [68.0, 80.0] | 0.00 [0.0, 1.9] |
| `bravansky` | Keyword-only | 48.50 [41.7, 55.4] | 43.50 [36.8, 50.4] | 0.00 [0.0, 1.9] |
| `bravansky` | Method A (hybrid) | 9.50 [6.2, 14.4] | 8.50 [5.4, 13.2] | 3.50 [1.7, 7.0] |
| `bravansky` | Method B (learned) | 23.50 [18.2, 29.8] | 20.00 [15.0, 26.1] | 3.00 [1.4, 6.4] |
| `bravansky` | Similarity-only | 52.00 [45.1, 58.8] | 44.50 [37.8, 51.4] | 2.50 [1.1, 5.7] |
| `bravansky` | DeBERTa fine-tuned | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] |
| `bravansky` | ProtectAI (DeBERTa-v3) | 9.50 [6.2, 14.4] | 9.50 [6.2, 14.4] | 2.50 [1.1, 5.7] |
| `bravansky` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 1.00 [0.3, 3.6] |
| `neuralchemy` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 64.50 [57.7, 70.8] | 0.00 [0.0, 1.9] |
| `neuralchemy` | Regex-only | 85.00 [79.4, 89.3] | 52.50 [45.6, 59.3] | 0.00 [0.0, 1.9] |
| `neuralchemy` | Keyword-only | 70.50 [63.8, 76.4] | 40.50 [33.9, 47.4] | 0.00 [0.0, 1.9] |
| `neuralchemy` | Method A (hybrid) | 49.50 [42.6, 56.4] | 30.00 [24.1, 36.7] | 3.50 [1.7, 7.0] |
| `neuralchemy` | Method B (learned) | 81.00 [75.0, 85.8] | 50.50 [43.6, 57.4] | 1.50 [0.5, 4.3] |
| `neuralchemy` | Similarity-only | 91.00 [86.2, 94.2] | 59.50 [52.6, 66.1] | 0.50 [0.1, 2.8] |
| `neuralchemy` | DeBERTa fine-tuned | 2.50 [1.1, 5.7] | 1.00 [0.3, 3.6] | 0.00 [0.0, 1.9] |
| `neuralchemy` | ProtectAI (DeBERTa-v3) | 4.50 [2.4, 8.3] | 2.00 [0.8, 5.0] | 4.50 [2.4, 8.3] |
| `neuralchemy` | Prompt Guard 2 | 1.00 [0.3, 3.6] | 0.00 [0.0, 1.9] | 1.00 [0.3, 3.6] |

## Table C — Four-source average and pooled

| Method | Macro Bypass % | Macro True ASR % | Macro FPR % | Pooled Bypass % (Wilson) | Pooled True ASR % (Wilson) | Pooled FPR % (Wilson) |
|--------|---------------:|-----------------:|------------:|------------------------:|--------------------------:|----------------------:|
| No sanitizer (baseline) | 100.00 | 58.12 | 0.00 | 100.00 [99.5, 100.0] | 58.13 [54.7, 61.5] | 0.00 [0.0, 0.5] |
| Regex-only | 84.12 | 47.62 | 0.00 | 84.12 [81.4, 86.5] | 47.62 [44.2, 51.1] | 0.00 [0.0, 0.5] |
| Keyword-only | 59.75 | 33.00 | 0.00 | 59.75 [56.3, 63.1] | 33.00 [29.8, 36.3] | 0.00 [0.0, 0.5] |
| Method A (hybrid) | 41.62 | 16.50 | 4.50 | 41.62 [38.3, 45.1] | 16.50 [14.1, 19.2] | 4.50 [3.3, 6.2] |
| Method B (learned) | 58.25 | 24.50 | 4.00 | 58.25 [54.8, 61.6] | 24.50 [21.6, 27.6] | 4.00 [2.8, 5.6] |
| Similarity-only | 76.38 | 38.12 | 2.50 | 76.38 [73.3, 79.2] | 38.12 [34.8, 41.5] | 2.50 [1.6, 3.8] |
| DeBERTa fine-tuned | 2.25 | 0.75 | 0.38 | 2.25 [1.4, 3.5] | 0.75 [0.3, 1.6] | 0.38 [0.1, 1.1] |
| ProtectAI (DeBERTa-v3) | 11.62 | 7.12 | 2.25 | 11.62 [9.6, 14.0] | 7.12 [5.5, 9.1] | 2.25 [1.4, 3.5] |
| Prompt Guard 2 | 3.62 | 1.88 | 1.00 | 3.62 [2.5, 5.2] | 1.88 [1.1, 3.1] | 1.00 [0.5, 2.0] |

## Table D — Deduplication

| Filter | Count |
|--------|------:|
| Cross-source MiniLM pairs compared | 5282 |
| Near-duplicates removed (cosine ≥ 0.9) | 3 |
| · removed from `simsonsun` | 0 |
| · removed from `mpdd` | 2 |
| · removed from `prompt_injection` | 0 |
| · removed from `bravansky` | 1 |
| Exact string overlaps (pre-filter inventory) |  |
| · exact overlap involving `simsonsun` | 0 |
| · exact overlap involving `bravansky` | 31 |
| · exact overlap involving `mpdd` | 4 |
| · exact overlap involving `prompt_injection` | 0 |
| Neuralchemy prompts dropped for overlap with four sources | 1 |

### Near-duplicate spot checks

| Removed source | Kept source | Cosine |
|----------------|-------------|-------:|
| `mpdd` | `simsonsun` | 1.000 |
| `mpdd` | `simsonsun` | 0.961 |
| `bravansky` | `simsonsun` | 0.909 |

## Table E — Train benign inventory by fold

| Fold | Train Alpaca | Val Alpaca | Fit total (train+val) | Train PI-benign | Val PI-benign |
|------|-------------:|-----------:|----------------------:|----------------:|--------------:|
| `simsonsun` | 200 | 100 | 300 | 0 | 0 |
| `mpdd` | 200 | 100 | 300 | 0 | 0 |
| `prompt_injection` | 200 | 100 | 300 | 0 | 0 |
| `bravansky` | 200 | 100 | 300 | 0 | 0 |
| `neuralchemy` | 200 | 100 | 300 | 0 | 0 |

Each fold fits Method B / similarity / DeBERTa on **300** Alpaca instruction prompts (train=200, val=100). **PI-benign is test-only** (never in train/val). Shared test benign = **200** PI-benign prompts, identical across **all** folds including Neuralchemy.

## Table F — Three-regime FPR + adaptive Bypass (matched LOO thresholds; edge = 80 technical prompts)

| Method | LOO-4 PI-benign FPR % | Neuralchemy-side FPR % | LOO-4 Edge FPR % | LOO-4 Adaptive Bypass % |
|--------|----------------------:|-----------------------:|-----------------:|-------------------------:|
| No sanitizer (baseline) | 0.00 | 0.00 | 0.00 | 100.00 |
| Regex-only | 0.00 | 0.00 | 2.50 | 95.21 |
| Keyword-only | 0.00 | 0.00 | 62.50 | 88.52 |
| Method A (hybrid) | 4.50 | 3.50 | 81.88 | 55.54 |
| Method B (learned) | 4.00 | 1.50 | 18.44 | 69.86 |
| Similarity-only | 2.50 | 0.50 | 10.62 | 72.00 |
| DeBERTa fine-tuned | 0.38 | 0.00 | 79.06 | 27.46 |
| ProtectAI (DeBERTa-v3) | 2.25 | 4.50 | 63.75 | 41.54 |
| Prompt Guard 2 | 1.00 | 1.00 | 87.50 | 35.00 |

Edge / adaptive use the **same** per-fold matched thresholds as Table A. ProtectAI / Prompt Guard adaptive Bypass is **gate-only** (True ASR not re-judged).

Sources: `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/loo_matched_fpr/metrics.csv`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/matched_fpr/metrics.csv`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/data/splits/source_holdout/manifest.json`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/loo_matched_fpr/robustness_edge_benign.csv`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/loo_matched_fpr/robustness_adaptive.csv`.
