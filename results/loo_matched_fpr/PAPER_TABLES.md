# LOO matched-FPR ≈ 5% — paper tables

Operating point: per-fold validation threshold chosen to **hit val FPR ≈ 5%** (closest). ProtectAI / Prompt Guard 2 keep published weights; only the score threshold is retuned. Soft features / Method A–B architecture unchanged.

**ID** column = **mixed suite matched-FPR** on the combined-dataset random split (`test.csv` → `results/matched_fpr/`, val FPR ≤ 5%). LOO columns use leave-one-source-out test sets (200 inj / 200 ben each).

## Table A — Main comparison (mixed suite matched-FPR / ID vs four-source LOO vs Neuralchemy)

| Method | ID Bypass % | ID FPR % | LOO-4 macro Bypass % | LOO-4 macro True ASR % | LOO-4 macro FPR % | LOO-4 pooled Bypass % (Wilson) | LOO-4 pooled True ASR % (Wilson) | LOO-4 pooled FPR % (Wilson) | Neuralchemy Bypass % (Wilson) | Neuralchemy True ASR % (Wilson) | Neuralchemy FPR % (Wilson) |
|--------|------------:|---------:|---------------------:|-----------------------:|------------------:|--------------------------------:|----------------------------------:|------------------------------:|-------------------------------:|---------------------------------:|-----------------------------:|
| No sanitizer (baseline) | — | — | 100.00 | 58.75 | 0.00 | 100.00 [99.5, 100.0] | 58.75 [55.3, 62.1] | 0.00 [0.0, 0.5] | 100.00 [98.1, 100.0] | 64.50 [57.7, 70.8] | 0.00 [0.0, 1.9] |
| Regex-only | — | — | 84.12 | 48.12 | 0.00 | 84.12 [81.4, 86.5] | 48.12 [44.7, 51.6] | 0.00 [0.0, 0.5] | 85.00 [79.4, 89.3] | 52.50 [45.6, 59.3] | 1.00 [0.3, 3.6] |
| Keyword-only | 58.33 [45.7, 69.9] | 0.00 [0.0, 7.9] | 59.75 | 33.50 | 0.00 | 59.75 [56.3, 63.1] | 33.50 [30.3, 36.8] | 0.00 [0.0, 0.5] | 70.50 [63.8, 76.4] | 40.50 [33.9, 47.4] | 1.50 [0.5, 4.3] |
| Method A (hybrid) | 11.67 [5.8, 22.2] | 6.67 [2.3, 17.9] | 34.62 | 13.75 | 16.62 | 34.62 [31.4, 38.0] | 13.75 [11.5, 16.3] | 16.62 [14.2, 19.4] | 20.00 [15.0, 26.1] | 11.00 [7.4, 16.1] | 87.00 [81.6, 91.0] |
| Method B (learned) | 13.33 [6.9, 24.2] | 0.00 [0.0, 7.9] | 47.62 | 19.00 | 11.38 | 47.62 [44.2, 51.1] | 19.00 [16.4, 21.9] | 11.38 [9.4, 13.8] | 26.00 [20.4, 32.5] | 14.50 [10.3, 20.0] | 56.50 [49.6, 63.2] |
| Similarity-only | 30.00 [19.9, 42.5] | 2.22 [0.4, 11.6] | 53.75 | 23.00 | 11.75 | 53.75 [50.3, 57.2] | 23.00 [20.2, 26.0] | 11.75 [9.7, 14.2] | 57.00 [50.1, 63.7] | 35.00 [28.7, 41.8] | 37.00 [30.6, 43.9] |
| DeBERTa fine-tuned | 3.33 [0.9, 11.4] | 0.00 [0.0, 7.9] | 1.38 | 0.62 | 13.50 | 1.38 [0.8, 2.4] | 0.62 [0.3, 1.5] | 13.50 [11.3, 16.0] | 1.50 [0.5, 4.3] | 0.50 [0.1, 2.8] | 60.50 [53.6, 67.0] |
| ProtectAI (DeBERTa-v3) | 10.00 [4.7, 20.1] | 6.67 [2.3, 17.9] | 5.62 | 2.75 | 35.25 | 5.62 [4.2, 7.4] | 2.75 [1.8, 4.1] | 35.25 [32.0, 38.6] | 4.50 [2.4, 8.3] | 2.00 [0.8, 5.0] | 12.50 [8.6, 17.8] |
| Prompt Guard 2 | 3.33 [0.9, 11.4] | 6.67 [2.3, 17.9] | 1.38 | 0.50 | 13.62 | 1.38 [0.8, 2.4] | 0.50 [0.2, 1.3] | 13.63 [11.4, 16.2] | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 85.00 [79.4, 89.3] |

LOO-4 **macro** = equal weight per held-out source. LOO-4 **pooled** = micro (800 inj / 800 ben) with Wilson 95% CI. Neuralchemy is the cleanest unseen external holdout and is **not** inside the four-source average.

## Table B — Per-fold main metrics with Wilson 95% CIs

| Fold | Method | Bypass % (Wilson) | True ASR % (Wilson) | FPR % (Wilson) |
|------|--------|------------------:|--------------------:|---------------:|
| `simsonsun` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 73.00 [66.5, 78.7] | 0.00 [0.0, 1.9] |
| `simsonsun` | Regex-only | 79.50 [73.4, 84.5] | 57.50 [50.6, 64.1] | 0.00 [0.0, 1.9] |
| `simsonsun` | Keyword-only | 50.50 [43.6, 57.4] | 39.50 [33.0, 46.4] | 0.00 [0.0, 1.9] |
| `simsonsun` | Method A (hybrid) | 12.00 [8.2, 17.2] | 6.00 [3.5, 10.2] | 10.50 [7.0, 15.5] |
| `simsonsun` | Method B (learned) | 12.00 [8.2, 17.2] | 6.00 [3.5, 10.2] | 11.00 [7.4, 16.1] |
| `simsonsun` | Similarity-only | 28.00 [22.2, 34.6] | 17.00 [12.4, 22.8] | 14.00 [9.9, 19.5] |
| `simsonsun` | DeBERTa fine-tuned | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 2.50 [1.1, 5.7] |
| `simsonsun` | ProtectAI (DeBERTa-v3) | 1.00 [0.3, 3.6] | 0.00 [0.0, 1.9] | 51.50 [44.6, 58.3] |
| `simsonsun` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 43.00 [36.3, 49.9] |
| `mpdd` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 50.50 [43.6, 57.4] | 0.00 [0.0, 1.9] |
| `mpdd` | Regex-only | 76.00 [69.6, 81.4] | 38.00 [31.6, 44.9] | 0.00 [0.0, 1.9] |
| `mpdd` | Keyword-only | 65.00 [58.2, 71.3] | 32.00 [25.9, 38.8] | 0.00 [0.0, 1.9] |
| `mpdd` | Method A (hybrid) | 48.00 [41.2, 54.9] | 27.00 [21.3, 33.5] | 7.50 [4.6, 12.0] |
| `mpdd` | Method B (learned) | 76.50 [70.2, 81.8] | 38.50 [32.0, 45.4] | 8.00 [5.0, 12.6] |
| `mpdd` | Similarity-only | 76.50 [70.2, 81.8] | 39.50 [33.0, 46.4] | 8.50 [5.4, 13.2] |
| `mpdd` | DeBERTa fine-tuned | 5.50 [3.1, 9.6] | 2.50 [1.1, 5.7] | 0.00 [0.0, 1.9] |
| `mpdd` | ProtectAI (DeBERTa-v3) | 20.50 [15.5, 26.6] | 10.00 [6.6, 14.9] | 15.50 [11.1, 21.2] |
| `mpdd` | Prompt Guard 2 | 5.50 [3.1, 9.6] | 2.00 [0.8, 5.0] | 2.00 [0.8, 5.0] |
| `prompt_injection` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 24.00 [18.6, 30.4] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Regex-only | 96.50 [93.0, 98.3] | 22.50 [17.3, 28.8] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Keyword-only | 75.00 [68.6, 80.5] | 19.00 [14.2, 25.0] | 0.00 [0.0, 1.9] |
| `prompt_injection` | Method A (hybrid) | 73.50 [67.0, 79.1] | 17.00 [12.4, 22.8] | 13.50 [9.4, 18.9] |
| `prompt_injection` | Method B (learned) | 91.00 [86.2, 94.2] | 22.50 [17.3, 28.8] | 16.00 [11.6, 21.7] |
| `prompt_injection` | Similarity-only | 97.00 [93.6, 98.6] | 24.00 [18.6, 30.4] | 6.50 [3.8, 10.8] |
| `prompt_injection` | DeBERTa fine-tuned | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 51.00 [44.1, 57.8] |
| `prompt_injection` | ProtectAI (DeBERTa-v3) | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 54.50 [47.6, 61.3] |
| `prompt_injection` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 7.50 [4.6, 12.0] |
| `bravansky` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 87.50 [82.2, 91.4] | 0.00 [0.0, 1.9] |
| `bravansky` | Regex-only | 84.50 [78.8, 88.9] | 74.50 [68.0, 80.0] | 0.00 [0.0, 1.9] |
| `bravansky` | Keyword-only | 48.50 [41.7, 55.4] | 43.50 [36.8, 50.4] | 0.00 [0.0, 1.9] |
| `bravansky` | Method A (hybrid) | 5.00 [2.7, 9.0] | 5.00 [2.7, 9.0] | 35.00 [28.7, 41.8] |
| `bravansky` | Method B (learned) | 11.00 [7.4, 16.1] | 9.00 [5.8, 13.8] | 10.50 [7.0, 15.5] |
| `bravansky` | Similarity-only | 13.50 [9.4, 18.9] | 11.50 [7.8, 16.7] | 18.00 [13.3, 23.9] |
| `bravansky` | DeBERTa fine-tuned | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 0.50 [0.1, 2.8] |
| `bravansky` | ProtectAI (DeBERTa-v3) | 1.00 [0.3, 3.6] | 1.00 [0.3, 3.6] | 19.50 [14.6, 25.5] |
| `bravansky` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 2.00 [0.8, 5.0] |
| `neuralchemy` | No sanitizer (baseline) | 100.00 [98.1, 100.0] | 64.50 [57.7, 70.8] | 0.00 [0.0, 1.9] |
| `neuralchemy` | Regex-only | 85.00 [79.4, 89.3] | 52.50 [45.6, 59.3] | 1.00 [0.3, 3.6] |
| `neuralchemy` | Keyword-only | 70.50 [63.8, 76.4] | 40.50 [33.9, 47.4] | 1.50 [0.5, 4.3] |
| `neuralchemy` | Method A (hybrid) | 20.00 [15.0, 26.1] | 11.00 [7.4, 16.1] | 87.00 [81.6, 91.0] |
| `neuralchemy` | Method B (learned) | 26.00 [20.4, 32.5] | 14.50 [10.3, 20.0] | 56.50 [49.6, 63.2] |
| `neuralchemy` | Similarity-only | 57.00 [50.1, 63.7] | 35.00 [28.7, 41.8] | 37.00 [30.6, 43.9] |
| `neuralchemy` | DeBERTa fine-tuned | 1.50 [0.5, 4.3] | 0.50 [0.1, 2.8] | 60.50 [53.6, 67.0] |
| `neuralchemy` | ProtectAI (DeBERTa-v3) | 4.50 [2.4, 8.3] | 2.00 [0.8, 5.0] | 12.50 [8.6, 17.8] |
| `neuralchemy` | Prompt Guard 2 | 0.00 [0.0, 1.9] | 0.00 [0.0, 1.9] | 85.00 [79.4, 89.3] |

## Table C — Four-source average and pooled

| Method | Macro Bypass % | Macro True ASR % | Macro FPR % | Pooled Bypass % (Wilson) | Pooled True ASR % (Wilson) | Pooled FPR % (Wilson) |
|--------|---------------:|-----------------:|------------:|------------------------:|--------------------------:|----------------------:|
| No sanitizer (baseline) | 100.00 | 58.75 | 0.00 | 100.00 [99.5, 100.0] | 58.75 [55.3, 62.1] | 0.00 [0.0, 0.5] |
| Regex-only | 84.12 | 48.12 | 0.00 | 84.12 [81.4, 86.5] | 48.12 [44.7, 51.6] | 0.00 [0.0, 0.5] |
| Keyword-only | 59.75 | 33.50 | 0.00 | 59.75 [56.3, 63.1] | 33.50 [30.3, 36.8] | 0.00 [0.0, 0.5] |
| Method A (hybrid) | 34.62 | 13.75 | 16.62 | 34.62 [31.4, 38.0] | 13.75 [11.5, 16.3] | 16.62 [14.2, 19.4] |
| Method B (learned) | 47.62 | 19.00 | 11.38 | 47.62 [44.2, 51.1] | 19.00 [16.4, 21.9] | 11.38 [9.4, 13.8] |
| Similarity-only | 53.75 | 23.00 | 11.75 | 53.75 [50.3, 57.2] | 23.00 [20.2, 26.0] | 11.75 [9.7, 14.2] |
| DeBERTa fine-tuned | 1.38 | 0.62 | 13.50 | 1.38 [0.8, 2.4] | 0.62 [0.3, 1.5] | 13.50 [11.3, 16.0] |
| ProtectAI (DeBERTa-v3) | 5.62 | 2.75 | 35.25 | 5.62 [4.2, 7.4] | 2.75 [1.8, 4.1] | 35.25 [32.0, 38.6] |
| Prompt Guard 2 | 1.38 | 0.50 | 13.62 | 1.38 [0.8, 2.4] | 0.50 [0.2, 1.3] | 13.63 [11.4, 16.2] |

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

| Fold | Train ChatGPT | Train PI-benign | Val ChatGPT | Val PI-benign | Fit total (train+val) |
|------|--------------:|----------------:|------------:|--------------:|----------------------:|
| `simsonsun` | 21 | 19 | 6 | 4 | 50 |
| `mpdd` | 25 | 15 | 4 | 6 | 50 |
| `prompt_injection` | 40 | 0 | 10 | 0 | 50 |
| `bravansky` | 19 | 21 | 2 | 8 | 50 |
| `neuralchemy` | 19 | 21 | 5 | 5 | 50 |

Each fold fits Method B / similarity / DeBERTa on **50** benign prompts (train+val; ChatGPT + leftover PI-benign). The `prompt_injection` fold uses **ChatGPT benign only** (PI native benign held out of train/val). Shared test benign = **200** PI-benign prompts, identical across four-source tests.

Sources: `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/loo_matched_fpr/metrics.csv`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/results/matched_fpr/metrics.csv`, `C:/Users/kulka/Downloads/Research_paper/Securing_LLMs/data/splits/source_holdout/manifest.json`.
