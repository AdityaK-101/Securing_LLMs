#!/usr/bin/env bash
# Short gate-only smoke (NOT the paper holdout pipeline).
# For AAAI tables:  python scripts/run_paper.py
set -euo pipefail
cd "$(dirname "$0")/.."

echo "NOTE: This is a smoke check. Paper numbers: python scripts/run_paper.py"
echo "      Cite results/loo_matched_fpr/PAPER_TABLES.md"
echo ""

echo "[1/4] Dataset sizes"
python scripts/print_dataset_sizes.py

echo "[2/4] Paraphrase gate-only (mixed test; not LOO matched-FPR)"
python scripts/eval_paraphrase_test.py

echo "[3/4] Judge human-validation score (offline)"
python scripts/score_judge_human_validation.py

echo "[4/4] Portable baselines (ProtectAI / Prompt Guard) — needs HF models"
python -m src.run_experiments --portable-baselines

echo "Done (smoke)."
echo "  Paper holdout:  python scripts/run_paper.py"
echo "  Consistency:    python scripts/audit_pipeline_consistency.py"
echo "  Ablations:      python -m src.run_experiments --hard-trigger-ablation"
