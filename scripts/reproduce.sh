#!/usr/bin/env bash
# Offline / gate-only checks. Does not overwrite paper ablation CSVs
# (those need a full target+judge run).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/3] Paraphrase gate-only (test injections)"
python scripts/eval_paraphrase_test.py

echo "[2/3] Judge human-validation score (offline)"
python scripts/score_judge_human_validation.py

echo "[3/3] Portable baselines (ProtectAI / Prompt Guard) — needs HF models"
python -m src.run_experiments --portable-baselines

echo "Done. Paper tables live in committed results/."
echo "  Full suites (target + judge): python -m src.run_experiments --suite a|b"
echo "  Ablations with True ASR:      python -m src.run_experiments --hard-trigger-ablation"
echo "  Ablations sanitizer-only:     python -m src.run_experiments --hard-trigger-ablation --no-llm"
echo "    (--no-llm zeros True ASR; it does not read previous CSVs.)"
