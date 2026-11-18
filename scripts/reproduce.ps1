# Short gate-only smoke (NOT the paper holdout pipeline).
# For AAAI tables:  python scripts/run_paper.py
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "NOTE: This is a smoke check. Paper numbers: python scripts/run_paper.py"
Write-Host "      Cite results/loo_matched_fpr/PAPER_TABLES.md"
Write-Host ""

Write-Host "[1/4] Dataset sizes"
python scripts/print_dataset_sizes.py

Write-Host "[2/4] Paraphrase gate-only (mixed test; not LOO matched-FPR)"
python scripts/eval_paraphrase_test.py

Write-Host "[3/4] Judge human-validation score (offline)"
python scripts/score_judge_human_validation.py

Write-Host "[4/4] Portable baselines (ProtectAI / Prompt Guard) — needs HF models"
python -m src.run_experiments --portable-baselines

Write-Host "Done (smoke)."
Write-Host "  Paper holdout:  python scripts/run_paper.py"
Write-Host "  Consistency:    python scripts/audit_pipeline_consistency.py"
Write-Host "  Ablations:      python -m src.run_experiments --hard-trigger-ablation"
