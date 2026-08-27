# Gate-only checks (paraphrase + judge HV offline; portable baselines need HF).
# Does not overwrite paper ablation CSVs (those need a full target+judge run).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "[1/3] Paraphrase gate-only (test injections)"
python scripts/eval_paraphrase_test.py

Write-Host "[2/3] Judge human-validation score (offline)"
python scripts/score_judge_human_validation.py

Write-Host "[3/3] Portable baselines (ProtectAI / Prompt Guard) — needs HF models"
python -m src.run_experiments --portable-baselines

Write-Host "Done. Paper tables live in committed results/."
Write-Host "  Full suites (target + judge): python -m src.run_experiments --suite a|b"
Write-Host "  Ablations with True ASR:      python -m src.run_experiments --hard-trigger-ablation"
Write-Host "  Ablations sanitizer-only:     python -m src.run_experiments --hard-trigger-ablation --no-llm"
Write-Host "    (--no-llm zeros True ASR; it does not read previous CSVs.)"
