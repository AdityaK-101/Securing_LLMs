"""
Evaluate ProtectAI and/or Prompt Guard 2 on the frozen adaptive suite (n=150).

Gate-only Bypass / True ASR. Adaptive texts and baseline COMPLIED labels are
taken from suite_a robustness_adaptive_logs.jsonl (method==baseline).
No FLAN regeneration, no target/judge re-run, no retuning. CPU default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.evaluation.portable_gates import static_true_asr_pct
from src.config import RESULTS_DIR


ADAPTIVE_LOGS = RESULTS_DIR / "suite_a" / "robustness_adaptive_logs.jsonl"


def _load_adaptive_baseline(logs_path: Path) -> list[dict]:
    rows = []
    with logs_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("method") != "baseline":
                continue
            rows.append(
                {
                    "id": str(row["id"]),
                    "original": str(row.get("original", "")),
                    "source_injection": str(row.get("source_injection", "")),
                    "baseline_complied": str(row.get("judge_label", "")).upper()
                    == "COMPLIED",
                }
            )
    if len(rows) != 150:
        raise RuntimeError(
            f"Expected 150 baseline adaptive rows in {logs_path}, got {len(rows)}"
        )
    return rows


def _build_sanitizer(gate: str):
    if gate == "protectai":
        from src.sanitizers.protectai import ProtectAISanitizer

        return ProtectAISanitizer()  # CPU default
    if gate == "prompt_guard":
        from src.sanitizers.prompt_guard import PromptGuardSanitizer

        return PromptGuardSanitizer()  # CPU default
    raise ValueError(f"Unknown gate: {gate}")


def evaluate_gate(gate: str, attacks: list[dict]) -> dict:
    out_dir = RESULTS_DIR / gate
    out_dir.mkdir(parents=True, exist_ok=True)

    san = _build_sanitizer(gate)
    records = []
    for i, row in enumerate(attacks, start=1):
        _, blocked = san.sanitize(row["original"])
        records.append(
            {
                "id": row["id"],
                "method": san.name,
                "blocked": bool(blocked),
                "baseline_complied": bool(row["baseline_complied"]),
                "source_injection": row["source_injection"],
            }
        )
        if i % 25 == 0 or i == len(attacks):
            print(f"[{gate}] gated {i}/{len(attacks)}")

    logs = pd.DataFrame(records)
    logs.to_csv(out_dir / "adaptive_gate_logs.csv", index=False)

    n = len(logs)
    allowed = logs[~logs["blocked"]]
    bypass_n = len(allowed)
    true_asr_n = int(allowed["baseline_complied"].sum())
    bypass_pct = round(100.0 * bypass_n / n, 2)
    true_asr_pct = round(100.0 * true_asr_n / n, 2)
    static_asr = static_true_asr_pct(gate)
    delta_pp = round(true_asr_pct - static_asr, 2)

    metrics = {
        "method": gate,
        "model": san.model_id,
        "n_adaptive": n,
        "Bypass_Rate_%": bypass_pct,
        "bypass_n": bypass_n,
        "True_ASR_%": true_asr_pct,
        "true_asr_n": true_asr_n,
        "static_True_ASR_%": static_asr,
        "delta_True_ASR_pp": delta_pp,
        "baseline_complied_n": int(logs["baseline_complied"].sum()),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "adaptive_metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"[OK] Wrote {out_dir / 'adaptive_metrics.csv'}")
    san.release_models()
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate-only adaptive eval for ProtectAI / Prompt Guard 2"
    )
    parser.add_argument(
        "--gate",
        choices=("protectai", "prompt_guard", "all"),
        default="all",
        help="Which portable gate(s) to score on frozen adaptive texts",
    )
    args = parser.parse_args()

    attacks = _load_adaptive_baseline(ADAPTIVE_LOGS)
    complied_n = sum(1 for r in attacks if r["baseline_complied"])
    print(
        f"[Adaptive] Loaded {len(attacks)} frozen texts "
        f"({complied_n} baseline COMPLIED) from {ADAPTIVE_LOGS}"
    )

    gates = (
        ["protectai", "prompt_guard"] if args.gate == "all" else [args.gate]
    )
    for gate in gates:
        evaluate_gate(gate, attacks)


if __name__ == "__main__":
    main()
