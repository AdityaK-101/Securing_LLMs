"""
Evaluate Meta Llama Prompt Guard 2 on the frozen test split + edge-benign set.

Gate-only for Bypass / FPR / Edge FPR. True ASR reuses baseline COMPLIED labels
from suite_a logs (greedy target+judge), so we do not retune or re-split.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.config import RESULTS_DIR, TEST_CSV
from src.evaluation.robustness import EDGE_BENIGN_PROMPTS
from src.sanitizers.prompt_guard import PromptGuardSanitizer


OUT_DIR = RESULTS_DIR / "prompt_guard"
LOGS_BASELINE = RESULTS_DIR / "suite_a" / "logs.jsonl"


def _baseline_complied_ids(logs_path: Path) -> set[str]:
    complied = set()
    with logs_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if (
                row.get("method") == "baseline"
                and row.get("label") == "injection"
                and str(row.get("judge_label", "")).upper() == "COMPLIED"
            ):
                complied.add(str(row["id"]))
    return complied


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    san = PromptGuardSanitizer()  # CPU default
    df = pd.read_csv(TEST_CSV)
    complied_ids = _baseline_complied_ids(LOGS_BASELINE)

    records = []
    for _, row in df.iterrows():
        pid = str(row["id"])
        prompt = str(row["prompt"])
        label = str(row["label"])
        _, blocked = san.sanitize(prompt)
        records.append(
            {
                "id": pid,
                "label": label,
                "method": san.name,
                "blocked": bool(blocked),
                "baseline_complied": pid in complied_ids,
            }
        )

    logs = pd.DataFrame(records)
    logs.to_csv(OUT_DIR / "test_gate_logs.csv", index=False)

    inj = logs[logs["label"] == "injection"]
    ben = logs[logs["label"] == "benign"]
    n_inj = len(inj)
    n_ben = len(ben)
    allowed_inj = inj[~inj["blocked"]]
    bypass_n = len(allowed_inj)
    true_asr_n = int(allowed_inj["baseline_complied"].sum())
    fp_n = int(ben["blocked"].sum())

    edge_blocked = 0
    for prompt in EDGE_BENIGN_PROMPTS:
        _, blocked = san.sanitize(prompt)
        if blocked:
            edge_blocked += 1
    n_edge = len(EDGE_BENIGN_PROMPTS)

    metrics = {
        "method": "prompt_guard",
        "model": san.model_id,
        "Bypass_Rate_%": round(100.0 * bypass_n / n_inj, 2),
        "bypass_n": bypass_n,
        "n_inj": n_inj,
        "True_ASR_%": round(100.0 * true_asr_n / n_inj, 2),
        "true_asr_n": true_asr_n,
        "FPR_%": round(100.0 * fp_n / n_ben, 2),
        "fp_n": fp_n,
        "n_ben": n_ben,
        "Edge_FPR_%": round(100.0 * edge_blocked / n_edge, 2),
        "edge_blocked": edge_blocked,
        "n_edge": n_edge,
    }
    pd.DataFrame([metrics]).to_csv(OUT_DIR / "metrics.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"[OK] Wrote {OUT_DIR / 'metrics.csv'}")
    san.release_models()


if __name__ == "__main__":
    main()
