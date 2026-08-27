"""Regenerate suite_a/suite_b robustness_note.txt from frozen CSVs.

Uses src/visualization/reporting.py (no full package import).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load_write_fn():
    """Load _write_robustness_note without importing heavy src deps."""
    cfg = ModuleType("src.config")
    cfg.JUDGE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    cfg.TARGET_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    sys.modules["src"] = ModuleType("src")
    sys.modules["src.config"] = cfg

    lat = ModuleType("src.evaluation.latency")

    def format_overhead_for_note(report: dict) -> list[str]:
        hw = report["hardware"]
        comps = report["components"]
        lines = [
            "",
            "5. RUNTIME / LATENCY OVERHEAD (MiniLM + DistilGPT-2)",
            "-" * 40,
            (
                f"  Hardware: CPU={hw.get('processor')}; RAM={hw.get('ram_gb')} GB; "
                f"CUDA={hw.get('cuda_available')}"
                + (f" ({hw.get('gpu_name')})" if hw.get("gpu_name") else "")
            ),
            "  Aux models (MiniLM, DistilGPT-2) run on CPU by design.",
            "  Canonical timings: results/suite_b/runtime_overhead.csv (paper runtime paragraph).",
        ]
        for key, label in (
            ("minilm_semantic", "MiniLM semantic"),
            ("distilgpt2_perplexity", "DistilGPT-2 perplexity"),
            ("minilm_plus_distilgpt2", "MiniLM + DistilGPT-2"),
            ("method_a_full_sanitize", "Method A full sanitize"),
        ):
            c = comps.get(key)
            if not c or c.get("mean_ms") is None:
                continue
            lines.append(
                f"  {label:24s}  mean={c['mean_ms']:.2f} ms/prompt  "
                f"(median={c['median_ms']:.2f}, p95={c['p95_ms']:.2f}, n={c['n']})"
            )
        return lines

    sys.modules["src.evaluation"] = ModuleType("src.evaluation")
    sys.modules["src.evaluation.latency"] = lat
    lat.format_overhead_for_note = format_overhead_for_note

    code = (ROOT / "src/visualization/reporting.py").read_text(encoding="utf-8")
    code = code.replace(
        "from ..config import JUDGE_MODEL_NAME, TARGET_MODEL_NAME",
        "from src.config import JUDGE_MODEL_NAME, TARGET_MODEL_NAME",
    )
    code = code.replace(
        "from ..evaluation.latency import format_overhead_for_note",
        "from src.evaluation.latency import format_overhead_for_note",
    )
    ns: dict = {}
    exec(compile(code, "reporting.py", "exec"), ns)
    return ns["_write_robustness_note"]


def main():
    write_note = _load_write_fn()
    overhead = json.loads(
        (ROOT / "results/suite_b/runtime_overhead.json").read_text(encoding="utf-8")
    )
    for name in (
        "runtime_overhead.json",
        "runtime_overhead.csv",
        "runtime_overhead.txt",
    ):
        shutil.copy2(ROOT / f"results/suite_b/{name}", ROOT / f"results/suite_a/{name}")
        print(f"copied {name} -> suite_a")

    para_all = pd.read_csv(ROOT / "results/robustness_paraphrase_test.csv")
    para_all.columns = [c.strip() for c in para_all.columns]

    def regen(suite: str, methods: list[str]) -> None:
        rd = ROOT / f"results/{suite}"
        metrics = pd.read_csv(rd / "metrics.csv")
        edge = pd.read_csv(rd / "robustness_edge_benign.csv")
        adaptive = pd.read_csv(rd / "robustness_adaptive.csv")
        for df in (metrics, edge, adaptive):
            df.columns = [c.strip() for c in df.columns]
        para = para_all[para_all["method"].isin(methods)].copy()
        order = {m: i for i, m in enumerate(methods)}
        para["_o"] = para["method"].map(order)
        para = para.sort_values("_o").drop(columns="_o")
        write_note(
            metrics,
            para,
            edge,
            adaptive,
            out_path=rd / "robustness_note.txt",
            overhead_report=overhead,
        )

    regen("suite_a", ["baseline", "regex", "keyword", "context_aware"])
    regen("suite_b", ["baseline", "regex", "keyword", "context_aware_learned"])

    banned = (
        "Regex sanitizer has the highest",
        "better (-)",
        "because of semantic similarity",
        "WHERE DEFENSES FAILED",
        "Multilingual attacks (German)",
    )
    for s in ("suite_a", "suite_b"):
        text = (ROOT / f"results/{s}/robustness_note.txt").read_text(encoding="utf-8")
        for b in banned:
            assert b not in text, f"{s} still contains: {b}"
        assert "20.98" in text and "14.73" not in text
        assert "Highest edge-case FPR among listed gates:" in text
        print(f"{s}: checks passed")


if __name__ == "__main__":
    main()
