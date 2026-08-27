"""
Runtime / latency overhead for auxiliary models used by context-aware sanitizers.

Reports average time per prompt for:
  - MiniLM (sentence-transformers/all-MiniLM-L6-v2) — semantic signal
  - DistilGPT-2 — perplexity signal

Also records the hardware used for the measurement.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..config import RANDOM_SEED, TEST_CSV, TRAIN_CSV


def get_hardware_info() -> dict:
    """Collect CPU / RAM / GPU details for the overhead report."""
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count_logical": None,
        "ram_gb": None,
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "torch_version": None,
        "aux_model_device": "cpu",  # MiniLM + DistilGPT-2 run on CPU in this project
    }

    try:
        import os

        info["cpu_count_logical"] = os.cpu_count()
    except Exception:
        pass

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        # Fallback: Windows / Unix without psutil
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            info["ram_gb"] = round(stat.ullTotalPhys / (1024 ** 3), 2)
        except Exception:
            pass

    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_gb"] = round(props.total_memory / (1024 ** 3), 2)
            except Exception:
                pass
    except Exception:
        pass

    return info


def _summarize(times_ms: np.ndarray) -> dict:
    if times_ms.size == 0:
        return {
            "n": 0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "std_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    return {
        "n": int(times_ms.size),
        "mean_ms": round(float(times_ms.mean()), 3),
        "median_ms": round(float(np.median(times_ms)), 3),
        "p95_ms": round(float(np.percentile(times_ms, 95)), 3),
        "std_ms": round(float(times_ms.std()), 3),
        "min_ms": round(float(times_ms.min()), 3),
        "max_ms": round(float(times_ms.max()), 3),
    }


def _load_prompts(csv_path: str, n: int, seed: int = RANDOM_SEED) -> list[str]:
    df = pd.read_csv(csv_path)
    prompts = df["prompt"].dropna().astype(str).tolist()
    if not prompts:
        raise ValueError(f"No prompts found in {csv_path}")
    rng = np.random.default_rng(seed)
    if len(prompts) <= n:
        return prompts
    idx = rng.choice(len(prompts), size=n, replace=False)
    return [prompts[i] for i in idx]


def run_runtime_overhead_benchmark(
    train_csv: str = TRAIN_CSV,
    prompt_csv: str = TEST_CSV,
    n_prompts: int = 50,
    warmup: int = 5,
    out_dir: Optional[str | Path] = None,
) -> dict:
    """
    Benchmark MiniLM and DistilGPT-2 per-prompt latency on CPU.

    Uses the same ContextAwareSanitizer code paths as Method A:
      - MiniLM  → _semantic_signal(prompt)
      - DistilGPT-2 → _perplexity_signal(prompt)

    Also reports full Method A sanitize() latency and lightweight
    regex / keyword baselines for context.
    """
    from ..sanitizers.context_aware import ContextAwareSanitizer
    from ..sanitizers.keyword import KeywordHeuristicSanitizer
    from ..sanitizers.regex import RegexSanitizer

    hardware = get_hardware_info()
    prompts = _load_prompts(prompt_csv, n=n_prompts)
    warmup_prompts = prompts[: min(warmup, len(prompts))]
    timed_prompts = prompts

    print("\n[Latency] Loading Method A (MiniLM + DistilGPT-2) for overhead timing...")
    san = ContextAwareSanitizer(train_csv=train_csv)
    regex_san = RegexSanitizer()
    keyword_san = KeywordHeuristicSanitizer()

    if san._embed_model is None:
        print("[Latency] Warning: MiniLM unavailable — semantic timings will be empty.")
    if san._ppl_model is None:
        print("[Latency] Warning: DistilGPT-2 unavailable — perplexity timings will be empty.")

    # Warmup (exclude from stats)
    for p in warmup_prompts:
        if san._embed_model is not None:
            san._semantic_signal(p)
        if san._ppl_model is not None:
            san._perplexity_signal(p)
        san.sanitize(p)
        regex_san.sanitize(p)
        keyword_san.sanitize(p)

    minilm_times = []
    ppl_times = []
    method_a_times = []
    regex_times = []
    keyword_times = []

    print(
        f"[Latency] Timing {len(timed_prompts)} prompts "
        f"(warmup={len(warmup_prompts)}) on {hardware.get('aux_model_device', 'cpu')}..."
    )

    for p in timed_prompts:
        if san._embed_model is not None:
            t0 = time.perf_counter()
            san._semantic_signal(p)
            minilm_times.append((time.perf_counter() - t0) * 1000.0)

        if san._ppl_model is not None:
            t0 = time.perf_counter()
            san._perplexity_signal(p)
            ppl_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        san.sanitize(p)
        method_a_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        regex_san.sanitize(p)
        regex_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        keyword_san.sanitize(p)
        keyword_times.append((time.perf_counter() - t0) * 1000.0)

    components = {
        "minilm_semantic": {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "device": "cpu",
            "path": "ContextAwareSanitizer._semantic_signal",
            **_summarize(np.asarray(minilm_times, dtype=np.float64)),
        },
        "distilgpt2_perplexity": {
            "model": "distilgpt2",
            "device": "cpu",
            "path": "ContextAwareSanitizer._perplexity_signal",
            **_summarize(np.asarray(ppl_times, dtype=np.float64)),
        },
        "method_a_full_sanitize": {
            "model": "context_aware (all 8 signals + hard triggers)",
            "device": "cpu",
            "path": "ContextAwareSanitizer.sanitize",
            **_summarize(np.asarray(method_a_times, dtype=np.float64)),
        },
        "regex_sanitize": {
            "model": "regex",
            "device": "cpu",
            "path": "RegexSanitizer.sanitize",
            **_summarize(np.asarray(regex_times, dtype=np.float64)),
        },
        "keyword_sanitize": {
            "model": "keyword",
            "device": "cpu",
            "path": "KeywordHeuristicSanitizer.sanitize",
            **_summarize(np.asarray(keyword_times, dtype=np.float64)),
        },
    }

    # Combined aux-model overhead estimate (MiniLM + DistilGPT-2), if both present
    if minilm_times and ppl_times:
        combined = np.asarray(minilm_times, dtype=np.float64) + np.asarray(
            ppl_times, dtype=np.float64
        )
        components["minilm_plus_distilgpt2"] = {
            "model": "MiniLM + DistilGPT-2 (sum of separate timings)",
            "device": "cpu",
            "path": "_semantic_signal + _perplexity_signal",
            **_summarize(combined),
        }

    report = {
        "hardware": hardware,
        "settings": {
            "train_csv": train_csv,
            "prompt_csv": prompt_csv,
            "n_prompts": len(timed_prompts),
            "warmup": len(warmup_prompts),
            "seed": RANDOM_SEED,
            "note": (
                "Auxiliary models (MiniLM, DistilGPT-2) are intentionally run on CPU "
                "in this project so GPU memory stays free for the target LLM / judge."
            ),
        },
        "components": components,
    }

    san.release_models()

    _print_overhead_summary(report)

    if out_dir is not None:
        save_runtime_overhead_report(report, out_dir)

    return report


def _print_overhead_summary(report: dict) -> None:
    hw = report["hardware"]
    print("\n--- Runtime Overhead (avg ms / prompt) ---")
    print(
        f"  Hardware: {hw.get('processor') or 'unknown CPU'} | "
        f"RAM={hw.get('ram_gb')} GB | "
        f"CUDA={hw.get('cuda_available')} "
        f"{('(' + str(hw.get('gpu_name')) + ')') if hw.get('gpu_name') else ''}"
    )
    print(f"  Aux models device: {hw.get('aux_model_device')} (by design)")
    for key in (
        "minilm_semantic",
        "distilgpt2_perplexity",
        "minilm_plus_distilgpt2",
        "method_a_full_sanitize",
        "regex_sanitize",
        "keyword_sanitize",
    ):
        c = report["components"].get(key)
        if not c or c.get("mean_ms") is None:
            continue
        print(
            f"  {key:28s}  mean={c['mean_ms']:8.2f} ms  "
            f"median={c['median_ms']:8.2f} ms  "
            f"p95={c['p95_ms']:8.2f} ms  (n={c['n']})"
        )


def save_runtime_overhead_report(report: dict, out_dir: str | Path) -> tuple[Path, Path]:
    """Write runtime_overhead.json and runtime_overhead.csv under out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "runtime_overhead.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = []
    for name, comp in report["components"].items():
        rows.append(
            {
                "component": name,
                "model": comp.get("model"),
                "device": comp.get("device"),
                "n": comp.get("n"),
                "mean_ms_per_prompt": comp.get("mean_ms"),
                "median_ms_per_prompt": comp.get("median_ms"),
                "p95_ms_per_prompt": comp.get("p95_ms"),
                "std_ms": comp.get("std_ms"),
                "min_ms": comp.get("min_ms"),
                "max_ms": comp.get("max_ms"),
            }
        )
    csv_path = out / "runtime_overhead.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # Human-readable note
    txt_path = out / "runtime_overhead.txt"
    hw = report["hardware"]
    lines = [
        "=" * 62,
        "  Runtime / Latency Overhead Report",
        "=" * 62,
        "",
        "HARDWARE",
        "-" * 40,
        f"  Platform : {hw.get('platform')}",
        f"  Python   : {hw.get('python')}",
        f"  CPU      : {hw.get('processor')}",
        f"  CPU cores: {hw.get('cpu_count_logical')}",
        f"  RAM      : {hw.get('ram_gb')} GB",
        f"  Torch    : {hw.get('torch_version')}",
        f"  CUDA     : {hw.get('cuda_available')}",
        f"  GPU      : {hw.get('gpu_name')}",
        f"  GPU mem  : {hw.get('gpu_memory_gb')} GB",
        f"  Aux device (MiniLM / DistilGPT-2): {hw.get('aux_model_device')}",
        "",
        "SETTINGS",
        "-" * 40,
        f"  Prompts timed : {report['settings'].get('n_prompts')}",
        f"  Warmup        : {report['settings'].get('warmup')}",
        f"  Prompt source : {report['settings'].get('prompt_csv')}",
        f"  Note          : {report['settings'].get('note')}",
        "",
        "LATENCY (ms per prompt)",
        "-" * 40,
    ]
    for name, comp in report["components"].items():
        if comp.get("mean_ms") is None:
            continue
        lines.append(
            f"  {name:28s}  mean={comp['mean_ms']:8.2f}  "
            f"median={comp['median_ms']:8.2f}  "
            f"p95={comp['p95_ms']:8.2f}  n={comp['n']}"
        )
    lines += ["", "=" * 62, ""]
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[OK] Saved: {json_path}")
    print(f"[OK] Saved: {csv_path}")
    print(f"[OK] Saved: {txt_path}")
    return json_path, csv_path


def format_overhead_for_note(report: dict) -> list[str]:
    """Lines to append into robustness_note.txt."""
    hw = report["hardware"]
    comps = report["components"]
    lines = [
        "",
        "5. RUNTIME / LATENCY OVERHEAD (MiniLM + DistilGPT-2)",
        "-" * 40,
        f"  Hardware: CPU={hw.get('processor')}; RAM={hw.get('ram_gb')} GB; "
        f"CUDA={hw.get('cuda_available')}"
        + (f" ({hw.get('gpu_name')})" if hw.get("gpu_name") else ""),
        "  Aux models (MiniLM, DistilGPT-2) run on CPU by design.",
        "  Paper runtime paragraph uses the suite_b overhead CSV as canonical.",
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
