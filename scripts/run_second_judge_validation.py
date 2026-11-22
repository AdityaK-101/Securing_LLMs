"""Second-family judge on the N=100 Table-1 human-validation pairs.

Loads Llama-3.2-3B-Instruct (override with --model) with the same COMPLIED/REFUSED
rubric as ``src.models.judge``, judges the frozen (prompt, reply) pairs, and
reports agreement + Cohen's κ for:
  human vs Qwen judge
  human vs second judge
  Qwen judge vs second judge

  python scripts/run_second_judge_validation.py
  python scripts/run_second_judge_validation.py --model mistralai/Mistral-7B-Instruct-v0.3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.judge import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    MAX_JUDGE_OUTPUT_CHARS,
    MAX_JUDGE_PROMPT_CHARS,
    parse_judge_output,
)
from src.utils.gpu import empty_cuda_cache

DATA = ROOT / "data" / "judge_human_validation"
RESULTS = ROOT / "results" / "judge_human_validation"
DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"


def _norm(label: str) -> str:
    h = (label or "").strip().upper()
    if h == "COMPILED":
        return "COMPLIED"
    return h


def _cohen_kappa(a: int, b: int, c: int, d: int) -> float:
    n = a + b + c + d
    if n == 0:
        return float("nan")
    po = (a + d) / n
    pe = ((a + b) * (a + c) + (c + d) * (b + d)) / (n * n)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def _pair_metrics(y1: list[str], y2: list[str]) -> dict:
    a = b = c = d = 0
    for u, v in zip(y1, y2):
        if u == "COMPLIED" and v == "COMPLIED":
            a += 1
        elif u == "COMPLIED" and v == "REFUSED":
            b += 1
        elif u == "REFUSED" and v == "COMPLIED":
            c += 1
        else:
            d += 1
    n = a + b + c + d
    agree = a + d
    return {
        "N": n,
        "n_agree": agree,
        "percent_agreement": round(100.0 * agree / n, 1) if n else float("nan"),
        "cohens_kappa": round(_cohen_kappa(a, b, c, d), 3),
        "CC": a,
        "CR": b,
        "RC": c,
        "RR": d,
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    return (100.0 * max(0.0, center - half), 100.0 * min(1.0, center + half))


def _truncate(prompt: str, output: str) -> tuple[str, str]:
    p = prompt[:MAX_JUDGE_PROMPT_CHARS]
    o = output[:MAX_JUDGE_OUTPUT_CHARS]
    if len(prompt) > MAX_JUDGE_PROMPT_CHARS:
        p += "\n[... truncated ...]"
    if len(output) > MAX_JUDGE_OUTPUT_CHARS:
        o += "\n[... truncated ...]"
    return p, o


def _load_pipe(model_id: str, *, load_4bit: bool = False):
    """Load like the Qwen target/judge path; optional 4-bit for 7B on 8GB GPUs."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()
    empty_cuda_cache()
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    use_cuda = torch.cuda.is_available()
    mode = (
        "cuda 4-bit"
        if load_4bit and use_cuda
        else "fp16 device_map=auto (GPU+CPU offload)"
        if use_cuda
        else "cpu fp32"
    )
    print(f"[SecondJudge] Loading {model_id} ({mode})...")
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    kwargs: dict = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    pipe_device = None
    if load_4bit and use_cuda:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        kwargs["device_map"] = "auto"
    elif use_cuda:
        # Full fp16 quality: spill layers to CPU RAM when VRAM < model size.
        kwargs["torch_dtype"] = torch.float16
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: "7GiB", "cpu": "48GiB"}
    else:
        kwargs["torch_dtype"] = torch.float32
        pipe_device = -1

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None

    pipe_kwargs: dict = {
        "task": "text-generation",
        "model": model,
        "tokenizer": tok,
        "return_full_text": False,
    }
    if pipe_device is not None:
        pipe_kwargs["device"] = pipe_device
    pipe = pipeline(**pipe_kwargs)
    print("[SecondJudge] Ready.")
    return pipe


def _judge_one(pipe, prompt: str, output: str) -> tuple[str, str]:
    p, o = _truncate(prompt, output)
    user = JUDGE_USER_TEMPLATE.format(original_prompt=p, llm_output=o)
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    formatted = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    try:
        raw = pipe(
            formatted,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=pipe.tokenizer.eos_token_id,
        )[0]["generated_text"].strip()
        return parse_judge_output(raw)
    except Exception as e:
        return "ERROR", f"[JUDGE_ERROR] {e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument(
        "--load-4bit",
        dest="load_4bit",
        action="store_true",
        help="Load with bitsandbytes 4-bit (needed for Mistral-7B on 8GB)",
    )
    ap.add_argument(
        "--no-load-4bit",
        dest="load_4bit",
        action="store_false",
        help="Disable 4-bit (fp16; will OOM for 7B on 8GB GPUs)",
    )
    ap.set_defaults(load_4bit=True)
    ap.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Reuse existing second_judge_labels.csv for this model when present",
    )
    ap.add_argument("--no-resume", dest="resume", action="store_false")
    args = ap.parse_args()
    load_4bit = bool(args.load_4bit)

    blind = list(
        csv.DictReader((DATA / "blind_labeling_sheet.csv").open(encoding="utf-8-sig"))
    )
    key = {
        r["sample_id"]: r
        for r in csv.DictReader((RESULTS / "judge_key.csv").open(encoding="utf-8-sig"))
    }
    if len(blind) != 100:
        raise SystemExit(f"Expected 100 blind rows, got {len(blind)}")

    out_labels = RESULTS / "second_judge_labels.csv"
    meta_path = RESULTS / "second_judge_meta.json"
    metrics_path = RESULTS / "second_judge_agreement.csv"
    summary_path = RESULTS / "second_judge_summary.txt"

    labels_by_id: dict[str, dict] = {}
    if args.resume and out_labels.is_file():
        for r in csv.DictReader(out_labels.open(encoding="utf-8")):
            if r.get("model") == args.model and _norm(r.get("second_judge_label", "")) in (
                "COMPLIED",
                "REFUSED",
            ):
                labels_by_id[r["sample_id"]] = r
        print(f"[SecondJudge] Resume: {len(labels_by_id)} cached labels for {args.model}")

    need = [r for r in blind if r["sample_id"] not in labels_by_id]
    pipe = None
    if need:
        pipe = _load_pipe(args.model, load_4bit=load_4bit)
        for i, r in enumerate(need, 1):
            sid = r["sample_id"]
            lab, reason = _judge_one(pipe, r["original"], r["llm_output"])
            lab = _norm(lab)
            if lab not in ("COMPLIED", "REFUSED"):
                lab = "REFUSED"
            labels_by_id[sid] = {
                "sample_id": sid,
                "prompt_id": r["prompt_id"],
                "fold": r.get("fold", ""),
                "model": args.model,
                "second_judge_label": lab,
                "second_judge_reasoning": reason,
            }
            if i % 5 == 0 or i == len(need) or i == 1:
                print(f"[SecondJudge] {i}/{len(need)}  last={sid}:{lab}", flush=True)
            # checkpoint only completed rows
            with out_labels.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "sample_id",
                        "prompt_id",
                        "fold",
                        "model",
                        "second_judge_label",
                        "second_judge_reasoning",
                    ],
                )
                w.writeheader()
                for row in blind:
                    if row["sample_id"] in labels_by_id:
                        w.writerow(labels_by_id[row["sample_id"]])
        if pipe is not None:
            del pipe
            empty_cuda_cache()
    else:
        print("[SecondJudge] All 100 labels already present.")

    human, qwen, second = [], [], []
    report_rows = []
    for r in blind:
        sid = r["sample_id"]
        h = _norm(r["human_label"])
        q = _norm(key[sid]["judge_label"])
        s = _norm(labels_by_id[sid]["second_judge_label"])
        if h not in ("COMPLIED", "REFUSED"):
            raise SystemExit(f"Missing human_label on sample_id={sid}")
        human.append(h)
        qwen.append(q)
        second.append(s)
        report_rows.append(
            {
                "sample_id": sid,
                "fold": r.get("fold", ""),
                "prompt_id": r["prompt_id"],
                "human": h,
                "qwen_judge": q,
                "second_judge": s,
                "human_qwen_agree": h == q,
                "human_second_agree": h == s,
                "qwen_second_agree": q == s,
            }
        )

    m_hq = _pair_metrics(human, qwen)
    m_hs = _pair_metrics(human, second)
    m_qs = _pair_metrics(qwen, second)
    for m, (y1, y2) in (
        (m_hq, (human, qwen)),
        (m_hs, (human, second)),
        (m_qs, (qwen, second)),
    ):
        lo, hi = _wilson(m["n_agree"], m["N"])
        m["wilson95_lo"] = round(lo, 1)
        m["wilson95_hi"] = round(hi, 1)

    with (RESULTS / "second_judge_pair_report.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        w.writeheader()
        w.writerows(report_rows)

    metrics_rows = [
        {"pair": "human_vs_qwen", "second_model": args.model, **m_hq},
        {"pair": "human_vs_second", "second_model": args.model, **m_hs},
        {"pair": "qwen_vs_second", "second_model": args.model, **m_qs},
    ]
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        w.writeheader()
        w.writerows(metrics_rows)

    meta = {
        "n": 100,
        "second_model": args.model,
        "qwen_model": "Qwen/Qwen2.5-3B-Instruct",
        "rubric": "same COMPLIED/REFUSED as src.models.judge",
        "pairs": metrics_rows,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    paper = (
        f"A second judge from a different family ({args.model.split('/')[-1]}) "
        f"scored the same $N=100$ pairs under the same rubric: human--Qwen agreement "
        f"${m_hq['percent_agreement']:.1f}\\%$ ($\\kappa={m_hq['cohens_kappa']:.2f}$), "
        f"human--second-judge ${m_hs['percent_agreement']:.1f}\\%$ "
        f"($\\kappa={m_hs['cohens_kappa']:.2f}$), and Qwen--second-judge "
        f"${m_qs['percent_agreement']:.1f}\\%$ ($\\kappa={m_qs['cohens_kappa']:.2f}$)."
    )
    summary = "\n".join(
        [
            "Second-family judge validation (N=100 Table 1 pairs)",
            "=" * 50,
            f"Second model          = {args.model}",
            f"human vs Qwen         = {m_hq['percent_agreement']:.1f}%  κ={m_hq['cohens_kappa']:.3f}  "
            f"Wilson[{m_hq['wilson95_lo']}, {m_hq['wilson95_hi']}]",
            f"human vs second       = {m_hs['percent_agreement']:.1f}%  κ={m_hs['cohens_kappa']:.3f}  "
            f"Wilson[{m_hs['wilson95_lo']}, {m_hs['wilson95_hi']}]",
            f"Qwen vs second        = {m_qs['percent_agreement']:.1f}%  κ={m_qs['cohens_kappa']:.3f}  "
            f"Wilson[{m_qs['wilson95_lo']}, {m_qs['wilson95_hi']}]",
            "",
            "Paper sentence (LaTeX draft):",
            paper,
            "",
        ]
    )
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
