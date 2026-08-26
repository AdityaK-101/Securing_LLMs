"""
Paraphrase probe on the 30 held-out test injections (Priority 3).

Same nine synonym substitutions as the train-pool diagnostic; gate-only Bypass.
Loads Method A and Method B together (shared MiniLM/DistilGPT-2). No LLM/judge.
"""

from __future__ import annotations

import json

from src.config import RESULTS_DIR, TEST_CSV, TRAIN_CSV
from src.evaluation.robustness import robustness_paraphrase
from src.visualization.plots import _plot_paraphrase_robustness


def main() -> None:
    out_dir = RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    para_df = robustness_paraphrase(
        prompt_csv=TEST_CSV,
        train_csv=TRAIN_CSV,
        n=None,
        include_method_a=True,
        include_method_b=True,
    )
    out_csv = out_dir / "robustness_paraphrase_test.csv"
    para_df.to_csv(out_csv, index=False)
    _plot_paraphrase_robustness(para_df, figures_dir=out_dir / "figures")

    print("\n--- Paraphrase Bypass (test injections) ---")
    cols = [
        "method",
        "orig_Bypass_Rate",
        "para_Bypass_Rate",
        "Bypass_Rate_delta",
        "n_prompts",
        "changed_n",
    ]
    print(para_df[cols].to_string(index=False))
    print(f"[OK] Wrote {out_csv}")
    print(json.dumps(para_df.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
