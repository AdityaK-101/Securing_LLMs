"""Shared paths, seeds, and output directories."""

import os
import random
from pathlib import Path

import numpy as np

# Fixed seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

try:
    import torch

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
except ImportError:
    pass

try:
    from transformers import set_seed as _hf_set_seed

    _hf_set_seed(RANDOM_SEED)
except ImportError:
    pass

_ROOT = os.path.join(os.path.dirname(__file__), "..")
PROJECT_ROOT = Path(_ROOT).resolve()

TEST_CSV = os.path.join(_ROOT, "data", "splits", "test.csv")
TRAIN_CSV = os.path.join(_ROOT, "data", "splits", "train.csv")
VAL_CSV = os.path.join(_ROOT, "data", "splits", "val.csv")
SPLITS_DIR = os.path.join(_ROOT, "data", "splits")
# Source-holdout protocol. Distinct from the mixed combined train/val/test above
# (v3.0 ~700-row corpus; rebuild via scripts/build_combined_dataset.py).
SOURCE_HOLDOUT_DIR = PROJECT_ROOT / "data" / "splits" / "source_holdout"
# Held-out Neuralchemy core/test slice. Not part of the mixed combined corpus.
EXTERNAL_TEST_MAIN_CSV = os.path.join(
    _ROOT, "data", "external", "neuralchemy", "test_main.csv"
)
EXTERNAL_TEST_CORE_CSV = os.path.join(
    _ROOT, "data", "external", "neuralchemy", "test_core.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
# Resumable greedy LLM/judge outcomes + MiniLM/PPL features (mentor speedup path)
PROMPT_CACHE_DIR = RESULTS_DIR / "prompt_cache"

# Model identifiers used by the experiment pipeline (keep display + load names aligned)
TARGET_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ATTACK_GENERATOR_MODEL_NAME = "google/flan-t5-base"
# Default batch size for greedy target-LLM generation into OutcomeCache
LLM_CACHE_BATCH_SIZE = int(os.environ.get("LLM_CACHE_BATCH_SIZE", "4"))
