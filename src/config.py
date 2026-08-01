"""Shared paths, seeds, and output directories."""

import os
import random
from pathlib import Path

import numpy as np

# Fixed seed for reproducibility (§4.4 functional test requirement)
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

_ROOT = os.path.join(os.path.dirname(__file__), "..")
PROJECT_ROOT = Path(_ROOT)

TEST_CSV = os.path.join(_ROOT, "data", "splits", "test.csv")
TRAIN_CSV = os.path.join(_ROOT, "data", "splits", "train.csv")
VAL_CSV = os.path.join(_ROOT, "data", "splits", "val.csv")
SPLITS_DIR = os.path.join(_ROOT, "data", "splits")

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# Model identifiers used by the experiment pipeline (keep display + load names aligned)
TARGET_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
JUDGE_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ATTACK_GENERATOR_MODEL_NAME = "google/flan-t5-base"
