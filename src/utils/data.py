"""Dataset split loading utilities."""

import os

import pandas as pd

from ..config import EXTERNAL_TEST_CORE_CSV, EXTERNAL_TEST_MAIN_CSV, SPLITS_DIR
from ..splits.source_holdout import load_source_holdout


def load_splits():
    """Return the frozen mixed-source ``(train, val, test)`` split."""
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    return train_df, val_df, test_df


def load_source_holdout_splits(source: str):
    """Return ``(train, val, test)`` for one held-out attack source.

    ``source`` is ``simsonsun``, ``mpdd``, ``prompt_injection``, ``bravansky``,
    or ``neuralchemy``.
    """
    return load_source_holdout(source)


def load_external_test(slice_name: str = "main") -> pd.DataFrame:
    """Load the Neuralchemy held-out test.

    slice_name ``main`` is the direct single-turn slice (test_main.csv).
    slice_name ``core`` is every kept row, including the appendix categories.
    """
    path = EXTERNAL_TEST_MAIN_CSV if slice_name == "main" else EXTERNAL_TEST_CORE_CSV
    if slice_name not in {"main", "core"}:
        raise ValueError("slice_name must be 'main' or 'core'")
    return pd.read_csv(path)
