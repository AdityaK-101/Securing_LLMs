"""Dataset split loading utilities."""

import os

import pandas as pd

from ..config import SPLITS_DIR


def load_splits():
    """Return (train_df, val_df, test_df) DataFrames."""
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    return train_df, val_df, test_df
