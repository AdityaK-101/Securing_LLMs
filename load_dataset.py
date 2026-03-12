"""Utility to load the train / validation / test splits."""

import os
import pandas as pd

SPLITS_DIR = os.path.join(os.path.dirname(__file__), "data", "splits")


def load_splits():
    """Return (train_df, val_df, test_df) DataFrames."""
    train_df = pd.read_csv(os.path.join(SPLITS_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(SPLITS_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    return train_df, val_df, test_df


if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
