"""CLI utility to load the train / validation / test splits."""

from src.utils.data import load_splits


if __name__ == "__main__":
    train, val, test = load_splits()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")
