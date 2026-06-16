"""
src/classifier.py
=================
Stretch Goal: Logistic Regression + TF-IDF classifier for injection detection.

Trains on data/splits/train.csv and evaluates on test.csv.
Reports accuracy, F1, and confusion matrix.
"""

import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)

RANDOM_SEED = 42

_ROOT      = os.path.join(os.path.dirname(__file__), "..")
TRAIN_CSV  = os.path.join(_ROOT, "data", "splits", "train.csv")
VAL_CSV    = os.path.join(_ROOT, "data", "splits", "val.csv")
TEST_CSV   = os.path.join(_ROOT, "data", "splits", "test.csv")


class InjectionClassifier:
    """
    Logistic Regression + TF-IDF binary classifier.
    Labels: injection=1, benign=0
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words="english",
            sublinear_tf=True,
        )
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_SEED,
            C=1.0,
            solver="lbfgs",
        )
        self.is_trained = False

    def _encode(self, prompts, fit=False):
        if fit:
            return self.vectorizer.fit_transform(prompts)
        return self.vectorizer.transform(prompts)

    def train(self, train_csv: str = TRAIN_CSV):
        df = pd.read_csv(train_csv)
        X = self._encode(df["prompt"].tolist(), fit=True)
        y = (df["label"] == "injection").astype(int)
        self.model.fit(X, y)
        self.is_trained = True

        # Report training accuracy
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"[Classifier] Training accuracy: {acc:.4f}")
        return acc

    def evaluate(self, csv_path: str):
        """Evaluate on a given CSV split. Returns metrics dict."""
        df = pd.read_csv(csv_path)
        X = self._encode(df["prompt"].tolist())
        y_true = (df["label"] == "injection").astype(int)
        y_pred = self.model.predict(X)

        acc  = accuracy_score(y_true, y_pred)
        f1   = f1_score(y_true, y_pred)
        cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
        report = classification_report(
            y_true, y_pred,
            target_names=["benign", "injection"]
        )
        tn, fp, fn, tp = cm.ravel()

        # Bypass Rate: fraction of injections the classifier failed to block
        bypass_rate = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        return {
            "accuracy":    round(acc, 4),
            "f1":          round(f1, 4),
            "Bypass_Rate": round(bypass_rate, 4),
            "ASR":         round(bypass_rate, 4),  # legacy alias
            "FPR":         round(fp / (fp + tn), 4) if (fp + tn) > 0 else 0.0,
            "TP": int(tp), "TN": int(tn),
            "FP": int(fp), "FN": int(fn),
            "confusion_matrix": cm.tolist(),
            "report": report,
        }

    def predict(self, prompt: str) -> tuple:
        """Returns (label: str, probability: float)"""
        vec = self.vectorizer.transform([prompt])
        pred = self.model.predict(vec)[0]
        prob = self.model.predict_proba(vec)[0][pred]
        label = "injection" if pred == 1 else "benign"
        return label, round(float(prob), 4)


def run_classifier_experiment(
    train_csv: str = TRAIN_CSV,
    test_csv:  str = TEST_CSV,
    val_csv:   str = VAL_CSV,
) -> dict:
    """Train classifier and report results on val + test sets."""
    clf = InjectionClassifier()

    print("\n[Classifier] Training Logistic Regression + TF-IDF...")
    clf.train(train_csv)

    print("[Classifier] Evaluating on validation set...")
    val_results = clf.evaluate(val_csv)

    print("[Classifier] Evaluating on test set...")
    test_results = clf.evaluate(test_csv)

    print("\n--- Classifier: Validation Set ---")
    print(val_results["report"])

    print("--- Classifier: Test Set ---")
    print(test_results["report"])

    return {
        "classifier": clf,
        "val":  val_results,
        "test": test_results,
    }
