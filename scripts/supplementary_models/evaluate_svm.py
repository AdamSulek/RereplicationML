#!/usr/bin/env python3
"""Evaluate a linear SVM artifact on the held-out predefined test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def infer_feature_column(frame: pd.DataFrame) -> str:
    for column in ("X", "X_morgan_radius_2_count"):
        if column in frame.columns:
            return column
    raise ValueError("No supported fingerprint column found. Pass --feature-column explicitly.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--feature-column", default=None)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data_dir / args.dataset / "raw.parquet")
    feature_column = args.feature_column or infer_feature_column(frame)
    test = frame[frame["split"] == "test"]
    if test.empty:
        raise ValueError("The predefined test split is empty.")

    model_path = args.model_path or Path("artifacts/svm") / args.dataset / "model.joblib"
    model = load(model_path)
    features = np.stack(test[feature_column].to_numpy()).astype(np.float32)
    labels = test["activity"].astype(int).to_numpy()
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    print(json.dumps({
        "dataset": args.dataset,
        "feature_column": feature_column,
        "model_path": str(model_path),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }, indent=2))


if __name__ == "__main__":
    main()
