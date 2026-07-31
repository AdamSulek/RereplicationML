#!/usr/bin/env python3
"""Train a Random Forest baseline on the repository's predefined split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


LOGGER = logging.getLogger(__name__)
PARAM_GRID = {
    "n_estimators": [300, 600],
    "max_depth": [None, 20],
    "max_features": ["sqrt", 0.3],
    "min_samples_leaf": [1, 3],
}


def infer_feature_column(frame: pd.DataFrame) -> str:
    for column in ("X", "X_morgan_radius_2_count"):
        if column in frame.columns:
            return column
    raise ValueError("No supported fingerprint column found. Pass --feature-column explicitly.")


def to_matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise KeyError(f"Column {column!r} is not present in the dataset")
    return np.stack(frame[column].to_numpy()).astype(np.float32)


def evaluate(model: RandomForestClassifier, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--feature-column")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/random_forest"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    data_path = args.data_dir / args.dataset / "raw.parquet"
    frame = pd.read_parquet(data_path)
    column = args.feature_column or infer_feature_column(frame)
    train = frame[frame["split"].str.startswith("train_")]
    validation = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty train_*, val, and test predefined splits.")

    x_train, y_train = to_matrix(train, column), train["activity"].astype(int).to_numpy()
    x_val, y_val = to_matrix(validation, column), validation["activity"].astype(int).to_numpy()
    x_test, y_test = to_matrix(test, column), test["activity"].astype(int).to_numpy()

    estimator = RandomForestClassifier(
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )
    search = GridSearchCV(
        estimator=estimator,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=args.seed),
        n_jobs=args.n_jobs,
        refit=True,
        verbose=1,
    )
    LOGGER.info("Training Random Forest with 10-fold stratified CV on %s", data_path)
    search.fit(x_train, y_train)

    results = {
        "dataset": args.dataset,
        "feature_column": column,
        "seed": args.seed,
        "best_cv_roc_auc": float(search.best_score_),
        "best_params": search.best_params_,
        "validation": evaluate(search.best_estimator_, x_val, y_val),
        "test": evaluate(search.best_estimator_, x_test, y_test),
    }
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    dump(search.best_estimator_, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    LOGGER.info("Validation ROC-AUC: %.4f", results["validation"]["roc_auc"])
    LOGGER.info("Test ROC-AUC: %.4f", results["test"]["roc_auc"])
    LOGGER.info("Saved model and metrics to %s", output_dir)


if __name__ == "__main__":
    main()
