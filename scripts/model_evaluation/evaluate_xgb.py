#!/usr/bin/env python3
"""Evaluate a saved XGBoost model on a labeled prepared-data split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.model_evaluation.metrics import screening_metrics


FEATURE_COLUMN = "X_morgan_radius_2_count"


def default_data_path(dataset: str, data_dir: Path) -> Path:
    if dataset == "mcf10a":
        return data_dir / "mcf10a" / "mcf10a_xgb_2048.parquet"
    if dataset == "sw480_clean":
        return data_dir / "sw480" / "sw480_clean_xgb.parquet"
    return data_dir / dataset / "raw.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480", "sw480_clean"], required=True)
    parser.add_argument("--split", required=True, help="Exact split label, e.g. val or test")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--feature-column", default=FEATURE_COLUMN)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    data_path = args.data_path or default_data_path(args.dataset, args.data_dir)
    frame = pd.read_parquet(
        data_path,
        columns=["smiles", "activity", "split", args.feature_column],
        filters=[("split", "=", args.split)],
    )
    if frame.empty:
        raise ValueError(f"Prepared split {args.split!r} is empty")
    features = np.stack(frame[args.feature_column].to_numpy()).astype(np.float32)
    labels = frame["activity"].astype(int).to_numpy()
    model = XGBClassifier()
    model.load_model(args.model_path)
    probabilities = model.predict_proba(features)[:, 1]
    result = {
        "dataset": args.dataset,
        "split": args.split,
        "feature_column": args.feature_column,
        "model_path": str(args.model_path.resolve()),
        "metrics": screening_metrics(labels, probabilities, threshold=args.threshold),
    }
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "smiles": frame["smiles"].astype(str),
                "y_true": labels,
                "prediction_score": probabilities,
            }
        ).to_csv(args.predictions, index=False)
        result["predictions"] = str(args.predictions.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
