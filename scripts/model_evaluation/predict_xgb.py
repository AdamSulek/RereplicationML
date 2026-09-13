#!/usr/bin/env python3
"""Score a small SMILES CSV with one saved XGBoost model."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

FEATURE_COLUMN = "X_morgan_radius_2_count"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data_preparation.generate_ecfp import FP_SIZE, configuration, fingerprint_smiles


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "smiles" not in frame:
        raise KeyError("Input CSV must contain a smiles column")
    if frame["smiles"].isna().any():
        raise ValueError("Input contains missing SMILES")
    if FEATURE_COLUMN in frame:
        values = [ast.literal_eval(value) if isinstance(value, str) else value for value in frame[FEATURE_COLUMN]]
        features = np.asarray(values, dtype=np.float32)
        if features.shape != (len(frame), FP_SIZE):
            raise ValueError(f"{FEATURE_COLUMN!r} must contain {FP_SIZE}-element vectors")
        feature_source = "precomputed input column"
    else:
        features = fingerprint_smiles(frame["smiles"].astype(str).tolist())
        feature_source = "generated from SMILES"
    booster = xgb.Booster()
    booster.load_model(args.model_path)
    result = frame.copy()
    result["prediction_score"] = booster.predict(xgb.DMatrix(features), validate_features=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path.resolve()),
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "features": configuration(),
                "feature_source": feature_source,
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
