#!/usr/bin/env python3
"""Score a small SMILES CSV with one saved XGBoost model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from scripts.data_preparation.generate_ecfp import configuration, fingerprint_smiles


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
    features = fingerprint_smiles(frame["smiles"].astype(str).tolist())
    model = XGBClassifier()
    model.load_model(args.model_path)
    result = frame.copy()
    result["prediction_score"] = model.predict_proba(features)[:, 1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "model_path": str(args.model_path.resolve()),
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
                "features": configuration(),
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
