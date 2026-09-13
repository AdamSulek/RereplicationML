#!/usr/bin/env python3
"""Compute native XGBoost TreeSHAP contributions for the selected models."""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import xgboost as xgb

from scripts.data_preparation.generate_ecfp import fingerprint_smiles
from scripts.train.train_xgb import feature_column, index_mask, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], required=True)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgb_full/shap"))
    parser.add_argument("--external", type=Path, default=Path("data/external_dataset_92_compound.csv"))
    parser.add_argument("--save-test-values", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def data_path(dataset: str) -> Path:
    if dataset == "mcf10a":
        return Path("data/mcf10a/mcf10a_xgb_2048.parquet")
    return Path("data/sw480/sw480_clean_xgb.parquet")


def main() -> None:
    args = parse_args()
    dataset_arg = argparse.Namespace(dataset=args.dataset)
    data = load_data(data_path(args.dataset), feature_column(dataset_arg))
    model_path = args.model_path or (Path("artifacts/xgb_full") / args.dataset / "scaffold/model.json")
    X = data["X"]
    splits = data["split"]
    test_idx = index_mask(splits, {"test"})

    fitted = xgb.XGBClassifier(n_jobs=args.n_jobs)
    fitted.load_model(model_path)
    booster = fitted.get_booster()
    test_contributions = booster.predict(
        xgb.DMatrix(X[test_idx]), pred_contribs=True, validate_features=False
    )
    test_values = test_contributions[:, :-1]

    output = args.output_dir / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    feature_summary = pd.DataFrame(
        {
            "feature_index": np.arange(test_values.shape[1]),
            "mean_abs_shap": np.abs(test_values).mean(axis=0),
            "mean_shap": test_values.mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False, kind="stable")
    feature_summary.insert(0, "shap_rank", np.arange(1, len(feature_summary) + 1))
    feature_summary.to_csv(output / "scaffold_test_feature_importance.csv", index=False)

    if args.save_test_values:
        np.savez_compressed(output / "scaffold_test_shap_values.npz", values=test_values, row_index=test_idx)

    external = pd.read_csv(args.external)
    external_values = fingerprint_smiles(external["smiles"].astype(str).tolist())
    external_contributions = booster.predict(
        xgb.DMatrix(external_values), pred_contribs=True, validate_features=False
    )[:, :-1]
    external_output = external.copy()
    external_output["shap_expected_value"] = float(test_contributions[:, -1].mean())
    external_output["shap_sum"] = external_contributions.sum(axis=1)
    external_output.to_csv(output / "external_shap_base.csv", index=False)
    np.savez_compressed(
        output / "external_shap_values.npz",
        values=external_contributions,
        feature_index=np.arange(external_contributions.shape[1]),
    )

    metadata = {
        "dataset": args.dataset,
        "protocol": "scaffold_selected_model",
        "feature_column": feature_column(dataset_arg),
        "feature_length": int(test_values.shape[1]),
        "model_path": str(model_path.resolve()),
        "test_rows": int(len(test_idx)),
        "external_rows": int(len(external)),
        "method": "XGBoost native pred_contribs (TreeSHAP)",
        "bias_column_excluded_from_feature_values": True,
        "outputs": {
            "test_feature_importance": str((output / "scaffold_test_feature_importance.csv").resolve()),
            "external_base": str((output / "external_shap_base.csv").resolve()),
            "external_values": str((output / "external_shap_values.npz").resolve()),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"dataset: {args.dataset}")
    print(f"test_rows: {len(test_idx)}")
    print(f"external_rows: {len(external)}")
    print(f"feature_length: {test_values.shape[1]}")
    print(f"top_feature: {int(feature_summary.iloc[0].feature_index)}")
    print(f"output: {output}")


if __name__ == "__main__":
    main()
