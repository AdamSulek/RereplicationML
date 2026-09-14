#!/usr/bin/env python3
"""Verify the final XGBoost input files before a reproduction run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


DATASETS = {
    "mcf10a": (
        Path("data/mcf10a/mcf10a_xgb_2048.parquet"),
        "X_morgan_radius_2_count",
    ),
    "sw480_clean": (
        Path("data/sw480/sw480_clean_xgb.parquet"),
        "X_morgan_radius_2_count",
    ),
}
EXPECTED_SPLITS = {"train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7", "train_8", "train_9", "val", "test"}


def audit_dataset(path: Path, feature_column: str) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)
    columns = parquet.schema_arrow.names
    if feature_column not in columns:
        raise KeyError(f"{feature_column!r} missing from {path}; columns={columns}")

    split_counts: dict[str, dict[str, int]] = {}
    total = 0
    feature_length: int | None = None
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=["activity", "split", feature_column],
    ):
        activities = batch.column(0).to_pylist()
        splits = batch.column(1).to_pylist()
        features = batch.column(2).to_pylist()
        if feature_length is None:
            feature_length = len(features[0])
        for vector in features:
            if len(vector) != feature_length:
                raise ValueError(f"Inconsistent feature length in {path}")
        for activity, split in zip(activities, splits):
            split = str(split)
            record = split_counts.setdefault(split, {"rows": 0, "positive": 0, "negative": 0})
            record["rows"] += 1
            if int(activity) == 1:
                record["positive"] += 1
            elif int(activity) == 0:
                record["negative"] += 1
            else:
                raise ValueError(f"Non-binary activity {activity!r} in {path}")
            total += 1

    missing_splits = sorted(EXPECTED_SPLITS - set(split_counts))
    unexpected_splits = sorted(set(split_counts) - EXPECTED_SPLITS)
    if missing_splits or unexpected_splits:
        raise ValueError(
            f"Invalid splits in {path}: missing={missing_splits}, unexpected={unexpected_splits}"
        )
    if feature_length != 2048:
        raise ValueError(f"Expected 2048 features in {path}, got {feature_length}")
    return {
        "path": str(path),
        "rows": total,
        "columns": columns,
        "feature_column": feature_column,
        "feature_length": feature_length,
        "splits": split_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, default=Path("data/external_dataset_92_compound.csv"))
    args = parser.parse_args()

    result = {dataset: audit_dataset(path, feature) for dataset, (path, feature) in DATASETS.items()}
    if args.external.is_file():
        import pandas as pd

        external = pd.read_csv(args.external)
        if external["smiles"].duplicated().any():
            raise ValueError("External dataset contains duplicate SMILES")
        result["external"] = {
            "path": str(args.external),
            "rows": len(external),
            "columns": external.columns.tolist(),
            "unique_smiles": int(external["smiles"].nunique()),
        }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
