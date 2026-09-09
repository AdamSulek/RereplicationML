#!/usr/bin/env python3
"""Attach existing Morgan/ECFP features to a filtered CSV dataset.

The source parquet already contains the feature column used by the prototype
XGBoost model.  This script joins those deterministic features by SMILES while
preserving every row and split in the clean CSV, including duplicate SMILES.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE_FEATURE_COLUMN = "X_morgan_radius_2_count"
OUTPUT_FEATURE_COLUMN = "X_morgan_radius_2_count"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--source-parquet", type=Path, required=True)
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--expected-feature-length", type=int, default=2048)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean = pd.read_csv(args.clean_csv, usecols=["smiles", "activity", "split"])
    if clean["smiles"].isna().any():
        raise ValueError("The clean CSV contains missing SMILES")

    # Keep all rows for a SMILES because the supplied split assignment may put
    # duplicate SMILES in different partitions.  The fingerprint is identical.
    pending: dict[str, list[tuple[int, int, str]]] = defaultdict(list)
    for row_index, row in clean.iterrows():
        pending[str(row.smiles)].append(
            (int(row_index), int(row.activity), str(row.split))
        )

    source = pq.ParquetFile(args.source_parquet)
    if SOURCE_FEATURE_COLUMN not in source.schema_arrow.names:
        raise KeyError(
            f"{SOURCE_FEATURE_COLUMN!r} is missing from {args.source_parquet}; "
            f"available columns: {source.schema_arrow.names}"
        )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    written = 0
    matched_smiles: set[str] = set()

    try:
        for batch in source.iter_batches(
            batch_size=args.batch_size,
            columns=["smiles", SOURCE_FEATURE_COLUMN],
        ):
            smiles = batch.column(0).to_pylist()
            features = batch.column(1).to_pylist()
            out_smiles: list[str] = []
            out_activity: list[int] = []
            out_split: list[str] = []
            out_index: list[int] = []
            out_features: list[list[int | float]] = []

            for molecule_smiles, fingerprint in zip(smiles, features):
                molecule_smiles = str(molecule_smiles)
                rows = pending.get(molecule_smiles)
                if rows is None or molecule_smiles in matched_smiles:
                    continue
                matched_smiles.add(molecule_smiles)
                for row_index, activity, split in rows:
                    if len(fingerprint) != args.expected_feature_length:
                        raise ValueError(
                            f"Unexpected feature length for {molecule_smiles}: "
                            f"{len(fingerprint)} != {args.expected_feature_length}"
                        )
                    out_smiles.append(molecule_smiles)
                    out_activity.append(activity)
                    out_split.append(split)
                    out_index.append(row_index)
                    out_features.append(fingerprint)

            if not out_smiles:
                continue

            table = pa.table(
                {
                    "source_row_index": out_index,
                    "smiles": out_smiles,
                    "activity": out_activity,
                    "split": out_split,
                    OUTPUT_FEATURE_COLUMN: out_features,
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(args.output_parquet, table.schema)
            writer.write_table(table)
            written += len(out_smiles)
    finally:
        if writer is not None:
            writer.close()

    missing = set(pending) - matched_smiles
    if missing:
        raise RuntimeError(
            f"Could not find {len(missing)} clean SMILES in the source parquet"
        )
    if written != len(clean):
        raise RuntimeError(f"Expected {len(clean)} rows, wrote {written}")

    print(f"wrote: {args.output_parquet}")
    print(f"rows: {written}")
    print(f"unique_smiles: {len(matched_smiles)}")
    print(f"source_feature_column: {SOURCE_FEATURE_COLUMN}")
    print(f"output_feature_column: {OUTPUT_FEATURE_COLUMN}")


if __name__ == "__main__":
    main()
