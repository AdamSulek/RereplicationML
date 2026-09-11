#!/usr/bin/env python3
"""Generate the exact count-ECFP representation used by final XGBoost models."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


FEATURE_COLUMN = "X_morgan_radius_2_count"
RADIUS = 2
FP_SIZE = 2048
INCLUDE_CHIRALITY = False
USE_BOND_TYPES = True
INCLUDE_RING_MEMBERSHIP = True
COUNT_SIMULATION = False


def make_generator():
    """Construct the fully specified RDKit Morgan fingerprint generator."""
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=INCLUDE_CHIRALITY,
        useBondTypes=USE_BOND_TYPES,
        includeRingMembership=INCLUDE_RING_MEMBERSHIP,
        countSimulation=COUNT_SIMULATION,
    )


def fingerprint_smiles(smiles: list[str]) -> np.ndarray:
    """Return dense integer count fingerprints, rejecting invalid SMILES."""
    molecules = [Chem.MolFromSmiles(value) for value in smiles]
    invalid = [index for index, molecule in enumerate(molecules) if molecule is None]
    if invalid:
        raise ValueError(f"Invalid SMILES at batch row(s): {invalid[:10]}")
    generator = make_generator()
    matrix = np.stack(
        [generator.GetCountFingerprintAsNumPy(molecule) for molecule in molecules]
    )
    if matrix.shape != (len(smiles), FP_SIZE):
        raise RuntimeError(f"Unexpected fingerprint matrix shape: {matrix.shape}")
    if not np.issubdtype(matrix.dtype, np.integer):
        raise TypeError(f"Expected integer count fingerprints, got {matrix.dtype}")
    return matrix


def configuration() -> dict[str, object]:
    return {
        "type": "Morgan/ECFP count fingerprint",
        "feature_column": FEATURE_COLUMN,
        "radius": RADIUS,
        "fp_size": FP_SIZE,
        "count": True,
        "include_chirality": INCLUDE_CHIRALITY,
        "use_bond_types": USE_BOND_TYPES,
        "include_ring_membership": INCLUDE_RING_MEMBERSHIP,
        "count_simulation": COUNT_SIMULATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    source = pq.ParquetFile(args.input)
    required = {"smiles", "activity", "split"}
    missing = required - set(source.schema_arrow.names)
    if missing:
        raise KeyError(f"Missing required input columns: {sorted(missing)}")
    total = source.metadata.num_rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    processed = 0
    started = time.monotonic()

    try:
        for batch in source.iter_batches(
            batch_size=args.batch_size,
            columns=["smiles", "activity", "split"],
        ):
            smiles = list(map(str, batch.column(0).to_pylist()))
            activity = batch.column(1).to_pylist()
            splits = list(map(str, batch.column(2).to_pylist()))
            features = fingerprint_smiles(smiles).tolist()
            table = pa.table(
                {
                    "source_row_index": list(range(processed, processed + len(smiles))),
                    "smiles": smiles,
                    "activity": activity,
                    "split": splits,
                    FEATURE_COLUMN: features,
                }
            )
            if writer is None:
                writer = pq.ParquetWriter(args.output, table.schema)
            writer.write_table(table)
            processed += len(smiles)
            elapsed = max(time.monotonic() - started, 1e-9)
            rate = processed / elapsed
            remaining = max(total - processed, 0) / max(rate, 1e-9)
            print(
                f"progress: {processed}/{total} ({100 * processed / total:6.2f}%) "
                f"rate={rate:,.0f}/s eta={remaining / 60:.1f} min",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(configuration(), indent=2) + "\n")
    print(f"completed: {processed}/{total}")
    print(f"output: {args.output}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
