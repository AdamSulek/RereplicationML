#!/usr/bin/env python3
"""Generate 2048-dimensional count ECFP features with progress reporting."""

from __future__ import annotations

import argparse
import sys
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
try:
    from skfp.fingerprints import ECFPFingerprint
except ModuleNotFoundError:
    sys.path.append(
        "/net/storage/pr3/plgrid/plggsanodrugs/miniconda/envs/savi/lib/python3.10/site-packages"
    )
    from skfp.fingerprints import ECFPFingerprint


FEATURE_COLUMN = "X_morgan_radius_2_count"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/mcf10a/raw.parquet"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mcf10a/mcf10a_xgb_2048.parquet"),
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()

    source = pq.ParquetFile(args.input)
    total = source.metadata.num_rows
    fingerprint = ECFPFingerprint(count=True, radius=2)
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
            features = [vector.tolist() for vector in fingerprint.transform(smiles)]
            table = pa.table(
                {
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

    print(f"completed: {processed}/{total}")
    print(f"output: {args.output}")
    print(f"feature_column: {FEATURE_COLUMN}")
    print("feature_length: 2048")


if __name__ == "__main__":
    main()
