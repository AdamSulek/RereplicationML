#!/usr/bin/env python3
"""Report dataset sizes, labels, and predefined split composition."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize_dataset(path: Path) -> dict[str, object]:
    frame = pd.read_parquet(path, columns=["activity", "split"])
    split_summary = (
        frame.groupby("split", dropna=False)["activity"]
        .agg(rows="size", positive="sum")
        .assign(negative=lambda table: table["rows"] - table["positive"])
        .sort_index()
    )
    return {
        "name": path.parent.name,
        "rows": len(frame),
        "positive": int(frame["activity"].sum()),
        "negative": int((frame["activity"] == 0).sum()),
        "splits": split_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize the repository's predefined dataset splits."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    dataset_paths = sorted(args.data_dir.glob("*/raw.parquet"))
    if not dataset_paths:
        raise FileNotFoundError(f"No raw.parquet files found in {args.data_dir}")

    for path in dataset_paths:
        summary = summarize_dataset(path)
        prevalence = 100 * summary["positive"] / summary["rows"]
        print(
            f"{summary['name']}: {summary['rows']:,} rows, "
            f"{summary['positive']:,} positive ({prevalence:.2f}%), "
            f"{summary['negative']:,} negative"
        )
        print(summary["splits"].to_string())
        print()


if __name__ == "__main__":
    main()
