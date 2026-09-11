#!/usr/bin/env python3
"""Audit pre-featurized binary-classification pickle splits."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path
from typing import Any


def read_split(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        objects = pickle.load(handle)
    if not isinstance(objects, list):
        raise TypeError(f"{path}: expected list, got {type(objects).__name__}")
    if not objects:
        raise ValueError(f"{path}: split is empty")

    positive = 0
    observed: set[float] = set()
    for index, item in enumerate(objects):
        if not hasattr(item, "y"):
            raise TypeError(f"{path}: object {index} has no y attribute")
        scalar = item.y
        if hasattr(scalar, "reshape"):
            scalar = scalar.reshape(-1)[0]
        label = float(scalar.item() if hasattr(scalar, "item") else scalar)
        observed.add(label)
        if label not in (0.0, 1.0):
            raise ValueError(f"{path}: object {index} has non-binary label {label!r}")
        positive += int(label)

    samples = len(objects)
    result = {
        "file": path.name,
        "samples": samples,
        "positive": positive,
        "negative": samples - positive,
        "prevalence": positive / samples,
        "labels": sorted(observed),
    }
    del objects
    gc.collect()
    return result


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = sum(row["samples"] for row in rows)
    positive = sum(row["positive"] for row in rows)
    return {
        "samples": samples,
        "positive": positive,
        "negative": samples - positive,
        "prevalence": positive / samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    expected_names = [f"train_{index}.p" for index in range(10)] + ["val.p", "test.p"]
    actual_names = sorted(path.name for path in args.dataset_dir.glob("*.p"))
    if sorted(expected_names) != actual_names:
        raise ValueError(
            f"Expected exactly {sorted(expected_names)}, found {actual_names}"
        )

    rows = []
    for name in expected_names:
        row = read_split(args.dataset_dir / name)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    train_rows = rows[:10]

    splits = {
        "train": aggregate(train_rows),
        "validation": aggregate([rows[10]]),
        "test": aggregate([rows[11]]),
    }
    report = {
        "dataset_directory": str(args.dataset_dir.resolve()),
        "files_verified": len(rows),
        "per_file": rows,
        "splits": splits,
        "checks": {
            "exact_12_file_manifest": True,
            "all_objects_expose_y": True,
            "labels_binary_0_1_only": True,
            "all_12_splits_nonempty": True,
            "per_file_object_counts_recorded": True,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("SPLIT_TOTALS " + json.dumps(splits, sort_keys=True), flush=True)
    print(f"REPORT {args.report.resolve()}", flush=True)


if __name__ == "__main__":
    main()
