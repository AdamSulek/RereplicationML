#!/usr/bin/env python3
"""Plot a deterministic UMAP projection from prepared molecular features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--feature-column", default="X_morgan_radius_2_count")
    parser.add_argument("--max-points", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import matplotlib.pyplot as plt
    from umap import UMAP

    frame = pd.read_parquet(args.input, columns=[args.feature_column, "activity", "split"])
    if len(frame) > args.max_points:
        frame = frame.sample(args.max_points, random_state=args.seed)
    matrix = np.stack(frame[args.feature_column].to_numpy()).astype(np.float32)
    embedding = UMAP(metric="jaccard", random_state=args.seed).fit_transform(matrix > 0)
    figure, axis = plt.subplots(figsize=(8, 6))
    scatter = axis.scatter(embedding[:, 0], embedding[:, 1], c=frame["activity"], s=5, alpha=0.6)
    axis.set(xlabel="UMAP 1", ylabel="UMAP 2", title="Prepared chemical space")
    figure.colorbar(scatter, ax=axis, label="Activity")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
