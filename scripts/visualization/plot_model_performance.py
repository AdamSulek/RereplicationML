#!/usr/bin/env python3
"""Plot manuscript model-performance bars from a tidy saved-metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    frame = pd.read_csv(args.input)
    required = {"model", "roc_auc", "pr_auc"}
    if missing := required - set(frame.columns):
        raise KeyError(f"Missing columns: {sorted(missing)}")
    plot = frame.set_index("model")[["roc_auc", "pr_auc"]].plot.bar(figsize=(8, 5))
    plot.set(ylabel="Score", ylim=(0, 1))
    plot.figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot.figure.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
