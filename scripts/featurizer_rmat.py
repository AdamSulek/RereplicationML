#!/usr/bin/env python3
"""Encode one predefined split as an R-MAT pickle cache."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pandas as pd
from huggingmolecules import RMatFeaturizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.data_dir / args.dataset / "raw.parquet")
    if args.split == "train":
        frame = frame[frame["split"].str.startswith("train_")]
    else:
        frame = frame[frame["split"] == args.split]
    if frame.empty:
        raise ValueError(f"The requested {args.split} split is empty.")

    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    encoded = featurizer.encode_smiles_list(
        frame["smiles"].tolist(),
        frame["activity"].astype(float).tolist(),
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("wb") as handle:
        pickle.dump(encoded, handle)


if __name__ == "__main__":
    main()
