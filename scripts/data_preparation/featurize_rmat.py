#!/usr/bin/env python3
"""Generate the split-preserving R-MAT pickle caches used by training."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_SRC = REPO_ROOT / "huggingmolecules" / "src"
if VENDORED_SRC.is_dir():
    sys.path.insert(0, str(VENDORED_SRC))
from huggingmolecules import RMatFeaturizer


SPLIT_NAMES = tuple([f"train_{index}" for index in range(10)] + ["val", "test"])


def select_split(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unsupported split {split_name!r}")
    selected = frame[frame["split"] == split_name]
    if selected.empty:
        raise ValueError(f"The requested split {split_name!r} is empty")
    return selected


def encode_split(
    featurizer: RMatFeaturizer,
    frame: pd.DataFrame,
    split_name: str,
    output_path: Path,
    *,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite")
    selected = select_split(frame, split_name)
    encoded = featurizer.encode_smiles_list(
        selected["smiles"].astype(str).tolist(),
        selected["activity"].astype(float).tolist(),
    )
    with output_path.open("wb") as handle:
        pickle.dump(encoded, handle)
    print(f"{split_name}: rows={len(selected)} output={output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480", "sw480_clean"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=SPLIT_NAMES, default=list(SPLIT_NAMES))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = args.input or args.data_dir / args.dataset / "raw.parquet"
    output_dir = args.output_dir or Path("pickle_dataloaders") / args.dataset
    frame = pd.read_parquet(input_path, columns=["smiles", "activity", "split"])
    if frame[["smiles", "activity", "split"]].isna().any().any():
        raise ValueError("Input contains missing SMILES, activity, or split values")
    labels = set(frame["activity"].astype(float).unique())
    if not labels <= {0.0, 1.0}:
        raise ValueError(f"Expected binary activity labels, got {sorted(labels)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    for split_name in args.splits:
        encode_split(
            featurizer,
            frame,
            split_name,
            output_dir / f"{split_name}.p",
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
