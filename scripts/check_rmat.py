#!/usr/bin/env python3
"""Evaluate an R-MAT artifact on the held-out predefined test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from huggingmolecules import RMatFeaturizer, RMatModel

from train_rmat_classifier import encode, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(args.data_dir / args.dataset / "raw.parquet")
    test = frame[frame["split"] == "test"]
    if test.empty:
        raise ValueError("The predefined test split is empty.")

    model_path = args.model_path or Path("artifacts/rmat") / args.dataset / "model.pt"
    checkpoint = torch.load(model_path, map_location=device)
    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    test_data = encode(featurizer, test)
    test_loader = featurizer.get_data_loader(test_data, batch_size=args.batch_size, shuffle=False)
    model = RMatModel.from_pretrained("rmat_4M").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(json.dumps({
        "dataset": args.dataset,
        "model_path": str(model_path),
        "test": evaluate(model, test_loader, device),
    }, indent=2))


if __name__ == "__main__":
    main()
