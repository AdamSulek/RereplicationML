#!/usr/bin/env python3
"""Score a small SMILES CSV with one saved R-MAT checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_SRC = REPO_ROOT / "huggingmolecules" / "src"
if VENDORED_SRC.is_dir():
    sys.path.insert(0, str(VENDORED_SRC))
sys.path.insert(0, str(REPO_ROOT))
from huggingmolecules import RMatFeaturizer
from scripts.model_evaluation.evaluate_rmat import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    if "smiles" not in frame:
        raise KeyError("Input CSV must contain a smiles column")
    if frame["smiles"].isna().any():
        raise ValueError("Input contains missing SMILES")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    encoded = featurizer.encode_smiles_list(frame["smiles"].astype(str).tolist())
    loader = featurizer.get_data_loader(encoded, batch_size=args.batch_size, shuffle=False)
    model, checkpoint = load_model(args.checkpoint, device)
    logits: list[float] = []
    with torch.no_grad():
        for batch in loader:
            logits.extend(model(batch.to(device)).reshape(-1).detach().cpu().tolist())
    result = frame.copy()
    result["prediction_score"] = torch.sigmoid(torch.tensor(logits)).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(
            {
                "model": "R-MAT rmat_4M",
                "checkpoint": str(args.checkpoint.resolve()),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "input": str(args.input.resolve()),
                "output": str(args.output.resolve()),
            },
            indent=2,
        ) + "\n"
    )


if __name__ == "__main__":
    main()
