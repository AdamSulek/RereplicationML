#!/usr/bin/env python3
"""Evaluate one saved R-MAT checkpoint on one pre-featurized labeled split."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDORED_SRC = REPO_ROOT / "huggingmolecules" / "src"
if VENDORED_SRC.is_dir():
    sys.path.insert(0, str(VENDORED_SRC))
sys.path.insert(0, str(REPO_ROOT))
from huggingmolecules import RMatFeaturizer, RMatModel
from scripts.model_evaluation.metrics import screening_metrics


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = RMatModel.from_pretrained("rmat_4M").to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_labeled(model, loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    logits: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.extend(model(batch).reshape(-1).detach().cpu().tolist())
            labels.extend(np.asarray(batch.y.detach().cpu()).reshape(-1).astype(int).tolist())
    probabilities = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    return np.asarray(labels, dtype=int), probabilities


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split-pickle", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--predictions", type=Path, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with args.split_pickle.open("rb") as handle:
        encoded = pickle.load(handle)
    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    loader = featurizer.get_data_loader(encoded, batch_size=args.batch_size, shuffle=False)
    model, checkpoint = load_model(args.checkpoint, device)
    labels, probabilities = predict_labeled(model, loader, device)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split_pickle": str(args.split_pickle.resolve()),
        "metrics": screening_metrics(labels, probabilities, threshold=args.threshold),
    }
    if args.predictions:
        args.predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"source_row_index": np.arange(len(labels)), "y_true": labels, "prediction_score": probabilities}
        ).to_csv(args.predictions, index=False)
        result["predictions"] = str(args.predictions.resolve())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
