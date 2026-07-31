#!/usr/bin/env python3
"""Fine-tune R-MAT for binary re-replication activity classification."""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingmolecules import RMatFeaturizer, RMatModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device: torch.device) -> dict[str, float]:
    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits.extend(model(batch).reshape(-1).detach().cpu().numpy())
            labels.extend(batch.y.reshape(-1).detach().cpu().numpy())

    labels_array = np.asarray(labels, dtype=int)
    probabilities = torch.sigmoid(torch.tensor(logits)).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(labels_array, probabilities)),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
    }


def encode(featurizer: RMatFeaturizer, frame: pd.DataFrame):
    return featurizer.encode_smiles_list(
        frame["smiles"].tolist(),
        frame["activity"].astype(float).tolist(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/rmat"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frame = pd.read_parquet(args.data_dir / args.dataset / "raw.parquet")
    train = frame[frame["split"].str.startswith("train_")]
    validation = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty train_*, val, and test predefined splits.")

    LOGGER.info("Loading R-MAT on %s and encoding the predefined splits", device)
    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    train_data = encode(featurizer, train)
    validation_data = encode(featurizer, validation)
    test_data = encode(featurizer, test)
    train_loader = featurizer.get_data_loader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = featurizer.get_data_loader(validation_data, batch_size=args.batch_size, shuffle=False)
    test_loader = featurizer.get_data_loader(test_data, batch_size=args.batch_size, shuffle=False)

    model = RMatModel.from_pretrained("rmat_4M").to(device)
    positives = int(train["activity"].sum())
    class_weight = (len(train) - positives) / max(positives, 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "model.pt"
    best_auc, stale_epochs = -np.inf, 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = loss_fn(model(batch).reshape(-1), batch.y.float().reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        validation_metrics = evaluate(model, validation_loader, device)
        LOGGER.info("Epoch %d validation ROC-AUC: %.4f", epoch, validation_metrics["roc_auc"])
        if validation_metrics["roc_auc"] > best_auc:
            best_auc, stale_epochs = validation_metrics["roc_auc"], 0
            torch.save({"model_state_dict": model.state_dict(), "args": vars(args)}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    results = {
        "dataset": args.dataset,
        "seed": args.seed,
        "best_validation_roc_auc": float(best_auc),
        "test": evaluate(model, test_loader, device),
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    LOGGER.info("Test ROC-AUC: %.4f", results["test"]["roc_auc"])


if __name__ == "__main__":
    main()
