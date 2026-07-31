"""Shared helpers for loading the repository's prepared classification datasets."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


SUPPORTED_FEATURE_COLUMNS = ("X", "X_morgan_radius_2_count")


def seed_everything(seed: int) -> None:
    """Set NumPy and Python random seeds for reproducible classical ML runs."""
    random.seed(seed)
    np.random.seed(seed)


def infer_feature_column(frame: pd.DataFrame) -> str:
    """Return the first supported feature column available in a dataset."""
    for column in SUPPORTED_FEATURE_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(
        "No supported feature column found. Pass an explicit feature column."
    )


def load_predefined_split(
    dataset: str,
    data_dir: Path = Path("data"),
    feature_column: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """Load combined training, validation, and held-out test matrices."""
    frame = pd.read_parquet(data_dir / dataset / "raw.parquet")
    column = feature_column or infer_feature_column(frame)
    train = frame[frame["split"].str.startswith("train_")]
    validation = frame[frame["split"] == "val"]
    test = frame[frame["split"] == "test"]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Expected non-empty train_*, val, and test predefined splits.")

    def matrix(partition: pd.DataFrame) -> np.ndarray:
        return np.stack(partition[column].to_numpy()).astype(np.float32)

    return (
        matrix(train),
        train["activity"].astype(int).to_numpy(),
        matrix(validation),
        validation["activity"].astype(int).to_numpy(),
        matrix(test),
        test["activity"].astype(int).to_numpy(),
        column,
    )
