#!/usr/bin/env python3
"""Run the 22 final XGBoost models on an external compound library."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

# Make imports work when launched as python scripts/file.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_xgb_full import SEED, feature_column, index_mask, load_data, model


FEATURE_LENGTH = 2048


def external_features(smiles: list[str]) -> sparse.csr_matrix:
    try:
        from skfp.fingerprints import ECFPFingerprint
    except ModuleNotFoundError:
        sys.path.append(
            "/net/storage/pr3/plgrid/plggsanodrugs/miniconda/envs/savi/lib/python3.10/site-packages"
        )
        from skfp.fingerprints import ECFPFingerprint

    vectors = ECFPFingerprint(count=True, radius=2).transform(smiles)
    matrix = sparse.csr_matrix(np.asarray(vectors, dtype=np.float32))
    if matrix.shape[1] != FEATURE_LENGTH:
        raise ValueError(f"Expected {FEATURE_LENGTH} features, got {matrix.shape[1]}")
    return matrix


def predict(model_object, features: sparse.csr_matrix) -> np.ndarray:
    return model_object.predict_proba(features)[:, 1]


def train_indices(splits: np.ndarray, fold: int | None) -> tuple[np.ndarray, np.ndarray]:
    if fold is None:
        train = index_mask(splits, {f"train_{i}" for i in range(10)})
        return train, np.empty(0, dtype=int)
    validation_fold = (fold + 1) % 10
    train = index_mask(
        splits,
        {f"train_{i}" for i in range(10) if i not in {fold, validation_fold}}
        | {"val", "test"},
    )
    return train, np.empty(0, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/external_dataset_92_compound.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/xgb_full/external_dataset_92_compound_predictions.csv"))
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    external = pd.read_csv(args.input)
    required = {"smiles"}
    missing = required - set(external.columns)
    if missing:
        raise KeyError(f"Missing external columns: {sorted(missing)}")
    if external["smiles"].isna().any():
        raise ValueError("External dataset contains missing SMILES")
    if external["smiles"].duplicated().any():
        raise ValueError("External dataset contains duplicate SMILES")

    external_matrix = external_features(external["smiles"].astype(str).tolist())
    result = external.copy()
    metadata: dict[str, object] = {"input": str(args.input.resolve()), "rows": len(result), "models": {}}

    for dataset in ("mcf10a", "sw480_clean"):
        data_path = Path("data/mcf10a/mcf10a_xgb_2048.parquet") if dataset == "mcf10a" else Path("data/sw480/sw480_clean_xgb.parquet")
        data = load_data(data_path, feature_column(argparse.Namespace(dataset=dataset)))
        params = json.loads(
            (Path("artifacts/xgb_full") / dataset / "scaffold/scaffold_summary.json").read_text()
        )["selected_params"]
        X = data["X"]
        y = data["y"]
        splits = data["split"]
        scaffold_train, _ = train_indices(splits, None)
        scaffold_model = model(params, y[scaffold_train], args.n_jobs)
        scaffold_model.fit(X[scaffold_train], y[scaffold_train])
        scaffold_probability = predict(scaffold_model, external_matrix)
        result[f"pred_prob_scaffold_{dataset}"] = scaffold_probability

        fold_probabilities = []
        for fold in range(10):
            fold_train, _ = train_indices(splits, fold)
            fold_model = model(params, y[fold_train], args.n_jobs)
            fold_model.fit(X[fold_train], y[fold_train])
            fold_probabilities.append(predict(fold_model, external_matrix))
        fold_matrix = np.vstack(fold_probabilities)
        result[f"pred_prob_kfold_mean_{dataset}"] = fold_matrix.mean(axis=0)
        result[f"pred_prob_kfold_std_{dataset}"] = fold_matrix.std(axis=0, ddof=1)
        metadata["models"][dataset] = {
            "scaffold": 1,
            "kfold": 10,
            "feature_column": feature_column(argparse.Namespace(dataset=dataset)),
            "feature_length": FEATURE_LENGTH,
            "params": params,
        }

    rank_columns = {
        "rank_scaffold_mcf10a": "pred_prob_scaffold_mcf10a",
        "rank_kfold_mcf10a": "pred_prob_kfold_mean_mcf10a",
        "rank_scaffold_sw480": "pred_prob_scaffold_sw480_clean",
        "rank_kfold_sw480": "pred_prob_kfold_mean_sw480_clean",
    }
    for rank_column, probability_column in rank_columns.items():
        order = result[probability_column].rank(method="first", ascending=False).astype(int)
        result[rank_column] = order

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    metadata["output"] = str(args.output.resolve())
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"wrote: {args.output}")
    print(f"rows: {len(result)}")
    print(f"models: 22 (2 scaffold + 20 kfold)")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
