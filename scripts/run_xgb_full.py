#!/usr/bin/env python3
"""Run the XGBoost scaffold and R-MAT-compatible 10-fold analyses."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_score, roc_auc_score
from xgboost import XGBClassifier


LOGGER = logging.getLogger(__name__)
SEED = 42
GRID = [
    {"learning_rate": lr, "max_depth": depth, "n_estimators": estimators}
    for lr in (0.05, 0.1)
    for depth in (5, 7)
    for estimators in (300, 600)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480_clean"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/xgb_full"))
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--selection-metric", choices=["roc_auc", "average_precision"], default="roc_auc")
    parser.add_argument("--mode", choices=["scaffold", "kfold", "all"], default="all")
    return parser.parse_args()


def data_path(args: argparse.Namespace) -> Path:
    if args.dataset == "mcf10a":
        return args.data_dir / "mcf10a" / "mcf10a_xgb_2048.parquet"
    return args.data_dir / "sw480" / "sw480_clean_xgb.parquet"


def feature_column(args: argparse.Namespace) -> str:
    return "X_morgan_radius_2_count"


def load_data(path: Path, feature: str) -> dict[str, object]:
    parquet = pq.ParquetFile(path)
    if feature not in parquet.schema_arrow.names:
        raise KeyError(f"Missing logical feature {feature!r}; columns={parquet.schema_arrow.names}")

    rows: list[int] = []
    smiles: list[str] = []
    activities: list[int] = []
    splits: list[str] = []
    sparse_rows: list[sparse.csr_matrix] = []
    for batch in parquet.iter_batches(
        batch_size=4096,
        columns=["smiles", "activity", "split", feature],
    ):
        # Nested/list parquet fields may expose the physical leaf name
        # ``element`` even when the logical pandas column has the requested name.
        columns = [batch.column(i).to_pylist() for i in range(batch.num_columns)]
        batch_smiles, batch_activity, batch_split, batch_features = columns
        matrix = sparse.csr_matrix(np.asarray(batch_features, dtype=np.float32))
        sparse_rows.append(matrix)
        start = len(rows)
        count = len(batch_features)
        rows.extend(range(start, start + count))
        smiles.extend(map(str, batch_smiles))
        activities.extend(np.asarray(batch_activity, dtype=int).tolist())
        splits.extend(map(str, batch_split))

    features = sparse.vstack(sparse_rows, format="csr")
    labels = np.asarray(activities, dtype=np.int8)
    split_array = np.asarray(splits, dtype=object)
    LOGGER.info("Loaded %s: rows=%d, features=%s, nnz=%d", path, len(labels), features.shape, features.nnz)
    return {"X": features, "y": labels, "split": split_array, "smiles": smiles}


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    prevalence = float(labels.mean())
    precision_recall = {}
    order = np.argsort(-probabilities, kind="stable")
    for percent in (1, 5, 10):
        count = max(1, int(np.ceil(len(labels) * percent / 100)))
        top_labels = labels[order[:count]]
        precision = float(top_labels.mean())
        precision_recall[f"precision_at_{percent}pct"] = precision
        precision_recall[f"ef_at_{percent}pct"] = float(precision / prevalence) if prevalence else 0.0
    result: dict[str, float | int] = {
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "pr_auc_enrichment": float(average_precision_score(labels, probabilities) / prevalence),
        "rows": int(len(labels)),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "prevalence": prevalence,
    }
    result.update(precision_recall)
    return result


def model(params: dict[str, object], labels: np.ndarray, n_jobs: int) -> XGBClassifier:
    positive = max(int(labels.sum()), 1)
    negative = max(int((labels == 0).sum()), 1)
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        scale_pos_weight=negative / positive,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=n_jobs,
        **params,
    )


def predict_and_save(
    fitted: XGBClassifier,
    data: dict[str, object],
    indices: np.ndarray,
    output_path: Path,
) -> dict[str, float | int]:
    X = data["X"]
    y = data["y"]
    probabilities = fitted.predict_proba(X[indices])[:, 1]
    selected_smiles = np.asarray(data["smiles"], dtype=object)[indices]
    selected_labels = y[indices]
    order = np.argsort(-probabilities, kind="stable")
    prediction_frame = pd.DataFrame(
        {
            "row_index": indices,
            "smiles": selected_smiles,
            "true_label": selected_labels,
            "predicted_probability": probabilities,
            "descending_rank": np.empty(len(indices), dtype=int),
        }
    )
    prediction_frame.loc[order, "descending_rank"] = np.arange(1, len(indices) + 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame.to_csv(output_path, index=False)
    return metrics(selected_labels, probabilities)


def index_mask(splits: np.ndarray, values: set[str]) -> np.ndarray:
    return np.flatnonzero(np.isin(splits, list(values)))


def run_scaffold(args: argparse.Namespace, data: dict[str, object], root: Path) -> dict[str, object]:
    splits = data["split"]
    train_idx = index_mask(splits, {f"train_{i}" for i in range(10)})
    val_idx = index_mask(splits, {"val"})
    test_idx = index_mask(splits, {"test"})
    X = data["X"]
    y = data["y"]
    validation_rows = []
    selected_model: XGBClassifier | None = None
    selected_params: dict[str, object] | None = None
    best_value = -np.inf

    for number, params in enumerate(GRID):
        fitted = model(params, y[train_idx], args.n_jobs)
        fitted.fit(X[train_idx], y[train_idx])
        probabilities = fitted.predict_proba(X[val_idx])[:, 1]
        row = {"config_id": number, **params, **metrics(y[val_idx], probabilities)}
        validation_rows.append(row)
        value = float(row[args.selection_metric])
        if value > best_value:
            best_value = value
            selected_model = fitted
            selected_params = params

    assert selected_model is not None and selected_params is not None
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(validation_rows).to_csv(root / "scaffold_validation_grid.csv", index=False)
    test_metrics = predict_and_save(selected_model, data, test_idx, root / "scaffold_test_predictions.csv")
    result = {
        "dataset": args.dataset,
        "protocol": "scaffold",
        "feature_column": feature_column(args),
        "selection_metric": args.selection_metric,
        "selected_params": selected_params,
        "validation_grid": str((root / "scaffold_validation_grid.csv").resolve()),
        "test_metrics": test_metrics,
        "note": "The scaffold test was evaluated once after validation-only selection.",
    }
    (root / "scaffold_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def run_kfold(args: argparse.Namespace, data: dict[str, object], params: dict[str, object], root: Path) -> list[dict[str, object]]:
    splits = data["split"]
    X = data["X"]
    y = data["y"]
    fold_rows = []
    for fold in range(10):
        validation_fold = (fold + 1) % 10
        test_idx = index_mask(splits, {f"train_{fold}"})
        val_idx = index_mask(splits, {f"train_{validation_fold}"})
        train_idx = index_mask(
            splits,
            {f"train_{i}" for i in range(10) if i not in {fold, validation_fold}}
            | {"val", "test"},
        )
        fitted = model(params, y[train_idx], args.n_jobs)
        fitted.fit(X[train_idx], y[train_idx])
        fold_dir = root / f"fold_{fold}"
        test_metrics = predict_and_save(fitted, data, test_idx, fold_dir / "predictions.csv")
        val_probabilities = fitted.predict_proba(X[val_idx])[:, 1]
        val_metrics = metrics(y[val_idx], val_probabilities)
        fold_result = {
            "dataset": args.dataset,
            "protocol": "rmat_compatible_kfold",
            "fold": fold,
            "test_split": f"train_{fold}",
            "validation_split": f"train_{validation_fold}",
            "training_splits": [f"train_{i}" for i in range(10) if i not in {fold, validation_fold}] + ["val", "test"],
            "params": params,
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "prediction_file": str((fold_dir / "predictions.csv").resolve()),
        }
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_result, indent=2) + "\n")
        fold_rows.append(fold_result)
        LOGGER.info("fold=%d ROC-AUC=%.6f AP=%.6f", fold, test_metrics["roc_auc"], test_metrics["average_precision"])

    metrics_names = ["roc_auc", "average_precision", "pr_auc_enrichment"] + [
        f"{kind}_{percent}pct" for kind in ("precision_at", "ef_at") for percent in (1, 5, 10)
    ]
    summary: dict[str, object] = {"dataset": args.dataset, "protocol": "rmat_compatible_kfold", "params": params, "folds": fold_rows}
    for name in metrics_names:
        values = np.asarray([row["test_metrics"][name] for row in fold_rows], dtype=float)
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_sd"] = float(values.std(ddof=1))
    root.mkdir(parents=True, exist_ok=True)
    (root / "kfold_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    pd.DataFrame([{**row["test_metrics"], "fold": row["fold"]} for row in fold_rows]).to_csv(root / "kfold_metrics.csv", index=False)
    return fold_rows


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    path = data_path(args)
    data = load_data(path, feature_column(args))
    root = args.output_dir / args.dataset
    scaffold_result: dict[str, object] | None = None
    if args.mode in {"scaffold", "all"}:
        scaffold_result = run_scaffold(args, data, root / "scaffold")
    if args.mode in {"kfold", "all"}:
        if scaffold_result is None:
            summary_path = root / "scaffold" / "scaffold_summary.json"
            if not summary_path.is_file():
                raise FileNotFoundError(
                    f"Missing scaffold summary {summary_path}; run --mode scaffold first"
                )
            scaffold_result = json.loads(summary_path.read_text())
        run_kfold(args, data, scaffold_result["selected_params"], root / "kfold")


if __name__ == "__main__":
    main()
