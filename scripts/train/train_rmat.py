#!/usr/bin/env python3
"""Fine-tune R-MAT from the repository's pre-featurized pickle datasets."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import pickle
import random
import sys
from pathlib import Path
from typing import Any, Sequence

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "pickle_dataloaders"
HUGGINGMOLECULES_SRC = REPO_ROOT / "huggingmolecules" / "src"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def source_file_assignment(dataset: str, split_mode: str, fold: int | None) -> dict[str, list[Path]]:
    dataset_dir = DATA_ROOT / dataset
    train_files = [dataset_dir / f"train_{index}.p" for index in range(10)]
    original_val = dataset_dir / "val.p"
    original_test = dataset_dir / "test.p"

    if split_mode == "scaffold":
        assignment = {
            "train": train_files,
            "validation": [original_val],
            "test": [original_test],
        }
    elif split_mode == "kfold":
        if fold is None or not 0 <= fold <= 9:
            raise ValueError("--fold must be an integer from 0 through 9 for kfold mode")
        validation_fold = (fold + 1) % 10
        assignment = {
            "train": [
                path
                for index, path in enumerate(train_files)
                if index not in {fold, validation_fold}
            ]
            + [original_val, original_test],
            "validation": [train_files[validation_fold]],
            "test": [train_files[fold]],
        }
    else:
        raise ValueError(f"Unsupported split mode: {split_mode}")

    assigned = {name: {path.resolve() for path in paths} for name, paths in assignment.items()}
    if split_mode == "scaffold":
        expected = {
            "train": {path.resolve() for path in train_files},
            "validation": {original_val.resolve()},
            "test": {original_test.resolve()},
        }
    else:
        validation_fold = (fold + 1) % 10
        expected = {
            "train": {
                path.resolve()
                for index, path in enumerate(train_files)
                if index not in {fold, validation_fold}
            } | {original_val.resolve(), original_test.resolve()},
            "validation": {train_files[validation_fold].resolve()},
            "test": {train_files[fold].resolve()},
        }
    if assigned != expected:
        raise RuntimeError(f"Unexpected exact R-MAT split assignment: {assignment}")
    if assigned["train"] & assigned["validation"] or assigned["train"] & assigned["test"]:
        raise RuntimeError(f"Overlapping R-MAT split assignment: {assignment}")
    if assigned["validation"] & assigned["test"]:
        raise RuntimeError(f"Validation/test overlap in R-MAT split assignment: {assignment}")

    missing = [path for paths in assignment.values() for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing source pickle(s): " + ", ".join(map(str, missing)))
    return assignment


def load_pickle(path: Path) -> list[Any]:
    LOGGER.info("Loading %s", path)
    with path.open("rb") as handle:
        value = pickle.load(handle)
    if not isinstance(value, list):
        raise TypeError(f"Expected {path} to contain a list, got {type(value).__name__}")
    if value and not hasattr(value[0], "y"):
        raise TypeError(f"Objects in {path} do not expose the expected y attribute")
    return value


def load_concat_dataset(paths: Sequence[Path]) -> tuple[Any, list[list[Any]]]:
    # Retain the component lists directly. ConcatDataset stores references and does
    # not allocate a second, multi-gigabyte list containing the same encodings.
    components = [load_pickle(path) for path in paths]
    if not components or any(len(component) == 0 for component in components):
        raise ValueError("Every assigned pickle must contain at least one encoding")
    return ConcatDataset(components), components


def label_counts(dataset: Any) -> dict[str, int]:
    positive = 0
    for item in dataset:
        label = float(np.asarray(item.y).reshape(-1)[0])
        if label not in (0.0, 1.0):
            raise ValueError(f"Expected binary labels 0/1, got {label}")
        positive += int(label)
    samples = len(dataset)
    return {"samples": samples, "positive": positive, "negative": samples - positive}


def evaluate(model, loader, device: torch.device, loss_fn) -> tuple[float, dict[str, Any]]:
    model.eval()
    logits: list[float] = []
    labels: list[float] = []
    loss_sum = 0.0
    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_logits = model(batch).reshape(-1)
            batch_labels = batch.y.float().reshape(-1)
            batch_size = batch_labels.numel()
            loss_sum += float(loss_fn(batch_logits, batch_labels).item()) * batch_size
            sample_count += batch_size
            logits.extend(batch_logits.detach().cpu().tolist())
            labels.extend(batch_labels.detach().cpu().tolist())

    labels_array = np.asarray(labels, dtype=int)
    probabilities = torch.sigmoid(torch.tensor(logits, dtype=torch.float32)).numpy()
    predictions = (probabilities >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels_array, predictions, labels=[0, 1]).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    metrics: dict[str, Any] = {
        "roc_auc": float(roc_auc_score(labels_array, probabilities)),
        "pr_auc": float(average_precision_score(labels_array, probabilities)),
        "accuracy": float(accuracy_score(labels_array, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels_array, predictions)),
        "precision": float(precision_score(labels_array, predictions, zero_division=0)),
        "recall": float(recall_score(labels_array, predictions, zero_division=0)),
        "f1": float(f1_score(labels_array, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels_array, predictions)),
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return loss_sum / sample_count, metrics


def binary_focal_loss(logits, targets, pos_weight, gamma: float):
    """Return mean weighted focal loss, computed stably from logits.

    For each sample, CE is the unweighted binary cross-entropy, p_t=exp(-CE),
    and WBCE is binary cross-entropy with the positive term multiplied by
    pos_weight. The loss is mean((1-p_t)**gamma * WBCE).
    """
    unweighted_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none"
    )
    weighted_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=pos_weight
    )
    probability_of_target = torch.exp(-unweighted_bce)
    return ((1.0 - probability_of_target).pow(gamma) * weighted_bce).mean()


def selection_improved(current: float, best: float, criterion: str) -> bool:
    """Return improvement direction for the selected early-stopping criterion."""
    if criterion == "val_loss":
        return current < best
    if criterion in {"val_roc_auc", "val_pr_auc"}:
        return current > best
    raise ValueError(f"Unsupported selection criterion: {criterion}")


def lr_tag(learning_rate: float) -> str:
    value = format(learning_rate, ".12g")
    return value.replace("e-0", "e-").replace("e+0", "e+")


def run_directory(args: argparse.Namespace) -> Path:
    path = args.output_root / args.dataset / args.split_mode
    if args.split_mode == "kfold":
        path /= f"fold_{args.fold}"
    return (
        path
        / args.loss
        / f"lr_{lr_tag(args.lr)}"
        / f"select_{args.selection_criterion}"
        / f"seed_{args.seed}"
    )


def ensure_collision_safe(path: Path) -> None:
    if path.exists():
        existing = list(path.iterdir()) if path.is_dir() else [path]
        if existing:
            names = ", ".join(sorted(item.name for item in existing))
            raise FileExistsError(f"Refusing to overwrite non-empty run path {path}: {names}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def checkpoint_payload(
    model,
    epoch: int,
    args: argparse.Namespace,
    validation_loss: float,
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    cli_arguments = {
        key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()
    }
    return {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "learning_rate": args.lr,
        "loss": args.loss,
        "selection_criterion": args.selection_criterion,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "seed": args.seed,
        "validation_loss": validation_loss,
        "validation_metrics": validation_metrics,
        "args": cli_arguments,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mcf10a", "sw480", "sw480_clean"], required=True)
    parser.add_argument("--split-mode", choices=["scaffold", "kfold"], required=True)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--loss", choices=["bce", "focal"], default="bce")
    parser.add_argument(
        "--selection-criterion",
        choices=["val_loss", "val_roc_auc", "val_pr_auc"],
        default="val_pr_auc",
        help="Validation criterion used for both checkpointing and early stopping.",
    )
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rmat_pickle"))
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Stop after training and checkpoint selection without loading test.p.",
    )
    args = parser.parse_args()
    if args.split_mode == "kfold" and (args.fold is None or not 0 <= args.fold <= 9):
        parser.error("--fold must be an integer from 0 through 9 for kfold mode")
    if args.split_mode == "scaffold" and args.fold is not None:
        parser.error("--fold is only valid with --split-mode kfold")
    if args.lr <= 0 or args.epochs <= 0 or args.batch_size <= 0 or args.patience <= 0:
        parser.error("--lr, --epochs, --batch-size, and --patience must be positive")
    if args.focal_gamma < 0:
        parser.error("--focal-gamma must be non-negative")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    assignment = source_file_assignment(args.dataset, args.split_mode, args.fold)
    output_dir = run_directory(args)
    ensure_collision_safe(output_dir)

    global np, torch, ConcatDataset
    global accuracy_score, average_precision_score, balanced_accuracy_score
    global confusion_matrix, f1_score, matthews_corrcoef
    global precision_score, recall_score, roc_auc_score
    import numpy as np
    import torch
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from torch.utils.data import ConcatDataset

    # Force the repository checkout to win over any installed package or the
    # checkout's outer namespace directory.
    sys.path.insert(0, str(HUGGINGMOLECULES_SRC))
    from huggingmolecules import RMatFeaturizer, RMatModel

    set_seed(args.seed)
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    LOGGER.info("Using device %s", device)

    train_dataset, train_components = load_concat_dataset(assignment["train"])
    validation_dataset, validation_components = load_concat_dataset(assignment["validation"])
    counts = {
        "train": label_counts(train_dataset),
        "validation": label_counts(validation_dataset),
    }
    if counts["train"]["positive"] == 0 or counts["train"]["negative"] == 0:
        raise ValueError("Training data must contain both classes")
    for split_name in ("train", "validation", "test"):
        LOGGER.info(
            "%s source file(s): %s",
            split_name,
            ", ".join(str(path) for path in assignment[split_name]),
        )
    LOGGER.info("loaded train rows: %d", counts["train"]["samples"])
    LOGGER.info("loaded validation rows: %d", counts["validation"]["samples"])
    LOGGER.info("test rows are not loaded until the selected checkpoint is final")

    # Atomically reserve this run's namespace after the initial collision check.
    # A concurrent identical invocation therefore fails instead of overwriting.
    output_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "learning_rate": args.lr,
        "loss": args.loss,
        "selection_criterion": args.selection_criterion,
        "focal_gamma": args.focal_gamma if args.loss == "focal" else None,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "seed": args.seed,
        "output_directory": str(output_dir.resolve()),
        "huggingmolecules_source": str(HUGGINGMOLECULES_SRC),
        "source_pickles": {
            split: [str(path.resolve()) for path in paths] for split, paths in assignment.items()
        },
        "loaded_counts": counts,
    }
    write_json(output_dir / "run_config.json", config)

    featurizer = RMatFeaturizer.from_pretrained("rmat_4M")
    train_loader = featurizer.get_data_loader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    validation_loader = featurizer.get_data_loader(
        validation_dataset, batch_size=args.batch_size, shuffle=False
    )
    model = RMatModel.from_pretrained("rmat_4M").to(device)
    model_device = next(model.parameters()).device
    LOGGER.info("R-MAT model parameter device: %s", model_device)
    if model_device != device:
        raise RuntimeError(f"R-MAT model is on {model_device}, expected {device}")
    pos_weight = counts["train"]["negative"] / counts["train"]["positive"]
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    evaluation_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    if args.loss == "bce":
        training_loss_fn = evaluation_loss_fn
    else:
        training_loss_fn = lambda logits, targets: binary_focal_loss(
            logits, targets, pos_weight_tensor, args.focal_gamma
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    history: list[dict[str, Any]] = []
    minimize = args.selection_criterion == "val_loss"
    best_selected_value = np.inf if minimize else -np.inf
    best_selected_epoch: int | None = None
    checkpoint_stem = {
        "val_loss": "val_loss",
        "val_roc_auc": "roc_auc",
        "val_pr_auc": "pr_auc",
    }[args.selection_criterion]
    selected_checkpoint_path = output_dir / f"best_{checkpoint_stem}.pt"
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_samples = 0
        for batch in train_loader:
            batch = batch.to(device)
            logits = model(batch).reshape(-1)
            labels = batch.y.float().reshape(-1)
            loss = training_loss_fn(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.item()) * labels.numel()
            train_samples += labels.numel()

        validation_loss, validation_metrics = evaluate(
            model, validation_loader, device, evaluation_loss_fn
        )
        criterion_values = {
            "val_loss": validation_loss,
            "val_roc_auc": validation_metrics["roc_auc"],
            "val_pr_auc": validation_metrics["pr_auc"],
        }
        selected_value = float(criterion_values[args.selection_criterion])
        improved = selection_improved(
            selected_value, best_selected_value, args.selection_criterion
        )
        if improved:
            best_selected_value = selected_value
            best_selected_epoch = epoch
            stale_epochs = 0
            torch.save(
                checkpoint_payload(model, epoch, args, validation_loss, validation_metrics),
                selected_checkpoint_path,
            )
        else:
            stale_epochs += 1

        config["selected_checkpoint"] = str(selected_checkpoint_path.resolve())
        config["best_selected_value"] = best_selected_value
        config["best_selected_epoch"] = best_selected_epoch
        write_json(output_dir / "run_config.json", config)
        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_samples,
            "validation_loss": validation_loss,
            "validation_metrics": validation_metrics,
            "selection_criterion": args.selection_criterion,
            "selected_value": selected_value,
            "selected_checkpoint_improved": improved,
            "best_selected_value": best_selected_value,
            "best_selected_epoch": best_selected_epoch,
            "stale_epochs": stale_epochs,
        }
        history.append(epoch_record)
        write_json(output_dir / "history.json", history)

        LOGGER.info(
            "Epoch %d train_loss=%.6f val_loss=%.6f ROC-AUC=%.6f PR-AUC=%.6f "
            "selected_by=%s improved=%s stale=%d/%d",
            epoch,
            epoch_record["train_loss"],
            validation_loss,
            validation_metrics["roc_auc"],
            validation_metrics["pr_auc"],
            args.selection_criterion,
            improved,
            stale_epochs,
            args.patience,
        )
        if stale_epochs >= args.patience:
            LOGGER.info(
                "Early stopping: %s did not improve for %d epochs",
                args.selection_criterion,
                stale_epochs,
            )
            break

    if args.skip_test_evaluation:
        LOGGER.info(
            "Finished training without test evaluation. Outputs are in %s", output_dir
        )
        return

    # Release the typically much larger training split before loading the test
    # pickle. The test set has not been loaded or evaluated before this point.
    del train_loader, validation_loader, train_dataset, validation_dataset
    del train_components, validation_components
    gc.collect()

    test_dataset, test_components = load_concat_dataset(assignment["test"])
    test_counts = label_counts(test_dataset)
    LOGGER.info("loaded test rows: %d", test_counts["samples"])
    config["loaded_counts"]["test"] = test_counts
    write_json(output_dir / "run_config.json", config)
    test_loader = featurizer.get_data_loader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    checkpoint = torch.load(selected_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_metrics = evaluate(model, test_loader, device, evaluation_loss_fn)
    selected_by = {
        "val_loss": "validation_loss",
        "val_roc_auc": "validation_roc_auc",
        "val_pr_auc": "validation_pr_auc",
    }[args.selection_criterion]
    LOGGER.info(
        "Test selected_by=%s epoch=%d rows=%d ROC-AUC=%.6f PR-AUC=%.6f",
        selected_by,
        checkpoint["epoch"],
        test_counts["samples"],
        test_metrics["roc_auc"],
        test_metrics["pr_auc"],
    )
    result = {
        "dataset": args.dataset,
        "split_mode": args.split_mode,
        "fold": args.fold,
        "selected_by": selected_by,
        "checkpoint": str(selected_checkpoint_path.resolve()),
        "checkpoint_epoch": checkpoint["epoch"],
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "test_counts": test_counts,
    }
    write_json(output_dir / "test_selected_checkpoint.json", result)
    write_json(output_dir / f"test_best_{checkpoint_stem}.json", result)

    del test_loader, test_dataset, test_components
    LOGGER.info("Finished. Outputs are in %s", output_dir)


if __name__ == "__main__":
    main()
