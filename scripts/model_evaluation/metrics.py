"""Shared binary-classification and virtual-screening metrics."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def screening_metrics(
    y_true: Sequence[int],
    y_score: Sequence[float],
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Compute the paper's threshold and ranking metrics.

    PR-AUC is sklearn average precision. EF cutoffs use ceil(N*fraction), and
    ties are resolved by stable descending score order.
    """
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("Labels and scores must be non-empty vectors of equal length")
    if not set(np.unique(labels)) <= {0, 1}:
        raise ValueError("Labels must be binary 0/1")
    if len(np.unique(labels)) != 2:
        raise ValueError("ROC-AUC requires both classes")
    if not np.isfinite(scores).all():
        raise ValueError("Scores must all be finite")

    predictions = (scores >= threshold).astype(int)
    prevalence = float(labels.mean())
    pr_auc = float(average_precision_score(labels, scores))
    result: dict[str, float | int] = {
        "rows": int(labels.size),
        "positive": int(labels.sum()),
        "negative": int((labels == 0).sum()),
        "prevalence": prevalence,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "pr_auc": pr_auc,
        "pr_auc_enrichment": pr_auc / prevalence if prevalence else 0.0,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "threshold": float(threshold),
    }
    order = np.argsort(-scores, kind="stable")
    for percent in (1, 5):
        count = max(1, math.ceil(labels.size * percent / 100))
        top_precision = float(labels[order[:count]].mean())
        result[f"precision_at_{percent}pct"] = top_precision
        result[f"ef_at_{percent}pct"] = top_precision / prevalence if prevalence else 0.0
        result[f"cutoff_at_{percent}pct"] = count
    return result
