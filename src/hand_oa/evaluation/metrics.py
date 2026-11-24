"""Evaluation metrics for 5-class KL grading and binary OA detection."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    num_classes: int = 5,
    oa_threshold_kl: int = 3,
) -> dict[str, Any]:
    """Compute per-class KL F1 scores and binary OA metrics (KL >= threshold vs healthy)."""
    labels = list(range(num_classes))
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    binary_true = (y_true >= oa_threshold_kl).astype(int)
    binary_pred = (y_pred >= oa_threshold_kl).astype(int)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class_f1": {f"KL{idx}": float(per_class_f1[idx]) for idx in labels},
        "binary_oa_sensitivity": float(recall_score(binary_true, binary_pred, zero_division=0)),
        "binary_oa_precision": float(precision_score(binary_true, binary_pred, zero_division=0)),
        "binary_oa_f1": float(f1_score(binary_true, binary_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }

    if y_prob is not None and len(np.unique(binary_true)) == 2:
        oa_prob = y_prob[:, oa_threshold_kl:].sum(axis=1)
        metrics["binary_oa_roc_auc"] = float(roc_auc_score(binary_true, oa_prob))
    else:
        metrics["binary_oa_roc_auc"] = None

    return metrics
