"""Metric helpers for binary deepfake detection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        confusion_matrix=confusion_matrix(y_true, y_pred).astype(int).tolist(),
    )
