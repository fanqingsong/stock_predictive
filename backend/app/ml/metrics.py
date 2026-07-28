"""Evaluation metrics for up/down classification."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _binary_pred(y_prob: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_prob, dtype=float) >= threshold).astype(float)


def _finite_or_none(value: float, scale: float = 1.0) -> Optional[float]:
    """Convert non-finite floats to None so metrics are JSON/JSONB-safe."""
    if value is None:
        return None
    num = float(value)
    if not np.isfinite(num):
        return None
    return round(num * scale, 4)


def direction_accuracy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Optional[float]:
    """Classification accuracy for up(1)/down(0) labels."""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = _binary_pred(y_prob, threshold)
    if len(y_true) == 0:
        return None
    return float(np.mean(y_true == y_hat) * 100.0)


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    y_hat = _binary_pred(y_prob, threshold)

    if len(y_true) == 0:
        return {
            "direction_acc": None,
            "precision_up": None,
            "recall_up": None,
            "f1_up": None,
            "avg_prob_up": None,
            "n_samples": 0,
            "n_up": 0,
            "n_down": 0,
        }

    tp = float(np.sum((y_hat == 1) & (y_true == 1)))
    fp = float(np.sum((y_hat == 1) & (y_true == 0)))
    fn = float(np.sum((y_hat == 0) & (y_true == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    if np.isnan(precision) or np.isnan(recall) or (precision + recall) == 0:
        f1 = float("nan")
    else:
        f1 = 2 * precision * recall / (precision + recall)

    avg_prob = float(np.mean(y_prob)) if len(y_prob) else float("nan")

    return {
        "direction_acc": _finite_or_none(direction_accuracy(y_true, y_prob, threshold)),
        "precision_up": _finite_or_none(precision, scale=100.0),
        "recall_up": _finite_or_none(recall, scale=100.0),
        "f1_up": _finite_or_none(f1, scale=100.0),
        "avg_prob_up": _finite_or_none(avg_prob),
        "n_samples": int(len(y_true)),
        "n_up": int(np.sum(y_true == 1)),
        "n_down": int(np.sum(y_true == 0)),
    }
