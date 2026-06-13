"""Probability aggregation helpers for whole-recording inference."""

from __future__ import annotations

import math

import numpy as np


AGGREGATION_MODES = (
    "mean",
    "log_mean",
    "median",
    "confidence_weighted",
    "top_confidence_mean",
    "vote",
)


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize class scores into a probability vector."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional array")

    total = float(np.sum(scores))
    if not np.isfinite(total) or total <= 0:
        return np.full(scores.shape, 1.0 / len(scores), dtype=np.float64)
    return scores / total


def aggregate_probabilities(
    probabilities: np.ndarray,
    mode: str = "mean",
    top_fraction: float = 0.25,
    epsilon: float = 1e-8,
) -> np.ndarray:
    """Aggregate per-window probabilities into one probability vector.

    ``mean`` preserves the historical behavior. The other modes are useful
    when many windows are idle/noisy and can dilute the whole-file prediction.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape (windows, classes)")
    if probabilities.shape[0] == 0 or probabilities.shape[1] == 0:
        raise ValueError("probabilities must include at least one window and class")
    if mode not in AGGREGATION_MODES:
        expected = ", ".join(AGGREGATION_MODES)
        raise ValueError(f"Unknown aggregation mode: {mode}. Expected one of: {expected}")

    if mode == "mean":
        return _normalize_scores(np.mean(probabilities, axis=0))

    if mode == "log_mean":
        scores = np.exp(np.mean(np.log(np.clip(probabilities, epsilon, 1.0)), axis=0))
        return _normalize_scores(scores)

    if mode == "median":
        return _normalize_scores(np.median(probabilities, axis=0))

    if mode == "confidence_weighted":
        weights = np.max(probabilities, axis=1)
        if float(np.sum(weights)) <= 0:
            return _normalize_scores(np.mean(probabilities, axis=0))
        return _normalize_scores(np.average(probabilities, axis=0, weights=weights))

    if mode == "top_confidence_mean":
        if not 0 < top_fraction <= 1:
            raise ValueError("top_fraction must be in the range (0, 1]")
        confidences = np.max(probabilities, axis=1)
        keep = max(1, int(math.ceil(len(confidences) * top_fraction)))
        indices = np.argpartition(confidences, -keep)[-keep:]
        return _normalize_scores(np.mean(probabilities[indices], axis=0))

    predicted_classes = np.argmax(probabilities, axis=1)
    votes = np.bincount(predicted_classes, minlength=probabilities.shape[1])
    return _normalize_scores(votes)
