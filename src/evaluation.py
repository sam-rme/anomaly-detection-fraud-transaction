from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import chi2
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute all evaluation metrics for a model.

    Args:
        y_true: Ground truth labels (0=normal, 1=fraud).
        scores: Anomaly scores (higher = more anomalous).
        threshold: Decision threshold. If None, best-F1 threshold is used.

    Returns:
        Dict with keys: pr_auc, roc_auc, f1, precision_at_100,
        recall_at_precision_09, threshold.
    """
    pr_auc = average_precision_score(y_true, scores)
    roc_auc = roc_auc_score(y_true, scores)

    if threshold is None:
        threshold = find_best_threshold(y_true, scores)

    preds = (scores >= threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    p_at_100 = precision_at_k(y_true, scores, k=100)
    recall_at_p09 = _recall_at_precision(y_true, scores, target_precision=0.9)

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "f1": f1,
        "precision_at_100": p_at_100,
        "recall_at_precision_09": recall_at_p09,
        "threshold": threshold,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for a scalar metric.

    Args:
        y_true: Ground truth labels.
        scores: Anomaly scores.
        metric_fn: Function (y_true, scores) -> float.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Significance level (0.05 -> 95% CI).
        seed: Random seed.

    Returns:
        Tuple (lower, upper) confidence bounds.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0:
            continue
        values.append(metric_fn(y_true[idx], scores[idx]))
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> float:
    """McNemar's test for paired classifier comparison (with continuity correction).

    Args:
        y_true: Ground truth labels.
        preds_a: Binary predictions from model A.
        preds_b: Binary predictions from model B.

    Returns:
        p-value.
    """
    b = int(((preds_a != y_true) & (preds_b == y_true)).sum())
    c = int(((preds_a == y_true) & (preds_b != y_true)).sum())
    if b + c == 0:
        return 1.0
    stat = (abs(b - c) - 1) ** 2 / (b + c) # Yates continuity correction
    return float(1 - chi2.cdf(stat, df=1))


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Apply Benjamini-Hochberg FDR correction.

    Args:
        p_values: List of raw p-values.
        alpha: Target FDR level.

    Returns:
        List of booleans — True means reject the null hypothesis.
    """
    n = len(p_values)
    order = np.argsort(p_values)
    sorted_p = np.array(p_values)[order]
    thresholds = (np.arange(1, n + 1) / n) * alpha

    reject_sorted = sorted_p <= thresholds
    if reject_sorted.any():
        last = int(np.where(reject_sorted)[0][-1])
        reject_sorted[: last + 1] = True

    result = [False] * n
    for rank, orig_idx in enumerate(order):
        result[orig_idx] = bool(reject_sorted[rank])
    return result


def find_best_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Find the threshold that maximises F1 on the provided split.

    Args:
        y_true: Ground truth labels.
        scores: Anomaly scores.

    Returns:
        Optimal threshold value.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
    return float(thresholds[np.argmax(f1)])


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int = 100) -> float:
    """Precision among the top-k highest-scored samples.

    Args:
        y_true: Ground truth labels.
        scores: Anomaly scores.
        k: Number of top samples to consider.

    Returns:
        Precision@k value.
    """
    top_k = np.argsort(scores)[-k:]
    return float(y_true[top_k].sum() / k)


def _recall_at_precision(
    y_true: np.ndarray, scores: np.ndarray, target_precision: float = 0.9
) -> float:
    """Maximum recall achievable at or above a target precision level."""
    precision, recall, _ = precision_recall_curve(y_true, scores)
    mask = precision >= target_precision
    return float(recall[mask].max()) if mask.any() else 0.0
