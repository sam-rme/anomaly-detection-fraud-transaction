from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot precision-recall curve for a single model.

    Args:
        y_true: Ground truth labels.
        scores: Anomaly scores.
        model_name: Used for the plot title and legend.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    precision, recall, _ = precision_recall_curve(y_true, scores)
    auc = average_precision_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, lw=2, label=f"{model_name} (PR-AUC={auc:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


def plot_score_distribution(
    scores_normal: np.ndarray,
    scores_fraud: np.ndarray,
    model_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot anomaly score distributions for normal vs fraud samples.

    Args:
        scores_normal: Scores for normal transactions.
        scores_fraud: Scores for fraud transactions.
        model_name: Used for the plot title.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scores_normal, bins=80, alpha=0.6, label="Normal", density=True, color="steelblue")
    ax.hist(scores_fraud, bins=80, alpha=0.6, label="Fraud", density=True, color="tomato")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(f"Score Distribution — {model_name}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


def plot_pr_curves_comparison(
    results: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path | None = None,
) -> plt.Figure:
    """Overlay PR curves for all models on a single figure.

    Args:
        results: Dict mapping model name to (y_true, scores).
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for model_name, (y_true, scores) in results.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        auc = average_precision_score(y_true, scores)
        ax.plot(recall, precision, lw=2, label=f"{model_name} ({auc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


def plot_imbalance_robustness(
    results: dict[str, list[float]],
    fraud_fractions: list[float],
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot PR-AUC vs fraud fraction for each model (Experiment 2).

    Args:
        results: Dict mapping model name to list of PR-AUC values (one per fraction).
        fraud_fractions: x-axis values matching the order in results lists.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    for model_name, pr_aucs in results.items():
        valid = [(f, v) for f, v in zip(fraud_fractions, pr_aucs) if not np.isnan(v)]
        if valid:
            fracs, vals = zip(*valid)
            ax.plot(fracs, vals, marker="o", lw=2, label=model_name)

    ax.set_xlabel("Fraud fraction in training set")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Imbalance Robustness — PR-AUC vs Fraud Fraction")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig


def plot_training_losses(
    train_losses: list[float],
    model_name: str,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot training losses over epochs.

    Args:
        train_losses: List of training losses per epoch.
        model_name: Used for the plot title.
        output_path: If provided, save the figure to this path.

    Returns:
        Matplotlib Figure.
    """
    epochs = [i for i in range(1,len(train_losses)+1)]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs,train_losses, color ='r')
    ax.set_xlabel('epochs')
    ax.set_ylabel('Train loss')
    ax.set_title('Training losses trough epochs')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if output_path is not None :
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    return fig
