"""Generate report figures from saved experiment CSVs.

Reads outputs/exp1_summary.csv and outputs/exp2_imbalance.csv and produces
publication-quality figures saved to outputs/figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import get_logger, load_config


METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "roc_auc": "ROC-AUC",
    "f1": "F1",
    "precision_at_100": "Precision@100",
    "recall_at_precision_09": "Recall@P=0.9",
}

# Stable colour per family — supervised, classical unsupervised, deep
FAMILY_COLOR = {
    "logistic_regression": "#1f77b4",
    "xgboost": "#1f77b4",
    "isolation_forest": "#2ca02c",
    "lof": "#2ca02c",
    "one_class_svm": "#2ca02c",
    "autoencoder": "#d62728",
    "vae": "#d62728",
    "deep_svdd": "#d62728",
}

MODEL_ORDER = [
    "logistic_regression", "xgboost",
    "isolation_forest", "lof", "one_class_svm",
    "autoencoder", "vae", "deep_svdd",
]


def _agg_exp1(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-seed results into mean ± std per (model, metric)."""
    metrics = list(METRIC_LABELS.keys())
    agg = df.groupby("model")[metrics].agg(["mean", "std"]).round(4)
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    return agg.reindex([m for m in MODEL_ORDER if m in agg.index])


def plot_exp1_pr_auc(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart of PR-AUC per model with std error bars, ordered as MODEL_ORDER."""
    agg = df.groupby("model")["pr_auc"].agg(["mean", "std"])
    agg = agg.reindex([m for m in MODEL_ORDER if m in agg.index])

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agg))
    colors = [FAMILY_COLOR[m] for m in agg.index]
    ax.bar(x, agg["mean"], yerr=agg["std"], color=colors, edgecolor="black",
           capsize=4, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(agg.index, rotation=20, ha="right")
    ax.set_ylabel("PR-AUC (mean ± std over 5 seeds)")
    ax.set_title("Experiment 1 — PR-AUC per model")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)

    # Family legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4", label="Supervised"),
        plt.Rectangle((0, 0), 1, 1, color="#2ca02c", label="Classical unsupervised"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Deep unsupervised"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exp1_metrics_grid(df: pd.DataFrame, out_path: Path) -> None:
    """One subplot per metric, bars per model with std error bars."""
    metrics = list(METRIC_LABELS.keys())
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        agg = df.groupby("model")[metric].agg(["mean", "std"])
        agg = agg.reindex([m for m in MODEL_ORDER if m in agg.index])
        x = np.arange(len(agg))
        colors = [FAMILY_COLOR[m] for m in agg.index]
        ax.bar(x, agg["mean"], yerr=agg["std"], color=colors,
               edgecolor="black", capsize=3, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(agg.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(METRIC_LABELS[metric])
        ax.grid(axis="y", alpha=0.3)

    axes[-1].axis("off")
    fig.suptitle("Experiment 1 — all metrics, mean ± std over 5 seeds", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_imbalance(df: pd.DataFrame, out_path: Path) -> None:
    """PR-AUC vs fraud_fraction, one line per model."""
    fig, ax = plt.subplots(figsize=(9, 5))

    for model in MODEL_ORDER:
        sub = df[df["model"] == model].sort_values("fraud_fraction")
        if sub.empty:
            continue
        valid = sub.dropna(subset=["pr_auc_mean"])
        if valid.empty:
            continue
        ax.errorbar(
            valid["fraud_fraction"], valid["pr_auc_mean"],
            yerr=valid["pr_auc_std"],
            marker="o", lw=2, capsize=3,
            color=FAMILY_COLOR[model], label=model, alpha=0.9,
        )

    ax.set_xlabel("Fraction of fraud labels available at training time")
    ax.set_ylabel("PR-AUC (mean ± std)")
    ax.set_title("Experiment 2 — Imbalance robustness")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_table(df: pd.DataFrame, out_path: Path) -> None:
    """Write a clean mean ± std table per model and metric to CSV."""
    agg = _agg_exp1(df)
    out = pd.DataFrame(index=agg.index)
    for metric, label in METRIC_LABELS.items():
        out[label] = agg.apply(
            lambda row, m=metric: f"{row[f'{m}_mean']:.3f} ± {row[f'{m}_std']:.3f}",
            axis=1,
        )
    out.to_csv(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate report figures.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output-dir", default=None,
                        help="Override outputs path (defaults to cfg['paths']['outputs']/figures).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger(__name__)

    outputs_dir = Path(args.output_dir) if args.output_dir else Path(cfg["paths"]["outputs"])
    figures_dir = outputs_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    exp1_path = outputs_dir / "exp1_summary.csv"
    exp2_path = outputs_dir / "exp2_imbalance.csv"

    if exp1_path.exists():
        logger.info("Reading %s", exp1_path)
        exp1 = pd.read_csv(exp1_path)
        plot_exp1_pr_auc(exp1, figures_dir / "exp1_pr_auc.png")
        plot_exp1_metrics_grid(exp1, figures_dir / "exp1_metrics_grid.png")
        write_summary_table(exp1, figures_dir / "exp1_summary_table.csv")
        logger.info("Exp1 figures + table saved to %s", figures_dir)
    else:
        logger.warning("Skipping Exp1 — %s not found.", exp1_path)

    if exp2_path.exists():
        logger.info("Reading %s", exp2_path)
        exp2 = pd.read_csv(exp2_path)
        plot_exp2_imbalance(exp2, figures_dir / "exp2_imbalance.png")
        logger.info("Exp2 figure saved to %s", figures_dir)
    else:
        logger.warning("Skipping Exp2 — %s not found.", exp2_path)


if __name__ == "__main__":
    main()
