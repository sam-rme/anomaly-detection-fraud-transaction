"""Main entry point for running all experiments."""
from __future__ import annotations

import os

# Must be set before any numpy/sklearn/xgboost import to prevent OpenBLAS/OpenMP
# thread-pool conflict that causes a segfault on macOS Apple Silicon.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import datetime
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow

from src.data import load_raw, make_splits, preprocess, subsample_fraud
from src.evaluation import compute_metrics, find_best_threshold
from src.models import (
    AutoencoderModel,
    DeepSVDDModel,
    IsolationForestModel,
    LOFModel,
    LogisticRegressionModel,
    OneClassSVMModel,
    VAEModel,
    XGBoostModel,
)
from src.plots import plot_pr_curve, plot_score_distribution, plot_training_losses
from src.utils import get_logger, load_config, set_seed


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "logistic_regression": LogisticRegressionModel,
    "xgboost": XGBoostModel,
    "isolation_forest": IsolationForestModel,
    "lof": LOFModel,
    "one_class_svm": OneClassSVMModel,
    "autoencoder": AutoencoderModel,
    "vae": VAEModel,
    "deep_svdd": DeepSVDDModel,
}

SUPERVISED_MODELS = {"logistic_regression", "xgboost"}
DEEP_MODELS = {"autoencoder", "vae", "deep_svdd"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_model(name: str, params: dict, seed: int) -> Any:
    """Instantiate a model from the registry; LOF receives no seed."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    if name == "lof":
        return MODEL_REGISTRY[name](**params)
    return MODEL_REGISTRY[name](**params, seed=seed)


def prepare_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Supervised models get the full set; unsupervised get only normal samples."""
    if model_name in SUPERVISED_MODELS:
        return X_train, y_train
    return X_train[y_train == 0], None


def setup_mlflow(cfg: dict) -> None:
    """Configure MLflow tracking URI (SQLite by default) and experiment name."""
    mlflow_cfg = cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get(
        "tracking_uri",
        f"sqlite:///{Path(cfg['paths']['mlruns']).absolute()}/mlflow.db",
    )
    Path(cfg["paths"]["mlruns"]).mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(mlflow_cfg["experiment_name"])


def _log_to_mongo(
    run_id: str,
    model_name: str,
    seed: int,
    experiment_tag: str,
    params: dict,
    metrics: dict[str, float],
    cfg: dict,
) -> None:
    """Insert a run summary document into MongoDB if a URI is configured.

    Fails silently so MongoDB being unavailable never breaks an experiment.
    """
    mongo_uri = cfg.get("mlflow", {}).get("mongodb_uri")
    if not mongo_uri:
        return
    try:
        from pymongo import MongoClient

        client = MongoClient(mongo_uri, serverSelectionTimeoutMS=3000)
        db_name = cfg.get("mlflow", {}).get("mongodb_db", "fraud_detection")
        db = client[db_name]
        db["runs"].insert_one(
            {
                "run_id": run_id,
                "model": model_name,
                "seed": seed,
                "experiment": experiment_tag,
                "params": params,
                "metrics": metrics,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
        )
    except Exception as exc:
        get_logger(__name__).warning("MongoDB logging skipped: %s", exc)


def _fit_and_score(
    model_name: str,
    seed: int,
    params: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, Any]:
    """Train a model and return (metrics, score_val, score_test, model).

    Threshold is chosen to maximise F1 on the validation set.
    The model is returned so callers can inspect attributes like train_losses_.
    """
    set_seed(seed)
    model = build_model(model_name, params, seed)
    X_fit, y_fit = prepare_training_data(X_train, y_train, model_name)

    if model_name in SUPERVISED_MODELS:
        model.fit(X_fit, y_fit, X_val, y_val)
    else:
        model.fit(X_fit)

    score_val = model.score_samples(X_val)
    score_test = model.score_samples(X_test)
    threshold = find_best_threshold(y_val, score_val)
    metrics = compute_metrics(y_test, score_test, threshold=threshold)
    return metrics, score_val, score_test, model


# ---------------------------------------------------------------------------
# Core loop — one (model, seed) pair
# ---------------------------------------------------------------------------


def single_run(
    model_name: str,
    seed: int,
    models_cfg: dict,
    splits: tuple,
    cfg: dict,
    experiment_tag: str = "exp1_baseline",
) -> dict[str, float]:
    """Train one model with one seed, log to MLflow + MongoDB, return test metrics.

    Args:
        model_name: Key in MODEL_REGISTRY.
        seed: Random seed.
        models_cfg: Per-model hyperparams from models.yaml.
        splits: (X_train, X_val, X_test, y_train, y_val, y_test).
        cfg: Global config dict.
        experiment_tag: Tag to group runs (e.g. 'exp1_baseline').

    Returns:
        Dict of metrics on the test set.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    params = models_cfg[model_name]

    with mlflow.start_run(run_name=f"{model_name}_seed{seed}") as run:
        mlflow.set_tags(
            {
                "model": model_name,
                "experiment": experiment_tag,
                "seed": str(seed),
                "family": "supervised" if model_name in SUPERVISED_MODELS else "unsupervised",
            }
        )
        mlflow.log_params({**params, "seed": seed})

        metrics, _score_val, score_test, model = _fit_and_score(
            model_name, seed, params,
            X_train, y_train, X_val, y_val, X_test, y_test,
        )
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            pr_path = tmp / f"pr_curve_{model_name}.png"
            fig = plot_pr_curve(y_test, score_test, model_name, output_path=pr_path)
            mlflow.log_artifact(str(pr_path))
            plt.close(fig)

            dist_path = tmp / f"score_dist_{model_name}.png"
            fig2 = plot_score_distribution(
                score_test[y_test == 0],
                score_test[y_test == 1],
                model_name,
                output_path=dist_path,
            )
            mlflow.log_artifact(str(dist_path))
            plt.close(fig2)

            losses = getattr(model, "train_losses", [])
            if losses:
                curve_path = tmp / f"training_curve_{model_name}.png"
                fig3 = plot_training_losses(losses, model_name, output_path=curve_path)
                mlflow.log_artifact(str(curve_path))
                plt.close(fig3)

        _log_to_mongo(
            run_id=run.info.run_id,
            model_name=model_name,
            seed=seed,
            experiment_tag=experiment_tag,
            params={**params, "seed": seed},
            metrics=metrics,
            cfg=cfg,
        )

    return metrics


# ---------------------------------------------------------------------------
# Experiment 1 — Baseline comparison
# ---------------------------------------------------------------------------


def run_experiment_1(cfg: dict, models_cfg: dict, models: list[str]) -> pd.DataFrame:
    """Exp1: 5 seeds × N models on the full training set. Primary benchmark table."""
    logger = get_logger(__name__)
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)

    df = load_raw(cfg["paths"]["data_raw"])
    X, y = preprocess(df)
    results: dict[str, list[dict]] = {m: [] for m in models}

    for seed in cfg["seeds"]:
        logger.info("=== Exp1 seed %d ===", seed)
        splits = make_splits(
            X, y,
            train_frac=cfg["split"]["train"],
            val_frac=cfg["split"]["val"],
            seed=seed,
        )
        for model_name in models:
            logger.info("  → %s", model_name)
            metrics = single_run(
                model_name, seed, models_cfg, splits, cfg, experiment_tag="exp1_baseline"
            )
            results[model_name].append(metrics)

    results_df = pd.DataFrame(
        [
            {"model": model, **m}
            for model, metric_list in results.items()
            for m in metric_list
        ]
    )
    csv_path = Path(cfg["paths"]["outputs"]) / "exp1_summary.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Exp1 results saved to %s", csv_path)
    return results_df


# ---------------------------------------------------------------------------
# Experiment 2 — Imbalance robustness
# ---------------------------------------------------------------------------


def run_experiment_2(cfg: dict, models_cfg: dict, models: list[str]) -> pd.DataFrame:
    """Exp2: PR-AUC vs fraud fraction seen during training.

    Unsupervised models are unaffected by fraud fraction (they never see fraud
    during training), so they will produce flat curves — an informative result.
    Supervised models should degrade as fewer fraud examples are available.
    """
    logger = get_logger(__name__)
    Path(cfg["paths"]["outputs"]).mkdir(parents=True, exist_ok=True)
    fraud_fractions: list[float] = cfg["experiment2"]["fraud_fractions"]

    df = load_raw(cfg["paths"]["data_raw"])
    X, y = preprocess(df)

    # {model: {fraction: [pr_auc_per_seed]}}
    agg: dict[str, dict[float, list[float]]] = {
        m: {f: [] for f in fraud_fractions} for m in models
    }

    for seed in cfg["seeds"]:
        logger.info("=== Exp2 seed %d ===", seed)
        splits = make_splits(
            X, y,
            train_frac=cfg["split"]["train"],
            val_frac=cfg["split"]["val"],
            seed=seed,
        )
        X_train, X_val, X_test, y_train, y_val, y_test = splits

        for frac in fraud_fractions:
            X_tr_f, y_tr_f = subsample_fraud(X_train, y_train, frac, seed)

            for model_name in models:
                # Supervised models with no fraud labels cannot be trained
                if model_name in SUPERVISED_MODELS and (y_tr_f == 1).sum() == 0:
                    logger.warning(
                        "Skipping %s at frac=0.0 — no fraud labels in training set",
                        model_name,
                    )
                    agg[model_name][frac].append(float("nan"))
                    continue

                logger.info("  frac=%.2f  %s", frac, model_name)
                params = models_cfg[model_name]
                experiment_tag = f"exp2_imb_{frac:.2f}"

                with mlflow.start_run(
                    run_name=f"{model_name}_seed{seed}_frac{frac:.2f}"
                ) as run:
                    mlflow.set_tags(
                        {
                            "model": model_name,
                            "experiment": experiment_tag,
                            "seed": str(seed),
                            "fraud_fraction": str(frac),
                        }
                    )
                    mlflow.log_params({**params, "seed": seed, "fraud_fraction": frac})

                    metrics, _, _, _ = _fit_and_score(
                        model_name, seed, params,
                        X_tr_f, y_tr_f, X_val, y_val, X_test, y_test,
                    )
                    mlflow.log_metrics(metrics)

                    _log_to_mongo(
                        run_id=run.info.run_id,
                        model_name=model_name,
                        seed=seed,
                        experiment_tag=experiment_tag,
                        params={**params, "seed": seed, "fraud_fraction": frac},
                        metrics=metrics,
                        cfg=cfg,
                    )

                agg[model_name][frac].append(metrics["pr_auc"])

    rows = [
        {
            "model": m,
            "fraud_fraction": frac,
            "pr_auc_mean": float(np.nanmean(agg[m][frac])),
            "pr_auc_std": float(np.nanstd(agg[m][frac])),
        }
        for m in models
        for frac in fraud_fractions
    ]
    results_df = pd.DataFrame(rows)
    csv_path = Path(cfg["paths"]["outputs"]) / "exp2_imbalance.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Exp2 results saved to %s", csv_path)
    return results_df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fraud detection experiments.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument("--experiment", type=int, choices=[1, 2], default=1)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of models to run. E.g. --models isolation_forest autoencoder",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    models_cfg = load_config(args.models_config)
    setup_mlflow(cfg)

    logger = get_logger(__name__)
    models = args.models or list(MODEL_REGISTRY.keys())
    logger.info(
        "Experiment %d | models=%s | seeds=%s",
        args.experiment, models, cfg["seeds"],
    )

    if args.experiment == 1:
        run_experiment_1(cfg, models_cfg, models)
    elif args.experiment == 2:
        run_experiment_2(cfg, models_cfg, models)


if __name__ == "__main__":
    main()
