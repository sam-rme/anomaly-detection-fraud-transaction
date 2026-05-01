"""Hyperparameter tuning with Optuna for unsupervised and deep models.

Optimises PR-AUC on the validation set (seed=0 split only).
Writes best params back to configs/models.yaml when done.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import yaml
from sklearn.metrics import average_precision_score

from src.data import load_raw, make_splits, preprocess
from src.models import (
    AutoencoderModel,
    DeepSVDDModel,
    IsolationForestModel,
    LOFModel,
    OneClassSVMModel,
    VAEModel,
)
from src.utils import get_logger, load_config, set_seed

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)

# Reduced epochs during tuning — full epochs run only in the final experiment
TUNE_EPOCHS = 30

# Max trials per model
N_TRIALS: dict[str, int] = {
    "isolation_forest": 60,
    "lof": 30,
    "one_class_svm": 30,
    "autoencoder": 60,
    "vae": 60,
    "deep_svdd": 60,
}

# OCSVM and LOF are O(n²) — subsample normals to keep tuning tractable
OCSVM_TUNE_SAMPLES = 10_000


# ---------------------------------------------------------------------------
# Objective functions
# ---------------------------------------------------------------------------


def _pr_auc(model: Any, X_val: np.ndarray, y_val: np.ndarray) -> float:
    scores = model.score_samples(X_val)
    return float(average_precision_score(y_val, scores))


def objective_isolation_forest(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "contamination": trial.suggest_float("contamination", 0.001, 0.05, log=True),
        "max_samples": trial.suggest_categorical("max_samples", ["auto", 0.5, 0.8]),
    }
    model = IsolationForestModel(**params, seed=0)
    model.fit(X_train_normal)
    return _pr_auc(model, X_val, y_val)


def objective_lof(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 100),
        "contamination": "auto",
        "novelty": True,
    }
    model = LOFModel(**params)
    model.fit(X_train_normal)
    return _pr_auc(model, X_val, y_val)


def objective_one_class_svm(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    params = {
        "nu": trial.suggest_float("nu", 0.01, 0.3, log=True),
        "kernel": "rbf",
        "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
    }
    # Subsample to keep each trial under ~1 min
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_train_normal), size=min(OCSVM_TUNE_SAMPLES, len(X_train_normal)), replace=False)
    model = OneClassSVMModel(**params)
    model.fit(X_train_normal[idx])
    return _pr_auc(model, X_val, y_val)


def objective_autoencoder(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    hidden_choice = trial.suggest_categorical(
        "hidden_dims",
        ["32-16", "64-32", "64-32-16", "128-64", "128-64-32"],
    )
    hidden_dims = [int(x) for x in hidden_choice.split("-")]
    params = {
        "hidden_dims": hidden_dims,
        "latent_dim": trial.suggest_categorical("latent_dim", [2, 4, 8, 12]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": TUNE_EPOCHS,
    }
    set_seed(0)
    model = AutoencoderModel(**params, seed=0)
    model.fit(X_train_normal)
    return _pr_auc(model, X_val, y_val)

def objective_vae(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    hidden_choice = trial.suggest_categorical(
        "hidden_dims",
        ["32-16", "64-32", "64-32-16", "128-64"],
    )
    hidden_dims = [int(x) for x in hidden_choice.split("-")]
    params = {
        "hidden_dims": hidden_dims,
        "latent_dim": trial.suggest_categorical("latent_dim", [2, 4, 8, 12]),
        "beta": trial.suggest_float("beta", 0.1, 5.0, log=True),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": TUNE_EPOCHS,
    }
    set_seed(0)
    model = VAEModel(**params, seed=0)
    model.fit(X_train_normal)
    return _pr_auc(model, X_val, y_val)


def objective_deep_svdd(
    trial: optuna.Trial,
    X_train_normal: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> float:
    hidden_choice = trial.suggest_categorical(
        "hidden_dims", ["32-16", "64-32", "64-32-16", "128-64-32", "128-64", "64-32-16-8"]
    )
    params = {
        "hidden_dims": [int(x) for x in hidden_choice.split("-")],
        "rep_dim": trial.suggest_categorical("rep_dim", [2, 4, 8, 16, 32]),
        "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
        "epochs": TUNE_EPOCHS,
    }
    set_seed(0)
    model = DeepSVDDModel(**params, seed=0)
    model.fit(X_train_normal)
    return _pr_auc(model, X_val, y_val)


OBJECTIVES = {
    "isolation_forest": objective_isolation_forest,
    "lof": objective_lof,
    "one_class_svm": objective_one_class_svm,
    "autoencoder": objective_autoencoder,
    "vae": objective_vae,
    "deep_svdd": objective_deep_svdd,
}


# ---------------------------------------------------------------------------
# Best params → models.yaml format
# ---------------------------------------------------------------------------


def _build_best_params(model_name: str, best: dict, current: dict) -> dict:
    """Merge Optuna best params into the current config dict for this model.

    Converts hidden_dims back from the 'a-b-c' string format used in the
    search space to a proper list, and restores fixed params (epochs, device…)
    that were not tuned.
    """
    merged = dict(current)
    for k, v in best.items():
        if k == "hidden_dims":
            merged["hidden_dims"] = [int(x) for x in v.split("-")]
        else:
            merged[k] = v
    # Restore full epochs for the actual experiment runs
    if "epochs" in merged:
        merged["epochs"] = current.get("epochs", 50)
    return merged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune hyperparameters with Optuna.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--models-config", default="configs/models.yaml")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(OBJECTIVES.keys()),
        help="Subset of models to tune (default: all 6).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    models_cfg: dict = load_config(args.models_config)

    logger.info("Loading data …")
    df = load_raw(cfg["paths"]["data_raw"])
    X, y = preprocess(df)
    splits = make_splits(X, y, train_frac=cfg["split"]["train"], val_frac=cfg["split"]["val"], seed=0)
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    X_train_normal = X_train[y_train == 0]
    logger.info("  train_normal=%d  val=%d  (seed=0)", len(X_train_normal), len(X_val))

    best_params: dict[str, dict] = {}

    for model_name in args.models:
        if model_name not in OBJECTIVES:
            logger.warning("No tuning objective for '%s' — skipping.", model_name)
            continue

        n_trials = N_TRIALS[model_name]
        logger.info("Tuning %s (%d trials) …", model_name, n_trials)

        objective = OBJECTIVES[model_name]
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial, obj=objective: obj(trial, X_train_normal, X_val, y_val),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        best = study.best_params
        best_auc = study.best_value
        logger.info("  best PR-AUC=%.4f  params=%s", best_auc, best)
        best_params[model_name] = _build_best_params(model_name, best, models_cfg[model_name])

    # Merge best params into models_cfg and write back
    for model_name, params in best_params.items():
        models_cfg[model_name] = params

    models_config_path = Path(args.models_config)
    with open(models_config_path, "w") as f:
        yaml.dump(models_cfg, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    logger.info("models.yaml updated with best params.")
    logger.info("Next step: python scripts/run_experiment.py --experiment 1")


if __name__ == "__main__":
    main()
