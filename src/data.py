from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw(path: str | Path) -> pd.DataFrame:
    """Load raw CSV and return a DataFrame.

    Args:
        path: Path to creditcard.csv.

    Returns:
        Raw DataFrame with all original columns.
    """
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and labels from the raw DataFrame.

    Drops Time (uninformative post-PCA), log-transforms Amount to reduce skew.
    Does NOT scale — StandardScaler must be fit on training normals only
    to avoid leakage (see make_splits).

    Args:
        df: Raw DataFrame.

    Returns:
        Tuple (X, y) where X is float32 of shape (n, 29) and y is int64 of shape (n,).
    """
    df = df.copy()
    df["Amount"] = np.log1p(df["Amount"])
    df = df.drop(columns=["Time"])
    X = df.drop(columns=["Class"]).to_numpy(dtype=np.float32)
    y = df["Class"].to_numpy(dtype=np.int64)
    return X, y


def split_indices(
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test index split (no scaling, no feature access).

    Used both by make_splits and by experiments that need to align auxiliary
    arrays (e.g. transaction amounts) with the same split.

    Args:
        y: Labels (used for stratification).
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set (remainder goes to test).
        seed: Random seed.

    Returns:
        Tuple (idx_train, idx_val, idx_test).
    """
    test_frac = 1.0 - train_frac - val_frac
    idx = np.arange(len(y))
    idx_train, idx_temp = train_test_split(
        idx, train_size=train_frac, stratify=y, random_state=seed
    )
    val_relative = val_frac / (val_frac + test_frac)
    idx_val, idx_test = train_test_split(
        idx_temp, train_size=val_relative, stratify=y[idx_temp], random_state=seed
    )
    return idx_train, idx_val, idx_test


def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    train_frac: float,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val/test split with standard scaling.

    StandardScaler is fit on normal (y==0) training samples only, then applied
    to all splits to prevent label and data leakage.

    Args:
        X: Feature matrix of shape (n, d).
        y: Labels of shape (n,) where 0=normal, 1=fraud.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set (remainder goes to test).
        seed: Random seed.

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test), all scaled.
    """
    idx_train, idx_val, idx_test = split_indices(y, train_frac, val_frac, seed)
    X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    scaler = StandardScaler()
    scaler.fit(X_train[y_train == 0])
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return X_train, X_val, X_test, y_train, y_val, y_test


def subsample_fraud(
    X_train: np.ndarray,
    y_train: np.ndarray,
    fraud_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only a fraction of fraud examples in the training set.

    Used for Experiment 2 (imbalance robustness). Normal samples are untouched.

    Args:
        X_train: Training features.
        y_train: Training labels.
        fraud_fraction: Fraction of fraud examples to keep in [0.0, 1.0].
        seed: Random seed.

    Returns:
        Subsampled (X_train, y_train).
    """
    rng = np.random.default_rng(seed)
    fraud_idx = np.where(y_train == 1)[0]
    normal_idx = np.where(y_train == 0)[0]

    n_keep = int(len(fraud_idx) * fraud_fraction)
    kept_fraud = (
        rng.choice(fraud_idx, size=n_keep, replace=False)
        if n_keep > 0
        else np.array([], dtype=int)
    )

    all_idx = np.sort(np.concatenate([normal_idx, kept_fraud]))
    return X_train[all_idx], y_train[all_idx]
