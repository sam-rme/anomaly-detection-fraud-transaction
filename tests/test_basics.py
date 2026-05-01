from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.data import make_splits, split_indices
from src.models import IsolationForestModel, LogisticRegressionModel


# Shared fixtures
@pytest.fixture
def small_dataset() -> tuple[np.ndarray, np.ndarray]:
    """400 samples, 10 features, ~10% fraud — small enough to be fast."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_informative=5,
        weights=[0.9, 0.1],
        random_state=0,
    )
    return X.astype(np.float32), y.astype(np.int64)


@pytest.fixture
def fitted_iforest(small_dataset: tuple) -> IsolationForestModel:
    """IsolationForest trained on normal samples only."""
    X, y = small_dataset
    model = IsolationForestModel(n_estimators=10, seed=0)
    X_normal = X[y==0] 
    model.fit(X_normal)
    return model

# Test 1 — score_samples output shape and variability
def test_score_samples_shape(fitted_iforest: IsolationForestModel, small_dataset: tuple) -> None:
    """score_samples() must return a 1-D array of length n_samples, with non-constant values."""
    X, y = small_dataset
    scores = fitted_iforest.score_samples(X)
    assert scores.shape == (X.shape[0],)
    assert not np.all(scores == scores[0])


# Test 2 — predict() returns only binary values
def test_predict_binary(fitted_iforest: IsolationForestModel, small_dataset: tuple) -> None:
    """predict() must return an array containing only 0 and 1."""
    X, y = small_dataset
    scores = fitted_iforest.score_samples(X)

    #Choosing a random threshold for classification (not important for this test, just need to call predict)
    threshold = np.median(scores)  
    preds = fitted_iforest.predict(X, threshold=threshold)

    assert np.all((preds == 0) | (preds == 1)) 


# Test 3 — fraud samples score higher than normal samples on average
def test_higher_score_means_anomalous(small_dataset: tuple) -> None:
    """A model trained on normals should assign higher scores to fraud on average."""
    X, y = small_dataset

    # Séparer en train/test manuellement (utilise les indices, pas make_splits)
    X_train_normal = X[y == 0][:200]
    X_test = X[200:]
    y_test = y[200:]

    model = IsolationForestModel(n_estimators=50, seed=0)
    model.fit(X_train_normal)
    scores = model.score_samples(X_test)

    mean_normal = scores[y_test==0].mean()
    mean_fraud  = scores[y_test==1].mean()  

    assert mean_fraud > mean_normal


# Test 4 — split_indices produces non-overlapping sets
def test_split_no_leakage(small_dataset: tuple) -> None:
    """train, val and test index sets must be disjoint and cover the full dataset."""
    X, y = small_dataset

    idx_train, idx_val, idx_test = split_indices(y, train_frac=0.8, val_frac=0.1, seed=0)

    # Aucune intersection entre les trois splits
    assert len(np.intersect1d(idx_train, idx_val))  == 0 
    assert len(np.intersect1d(idx_train, idx_test)) == 0  
    assert len(np.intersect1d(idx_val,   idx_test)) == 0 

    # Les trois splits réunis couvrent bien tout le dataset
    all_indices = np.sort(np.concatenate([idx_train, idx_val, idx_test]))
    assert all_indices.shape[0] == len(y)  
