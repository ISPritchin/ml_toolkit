"""Общие fixtures/helpers для смоук-тестов пресетов ml_toolkit/presets/classification/high_pr_auc/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

BASE_PARAMS = {'iterations': 50, 'verbose': 0, 'random_seed': 42}


@pytest.fixture
def binary_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series((rng.random(n_train) < 0.15).astype(int))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series((rng.random(n_valid) < 0.15).astype(int))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def binary_data_with_cat():
    """binary_data + один категориальный признак (для тестов cat_features)."""
    rng = np.random.default_rng(7)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(4)]

    def _make(n, seed):
        r = np.random.default_rng(seed)
        X = pd.DataFrame(r.normal(size=(n, 4)), columns=cols)
        X['cat_col'] = r.choice(['a', 'b', 'c'], size=n)
        y = pd.Series((r.random(n) < 0.15).astype(int))
        return X, y

    X_train, y_train = _make(n_train, 70)
    X_valid, y_valid = _make(n_valid, 71)
    return X_train, y_train, X_valid, y_valid


def assert_valid_proba(model, X_valid):
    proba = model.predict_proba(X_valid)
    assert proba.shape == (len(X_valid),)
    assert np.all((proba >= 0) & (proba <= 1))
