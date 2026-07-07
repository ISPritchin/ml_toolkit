"""Общие fixtures для tests/model_explainer/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Быстрые параметры CatBoost (без Optuna) — используются во всех фабриках пресетов.
FAST_CATBOOST_PARAMS = {
    'iterations': 40, 'max_depth': 3, 'learning_rate': 0.2,
    'verbose': 0, 'random_seed': 42,
}


@pytest.fixture
def classification_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 100
    cols = [f'f{i}' for i in range(5)]

    def _make(n):
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        logit = 1.4 * X['f0'] + 0.8 * X['f1'] - 0.3 * X['f2']
        proba = 1 / (1 + np.exp(-logit))
        y = pd.Series((rng.random(n) < proba * 0.5).astype(int))
        return X, y

    X_train, y_train = _make(n_train)
    X_valid, y_valid = _make(n_valid)
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(1)
    n_train, n_valid = 300, 100
    cols = [f'f{i}' for i in range(5)]

    def _make(n):
        X = pd.DataFrame(rng.normal(size=(n, 5)), columns=cols)
        y = pd.Series(2.0 * X['f0'] - 1.5 * X['f1'] + rng.normal(scale=0.3, size=n))
        return X, y

    X_train, y_train = _make(n_train)
    X_valid, y_valid = _make(n_valid)
    return X_train, y_train, X_valid, y_valid


def assert_valid_importance(imp: pd.Series, feature_names: list[str]) -> None:
    assert set(imp.index) == set(feature_names)
    assert not imp.isna().any()
    assert list(imp.index) == list(imp.sort_values(ascending=False).index)


def assert_valid_contribution(contrib: pd.Series, feature_names: list[str]) -> None:
    assert set(contrib.index) == set(feature_names)
    assert not contrib.isna().any()
