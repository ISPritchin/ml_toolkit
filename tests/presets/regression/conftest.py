"""Общие fixtures/helpers для смоук-тестов пресетов ml_toolkit/presets/regression/."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

BASE_PARAMS = {'iterations': 50, 'verbose': 0, 'random_seed': 42}


@pytest.fixture
def regression_data():
    rng = np.random.default_rng(0)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(X_train['f0'] * 2.0 - X_train['f1'] + rng.normal(scale=0.5, size=n_train))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(X_valid['f0'] * 2.0 - X_valid['f1'] + rng.normal(scale=0.5, size=n_valid))
    return X_train, y_train, X_valid, y_valid


@pytest.fixture
def positive_regression_data():
    """Строго положительный таргет — для Tweedie/лог-трансформов/относительных ошибок."""
    rng = np.random.default_rng(1)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(np.exp(0.5 * X_train['f0'] + rng.normal(scale=0.3, size=n_train)))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(np.exp(0.5 * X_valid['f0'] + rng.normal(scale=0.3, size=n_valid)))
    return X_train, y_train, X_valid, y_valid


def assert_valid_predictions(model, X_valid, y_valid=None):
    pred = model.predict(X_valid)
    assert pred.shape == (len(X_valid),)
    assert np.all(np.isfinite(pred))
    return pred
