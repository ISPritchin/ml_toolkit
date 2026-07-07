"""Общие fixtures для смоук-тестов пресетов ml_toolkit/presets/classification/multiclass_imbalance/."""

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
def multiclass_data():
    rng = np.random.default_rng(1)
    n_train, n_valid = 300, 80
    cols = [f'f{i}' for i in range(5)]
    probs = [0.6, 0.25, 0.1, 0.05]
    X_train = pd.DataFrame(rng.normal(size=(n_train, 5)), columns=cols)
    y_train = pd.Series(rng.choice([0, 1, 2, 3], size=n_train, p=probs))
    X_valid = pd.DataFrame(rng.normal(size=(n_valid, 5)), columns=cols)
    y_valid = pd.Series(rng.choice([0, 1, 2, 3], size=n_valid, p=probs))
    return X_train, y_train, X_valid, y_valid
